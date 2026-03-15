"""Data pipeline: download, filter, split, and create DataLoaders.

All functions accept a `config` dict. No internal configuration.
"""

import logging
import os
from typing import Callable, Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from tokenizer import ChemBPETokenizer

logger = logging.getLogger("smiles_iupac")


# ── Download & Prepare ────────────────────────────────────────────────────────


def download_and_prepare(config: dict) -> pd.DataFrame:
    """Download dataset from HuggingFace and apply filtering.

    Streams `config['max_samples']` rows from the hub, filters invalid rows,
    and saves to parquet. If filtered parquet already exists, loads from disk.

    Args:
        config: Configuration dictionary with keys: dataset_id, data_dir,
                max_samples, max_smiles_len, max_iupac_len.

    Returns:
        Filtered pandas DataFrame with columns: CID, SMILES_Canonical, iupac.
    """
    filtered_path = os.path.join(config["data_dir"], "filtered.parquet")

    if os.path.exists(filtered_path):
        logger.info(f"Filtered data already exists: '{filtered_path}', loading from disk.")
        df = pd.read_parquet(filtered_path)
        logger.info(f"Loaded {len(df):,} rows from '{filtered_path}'")
        if len(df) > config["max_samples"]:
            df = df.head(config["max_samples"]).reset_index(drop=True)
            logger.info(f"Subsampled to {config['max_samples']:,} rows for this run")
        return df

    raw_path = os.path.join(config["data_dir"], "raw.parquet")
    if os.path.exists(raw_path):
        logger.info(f"Raw data exists: '{raw_path}', loading from disk.")
        df = pd.read_parquet(raw_path)
        if len(df) > config["max_samples"]:
            df = df.head(config["max_samples"])
            logger.info(f"Subsampled raw data to {config['max_samples']:,} rows for this run")
    else:
        logger.info(f"Streaming {config['max_samples']:,} samples from HuggingFace...")
        from datasets import load_dataset

        os.makedirs(config["data_dir"], exist_ok=True)
        ds = load_dataset(config["dataset_id"], streaming=True)
        split = ds["train"].take(config["max_samples"])

        columns = ["CID", "SMILES_Canonical", "iupac"]
        records: List[Dict] = []
        for i, sample in enumerate(split):
            records.append({c: sample.get(c) for c in columns})
            if (i + 1) % 10_000 == 0:
                logger.info(f"  {i + 1:,} / {config['max_samples']:,}")

        df = pd.DataFrame(records)
        df.to_parquet(raw_path, index=False)
        logger.info(
            f"Saved {len(df):,} raw rows -> '{raw_path}' "
            f"({os.path.getsize(raw_path) / 1e6:.1f} MB)"
        )

    initial_count = len(df)

    # Drop rows with null/empty SMILES or IUPAC
    df = df.dropna(subset=["SMILES_Canonical", "iupac"])
    df = df[df["SMILES_Canonical"].str.strip().astype(bool)]
    df = df[df["iupac"].str.strip().astype(bool)]

    # Try RDKit validation
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        logger.info("RDKit found. Running molecular validation...")
        valid_indices = []
        canonical_smiles = []
        for idx, smi in df["SMILES_Canonical"].items():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mw = Descriptors.MolWt(mol)
            if mw >= 1500:
                continue
            canonical_smiles.append(Chem.MolToSmiles(mol))
            valid_indices.append(idx)

        df = df.loc[valid_indices].copy()
        df["SMILES_Canonical"] = canonical_smiles
    except ImportError:
        logger.warning(
            "RDKit not found. Skipping molecular validation. "
            "Install with: pip install rdkit-pypi"
        )

    # String-length filters
    df = df[df["SMILES_Canonical"].str.len() <= config["max_smiles_len"]]
    df = df[df["iupac"].str.len() <= config["max_iupac_len"]]
    df = df.reset_index(drop=True)

    logger.info(f"Filtered: {initial_count:,} -> {len(df):,} valid samples")
    df.to_parquet(filtered_path, index=False)
    logger.info(f"Dataset saved to '{filtered_path}'")

    return df


# ── Dataset Class ─────────────────────────────────────────────────────────────


class SMILESIUPACDataset(Dataset):
    """PyTorch Dataset for SMILES <-> IUPAC translation pairs.

    Args:
        df: DataFrame with columns SMILES_Canonical and iupac.
        direction: Either 'smiles2iupac' or 'iupac2smiles'.
    """

    def __init__(self, df: pd.DataFrame, direction: str) -> None:
        if direction not in ("smiles2iupac", "iupac2smiles"):
            raise ValueError(f"direction must be 'smiles2iupac' or 'iupac2smiles', got '{direction}'")
        self.df = df.reset_index(drop=True)
        self.direction = direction

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        if self.direction == "smiles2iupac":
            src, tgt = str(row["SMILES_Canonical"]), str(row["iupac"])
        else:
            src, tgt = str(row["iupac"]), str(row["SMILES_Canonical"])

        cid = int(row["CID"]) if pd.notna(row["CID"]) else -1
        return {"src": src, "tgt": tgt, "cid": cid}


# ── Collate Function ─────────────────────────────────────────────────────────


def create_collate_fn(
    src_tokenizer: ChemBPETokenizer,
    tgt_tokenizer: ChemBPETokenizer,
    max_seq_len: int,
) -> Callable:
    """Create a collate function that tokenizes, pads, and creates masks.

    Args:
        src_tokenizer: Tokenizer for source sequences.
        tgt_tokenizer: Tokenizer for target sequences.
        max_seq_len: Maximum token sequence length (truncate beyond this).

    Returns:
        A collate function for use with DataLoader.
    """

    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        src_ids_list: List[List[int]] = []
        tgt_input_list: List[List[int]] = []
        tgt_label_list: List[List[int]] = []

        for sample in batch:
            s_ids = src_tokenizer.encode(sample["src"])[:max_seq_len]
            t_ids = tgt_tokenizer.encode(sample["tgt"])[: max_seq_len - 2]

            # Decoder input: [<sos>, tok1, ..., tokN]
            tgt_input = [tgt_tokenizer.sos_id] + t_ids
            # Labels: [tok1, ..., tokN, <eos>]
            tgt_label = t_ids + [tgt_tokenizer.eos_id]

            src_ids_list.append(s_ids)
            tgt_input_list.append(tgt_input)
            tgt_label_list.append(tgt_label)

        # Pad to max length in this batch
        src_max = max(len(s) for s in src_ids_list)
        tgt_max = max(len(t) for t in tgt_input_list)

        src_padded = []
        src_mask = []
        tgt_in_padded = []
        tgt_lbl_padded = []
        tgt_mask = []

        for s_ids, t_in, t_lbl in zip(src_ids_list, tgt_input_list, tgt_label_list):
            s_pad_len = src_max - len(s_ids)
            t_pad_len = tgt_max - len(t_in)

            src_padded.append(s_ids + [src_tokenizer.pad_id] * s_pad_len)
            src_mask.append([False] * len(s_ids) + [True] * s_pad_len)

            tgt_in_padded.append(t_in + [tgt_tokenizer.pad_id] * t_pad_len)
            tgt_lbl_padded.append(t_lbl + [tgt_tokenizer.pad_id] * t_pad_len)
            tgt_mask.append([False] * len(t_in) + [True] * t_pad_len)

        return {
            "src_ids": torch.tensor(src_padded, dtype=torch.long),
            "tgt_input_ids": torch.tensor(tgt_in_padded, dtype=torch.long),
            "tgt_label_ids": torch.tensor(tgt_lbl_padded, dtype=torch.long),
            "src_padding_mask": torch.tensor(src_mask, dtype=torch.bool),
            "tgt_padding_mask": torch.tensor(tgt_mask, dtype=torch.bool),
        }

    return collate_fn


# ── DataLoader Creation ──────────────────────────────────────────────────────


def make_dataloaders(
    df: pd.DataFrame,
    direction: str,
    src_tokenizer: ChemBPETokenizer,
    tgt_tokenizer: ChemBPETokenizer,
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders.

    Args:
        df: Filtered DataFrame.
        direction: 'smiles2iupac' or 'iupac2smiles'.
        src_tokenizer: Source tokenizer.
        tgt_tokenizer: Target tokenizer.
        config: Configuration dictionary.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    dataset = SMILESIUPACDataset(df, direction)
    n = len(dataset)

    val_size = int(n * config["val_split"])
    test_size = int(n * config["test_split"])
    train_size = n - val_size - test_size

    generator = torch.Generator().manual_seed(config["seed"])
    train_subset, val_subset, test_subset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    collate_fn = create_collate_fn(src_tokenizer, tgt_tokenizer, config["max_seq_len"])
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=pin,
    )

    logger.info(
        f"DataLoaders ({direction}): "
        f"train={train_size:,}, val={val_size:,}, test={test_size:,}"
    )

    return train_loader, val_loader, test_loader
