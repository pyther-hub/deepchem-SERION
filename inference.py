"""Inference module: load a trained checkpoint and translate arbitrary inputs.

Caches loaded models and tokenizers at module level so repeated calls
don't reload from disk every time.
"""

import logging
import os
from typing import Dict, List, Tuple

import torch

from model import Seq2SeqTransformer
from tokenizer import ChemBPETokenizer

logger = logging.getLogger("smiles_iupac")

# Module-level cache: {direction -> (model, src_tok, tgt_tok)}
_cache: Dict[str, Tuple[Seq2SeqTransformer, ChemBPETokenizer, ChemBPETokenizer]] = {}


def _load_or_cache(
    direction: str, config: dict, device: torch.device
) -> Tuple[Seq2SeqTransformer, ChemBPETokenizer, ChemBPETokenizer]:
    """Load model and tokenizers, using cache if available.

    Args:
        direction: 'smiles2iupac' or 'iupac2smiles'.
        config: Configuration dictionary.
        device: Device to load model onto.

    Returns:
        Tuple of (model, src_tokenizer, tgt_tokenizer).
    """
    cache_key = f"{direction}_{device}"
    if cache_key in _cache:
        return _cache[cache_key]

    # Load tokenizers
    smiles_tok = ChemBPETokenizer.load(
        os.path.join(config["tokenizer_dir"], "smiles.json")
    )
    iupac_tok = ChemBPETokenizer.load(
        os.path.join(config["tokenizer_dir"], "iupac.json")
    )

    if direction == "smiles2iupac":
        src_tok, tgt_tok = smiles_tok, iupac_tok
    else:
        src_tok, tgt_tok = iupac_tok, smiles_tok

    # Load model
    ckpt_path = os.path.join(config["checkpoint_dir"], direction, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = Seq2SeqTransformer(
        src_vocab_size=src_tok.vocab_size,
        tgt_vocab_size=tgt_tok.vocab_size,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_seq_len=config["max_seq_len"],
    )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded model for {direction} from {ckpt_path}")
    _cache[cache_key] = (model, src_tok, tgt_tok)
    return model, src_tok, tgt_tok


def translate(
    input_str: str, direction: str, config: dict, device: torch.device
) -> str:
    """Translate a single input string.

    Args:
        input_str: Source string (SMILES or IUPAC name).
        direction: 'smiles2iupac' or 'iupac2smiles'.
        config: Configuration dictionary.
        device: Device to run inference on.

    Returns:
        Translated string.
    """
    model, src_tok, tgt_tok = _load_or_cache(direction, config, device)

    # Tokenize input
    src_ids = src_tok.encode(input_str)
    src_ids = src_ids[: config["max_seq_len"]]

    # Create tensors
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = torch.zeros(1, len(src_ids), dtype=torch.bool, device=device)

    # Greedy decode
    pred_ids = model.greedy_decode(
        src_ids=src_tensor,
        src_padding_mask=src_mask,
        max_len=config["max_seq_len"],
        sos_id=tgt_tok.sos_id,
        eos_id=tgt_tok.eos_id,
        device=device,
    )

    return tgt_tok.decode(pred_ids[0])


def translate_batch(
    input_list: List[str], direction: str, config: dict, device: torch.device
) -> List[str]:
    """Translate a batch of input strings.

    Args:
        input_list: List of source strings.
        direction: 'smiles2iupac' or 'iupac2smiles'.
        config: Configuration dictionary.
        device: Device to run inference on.

    Returns:
        List of translated strings.
    """
    model, src_tok, tgt_tok = _load_or_cache(direction, config, device)

    # Tokenize all inputs
    all_src_ids = []
    max_len = 0
    for s in input_list:
        ids = src_tok.encode(s)[: config["max_seq_len"]]
        all_src_ids.append(ids)
        max_len = max(max_len, len(ids))

    # Pad to max length in batch
    padded = []
    masks = []
    for ids in all_src_ids:
        pad_len = max_len - len(ids)
        padded.append(ids + [src_tok.pad_id] * pad_len)
        masks.append([False] * len(ids) + [True] * pad_len)

    src_tensor = torch.tensor(padded, dtype=torch.long, device=device)
    src_mask = torch.tensor(masks, dtype=torch.bool, device=device)

    # Greedy decode
    pred_ids = model.greedy_decode(
        src_ids=src_tensor,
        src_padding_mask=src_mask,
        max_len=config["max_seq_len"],
        sos_id=tgt_tok.sos_id,
        eos_id=tgt_tok.eos_id,
        device=device,
    )

    return [tgt_tok.decode(ids) for ids in pred_ids]
