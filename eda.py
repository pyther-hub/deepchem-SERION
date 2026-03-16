"""
EDA (Exploratory Data Analysis) for SMILES ↔ IUPAC Dataset
Analyzes sequence lengths (character & token-level), vocabulary stats, and distributions
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import Counter

# ============================================================================
# SECTION 1: CONFIG & SETUP
# ============================================================================

CONFIG = {
    "data_dir": None,  # SET THIS: path to parquet files or raw data directory
    "smiles_vocab_size": 3000,
    "iupac_vocab_size": 5000,
    "seed": 42,
    "output_dir": "eda_output",
}

# ============================================================================
# SECTION 2: LOGGING & UTILITIES
# ============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to console and file."""
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("EDA")
    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(os.path.join(output_dir, "eda.log"))
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

# ============================================================================
# SECTION 3: DATA LOADING
# ============================================================================

def load_data(data_dir: str, logger: logging.Logger) -> pd.DataFrame:
    """Load data from parquet files."""
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all parquet files
    parquet_files = list(data_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    logger.info(f"Found {len(parquet_files)} parquet file(s)")

    # Load all parquet files
    dfs = []
    for pf in parquet_files:
        logger.info(f"Loading {pf.name}...")
        df = pd.read_parquet(pf)
        dfs.append(df)
        logger.info(f"  Loaded {len(df)} samples")

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total samples: {len(df)}")

    return df

def validate_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Validate and clean data."""
    logger.info("\n=== DATA VALIDATION ===")

    # Check required columns
    required_cols = ["SMILES_Canonical", "iupac"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        logger.info(f"Available columns: {list(df.columns)}")

    # Remove rows with missing SMILES or IUPAC
    initial_len = len(df)
    df = df.dropna(subset=["SMILES_Canonical", "iupac"])
    logger.info(f"Removed {initial_len - len(df)} rows with missing values")

    # Remove empty strings
    df = df[(df["SMILES_Canonical"].str.len() > 0) & (df["iupac"].str.len() > 0)]
    logger.info(f"After removing empty strings: {len(df)} samples")

    return df

# ============================================================================
# SECTION 4: CHARACTER-LEVEL ANALYSIS
# ============================================================================

def analyze_character_lengths(df: pd.DataFrame, logger: logging.Logger) -> Dict:
    """Analyze character-level sequence lengths."""
    logger.info("\n=== CHARACTER-LEVEL ANALYSIS ===")

    smiles_lengths = df["SMILES_Canonical"].str.len()
    iupac_lengths = df["iupac"].str.len()

    stats = {
        "smiles": {
            "min": smiles_lengths.min(),
            "max": smiles_lengths.max(),
            "mean": smiles_lengths.mean(),
            "median": smiles_lengths.median(),
            "std": smiles_lengths.std(),
            "25th": smiles_lengths.quantile(0.25),
            "75th": smiles_lengths.quantile(0.75),
            "95th": smiles_lengths.quantile(0.95),
        },
        "iupac": {
            "min": iupac_lengths.min(),
            "max": iupac_lengths.max(),
            "mean": iupac_lengths.mean(),
            "median": iupac_lengths.median(),
            "std": iupac_lengths.std(),
            "25th": iupac_lengths.quantile(0.25),
            "75th": iupac_lengths.quantile(0.75),
            "95th": iupac_lengths.quantile(0.95),
        }
    }

    # Log statistics
    logger.info("\nSMILES Character Lengths:")
    for key, val in stats["smiles"].items():
        logger.info(f"  {key:8s}: {val:.2f}")

    logger.info("\nIUPAC Character Lengths:")
    for key, val in stats["iupac"].items():
        logger.info(f"  {key:8s}: {val:.2f}")

    return stats, {"smiles": smiles_lengths, "iupac": iupac_lengths}

# ============================================================================
# SECTION 5: TOKENIZER & TOKEN-LEVEL ANALYSIS
# ============================================================================

def build_vocab(texts: List[str], vocab_size: int, regex_pattern: str) -> Dict[str, int]:
    """Build BPE-style vocabulary from texts (simplified)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

    # Create tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(regex_pattern, behavior="isolated"),
    ])

    # Train on texts (write to temp file)
    temp_file = "/tmp/texts.txt"
    with open(temp_file, "w") as f:
        for text in texts:
            f.write(text + "\n")

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],
        show_progress=False,
    )
    tokenizer.train([temp_file], trainer)

    # Extract vocab
    vocab = tokenizer.get_vocab()
    os.remove(temp_file)

    return vocab, tokenizer

def analyze_tokens(df: pd.DataFrame, logger: logging.Logger, config: Dict) -> Dict:
    """Tokenize data and analyze token-level statistics."""
    logger.info("\n=== TOKENIZER TRAINING & TOKEN-LEVEL ANALYSIS ===")

    # SMILES regex pattern
    smiles_pattern = r"(\[[^\]]+\]|Br|Cl|Si|Se|@@|@|[=#$:\/\\]|[A-Z][a-z]?|[0-9]|[().])"

    # IUPAC regex pattern
    iupac_pattern = r"(cyclo|methyl|ethyl|propyl|butyl|pentyl|hexyl|phenyl|amino|hydroxy|oxo|chloro|bromo|fluoro|nitro|oxy|thio|one|al|ol|ane|ene|yne|oic|amide|amine|acid|di|tri|tetra|penta|hexa|mono|\(|\)|\[|\]|,|-|[0-9]+|[a-zA-Z])"

    logger.info("Training SMILES tokenizer...")
    smiles_vocab, smiles_tokenizer = build_vocab(
        df["SMILES_Canonical"].tolist(),
        config["smiles_vocab_size"],
        smiles_pattern
    )

    logger.info("Training IUPAC tokenizer...")
    iupac_vocab, iupac_tokenizer = build_vocab(
        df["iupac"].tolist(),
        config["iupac_vocab_size"],
        iupac_pattern
    )

    # Tokenize all data
    smiles_token_lengths = []
    iupac_token_lengths = []

    for _, row in df.iterrows():
        smiles_enc = smiles_tokenizer.encode(row["SMILES_Canonical"])
        iupac_enc = iupac_tokenizer.encode(row["iupac"])
        smiles_token_lengths.append(len(smiles_enc.tokens))
        iupac_token_lengths.append(len(iupac_enc.tokens))

    smiles_token_lengths = np.array(smiles_token_lengths)
    iupac_token_lengths = np.array(iupac_token_lengths)

    # Vocab stats
    logger.info(f"\nSMILES Vocabulary Size: {len(smiles_vocab)}")
    logger.info(f"IUPAC Vocabulary Size: {len(iupac_vocab)}")

    # Token length stats
    stats = {
        "smiles": {
            "vocab_size": len(smiles_vocab),
            "min_tokens": smiles_token_lengths.min(),
            "max_tokens": smiles_token_lengths.max(),
            "mean_tokens": smiles_token_lengths.mean(),
            "median_tokens": np.median(smiles_token_lengths),
            "std_tokens": smiles_token_lengths.std(),
        },
        "iupac": {
            "vocab_size": len(iupac_vocab),
            "min_tokens": iupac_token_lengths.min(),
            "max_tokens": iupac_token_lengths.max(),
            "mean_tokens": iupac_token_lengths.mean(),
            "median_tokens": np.median(iupac_token_lengths),
            "std_tokens": iupac_token_lengths.std(),
        }
    }

    logger.info("\nSMILES Token Lengths:")
    for key, val in stats["smiles"].items():
        logger.info(f"  {key:15s}: {val}")

    logger.info("\nIUPAC Token Lengths:")
    for key, val in stats["iupac"].items():
        logger.info(f"  {key:15s}: {val}")

    return stats, {
        "smiles": smiles_token_lengths,
        "iupac": iupac_token_lengths,
    }, {
        "smiles": smiles_vocab,
        "iupac": iupac_vocab,
    }

# ============================================================================
# SECTION 6: VISUALIZATION & REPORTING
# ============================================================================

def create_visualizations(
    char_stats: Dict,
    char_lengths: Dict,
    token_stats: Dict,
    token_lengths: Dict,
    output_dir: str,
    logger: logging.Logger
) -> None:
    """Create plots and save as PNG."""
    os.makedirs(output_dir, exist_ok=True)

    sns.set_style("whitegrid")

    # ---- Figure 1: Character-level Length Distributions ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(char_lengths["smiles"], bins=50, alpha=0.7, color="blue", edgecolor="black")
    axes[0].set_xlabel("Character Length", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("SMILES Character Length Distribution", fontsize=13, fontweight="bold")
    axes[0].axvline(char_stats["smiles"]["mean"], color="red", linestyle="--", label=f"Mean: {char_stats['smiles']['mean']:.1f}")
    axes[0].axvline(char_stats["smiles"]["median"], color="green", linestyle="--", label=f"Median: {char_stats['smiles']['median']:.1f}")
    axes[0].legend()

    axes[1].hist(char_lengths["iupac"], bins=50, alpha=0.7, color="orange", edgecolor="black")
    axes[1].set_xlabel("Character Length", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("IUPAC Character Length Distribution", fontsize=13, fontweight="bold")
    axes[1].axvline(char_stats["iupac"]["mean"], color="red", linestyle="--", label=f"Mean: {char_stats['iupac']['mean']:.1f}")
    axes[1].axvline(char_stats["iupac"]["median"], color="green", linestyle="--", label=f"Median: {char_stats['iupac']['median']:.1f}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_character_length_distributions.png"), dpi=150, bbox_inches="tight")
    logger.info("Saved: 01_character_length_distributions.png")
    plt.close()

    # ---- Figure 2: Token-level Length Distributions ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(token_lengths["smiles"], bins=50, alpha=0.7, color="blue", edgecolor="black")
    axes[0].set_xlabel("Token Length", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("SMILES Token Length Distribution", fontsize=13, fontweight="bold")
    axes[0].axvline(token_stats["smiles"]["mean_tokens"], color="red", linestyle="--", label=f"Mean: {token_stats['smiles']['mean_tokens']:.1f}")
    axes[0].axvline(token_stats["smiles"]["median_tokens"], color="green", linestyle="--", label=f"Median: {token_stats['smiles']['median_tokens']:.1f}")
    axes[0].legend()

    axes[1].hist(token_lengths["iupac"], bins=50, alpha=0.7, color="orange", edgecolor="black")
    axes[1].set_xlabel("Token Length", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("IUPAC Token Length Distribution", fontsize=13, fontweight="bold")
    axes[1].axvline(token_stats["iupac"]["mean_tokens"], color="red", linestyle="--", label=f"Mean: {token_stats['iupac']['mean_tokens']:.1f}")
    axes[1].axvline(token_stats["iupac"]["median_tokens"], color="green", linestyle="--", label=f"Median: {token_stats['iupac']['median_tokens']:.1f}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_token_length_distributions.png"), dpi=150, bbox_inches="tight")
    logger.info("Saved: 02_token_length_distributions.png")
    plt.close()

    # ---- Figure 3: Character vs Token Length Scatter ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(char_lengths["smiles"], token_lengths["smiles"], alpha=0.3, s=10, color="blue")
    axes[0].set_xlabel("Character Length", fontsize=12)
    axes[0].set_ylabel("Token Length", fontsize=12)
    axes[0].set_title("SMILES: Character vs Token Length", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(char_lengths["iupac"], token_lengths["iupac"], alpha=0.3, s=10, color="orange")
    axes[1].set_xlabel("Character Length", fontsize=12)
    axes[1].set_ylabel("Token Length", fontsize=12)
    axes[1].set_title("IUPAC: Character vs Token Length", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_char_vs_token_scatter.png"), dpi=150, bbox_inches="tight")
    logger.info("Saved: 03_char_vs_token_scatter.png")
    plt.close()

    # ---- Figure 4: Box plots ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Character lengths
    axes[0, 0].boxplot([char_lengths["smiles"]], labels=["SMILES"])
    axes[0, 0].set_ylabel("Character Length", fontsize=11)
    axes[0, 0].set_title("SMILES Character Length (Box)", fontsize=12, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    axes[0, 1].boxplot([char_lengths["iupac"]], labels=["IUPAC"])
    axes[0, 1].set_ylabel("Character Length", fontsize=11)
    axes[0, 1].set_title("IUPAC Character Length (Box)", fontsize=12, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Token lengths
    axes[1, 0].boxplot([token_lengths["smiles"]], labels=["SMILES"])
    axes[1, 0].set_ylabel("Token Length", fontsize=11)
    axes[1, 0].set_title("SMILES Token Length (Box)", fontsize=12, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].boxplot([token_lengths["iupac"]], labels=["IUPAC"])
    axes[1, 1].set_ylabel("Token Length", fontsize=11)
    axes[1, 1].set_title("IUPAC Token Length (Box)", fontsize=12, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_box_plots.png"), dpi=150, bbox_inches="tight")
    logger.info("Saved: 04_box_plots.png")
    plt.close()

def create_summary_tables(
    char_stats: Dict,
    token_stats: Dict,
    vocabs: Dict,
    output_dir: str,
    logger: logging.Logger
) -> None:
    """Create and save summary tables as CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # ---- Table 1: Character-level Statistics ----
    char_df = pd.DataFrame({
        "Metric": ["Min", "Max", "Mean", "Median", "Std", "Q25", "Q75", "Q95"],
        "SMILES": [
            char_stats["smiles"]["min"],
            char_stats["smiles"]["max"],
            f"{char_stats['smiles']['mean']:.2f}",
            f"{char_stats['smiles']['median']:.2f}",
            f"{char_stats['smiles']['std']:.2f}",
            f"{char_stats['smiles']['25th']:.2f}",
            f"{char_stats['smiles']['75th']:.2f}",
            f"{char_stats['smiles']['95th']:.2f}",
        ],
        "IUPAC": [
            char_stats["iupac"]["min"],
            char_stats["iupac"]["max"],
            f"{char_stats['iupac']['mean']:.2f}",
            f"{char_stats['iupac']['median']:.2f}",
            f"{char_stats['iupac']['std']:.2f}",
            f"{char_stats['iupac']['25th']:.2f}",
            f"{char_stats['iupac']['75th']:.2f}",
            f"{char_stats['iupac']['95th']:.2f}",
        ]
    })

    char_df.to_csv(os.path.join(output_dir, "character_level_stats.csv"), index=False)
    logger.info("Saved: character_level_stats.csv")
    logger.info("\n" + char_df.to_string(index=False))

    # ---- Table 2: Token-level Statistics ----
    token_df = pd.DataFrame({
        "Metric": ["Vocab Size", "Min Tokens", "Max Tokens", "Mean Tokens", "Median Tokens", "Std Tokens"],
        "SMILES": [
            token_stats["smiles"]["vocab_size"],
            token_stats["smiles"]["min_tokens"],
            token_stats["smiles"]["max_tokens"],
            f"{token_stats['smiles']['mean_tokens']:.2f}",
            f"{token_stats['smiles']['median_tokens']:.2f}",
            f"{token_stats['smiles']['std_tokens']:.2f}",
        ],
        "IUPAC": [
            token_stats["iupac"]["vocab_size"],
            token_stats["iupac"]["min_tokens"],
            token_stats["iupac"]["max_tokens"],
            f"{token_stats['iupac']['mean_tokens']:.2f}",
            f"{token_stats['iupac']['median_tokens']:.2f}",
            f"{token_stats['iupac']['std_tokens']:.2f}",
        ]
    })

    token_df.to_csv(os.path.join(output_dir, "token_level_stats.csv"), index=False)
    logger.info("\n" + token_df.to_string(index=False))
    logger.info("Saved: token_level_stats.csv")

    # ---- Table 3: Top tokens by frequency ----
    # This would require re-encoding and counting; simplified for now
    vocab_df = pd.DataFrame({
        "Direction": ["SMILES", "IUPAC"],
        "Vocab Size": [
            token_stats["smiles"]["vocab_size"],
            token_stats["iupac"]["vocab_size"]
        ]
    })

    vocab_df.to_csv(os.path.join(output_dir, "vocabulary_summary.csv"), index=False)
    logger.info("\n" + vocab_df.to_string(index=False))
    logger.info("Saved: vocabulary_summary.csv")

# ============================================================================
# SECTION 7: MAIN EXECUTION
# ============================================================================

def main() -> None:
    """Main EDA execution."""
    # Setup
    set_seed(CONFIG["seed"])
    logger = setup_logging(CONFIG["output_dir"])

    logger.info("="*70)
    logger.info("SMILES ↔ IUPAC Dataset EDA")
    logger.info("="*70)

    # Validate config
    if CONFIG["data_dir"] is None:
        logger.error("ERROR: data_dir is not set in CONFIG")
        logger.error("Please update CONFIG['data_dir'] with path to parquet files")
        sys.exit(1)

    logger.info(f"\nData directory: {CONFIG['data_dir']}")
    logger.info(f"Output directory: {CONFIG['output_dir']}")

    # Load and validate data
    df = load_data(CONFIG["data_dir"], logger)
    df = validate_data(df, logger)

    # Character-level analysis
    char_stats, char_lengths = analyze_character_lengths(df, logger)

    # Token-level analysis & tokenizer training
    token_stats, token_lengths, vocabs = analyze_tokens(df, logger, CONFIG)

    # Visualizations
    create_visualizations(
        char_stats, char_lengths, token_stats, token_lengths,
        CONFIG["output_dir"], logger
    )

    # Summary tables
    create_summary_tables(char_stats, token_stats, vocabs, CONFIG["output_dir"], logger)

    logger.info("\n" + "="*70)
    logger.info("EDA Complete!")
    logger.info(f"Output saved to: {CONFIG['output_dir']}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
