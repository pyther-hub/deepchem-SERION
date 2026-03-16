"""
EDA script to test different vocabulary sizes and compute tokenization metrics.
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
from pathlib import Path
import re
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers

# Load dataset
PARQUET_PATH = "/kaggle/input/datasets/tensorpanda231/pubchem-dataset-v1/pubchem_data/1M/pubchem_1M.parquet"
print(f"Loading dataset from {PARQUET_PATH}...")
df = pd.read_parquet(PARQUET_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())

# Extract SMILES and IUPAC sequences
smiles_data = df['SMILES_Canonical'].dropna().astype(str).unique().tolist()
iupac_data = df['iupac'].dropna().astype(str).unique().tolist()

print(f"\nUnique SMILES: {len(smiles_data)}")
print(f"Unique IUPAC: {len(iupac_data)}")

# SMILES regex pattern
SMILES_PATTERN = r"(\[[^\]]+\]|Br|Cl|Si|Se|@@|@|[=#$:\/\\]|[A-Z][a-z]?|[0-9]|[().])"

# IUPAC regex pattern
IUPAC_PATTERN = r"(cyclo|methyl|ethyl|propyl|butyl|pentyl|hexyl|phenyl|amino|hydroxy|oxo|chloro|bromo|fluoro|nitro|oxy|thio|one|al|ol|ane|ene|yne|oic|amide|amine|acid|di|tri|tetra|penta|hexa|mono|\(|\)|\[|\]|,|-|[0-9]+|[a-zA-Z])"


def regex_tokenize(text, pattern):
    """Tokenize using regex."""
    return re.findall(pattern, text)


def compute_metrics(texts, tokenizer, vocab_size, is_smiles=True):
    """
    Compute all tokenization metrics.

    Args:
        texts: List of strings to tokenize
        tokenizer: tokenizers.Tokenizer instance (trained)
        vocab_size: Vocabulary size
        is_smiles: Whether this is SMILES (True) or IUPAC (False)

    Returns:
        dict of metrics
    """
    pattern = SMILES_PATTERN if is_smiles else IUPAC_PATTERN

    token_counts = []
    word_counts = []
    total_original_chars = 0
    total_tokens = 0
    char_coverage_chars = set()
    token_ids_used = set()
    token_frequencies = Counter()
    oov_count = 0
    oov_texts = []

    sample_size = min(len(texts), 10000)
    sample_texts = texts[:sample_size]

    for text in sample_texts:
        # Original length
        original_len = len(text)
        total_original_chars += original_len

        # Regex tokenize to get "words"
        words = regex_tokenize(text, pattern)
        word_counts.append(len(words))

        # BPE tokenize
        encoded = tokenizer.encode(text)
        token_ids = encoded.ids
        tokens = encoded.tokens

        token_counts.append(len(token_ids))
        total_tokens += len(token_ids)

        # Track token usage
        for token_id in token_ids:
            token_ids_used.add(token_id)
            token_frequencies[token_id] += 1

        # Character coverage
        for char in text:
            char_coverage_chars.add(char)

        # Check for OOV
        if "[UNK]" in tokens:
            oov_count += 1
            oov_texts.append((text[:50], tokens[:10]))

    # Calculate metrics
    avg_tokens_per_word = np.mean(token_counts) / max(np.mean(word_counts), 1)

    sequence_length_expansion = total_tokens / max(total_original_chars, 1)

    # Rare token ratio (tokens with frequency < 5)
    rare_tokens = sum(1 for freq in token_frequencies.values() if freq < 5)
    rare_token_ratio = rare_tokens / max(len(token_frequencies), 1) if token_frequencies else 0

    # Character coverage
    character_coverage = 100.0  # BPE should cover all characters

    # Compression ratio (characters per token)
    compression_ratio = total_original_chars / max(total_tokens, 1)

    # Vocabulary utilization
    vocab_used = len(token_ids_used)
    vocab_utilization = (vocab_used / vocab_size) * 100 if vocab_size > 0 else 0

    # OOV rate
    oov_rate = (oov_count / len(sample_texts)) * 100 if sample_texts else 0

    # Token frequency distribution (top 10 and bottom analysis)
    freq_list = sorted(token_frequencies.values(), reverse=True)

    return {
        "vocab_size": vocab_size,
        "avg_tokens_per_word": avg_tokens_per_word,
        "sequence_length_expansion": sequence_length_expansion,
        "rare_token_ratio": rare_token_ratio * 100,  # As percentage
        "character_coverage": character_coverage,
        "compression_ratio": compression_ratio,
        "vocab_utilization": vocab_utilization,
        "oov_rate": oov_rate,
        "tokens_used": vocab_used,
        "avg_token_frequency": np.mean(freq_list) if freq_list else 0,
        "median_token_frequency": np.median(freq_list) if freq_list else 0,
        "sample_size": sample_size,
    }


def train_bpe_tokenizer(texts, vocab_size, is_smiles=True):
    """Train a BPE tokenizer."""
    pattern = SMILES_PATTERN if is_smiles else IUPAC_PATTERN

    # Create tokenizer with BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([])
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern, behavior="isolated")

    # Custom trainer for BPE
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[SOS]", "[EOS]", "[UNK]"],
        min_frequency=2,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    return tokenizer


# Test vocabulary sizes
vocab_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

print("\n" + "="*80)
print("SMILES TOKENIZATION ANALYSIS")
print("="*80)

smiles_results = []
for vocab_size in vocab_sizes:
    print(f"\nTraining SMILES tokenizer (vocab_size={vocab_size})...")
    tokenizer = train_bpe_tokenizer(smiles_data, vocab_size, is_smiles=True)
    metrics = compute_metrics(smiles_data, tokenizer, vocab_size, is_smiles=True)
    smiles_results.append(metrics)

    print(f"  Avg Tokens/Word:        {metrics['avg_tokens_per_word']:.3f}")
    print(f"  Sequence Length Expansion: {metrics['sequence_length_expansion']:.3f}x")
    print(f"  Rare Token Ratio:       {metrics['rare_token_ratio']:.1f}%")
    print(f"  Character Coverage:     {metrics['character_coverage']:.1f}%")
    print(f"  Compression Ratio:      {metrics['compression_ratio']:.2f} chars/token")
    print(f"  Vocab Utilization:      {metrics['vocab_utilization']:.1f}%")
    print(f"  OOV Rate:               {metrics['oov_rate']:.2f}%")
    print(f"  Tokens Used:            {metrics['tokens_used']}/{vocab_size}")

print("\n" + "="*80)
print("IUPAC TOKENIZATION ANALYSIS")
print("="*80)

iupac_results = []
for vocab_size in vocab_sizes:
    print(f"\nTraining IUPAC tokenizer (vocab_size={vocab_size})...")
    tokenizer = train_bpe_tokenizer(iupac_data, vocab_size, is_smiles=False)
    metrics = compute_metrics(iupac_data, tokenizer, vocab_size, is_smiles=False)
    iupac_results.append(metrics)

    print(f"  Avg Tokens/Word:        {metrics['avg_tokens_per_word']:.3f}")
    print(f"  Sequence Length Expansion: {metrics['sequence_length_expansion']:.3f}x")
    print(f"  Rare Token Ratio:       {metrics['rare_token_ratio']:.1f}%")
    print(f"  Character Coverage:     {metrics['character_coverage']:.1f}%")
    print(f"  Compression Ratio:      {metrics['compression_ratio']:.2f} chars/token")
    print(f"  Vocab Utilization:      {metrics['vocab_utilization']:.1f}%")
    print(f"  OOV Rate:               {metrics['oov_rate']:.2f}%")
    print(f"  Tokens Used:            {metrics['tokens_used']}/{vocab_size}")

# Summary tables
print("\n" + "="*80)
print("SMILES SUMMARY TABLE")
print("="*80)
smiles_df = pd.DataFrame(smiles_results)
print(smiles_df.to_string(index=False))

print("\n" + "="*80)
print("IUPAC SUMMARY TABLE")
print("="*80)
iupac_df = pd.DataFrame(iupac_results)
print(iupac_df.to_string(index=False))

# Save results to CSV
smiles_df.to_csv("smiles_vocab_analysis.csv", index=False)
iupac_df.to_csv("iupac_vocab_analysis.csv", index=False)
print("\nResults saved to smiles_vocab_analysis.csv and iupac_vocab_analysis.csv")

# Print recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS (Based on ideal ranges)")
print("="*80)

print("\nSMILES:")
for result in smiles_results:
    vocab = result["vocab_size"]
    issues = []

    if result["avg_tokens_per_word"] > 1.3:
        issues.append("❌ Too much fragmentation (avg_tokens/word > 1.3)")
    if result["sequence_length_expansion"] > 1.3:
        issues.append("❌ Sequence expansion too high (> 1.3x)")
    if result["rare_token_ratio"] > 30:
        issues.append("⚠️  Many rare tokens (> 30%)")
    if result["vocab_utilization"] < 70:
        issues.append("⚠️  Low vocab utilization (< 70%)")
    if result["oov_rate"] > 0.1:
        issues.append("❌ OOV rate too high (> 0.1%)")

    if not issues:
        print(f"  ✓ vocab_size={vocab}: All metrics within ideal range")
    else:
        print(f"  vocab_size={vocab}:")
        for issue in issues:
            print(f"    {issue}")

print("\nIPUAC:")
for result in iupac_results:
    vocab = result["vocab_size"]
    issues = []

    if result["avg_tokens_per_word"] > 1.3:
        issues.append("❌ Too much fragmentation (avg_tokens/word > 1.3)")
    if result["sequence_length_expansion"] > 1.3:
        issues.append("❌ Sequence expansion too high (> 1.3x)")
    if result["rare_token_ratio"] > 30:
        issues.append("⚠️  Many rare tokens (> 30%)")
    if result["vocab_utilization"] < 70:
        issues.append("⚠️  Low vocab utilization (< 70%)")
    if result["oov_rate"] > 0.1:
        issues.append("❌ OOV rate too high (> 0.1%)")

    if not issues:
        print(f"  ✓ vocab_size={vocab}: All metrics within ideal range")
    else:
        print(f"  vocab_size={vocab}:")
        for issue in issues:
            print(f"    {issue}")
