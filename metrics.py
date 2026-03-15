"""Validation and evaluation metrics for seq2seq models.

Provides token-level accuracy, exact match accuracy, and BLEU-4 score.
"""

from typing import List, Tuple
import torch
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

try:
    import nltk
    nltk.download("punkt", quiet=True)
except Exception:
    pass


def compute_token_level_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int = 0,
) -> float:
    """Compute token-level accuracy for teacher forcing.

    Args:
        logits: Model output logits of shape (batch * seq_len, vocab_size).
        targets: Target token IDs of shape (batch * seq_len).
        pad_id: Padding token ID to ignore.

    Returns:
        Token-level accuracy as a percentage (0-100).
    """
    predictions = torch.argmax(logits, dim=-1)
    mask = targets != pad_id

    if mask.sum() == 0:
        return 0.0

    correct = (predictions == targets) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    return accuracy * 100


def compute_exact_match_accuracy(
    predictions: List[str],
    targets: List[str],
) -> float:
    """Compute exact match accuracy (complete sequence accuracy).

    Args:
        predictions: List of predicted sequences (after decoding).
        targets: List of target sequences (after decoding).

    Returns:
        Exact match accuracy as a percentage (0-100).
    """
    if len(predictions) == 0:
        return 0.0

    matches = sum(
        p.strip() == t.strip()
        for p, t in zip(predictions, targets)
    )
    return (matches / len(predictions)) * 100


def compute_bleu_score(
    predictions: List[str],
    targets: List[str],
    max_n: int = 4,
) -> float:
    """Compute BLEU-4 score using NLTK.

    Args:
        predictions: List of predicted sequences.
        targets: List of target sequences (each as a single reference).
        max_n: Maximum n-gram order (default: 4 for BLEU-4).

    Returns:
        BLEU score as a percentage (0-100).
    """
    if len(predictions) == 0:
        return 0.0

    # Split sequences into tokens (character-level for SMILES/IUPAC)
    pred_tokens = [list(p.strip()) for p in predictions]
    target_tokens = [[list(t.strip())] for t in targets]  # NLTK expects list of references

    # Use smoothing function to handle zero counts
    smoothing_function = SmoothingFunction().method1

    try:
        bleu = corpus_bleu(
            target_tokens,
            pred_tokens,
            weights=tuple([1.0 / max_n] * max_n),
            smoothing_function=smoothing_function,
        )
        return bleu * 100
    except Exception:
        return 0.0


def compute_token_level_accuracy_sequences(
    predictions: List[str],
    targets: List[str],
) -> float:
    """Compute token-level accuracy for sequences after decoding.

    Compares character-by-character accuracy excluding padding.

    Args:
        predictions: List of predicted sequences (after decoding).
        targets: List of target sequences (after decoding).

    Returns:
        Token-level accuracy as a percentage (0-100).
    """
    if len(predictions) == 0:
        return 0.0

    total_tokens = 0
    correct_tokens = 0

    for pred, target in zip(predictions, targets):
        pred_str = pred.strip()
        target_str = target.strip()

        # Character-level comparison
        min_len = min(len(pred_str), len(target_str))
        correct_tokens += sum(
            p == t for p, t in zip(pred_str, target_str)
        )
        total_tokens += max(len(pred_str), len(target_str))

    if total_tokens == 0:
        return 0.0

    return (correct_tokens / total_tokens) * 100
