"""Test-set evaluation: exact match accuracy and sample predictions.

Loads the best checkpoint, runs greedy decoding on the test set,
and reports accuracy with sample predictions.
"""

import logging
import os
import random
from typing import List

import pandas as pd
import torch

from dataset import make_dataloaders
from model import Seq2SeqTransformer
from tokenizer import ChemBPETokenizer
from metrics import (
    compute_exact_match_accuracy,
    compute_bleu_score,
    compute_token_level_accuracy_sequences,
)

logger = logging.getLogger("smiles_iupac")


def _load_model_from_checkpoint(
    ckpt_path: str,
    src_vocab_size: int,
    tgt_vocab_size: int,
    config: dict,
    device: torch.device,
) -> Seq2SeqTransformer:
    """Load a model from a checkpoint file.

    Args:
        ckpt_path: Path to checkpoint .pt file.
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        config: Configuration dictionary.
        device: Device to load model onto.

    Returns:
        Model with loaded weights in eval mode.
    """
    model = Seq2SeqTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
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
    return model


def _get_tokenizers(
    direction: str, config: dict
) -> tuple[ChemBPETokenizer, ChemBPETokenizer]:
    """Load source and target tokenizers based on direction."""
    smiles_tok = ChemBPETokenizer.load(
        os.path.join(config["tokenizer_dir"], "smiles.json")
    )
    iupac_tok = ChemBPETokenizer.load(
        os.path.join(config["tokenizer_dir"], "iupac.json")
    )
    if direction == "smiles2iupac":
        return smiles_tok, iupac_tok
    else:
        return iupac_tok, smiles_tok


def evaluate_model(direction: str, config: dict, device: torch.device) -> None:
    """Evaluate a trained model on the test set.

    Computes exact match accuracy, logs sample predictions, and saves
    full predictions to CSV.

    Args:
        direction: 'smiles2iupac' or 'iupac2smiles'.
        config: Configuration dictionary.
        device: Device to run evaluation on.
    """
    # Load data and tokenizers
    df = pd.read_parquet(os.path.join(config["data_dir"], "filtered.parquet"))
    if len(df) > config["max_samples"]:
        df = df.head(config["max_samples"]).reset_index(drop=True)
        logger.info(f"Subsampled to {config['max_samples']:,} rows for this run")
    src_tok, tgt_tok = _get_tokenizers(direction, config)
    _, _, test_loader = make_dataloaders(df, direction, src_tok, tgt_tok, config)

    # Load model
    ckpt_path = os.path.join(config["checkpoint_dir"], direction, "best_model.pt")
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    model = _load_model_from_checkpoint(
        ckpt_path, src_tok.vocab_size, tgt_tok.vocab_size, config, device
    )
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # Run greedy decoding on test set
    all_sources: List[str] = []
    all_targets: List[str] = []
    all_predictions: List[str] = []

    for batch in test_loader:
        src_ids = batch["src_ids"].to(device)
        src_mask = batch["src_padding_mask"].to(device)
        tgt_labels = batch["tgt_label_ids"]

        # Decode predictions
        pred_ids = model.greedy_decode(
            src_ids=src_ids,
            src_padding_mask=src_mask,
            max_len=config["max_seq_len"],
            sos_id=tgt_tok.sos_id,
            eos_id=tgt_tok.eos_id,
            device=device,
        )

        # Decode token IDs back to strings
        for i in range(src_ids.size(0)):
            # Source string
            src_tokens = src_ids[i].tolist()
            src_str = src_tok.decode(src_tokens)

            # Target string (from labels, strip pad and eos)
            tgt_tokens = tgt_labels[i].tolist()
            tgt_str = tgt_tok.decode(tgt_tokens)

            # Prediction string
            pred_str = tgt_tok.decode(pred_ids[i])

            all_sources.append(src_str)
            all_targets.append(tgt_str)
            all_predictions.append(pred_str)

    # Compute metrics
    matches = [p.strip() == t.strip() for p, t in zip(all_predictions, all_targets)]
    correct = sum(matches)
    total = len(matches)

    # 1. Exact match accuracy (complete accuracy)
    exact_match_acc = compute_exact_match_accuracy(all_predictions, all_targets)

    # 2. Partial sentence accuracy (token-level)
    partial_sentence_acc = compute_token_level_accuracy_sequences(
        all_predictions, all_targets
    )

    # 3. BLEU-4 score
    bleu_score = compute_bleu_score(all_predictions, all_targets)

    logger.info("")
    logger.info("Test Set Metrics:")
    logger.info("-" * 40)
    logger.info(f"Complete (Exact Match) Accuracy: {exact_match_acc:.2f}% ({correct} / {total})")
    logger.info(f"Partial Sentence (Token-Level) Accuracy: {partial_sentence_acc:.2f}%")
    logger.info(f"BLEU-4 Score: {bleu_score:.2f}")

    # Log 20 random sample predictions
    logger.info("")
    header = f"| {'#':>2s} | {'Source':24s} | {'Target':28s} | {'Prediction':28s} | Match |"
    separator = f"|{'-' * 4}|{'-' * 26}|{'-' * 30}|{'-' * 30}|{'-' * 7}|"
    logger.info(header)
    logger.info(separator)

    sample_size = min(20, total)
    indices = random.sample(range(total), sample_size)
    for rank, idx in enumerate(indices, 1):
        src_trunc = all_sources[idx][:22]
        tgt_trunc = all_targets[idx][:26]
        pred_trunc = all_predictions[idx][:26]
        match_sym = "✓" if matches[idx] else "✗"
        logger.info(
            f"| {rank:2d} | {src_trunc:24s} | {tgt_trunc:28s} | {pred_trunc:28s} | {match_sym:>5s} |"
        )

    # Save full predictions to CSV
    os.makedirs(config["results_dir"], exist_ok=True)
    csv_path = os.path.join(config["results_dir"], f"{direction}_predictions.csv")
    results_df = pd.DataFrame(
        {
            "source": all_sources,
            "target": all_targets,
            "prediction": all_predictions,
            "match": matches,
        }
    )
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Predictions saved to {csv_path}")
