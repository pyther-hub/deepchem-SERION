"""Training and validation functions for seq2seq models.

Provides train_one_epoch, validate_one_epoch, and the Noam LR lambda.
Called from main.py which owns the training loop.
"""

import logging
import time
from typing import Callable, List, Tuple

import torch
import torch.nn as nn

from model import Seq2SeqTransformer
from metrics import (
    compute_token_level_accuracy,
    compute_exact_match_accuracy,
    compute_bleu_score,
    compute_token_level_accuracy_sequences,
)
from tokenizer import ChemBPETokenizer

logger = logging.getLogger("smiles_iupac")


def noam_lr_lambda(warmup_steps: int, d_model: int) -> Callable[[int], float]:
    """Return a lambda for Noam-style LR scheduling.

    Linear warmup for warmup_steps, then inverse square root decay.

    Args:
        warmup_steps: Number of warmup steps.
        d_model: Model dimension (used for scaling).

    Returns:
        Lambda function for LambdaLR.
    """

    def lr_lambda(step: int) -> float:
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

    return lr_lambda


def train_one_epoch(
    model: Seq2SeqTransformer,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: dict,
    epoch: int,
    num_epochs: int,
    verbose: bool = False,
) -> float:
    """Run a single training epoch.

    Args:
        model: The seq2seq model.
        train_loader: Training DataLoader.
        optimizer: Optimizer.
        scheduler: LR scheduler (stepped per batch).
        criterion: Loss function.
        scaler: GradScaler for mixed precision.
        device: Device.
        config: Configuration dictionary.
        epoch: Current epoch number (1-indexed).
        num_epochs: Total number of epochs.
        verbose: If True, log every batch (used for first epoch).

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    use_amp = device.type == "cuda"
    num_batches = len(train_loader)

    train_loss_sum = 0.0
    train_tokens = 0
    epoch_start = time.time()

    for batch_idx, batch in enumerate(train_loader, 1):
        step_start = time.time()

        src_ids = batch["src_ids"].to(device)
        tgt_input = batch["tgt_input_ids"].to(device)
        tgt_labels = batch["tgt_label_ids"].to(device)
        src_mask = batch["src_padding_mask"].to(device)
        tgt_mask = batch["tgt_padding_mask"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(src_ids, tgt_input, src_mask, tgt_mask)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_labels.reshape(-1),
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        non_pad = (tgt_labels != 0).sum().item()
        train_loss_sum += loss.item() * non_pad
        train_tokens += non_pad

        if verbose and batch_idx % 100 == 0:
            step_time = time.time() - step_start
            elapsed = time.time() - epoch_start
            logger.info(
                f"  Epoch {epoch:02d} | "
                f"Batch {batch_idx:>4d}/{num_batches} | "
                f"Loss: {loss.item():.4f} | "
                f"Step: {step_time:.2f}s | "
                f"Elapsed: {elapsed:.1f}s"
            )

    return train_loss_sum / max(train_tokens, 1)


def validate_one_epoch(
    model: Seq2SeqTransformer,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tgt_tokenizer: ChemBPETokenizer = None,
    src_tokenizer: ChemBPETokenizer = None,
) -> Tuple[float, dict]:
    """Run a single validation pass and compute metrics.

    Args:
        model: The seq2seq model.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: Device.
        tgt_tokenizer: Target tokenizer for decoding (optional).
        src_tokenizer: Source tokenizer for decoding (optional).

    Returns:
        Tuple of (average validation loss, metrics dict).
        Metrics dict contains: teacher_forcing_acc, exact_match_acc,
        partial_sentence_acc, bleu_score.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Metrics accumulation
    teacher_forcing_logits = []
    teacher_forcing_labels = []
    all_predictions: List[str] = []
    all_targets: List[str] = []

    with torch.no_grad():
        for batch in val_loader:
            src_ids = batch["src_ids"].to(device)
            tgt_input = batch["tgt_input_ids"].to(device)
            tgt_labels = batch["tgt_label_ids"].to(device)
            src_mask = batch["src_padding_mask"].to(device)
            tgt_mask = batch["tgt_padding_mask"].to(device)

            logits = model(src_ids, tgt_input, src_mask, tgt_mask)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_labels.reshape(-1),
            )

            non_pad = (tgt_labels != 0).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

            # Accumulate logits and labels for teacher forcing accuracy
            teacher_forcing_logits.append(logits.reshape(-1, logits.size(-1)))
            teacher_forcing_labels.append(tgt_labels.reshape(-1))

            # Decode predictions if tokenizers are provided
            if tgt_tokenizer is not None:
                pred_ids = model.greedy_decode(
                    src_ids=src_ids,
                    src_padding_mask=src_mask,
                    max_len=tgt_labels.size(1),
                    sos_id=tgt_tokenizer.sos_id,
                    eos_id=tgt_tokenizer.eos_id,
                    device=device,
                )

                # Decode to strings
                for i in range(src_ids.size(0)):
                    tgt_tokens = tgt_labels[i].tolist()
                    tgt_str = tgt_tokenizer.decode(tgt_tokens)
                    pred_str = tgt_tokenizer.decode(pred_ids[i])

                    all_targets.append(tgt_str)
                    all_predictions.append(pred_str)

    # Compute metrics
    metrics = {}

    # 1. Teacher forcing output accuracy (token-level from logits)
    if teacher_forcing_logits:
        logits_combined = torch.cat(teacher_forcing_logits, dim=0)
        labels_combined = torch.cat(teacher_forcing_labels, dim=0)
        metrics["teacher_forcing_acc"] = compute_token_level_accuracy(
            logits_combined, labels_combined, pad_id=0
        )
    else:
        metrics["teacher_forcing_acc"] = 0.0

    # 2. Complete accuracy (exact match on decoded sequences)
    if all_predictions:
        metrics["exact_match_acc"] = compute_exact_match_accuracy(
            all_predictions, all_targets
        )
    else:
        metrics["exact_match_acc"] = 0.0

    # 3. Partial sentence accuracy (token-level on decoded sequences)
    if all_predictions:
        metrics["partial_sentence_acc"] = compute_token_level_accuracy_sequences(
            all_predictions, all_targets
        )
    else:
        metrics["partial_sentence_acc"] = 0.0

    # 4. BLEU-4 score
    if all_predictions:
        metrics["bleu_score"] = compute_bleu_score(all_predictions, all_targets)
    else:
        metrics["bleu_score"] = 0.0

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, metrics
