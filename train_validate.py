"""Training and validation functions for seq2seq models.

Provides train_one_epoch, validate_one_epoch, and the Noam LR lambda.
Called from main.py which owns the training loop.
"""

import time
from typing import Callable, List, Tuple

import torch
import torch.nn as nn

from model import Seq2SeqTransformer
from metrics import (
    compute_exact_match_accuracy,
    compute_bleu_score,
    compute_token_level_accuracy_sequences,
)
from tokenizer import ChemBPETokenizer


def cosine_annealing_with_warmup(
    warmup_steps: int, total_steps: int
) -> Callable[[int], float]:
    """Return a lambda for cosine annealing LR scheduling with linear warmup.

    Linear warmup for warmup_steps, then cosine annealing decay to near-zero.

    Args:
        warmup_steps: Number of linear warmup steps.
        total_steps: Total number of training steps.

    Returns:
        Lambda function for LambdaLR.
    """
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

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
    """Run a single training epoch with gradient accumulation.

    Args:
        model: The seq2seq model.
        train_loader: Training DataLoader.
        optimizer: Optimizer.
        scheduler: LR scheduler (stepped per accumulation step).
        criterion: Loss function.
        scaler: GradScaler for mixed precision.
        device: Device.
        config: Configuration dictionary (must include 'grad_clip' and optionally 'gradient_accumulation_steps').
        epoch: Current epoch number (1-indexed).
        num_epochs: Total number of epochs.
        verbose: If True, log every batch (used for first epoch).

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    use_amp = device.type == "cuda"
    num_batches = len(train_loader)
    grad_accum_steps = config.get("gradient_accumulation_steps", 1)

    train_loss_sum = 0.0
    train_tokens = 0
    epoch_start = time.time()
    accum_loss = 0.0
    accum_tokens = 0

    for batch_idx, batch in enumerate(train_loader, 1):
        step_start = time.time()

        src_ids = batch["src_ids"].to(device)
        tgt_input = batch["tgt_input_ids"].to(device)
        tgt_labels = batch["tgt_label_ids"].to(device)
        src_mask = batch["src_padding_mask"].to(device)
        tgt_mask = batch["tgt_padding_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(src_ids, tgt_input, src_mask, tgt_mask)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_labels.reshape(-1),
            )

        # Scale loss for gradient accumulation
        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()

        non_pad = (tgt_labels != 0).sum().item()
        accum_loss += loss.item() * non_pad
        accum_tokens += non_pad
        train_loss_sum += loss.item() * non_pad
        train_tokens += non_pad

        # Optimizer step every grad_accum_steps batches
        if batch_idx % grad_accum_steps == 0 or batch_idx == num_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            accum_loss = 0.0
            accum_tokens = 0

        if verbose and batch_idx % 100 == 0:
            step_time = time.time() - step_start
            elapsed = time.time() - epoch_start
            print(
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
    compute_decode_metrics: bool = True,
) -> Tuple[float, dict]:
    """Run a single validation pass and compute metrics.

    Args:
        model: The seq2seq model.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: Device.
        tgt_tokenizer: Target tokenizer for decoding (optional).
        src_tokenizer: Source tokenizer for decoding (optional).
        compute_decode_metrics: If True, compute expensive decode metrics (exact match, BLEU).
                                If False, only compute teacher forcing accuracy (faster).

    Returns:
        Tuple of (average validation loss, metrics dict).
        Metrics dict contains: teacher_forcing_acc. If compute_decode_metrics=True,
        also includes: exact_match_acc, partial_sentence_acc, bleu_score.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Metrics accumulation — incremental to avoid OOM
    tf_correct = 0
    tf_total = 0
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

            # Compute teacher forcing accuracy incrementally (avoid OOM from accumulating logits)
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_labels = tgt_labels.reshape(-1)
            preds = torch.argmax(flat_logits, dim=-1)
            mask = flat_labels != 0
            tf_correct += ((preds == flat_labels) & mask).sum().item()
            tf_total += mask.sum().item()

            # Decode predictions if tokenizers are provided and metrics requested
            if tgt_tokenizer is not None and compute_decode_metrics:
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

    # 1. Teacher forcing output accuracy (token-level, computed incrementally)
    metrics["teacher_forcing_acc"] = (tf_correct / tf_total * 100) if tf_total > 0 else 0.0

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
