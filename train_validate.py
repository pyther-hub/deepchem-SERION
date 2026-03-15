"""Training loop and validation logic for seq2seq models.

Single file containing both training and validation. Called from main.py
with a direction parameter to train either SMILES->IUPAC or IUPAC->SMILES.
"""

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import make_dataloaders
from model import Seq2SeqTransformer, model_summary
from tokenizer import ChemBPETokenizer
from utils import Timer, format_time

logger = logging.getLogger("smiles_iupac")


def _get_tokenizers(
    direction: str, config: dict
) -> tuple[ChemBPETokenizer, ChemBPETokenizer]:
    """Load source and target tokenizers based on direction.

    Args:
        direction: 'smiles2iupac' or 'iupac2smiles'.
        config: Configuration dictionary.

    Returns:
        Tuple of (src_tokenizer, tgt_tokenizer).
    """
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


def _build_model(
    src_tokenizer: ChemBPETokenizer,
    tgt_tokenizer: ChemBPETokenizer,
    config: dict,
    device: torch.device,
) -> Seq2SeqTransformer:
    """Build and initialize the Seq2SeqTransformer model.

    Args:
        src_tokenizer: Source tokenizer.
        tgt_tokenizer: Target tokenizer.
        config: Configuration dictionary.
        device: Device to place model on.

    Returns:
        Model moved to device.
    """
    model = Seq2SeqTransformer(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_seq_len=config["max_seq_len"],
    )
    return model.to(device)


def _noam_lr_lambda(warmup_steps: int, d_model: int):
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


def validate_one_epoch(
    model: Seq2SeqTransformer,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run a single validation pass.

    Args:
        model: The seq2seq model.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: Device.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            src_ids = batch["src_ids"].to(device)
            tgt_input = batch["tgt_input_ids"].to(device)
            tgt_labels = batch["tgt_label_ids"].to(device)
            src_mask = batch["src_padding_mask"].to(device)
            tgt_mask = batch["tgt_padding_mask"].to(device)

            logits = model(src_ids, tgt_input, src_mask, tgt_mask)
            # logits: (batch, tgt_len, vocab) -> (batch*tgt_len, vocab)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_labels.reshape(-1),
            )

            # Count non-pad tokens for proper averaging
            non_pad = (tgt_labels != 0).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

    return total_loss / max(total_tokens, 1)


def train_model(direction: str, config: dict, device: torch.device) -> str:
    """Train a seq2seq model for the given direction.

    This is the main function called from main.py. It handles data loading,
    model creation, training loop with early stopping, and checkpointing.

    Args:
        direction: 'smiles2iupac' or 'iupac2smiles'.
        config: Configuration dictionary.
        device: Device to train on.

    Returns:
        Path to the best checkpoint file.
    """
    import pandas as pd

    # Load data and tokenizers
    df = pd.read_parquet(os.path.join(config["data_dir"], "filtered.parquet"))
    if len(df) > config["max_samples"]:
        df = df.head(config["max_samples"]).reset_index(drop=True)
        logger.info(f"Subsampled to {config['max_samples']:,} rows for this run")
    src_tok, tgt_tok = _get_tokenizers(direction, config)
    train_loader, val_loader, _ = make_dataloaders(
        df, direction, src_tok, tgt_tok, config
    )

    # Build model
    model = _build_model(src_tok, tgt_tok, config, device)
    summary = model_summary(model)
    logger.info(summary)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model: {total / 1e6:.1f}M parameters ({trainable / 1e6:.1f}M trainable)"
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_noam_lr_lambda(config["warmup_steps"], config["d_model"]),
    )

    # Loss
    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_tok.pad_id,
        label_smoothing=config["label_smoothing"],
    )

    # Checkpoint directory
    ckpt_dir = os.path.join(config["checkpoint_dir"], direction)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "best_model.pt")

    # Mixed precision
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training state
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config["num_epochs"] + 1):
        with Timer() as epoch_timer:
            # ── Train ─────────────────────────────────────
            model.train()
            train_loss_sum = 0.0
            train_tokens = 0

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch:02d}/{config['num_epochs']:02d}",
                leave=False,
            )
            for batch in progress:
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
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["grad_clip"]
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                non_pad = (tgt_labels != 0).sum().item()
                train_loss_sum += loss.item() * non_pad
                train_tokens += non_pad

                progress.set_postfix(loss=f"{loss.item():.3f}")

            train_loss = train_loss_sum / max(train_tokens, 1)

            # ── Validate ──────────────────────────────────
            val_loss = validate_one_epoch(model, val_loader, criterion, device)

        # ── Logging ───────────────────────────────────────
        current_lr = optimizer.param_groups[0]["lr"]
        is_best = val_loss < best_val_loss
        best_marker = "✓" if is_best else ""

        logger.info(
            f"Epoch {epoch:02d}/{config['num_epochs']:02d} | "
            f"Train Loss: {train_loss:.3f} | "
            f"Val Loss: {val_loss:.3f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_timer} | "
            f"Best: {best_marker}"
        )

        # ── Checkpointing ────────────────────────────────
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss if not is_best else val_loss,
            "config": config,
        }

        # Save epoch checkpoint
        epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch:02d}.pt")
        torch.save(checkpoint, epoch_path)

        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(checkpoint, best_ckpt_path)
        else:
            patience_counter += 1

        # ── Early Stopping ────────────────────────────────
        if patience_counter >= config["patience"]:
            logger.info(
                f"Early stopping at epoch {epoch}. "
                f"Best val loss: {best_val_loss:.3f}"
            )
            break

    if patience_counter < config["patience"]:
        logger.info(
            f"Training complete. Best val loss: {best_val_loss:.3f}"
        )

    return best_ckpt_path
