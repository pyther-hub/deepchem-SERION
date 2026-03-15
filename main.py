"""SMILES <-> IUPAC Seq2Seq Translation Pipeline.

Only entry point. All configuration is defined here.
Run with: python main.py
"""

import os

import pandas as pd
import torch
import torch.nn as nn

from utils import set_seed, get_device, setup_logging, create_dirs, Timer
from dataset import download_and_prepare, make_dataloaders
from tokenizer import ChemBPETokenizer
from model import Seq2SeqTransformer
from train_validate import train_one_epoch, validate_one_epoch, noam_lr_lambda
from evaluate import evaluate_model

# ═══════════════════════════════════════════════════════════════════════════════
# Environment Detection
# ═══════════════════════════════════════════════════════════════════════════════

ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
_BASE_DIR = "/kaggle/working/" if ON_KAGGLE else ""

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

config = {
    # ── Data ──────────────────────────────────────────────
    "dataset_id":        "hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC",
    "data_dir":          os.path.join(_BASE_DIR, "dataset/pubchem_data/"),
    "max_samples":       100_000,
    "max_smiles_len":    200,
    "max_iupac_len":     300,
    "val_split":         0.1,
    "test_split":        0.05,

    # ── Tokenizer ─────────────────────────────────────────
    "smiles_vocab_size": 3000,
    "iupac_vocab_size":  5000,
    "bpe_min_frequency": 2,
    "tokenizer_dir":     os.path.join(_BASE_DIR, "tokenizers/"),

    # ── Model ─────────────────────────────────────────────
    "d_model":           256,
    "nhead":             8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 6,
    "dim_feedforward":   1024,
    "dropout":           0.1,
    "max_seq_len":       512,

    # ── Training ──────────────────────────────────────────
    "batch_size":        64,
    "num_epochs":        30,
    "learning_rate":     3e-4,
    "warmup_steps":      4000,
    "grad_clip":         1.0,
    "label_smoothing":   0.1,
    "patience":          5,
    "seed":              42,

    # ── Paths ─────────────────────────────────────────────
    "checkpoint_dir":    os.path.join(_BASE_DIR, "checkpoints/"),
    "log_dir":           os.path.join(_BASE_DIR, "logs/"),
    "results_dir":       os.path.join(_BASE_DIR, "results/"),
}

# ═══════════════════════════════════════════════════════════════════════════════
# Run Settings
# ═══════════════════════════════════════════════════════════════════════════════

direction = "iupac2smiles"      # "iupac2smiles" or "smiles2iupac"
DOWNLOAD_DATA = True            # True: stream from HuggingFace | False: load parquet
TEST_RUN = False                # Tiny model for sanity checking

if TEST_RUN:
    config.update({
        "max_samples":        1_000,
        "smiles_vocab_size":  500,
        "iupac_vocab_size":   500,
        "d_model":            32,
        "nhead":              2,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward":    64,
        "batch_size":         16,
        "num_epochs":         5,
        "warmup_steps":       50,
        "patience":           3,
    })

# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Setup ─────────────────────────────────────────────────────────────────
    set_seed(config["seed"])
    create_dirs(
        config["data_dir"], config["checkpoint_dir"],
        config["log_dir"], config["results_dir"],
        config["tokenizer_dir"],
    )
    logger = setup_logging(config["log_dir"], "smiles_iupac")
    device = get_device()

    logger.info("=" * 60)
    logger.info("SMILES <-> IUPAC Seq2Seq Translation Pipeline")
    logger.info(f"Direction: {direction}")
    if TEST_RUN:
        logger.info("** TEST RUN — reduced config **")
    logger.info("=" * 60)

    # ── Step 1: Load Dataset ──────────────────────────────────────────────────
    logger.info("")
    logger.info("STEP 1: Load Dataset")
    logger.info("-" * 40)

    if DOWNLOAD_DATA:
        df = download_and_prepare(config)
    else:
        filtered_path = os.path.join(config["data_dir"], "filtered.parquet")
        df = pd.read_parquet(filtered_path)
        if len(df) > config["max_samples"]:
            df = df.head(config["max_samples"]).reset_index(drop=True)
        logger.info(f"Loaded {len(df):,} samples from {filtered_path}")

    logger.info(f"Dataset ready: {len(df):,} samples")

    # ── Step 2: Tokenizers ────────────────────────────────────────────────────
    logger.info("")
    logger.info("STEP 2: Tokenizers")
    logger.info("-" * 40)

    smiles_tok_path = os.path.join(config["tokenizer_dir"], "smiles.json")
    iupac_tok_path = os.path.join(config["tokenizer_dir"], "iupac.json")

    if os.path.exists(smiles_tok_path):
        smiles_tok = ChemBPETokenizer.load(smiles_tok_path)
        logger.info(f"Loaded SMILES tokenizer (vocab={smiles_tok.vocab_size})")
    else:
        logger.info(f"Training SMILES tokenizer (vocab_size={config['smiles_vocab_size']})...")
        smiles_tok = ChemBPETokenizer("smiles")
        smiles_texts = df["SMILES_Canonical"].dropna().tolist()
        logger.info(f"Filtered to {len(smiles_texts):,} valid SMILES (removed {df['SMILES_Canonical'].isna().sum()} NaN)")
        smiles_tok.train(
            texts=smiles_texts,
            vocab_size=config["smiles_vocab_size"],
            min_freq=config["bpe_min_frequency"],
        )
        smiles_tok.save(smiles_tok_path)
        logger.info(f"Saved SMILES tokenizer (vocab={smiles_tok.vocab_size})")

    if os.path.exists(iupac_tok_path):
        iupac_tok = ChemBPETokenizer.load(iupac_tok_path)
        logger.info(f"Loaded IUPAC tokenizer (vocab={iupac_tok.vocab_size})")
    else:
        logger.info(f"Training IUPAC tokenizer (vocab_size={config['iupac_vocab_size']})...")
        iupac_tok = ChemBPETokenizer("iupac")
        iupac_texts = df["iupac"].dropna().tolist()
        logger.info(f"Filtered to {len(iupac_texts):,} valid IUPAC (removed {df['iupac'].isna().sum()} NaN)")
        iupac_tok.train(
            texts=iupac_texts,
            vocab_size=config["iupac_vocab_size"],
            min_freq=config["bpe_min_frequency"],
        )
        iupac_tok.save(iupac_tok_path)
        logger.info(f"Saved IUPAC tokenizer (vocab={iupac_tok.vocab_size})")

    if direction == "smiles2iupac":
        src_tok, tgt_tok = smiles_tok, iupac_tok
    else:
        src_tok, tgt_tok = iupac_tok, smiles_tok

    # ── Step 3: DataLoaders ───────────────────────────────────────────────────
    logger.info("")
    logger.info("STEP 3: DataLoaders")
    logger.info("-" * 40)

    train_loader, val_loader, test_loader = make_dataloaders(
        df, direction, src_tok, tgt_tok, config
    )

    # ── Step 4: Model ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("STEP 4: Model")
    logger.info("-" * 40)

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
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters:     {total_params:>12,}")
    logger.info(f"Trainable parameters: {trainable_params:>12,}")
    logger.info(f"Model size:           ~{total_params * 4 / 1e6:.1f} MB (float32)")

    # ── Step 5: Training Setup & Details ──────────────────────────────────────
    logger.info("")
    logger.info("STEP 5: Training Details")
    logger.info("-" * 40)

    use_amp = device.type == "cuda"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=noam_lr_lambda(config["warmup_steps"], config["d_model"]),
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_tok.pad_id,
        label_smoothing=config["label_smoothing"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    logger.info(f"  Direction:       {direction}")
    logger.info(f"  Device:          {device}")
    logger.info(f"  Train batches:   {len(train_loader)}")
    logger.info(f"  Val batches:     {len(val_loader)}")
    logger.info(f"  Test batches:    {len(test_loader)}")
    logger.info(f"  Batch size:      {config['batch_size']}")
    logger.info(f"  Epochs:          {config['num_epochs']}")
    logger.info(f"  Learning rate:   {config['learning_rate']:.2e}")
    logger.info(f"  Warmup steps:    {config['warmup_steps']}")
    logger.info(f"  Label smoothing: {config['label_smoothing']}")
    logger.info(f"  Grad clip:       {config['grad_clip']}")
    logger.info(f"  Mixed precision: {use_amp}")
    logger.info(f"  Patience:        {config['patience']}")

    # ── Step 6: Training Loop ─────────────────────────────────────────────────
    logger.info("")
    logger.info("STEP 6: Training")
    logger.info("=" * 60)

    ckpt_dir = os.path.join(config["checkpoint_dir"], direction)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "best_model.pt")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config["num_epochs"] + 1):
        verbose = (epoch == 1)
        compute_full_metrics = (epoch >= 10)  # Full decode metrics only after epoch 10

        with Timer() as epoch_timer:
            train_loss = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                scaler=scaler,
                device=device,
                config=config,
                epoch=epoch,
                num_epochs=config["num_epochs"],
                verbose=verbose,
            )
            val_loss, val_metrics = validate_one_epoch(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                tgt_tokenizer=tgt_tok,
                src_tokenizer=src_tok,
                compute_decode_metrics=compute_full_metrics,
            )

        current_lr = optimizer.param_groups[0]["lr"]
        is_best = val_loss < best_val_loss

        logger.info(
            f"Epoch {epoch:02d}/{config['num_epochs']:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_timer}"
            f"{' | * Best' if is_best else ''}"
        )

        # Log metrics (decode metrics only available after epoch 10)
        if compute_full_metrics:
            logger.info(
                f"  Validation Metrics | "
                f"Teacher Forcing Acc: {val_metrics['teacher_forcing_acc']:.2f}% | "
                f"Complete Acc: {val_metrics['exact_match_acc']:.2f}% | "
                f"Partial Acc: {val_metrics['partial_sentence_acc']:.2f}% | "
                f"BLEU-4: {val_metrics['bleu_score']:.2f}"
            )
        else:
            logger.info(
                f"  Validation Metrics | "
                f"Teacher Forcing Acc: {val_metrics['teacher_forcing_acc']:.2f}%"
            )

        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "config": config,
            }, best_ckpt_path)
            logger.info(f"  Saved best model -> {best_ckpt_path}")
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            logger.info(
                f"Early stopping at epoch {epoch}. "
                f"Best val loss: {best_val_loss:.4f}"
            )
            break

    if patience_counter < config["patience"]:
        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

    # ── Step 7: Final Evaluation ──────────────────────────────────────────────
    logger.info("")
    logger.info("STEP 7: Final Evaluation on Test Set")
    logger.info("=" * 60)

    evaluate_model(direction=direction, config=config, device=device)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)
