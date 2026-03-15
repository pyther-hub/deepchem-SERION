"""SMILES <-> IUPAC Seq2Seq Translation Pipeline.

This is the ONLY entry point. All configuration is defined here.
Run with: python main.py
"""

import os

import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0: Environment Detection
# ═══════════════════════════════════════════════════════════════════════════════

ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
if ON_KAGGLE:
    print("Running on Kaggle")
    _BASE_DIR = "/kaggle/working/"
else:
    _BASE_DIR = ""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Configuration Dictionary
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
# SECTION 2: Pipeline Execution Flags
# ═══════════════════════════════════════════════════════════════════════════════

RUN_DATA_PREPARATION   = True
RUN_TOKENIZER_TRAINING = True

RUN_TRAIN_FORWARD      = True       # SMILES -> IUPAC
RUN_TRAIN_REVERSE      = True       # IUPAC -> SMILES

RUN_EVALUATE_FORWARD   = True
RUN_EVALUATE_REVERSE   = True

RUN_INFERENCE_EXAMPLES = True

# ── Test Run Override ────────────────────────────────────────
TEST_RUN = False  # Sanity check: tiny model, 1k samples, 5 epochs, CPU

if TEST_RUN:
    config.update({
        "max_samples":        1_000,
        "max_smiles_len":     100,
        "max_iupac_len":      150,
        "smiles_vocab_size":  500,
        "iupac_vocab_size":   500,
        "bpe_min_frequency":  1,
        "d_model":            32,
        "nhead":              2,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward":    64,
        "max_seq_len":        128,
        "batch_size":         16,
        "num_epochs":         5,
        "learning_rate":      1e-3,
        "warmup_steps":       50,
        "patience":           3,
        "tokenizer_dir":      os.path.join(_BASE_DIR, "tokenizers_test/"),
        "checkpoint_dir":     os.path.join(_BASE_DIR, "checkpoints_test/"),
        "log_dir":            os.path.join(_BASE_DIR, "logs_test/"),
        "results_dir":        os.path.join(_BASE_DIR, "results_test/"),
    })

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Pipeline Orchestration
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from utils import set_seed, get_device, setup_logging, create_dirs
    from dataset import download_and_prepare
    from tokenizer import ChemBPETokenizer
    from train_validate import train_model
    from evaluate import evaluate_model
    from inference import translate

    # ── Setup ─────────────────────────────────────────────
    set_seed(config["seed"])
    create_dirs(
        config["data_dir"],
        config["checkpoint_dir"],
        config["log_dir"],
        config["results_dir"],
        config["tokenizer_dir"],
    )
    logger = setup_logging(config["log_dir"], "smiles_iupac")
    device = get_device()

    df = None  # Will be set by data preparation step

    # ── Step 1: Data Preparation ──────────────────────────
    if RUN_DATA_PREPARATION:
        logger.info("=" * 60)
        logger.info("STEP 1: Data Preparation")
        logger.info("=" * 60)
        df = download_and_prepare(config)
        logger.info(f"Dataset ready: {len(df):,} samples")

    # ── Step 2: Train Tokenizers ──────────────────────────
    if RUN_TOKENIZER_TRAINING:
        logger.info("=" * 60)
        logger.info("STEP 2: Training BPE Tokenizers")
        logger.info("=" * 60)

        smiles_tok_path = os.path.join(config["tokenizer_dir"], "smiles.json")
        iupac_tok_path = os.path.join(config["tokenizer_dir"], "iupac.json")

        # Load filtered data if not already in memory
        if df is None:
            df = pd.read_parquet(
                os.path.join(config["data_dir"], "filtered.parquet")
            )

        # SMILES tokenizer
        if os.path.exists(smiles_tok_path):
            logger.info(f"SMILES tokenizer already exists: {smiles_tok_path}")
            smiles_tok = ChemBPETokenizer.load(smiles_tok_path)
        else:
            logger.info(
                f"Training SMILES tokenizer (vocab_size={config['smiles_vocab_size']})..."
            )
            smiles_tok = ChemBPETokenizer("smiles")
            smiles_tok.train(
                texts=df["SMILES_Canonical"].tolist(),
                vocab_size=config["smiles_vocab_size"],
                min_freq=config["bpe_min_frequency"],
            )
            smiles_tok.save(smiles_tok_path)
            logger.info(
                f"SMILES tokenizer saved (vocab_size={smiles_tok.vocab_size})"
            )

        # IUPAC tokenizer
        if os.path.exists(iupac_tok_path):
            logger.info(f"IUPAC tokenizer already exists: {iupac_tok_path}")
            iupac_tok = ChemBPETokenizer.load(iupac_tok_path)
        else:
            logger.info(
                f"Training IUPAC tokenizer (vocab_size={config['iupac_vocab_size']})..."
            )
            iupac_tok = ChemBPETokenizer("iupac")
            iupac_tok.train(
                texts=df["iupac"].tolist(),
                vocab_size=config["iupac_vocab_size"],
                min_freq=config["bpe_min_frequency"],
            )
            iupac_tok.save(iupac_tok_path)
            logger.info(
                f"IUPAC tokenizer saved (vocab_size={iupac_tok.vocab_size})"
            )

        logger.info("Tokenizers ready.")

    # ── Step 3: Train Forward Model (SMILES -> IUPAC) ────
    if RUN_TRAIN_FORWARD:
        logger.info("=" * 60)
        logger.info("STEP 3: Training SMILES -> IUPAC Model")
        logger.info("=" * 60)
        best_ckpt_fwd = train_model(
            direction="smiles2iupac", config=config, device=device
        )
        logger.info(f"Best forward checkpoint: {best_ckpt_fwd}")

    # ── Step 4: Train Reverse Model (IUPAC -> SMILES) ────
    if RUN_TRAIN_REVERSE:
        logger.info("=" * 60)
        logger.info("STEP 4: Training IUPAC -> SMILES Model")
        logger.info("=" * 60)
        best_ckpt_rev = train_model(
            direction="iupac2smiles", config=config, device=device
        )
        logger.info(f"Best reverse checkpoint: {best_ckpt_rev}")

    # ── Step 5: Evaluate Forward Model ────────────────────
    if RUN_EVALUATE_FORWARD:
        logger.info("=" * 60)
        logger.info("STEP 5: Evaluating SMILES -> IUPAC Model")
        logger.info("=" * 60)
        evaluate_model(direction="smiles2iupac", config=config, device=device)

    # ── Step 6: Evaluate Reverse Model ────────────────────
    if RUN_EVALUATE_REVERSE:
        logger.info("=" * 60)
        logger.info("STEP 6: Evaluating IUPAC -> SMILES Model")
        logger.info("=" * 60)
        evaluate_model(direction="iupac2smiles", config=config, device=device)

    # ── Step 7: Inference Examples ─────────────────────────
    if RUN_INFERENCE_EXAMPLES:
        logger.info("=" * 60)
        logger.info("STEP 7: Running Inference Examples")
        logger.info("=" * 60)

        test_smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CC(C)O", "C=CC=C"]
        logger.info("SMILES -> IUPAC translations:")
        for smi in test_smiles:
            result = translate(
                smi, direction="smiles2iupac", config=config, device=device
            )
            logger.info(f"  {smi:20s} -> {result}")

        test_iupac = ["ethanol", "acetic acid", "benzene", "propan-2-ol"]
        logger.info("IUPAC -> SMILES translations:")
        for name in test_iupac:
            result = translate(
                name, direction="iupac2smiles", config=config, device=device
            )
            logger.info(f"  {name:20s} -> {result}")

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)
