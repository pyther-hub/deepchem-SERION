"""Shared utility functions: seeding, device selection, logging, timing, masks."""

import logging
import os
import random
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Set seeds for random, numpy, torch, and torch.cuda for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return cuda device if available, else cpu. Log which device is being used."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        logging.getLogger("smiles_iupac").info(
            f"Using device: cuda ({gpu_name}, {gpu_mem:.1f} GB)"
        )
    else:
        device = torch.device("cpu")
        logging.getLogger("smiles_iupac").info("Using device: cpu")
    return device


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(seconds: float) -> str:
    """Convert seconds to 'Xh Ym Zs' or 'Xm Ys' format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def create_dirs(*dirs: str) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def setup_logging(log_dir: str, name: str) -> logging.Logger:
    """Configure Python logging to file + console. Return the logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """Generate an upper-triangular causal mask for the Transformer decoder.

    Returns a float mask of shape (size, size) where masked positions are -inf
    and allowed positions are 0.0. Compatible with PyTorch's Transformer API.
    """
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


class Timer:
    """Simple context-manager timer."""

    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.time() - self.start

    def __str__(self) -> str:
        return format_time(self.elapsed)
