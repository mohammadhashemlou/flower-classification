"""Utility helpers — seeding, plotting, logging."""

from __future__ import annotations

import os
import random
import numpy as np
import tensorflow as tf

from pathlib import Path
from typing import Optional


def set_global_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_oxford_flowers_metadata(dataset_dir: Path):
    """Load labels and train/val/test splits from the Oxford Flowers 102 .mat files.

    Returns:
        labels: np.ndarray of shape (8189,) — 0-based class indices
        train_ids, valid_ids, test_ids: np.ndarray of image IDs (1-based)
    """
    from scipy.io import loadmat

    labels = loadmat(dataset_dir / "imagelabels.mat")["labels"].flatten()
    labels = labels.astype(np.int32) - 1  # convert to 0-based

    setid = loadmat(dataset_dir / "setid.mat")
    train_ids = setid["trnid"].flatten().astype(np.int32)
    valid_ids = setid["valid"].flatten().astype(np.int32)
    test_ids = setid["tstid"].flatten().astype(np.int32)

    return labels, train_ids, valid_ids, test_ids


def image_id_to_path(images_dir: Path, image_id: int) -> str:
    """Convert a 1-based image ID to its file path string."""
    return str(images_dir / f"image_{image_id:05d}.jpg")


def plot_training_history(
    history_csv: Path,
    save_path: Optional[Path] = None,
    title: str = "Training History",
) -> None:
    """Plot accuracy & loss curves from a CSVLogger output file."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(history_csv)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(df["accuracy"], label="Train", linewidth=1.5)
    axes[0].plot(df["val_accuracy"], label="Validation", linewidth=1.5)
    axes[0].set_title(f"{title} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(df["loss"], label="Train", linewidth=1.5)
    axes[1].plot(df["val_loss"], label="Validation", linewidth=1.5)
    axes[1].set_title(f"{title} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
