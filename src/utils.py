"""Utility helpers — seeding, plotting, logging."""

from __future__ import annotations

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

from pathlib import Path
from typing import Optional


def set_global_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        keras.utils.set_random_seed(seed)
    except Exception:
        pass


def configure_gpu(memory_growth: bool = True) -> list:
    """Enable GPU memory growth and report available devices.

    Without memory growth TensorFlow grabs the entire GPU memory on first
    use, which makes the machine unusable and blocks other processes.
    Enabling growth allocates on demand instead.

    Args:
        memory_growth: Whether to enable per-device memory growth.

    Returns:
        List of physical GPU devices TensorFlow can see.
    """
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print("⚠  No GPU detected — training will run on CPU.")
        return gpus

    if memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                # Already initialised — growth must be set before first use
                print(f"  (memory growth not applied: {exc})")

    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        name = details.get("device_name", gpu.name)
        print(f"✓ GPU available: {name}")

    return gpus


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
) -> Path:
    """Plot accuracy & loss curves from a CSVLogger output file.

    Always renders to a PNG file. The non-interactive Agg backend is forced
    so this works in headless contexts (CI, servers, background runs) where
    a GUI backend would block forever on ``plt.show()``.

    Args:
        history_csv: Path to the ``history.csv`` written by CSVLogger.
        save_path: Output image path. Defaults to ``training_curves.png``
            next to the CSV.
        title: Plot title.

    Returns:
        The path the figure was written to.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(history_csv)
    out_path = Path(save_path) if save_path else Path(history_csv).parent / "training_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(df["epoch"] + 1, df["accuracy"], label="Train", linewidth=1.5)
    axes[0].plot(df["epoch"] + 1, df["val_accuracy"], label="Validation", linewidth=1.5)
    axes[0].set_title(f"{title} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(df["epoch"] + 1, df["loss"], label="Train", linewidth=1.5)
    axes[1].plot(df["epoch"] + 1, df["val_loss"], label="Validation", linewidth=1.5)
    axes[1].set_title(f"{title} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Training curves saved to: {out_path}")
    return out_path
