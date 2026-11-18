"""Seeding, GPU setup and plotting helpers."""

from __future__ import annotations

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

from pathlib import Path
from typing import Optional


def set_global_seed(seed: int = 42) -> None:
    """Seed python, numpy and tensorflow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        keras.utils.set_random_seed(seed)
    except Exception:
        pass


def configure_gpu(memory_growth: bool = True) -> list:
    """Turn on GPU memory growth and report what is available.

    Without memory growth TensorFlow takes the whole GPU on first use, which
    blocks other processes. Growth allocates as needed instead.
    """
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print("No GPU detected, training will run on CPU.")
        return gpus

    if memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                # Growth has to be set before the device is first used
                print(f"  (memory growth not applied: {exc})")

    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        name = details.get("device_name", gpu.name)
        print(f"GPU available: {name}")

    return gpus


def load_oxford_flowers_metadata(dataset_dir: Path):
    """Read labels and splits from the Oxford Flowers 102 .mat files.

    Returns:
        labels: (8189,) 0-based class indices
        train_ids, valid_ids, test_ids: 1-based image IDs
    """
    from scipy.io import loadmat

    labels = loadmat(dataset_dir / "imagelabels.mat")["labels"].flatten()
    labels = labels.astype(np.int32) - 1

    setid = loadmat(dataset_dir / "setid.mat")
    train_ids = setid["trnid"].flatten().astype(np.int32)
    valid_ids = setid["valid"].flatten().astype(np.int32)
    test_ids = setid["tstid"].flatten().astype(np.int32)

    return labels, train_ids, valid_ids, test_ids


def image_id_to_path(images_dir: Path, image_id: int) -> str:
    """Map a 1-based image ID to its file path."""
    return str(images_dir / f"image_{image_id:05d}.jpg")


def plot_training_history(
    history_csv: Path,
    save_path: Optional[Path] = None,
    title: str = "Training History",
) -> Path:
    """Write accuracy and loss curves to a PNG.

    The Agg backend is forced so this also works on machines without a display,
    where a GUI backend would hang on plt.show().
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(history_csv)
    out_path = Path(save_path) if save_path else Path(history_csv).parent / "training_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(df["epoch"] + 1, df["accuracy"], label="Train", linewidth=1.5)
    axes[0].plot(df["epoch"] + 1, df["val_accuracy"], label="Validation", linewidth=1.5)
    axes[0].set_title(f"{title} - Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["epoch"] + 1, df["loss"], label="Train", linewidth=1.5)
    axes[1].plot(df["epoch"] + 1, df["val_loss"], label="Validation", linewidth=1.5)
    axes[1].set_title(f"{title} - Loss")
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
