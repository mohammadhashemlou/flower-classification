"""Data loading, preprocessing, and augmentation pipeline.

Builds efficient tf.data.Dataset pipelines for the Oxford Flowers 102
dataset with optional data augmentation for training.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore

from pathlib import Path
from typing import Tuple

from .config import Config
from .utils import load_oxford_flowers_metadata, image_id_to_path


# ── Image preprocessing ──────────────────────────────────────────────


def load_and_preprocess_image(
    image_path: tf.Tensor,
    label: tf.Tensor,
    img_height: int = 224,
    img_width: int = 224,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read, decode, resize, and normalize a single image.

    Args:
        image_path: String tensor with the file path.
        label: Integer tensor with the class index (0-based).
        img_height: Target height in pixels.
        img_width: Target width in pixels.

    Returns:
        (image, label) tuple — image is float32 in [0, 1], shape (H, W, 3).
    """
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_pad(image, img_height, img_width)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)
    return image, label


# ── Augmentation ─────────────────────────────────────────────────────


def build_augmentation_layer(img_height: int = 224, img_width: int = 224) -> keras.Model:
    """Create a Keras data augmentation model for training.

    This is meant to be placed at the START of a model (not in the data pipeline),
    so it receives batched input of shape (batch, H, W, 3).

    Includes horizontal flip, rotation, zoom, and contrast shifts.
    """
    return keras.Sequential(
        [
            layers.Input(shape=(img_height, img_width, 3)),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
        ],
        name="data_augmentation",
    )


# ── Pipeline builder ────────────────────────────────────────────────


def create_data_pipeline(
    cfg: Config,
    split: str = "train",
    augment: bool = False,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset for one of the train / val / test splits.

    Args:
        cfg: Project Config instance.
        split: One of "train", "val", "test".
        augment: Whether to apply on-the-fly data augmentation (training only).

    Returns:
        A batched, prefetched tf.data.Dataset yielding (images, labels).
    """
    labels, train_ids, valid_ids, test_ids = load_oxford_flowers_metadata(cfg.dataset_dir)

    if split == "train":
        ids = train_ids
        y = labels[train_ids - 1]
    elif split == "val":
        ids = valid_ids
        y = labels[valid_ids - 1]
    elif split == "test":
        ids = test_ids
        y = labels[test_ids - 1]
    else:
        raise ValueError(f"Unknown split: {split!r}. Use 'train', 'val', or 'test'.")

    paths = [image_id_to_path(cfg.images_dir, int(i)) for i in ids]

    dataset = tf.data.Dataset.from_tensor_slices((paths, y.astype(np.int32)))

    if split == "train":
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=cfg.seed,
            reshuffle_each_iteration=True,
        )

    # Map preprocessing
    def _preprocess(path, label):
        return load_and_preprocess_image(path, label, cfg.img_height, cfg.img_width)

    dataset = dataset.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # NOTE: Augmentation is applied INSIDE the model (via build_augmentation_layer),
    # not in the data pipeline. This avoids shape issues with tf.data.Dataset.map
    # and is the recommended Keras approach.

    dataset = dataset.batch(cfg.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_all_pipelines(
    cfg: Config,
    augment_train: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Convenience: return (train_ds, val_ds, test_ds) in one call."""
    train_ds = create_data_pipeline(cfg, split="train", augment=augment_train)
    val_ds = create_data_pipeline(cfg, split="val", augment=False)
    test_ds = create_data_pipeline(cfg, split="test", augment=False)
    return train_ds, val_ds, test_ds


def compute_class_weights(cfg: Config) -> dict:
    """Compute class weights to handle class imbalance.

    Returns a dict mapping class index → weight, suitable for model.fit(class_weight=...).
    """
    labels, train_ids, _, _ = load_oxford_flowers_metadata(cfg.dataset_dir)
    train_labels = labels[train_ids - 1]

    unique, counts = np.unique(train_labels, return_counts=True)
    total = len(train_labels)
    num_classes = len(unique)

    weights = total / (num_classes * counts)
    return {int(cls): float(w) for cls, w in zip(unique, weights)}
