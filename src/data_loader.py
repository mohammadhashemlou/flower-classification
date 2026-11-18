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


def decode_image_uint8(
    image_path: tf.Tensor,
    label: tf.Tensor,
    img_height: int = 224,
    img_width: int = 224,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Decode + resize an image but keep it as uint8.

    Keeping decoded images in uint8 uses 4x less memory than float32, which
    makes it practical to cache the whole train/val split in RAM. JPEG
    decoding is by far the most expensive part of the input pipeline, so
    caching it typically gives a 3-5x speedup on small datasets like
    Oxford Flowers 102 (~1k training images).

    Args:
        image_path: String tensor with the file path.
        label: Integer tensor with the class index (0-based).
        img_height: Target height in pixels.
        img_width: Target width in pixels.

    Returns:
        (image, label) tuple — image is uint8, shape (H, W, 3).
    """
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_pad(image, img_height, img_width)
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
    cache: bool = True,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset for one of the train / val / test splits.

    Pipeline stages, in order:

    1. ``read_file`` + ``decode_jpeg`` + ``resize`` → uint8  (expensive)
    2. ``cache()`` in RAM — done once, reused for every epoch
    3. ``shuffle()`` (train only) — reshuffles every epoch, after the cache
    4. cast to float32 and scale to [0, 1]
    5. ``batch()`` + ``prefetch()``

    Caching *before* the shuffle is deliberate: it keeps per-epoch
    reshuffling while avoiding repeated JPEG decodes.

    Args:
        cfg: Project Config instance.
        split: One of "train", "val", "test".
        augment: Reserved for future pipeline-level augmentation. Currently
            augmentation is applied inside the model, so this is unused.
        cache: Cache the decoded uint8 images in RAM. Disable for the full
            test split if memory is tight.

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

    # Stage 1 — expensive decode/resize, kept in uint8
    dataset = dataset.map(
        lambda p, l: decode_image_uint8(p, l, cfg.img_height, cfg.img_width),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Stage 2 — cache decoded images in RAM (no-op on later epochs)
    if cache:
        dataset = dataset.cache()

    # Stage 3 — shuffle after caching so each epoch sees a new order
    if split == "train":
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=cfg.seed,
            reshuffle_each_iteration=True,
        )

    # Stage 4 — convert to float32 in [0, 1]
    # NOTE: Augmentation is applied INSIDE the model (build_augmentation_layer),
    # so the cached pixels stay deterministic and reusable across epochs.
    dataset = dataset.map(
        lambda image, label: (tf.cast(image, tf.float32) / 255.0, label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(cfg.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_all_pipelines(
    cfg: Config,
    augment_train: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Convenience: return (train_ds, val_ds, test_ds) in one call.

    Train and val are cached in RAM (small splits, read many times).
    The test split is *not* cached — it holds ~6k images and is only
    iterated once, so caching it would waste memory for no gain.
    """
    train_ds = create_data_pipeline(cfg, split="train", augment=augment_train, cache=True)
    val_ds = create_data_pipeline(cfg, split="val", augment=False, cache=True)
    test_ds = create_data_pipeline(cfg, split="test", augment=False, cache=False)
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
