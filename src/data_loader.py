"""tf.data pipelines for Oxford Flowers 102."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore

from .config import Config
from .utils import image_id_to_path, load_oxford_flowers_metadata

VALID_SPLITS = ("train", "val", "test")


def decode_image_uint8(
    image_path: tf.Tensor,
    label: tf.Tensor,
    img_height: int = 224,
    img_width: int = 224,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Decode and resize one JPEG, keeping uint8.

    uint8 keeps the cached copy small enough to fit the whole dataset in RAM.
    """
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_pad(image, img_height, img_width)
    return image, tf.cast(label, tf.int32)


def build_augmentation_layer(img_height: int = 224, img_width: int = 224) -> keras.Model:
    """Augmentation layers, meant to sit at the start of a model.

    Takes a batch of (H, W, 3). No-ops at inference time.
    """
    return keras.Sequential(
        [
            layers.Input(shape=(img_height, img_width, 3)),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.2),
        ],
        name="data_augmentation",
    )


def _build_pipeline_from_ids(
    cfg: Config,
    ids: np.ndarray,
    labels: np.ndarray,
    shuffle: bool,
    cache: bool,
) -> tf.data.Dataset:
    """Decode, cache, shuffle, normalise, batch."""
    y = labels[ids - 1].astype(np.int32)
    paths = [image_id_to_path(cfg.images_dir, int(i)) for i in ids]

    dataset = tf.data.Dataset.from_tensor_slices((paths, y))
    dataset = dataset.map(
        lambda path, label: decode_image_uint8(path, label, cfg.img_height, cfg.img_width),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if cache:
        dataset = dataset.cache()

    # Shuffle after the cache so every epoch gets a new order without re-decoding.
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=cfg.seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda image, label: (tf.cast(image, tf.float32) / 255.0, label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)


def create_data_pipeline(
    cfg: Config,
    split: str = "train",
    cache: bool = True,
) -> tf.data.Dataset:
    """Build the pipeline for one split.

    Augmentation is not applied here. It lives inside the model so the exported
    checkpoint carries its own preprocessing.

    Args:
        cfg: Project config.
        split: "train", "val" or "test".
        cache: Keep decoded images in RAM.

    Raises:
        ValueError: On an unknown split name, before reading anything from disk.
    """
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Unknown split: {split!r}. Use one of: {', '.join(VALID_SPLITS)}."
        )

    labels, train_ids, valid_ids, test_ids = load_oxford_flowers_metadata(cfg.dataset_dir)
    ids = {"train": train_ids, "val": valid_ids, "test": test_ids}[split]

    return _build_pipeline_from_ids(
        cfg,
        ids=np.sort(ids),
        labels=labels,
        shuffle=(split == "train"),
        cache=cache,
    )


def create_all_pipelines(
    cfg: Config,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Return (train, val, test) pipelines."""
    train_ds = create_data_pipeline(cfg, split="train", cache=True)
    val_ds = create_data_pipeline(cfg, split="val", cache=True)
    test_ds = create_data_pipeline(cfg, split="test", cache=True)
    return train_ds, val_ds, test_ds


def create_trainval_pipelines(
    cfg: Config,
    val_fraction: float = 0.1,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Merge the train and val splits, holding out a slice for monitoring.

    The official training split has only 1,020 images, which is what limits
    accuracy. Merging val into it gives 2,040. A deterministic slice is kept
    aside for early stopping. The test split is never used here.

    Returns:
        (train_ds, monitor_ds, test_ds)
    """
    labels, train_ids, valid_ids, test_ids = load_oxford_flowers_metadata(cfg.dataset_dir)

    merged = np.concatenate([train_ids, valid_ids])
    rng = np.random.default_rng(cfg.seed)
    merged = merged[rng.permutation(len(merged))]

    n_monitor = max(1, int(round(len(merged) * val_fraction)))
    monitor_ids = np.sort(merged[:n_monitor])
    fit_ids = np.sort(merged[n_monitor:])

    train_ds = _build_pipeline_from_ids(cfg, fit_ids, labels, shuffle=True, cache=True)
    monitor_ds = _build_pipeline_from_ids(cfg, monitor_ids, labels, shuffle=False, cache=True)
    test_ds = _build_pipeline_from_ids(cfg, test_ids, labels, shuffle=False, cache=True)

    print(
        f"train+val mode: {len(fit_ids)} train / {len(monitor_ids)} monitor "
        f"/ {len(test_ids)} test images"
    )
    return train_ds, monitor_ds, test_ds


def compute_class_weights(cfg: Config) -> dict:
    """Inverse-frequency class weights for the training split."""
    labels, train_ids, _, _ = load_oxford_flowers_metadata(cfg.dataset_dir)
    train_labels = labels[train_ids - 1]

    unique, counts = np.unique(train_labels, return_counts=True)
    weights = len(train_labels) / (len(unique) * counts)
    return {int(cls): float(w) for cls, w in zip(unique, weights)}


def compute_class_weights_trainval(cfg: Config, val_fraction: float = 0.1) -> dict:
    """Inverse-frequency class weights for the merged train+val pool."""
    labels, train_ids, valid_ids, _ = load_oxford_flowers_metadata(cfg.dataset_dir)

    merged = np.concatenate([train_ids, valid_ids])
    rng = np.random.default_rng(cfg.seed)
    merged = merged[rng.permutation(len(merged))]

    n_monitor = max(1, int(round(len(merged) * val_fraction)))
    fit_labels = labels[merged[n_monitor:] - 1]

    unique, counts = np.unique(fit_labels, return_counts=True)
    weights = len(fit_labels) / (len(unique) * counts)
    return {int(cls): float(w) for cls, w in zip(unique, weights)}
