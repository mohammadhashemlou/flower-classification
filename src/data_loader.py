"""Data loading, preprocessing, and augmentation pipeline.

Builds efficient tf.data.Dataset pipelines for the Oxford Flowers 102
dataset with optional data augmentation for training.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore

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

    The transforms are deliberately strong. Oxford Flowers 102 has only
    ~10 training images per class, so the model memorises the training set
    within a few epochs unless the input distribution is aggressively
    perturbed. Geometry (flip / rotation / zoom / translation) plus
    photometric (contrast) jitter keeps the validation gap in check.

    All layers are no-ops at inference time, so they are safe to leave in
    the exported model.
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


# ── Pipeline builder ────────────────────────────────────────────────

VALID_SPLITS = ("train", "val", "test")


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

    Raises:
        ValueError: If ``split`` is not one of "train", "val", "test".
            This is checked *before* touching the dataset on disk, so a typo
            fails with a clear message instead of a confusing file error.
    """
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Unknown split: {split!r}. Use one of: {', '.join(VALID_SPLITS)}."
        )

    labels, train_ids, valid_ids, test_ids = load_oxford_flowers_metadata(cfg.dataset_dir)
    ids = {
        "train": train_ids,
        "val": valid_ids,
        "test": test_ids,
    }[split]

    return _build_pipeline_from_ids(
        cfg,
        ids=np.sort(ids),
        labels=labels,
        shuffle=(split == "train"),
        cache=cache,
    )


def create_all_pipelines(
    cfg: Config,
    augment_train: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Convenience: return (train_ds, val_ds, test_ds) in one call.

    Train, val, and test are all cached in RAM. At 224×224×3 uint8 that is
    ~150 KB per image: 1.0k train + 1.0k val + 6.1k test ≈ 1.2 GB, which is
    cheap compared to re-decoding 6k JPEGs on every evaluation pass.
    """
    train_ds = create_data_pipeline(cfg, split="train", augment=augment_train, cache=True)
    val_ds = create_data_pipeline(cfg, split="val", augment=False, cache=True)
    test_ds = create_data_pipeline(cfg, split="test", augment=False, cache=True)
    return train_ds, val_ds, test_ds


def compute_class_weights(cfg: Config) -> dict:
    """Compute class weights to handle class imbalance.

    Oxford Flowers 102 is fairly balanced, but a few classes have noticeably
    fewer training images. Weighting by inverse frequency stops those classes
    from being ignored.

    Args:
        cfg: Project configuration.

    Returns:
        A dict mapping class index -> weight, suitable for
        ``model.fit(class_weight=...)``.
    """
    labels, train_ids, _, _ = load_oxford_flowers_metadata(cfg.dataset_dir)
    train_labels = labels[train_ids - 1]

    unique, counts = np.unique(train_labels, return_counts=True)
    total = len(train_labels)
    num_classes = len(unique)

    weights = total / (num_classes * counts)
    return {int(cls): float(w) for cls, w in zip(unique, weights)}


def _build_pipeline_from_ids(
    cfg: Config,
    ids: np.ndarray,
    labels: np.ndarray,
    shuffle: bool,
    cache: bool,
) -> tf.data.Dataset:
    """Build a pipeline from explicit image IDs.

    Shared by ``create_data_pipeline`` and ``create_trainval_pipelines`` so
    both use exactly the same decode / cache / normalise / batch stages.
    """
    y = labels[ids - 1].astype(np.int32)
    paths = [image_id_to_path(cfg.images_dir, int(i)) for i in ids]

    dataset = tf.data.Dataset.from_tensor_slices((paths, y))

    dataset = dataset.map(
        lambda path, label: decode_image_uint8(path, label, cfg.img_height, cfg.img_width),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if cache:
        dataset = dataset.cache()

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


def create_trainval_pipelines(
    cfg: Config,
    val_fraction: float = 0.1,
    augment_train: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Train on the official train+val images, holding out a small monitor set.

    Oxford Flowers 102 ships only 1,020 training images for 102 classes
    (~10 per class). That is few enough that any reasonable model reaches
    100% training accuracy within a handful of epochs, which caps
    validation accuracy around 86%. Merging the official val split into
    training doubles the data to 2,040 images — the single largest accuracy
    gain available without outside data.

    A deterministic ``val_fraction`` slice of the merged set is held out
    purely to monitor training and drive early stopping / checkpointing.
    Final numbers must still be reported on the untouched official **test**
    split, which this function never touches.

    Args:
        cfg: Project configuration.
        val_fraction: Fraction of train+val reserved for monitoring.
        augment_train: Unused; augmentation lives inside the model.

    Returns:
        (train_ds, monitor_ds, test_ds)
    """
    labels, train_ids, valid_ids, test_ids = load_oxford_flowers_metadata(cfg.dataset_dir)

    merged = np.concatenate([train_ids, valid_ids])

    # Deterministic split of the merged pool
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(merged))
    merged = merged[order]

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


def compute_class_weights_trainval(cfg: Config, val_fraction: float = 0.1) -> dict:
    """Class weights for the merged train+val pool (monitor slice excluded)."""
    labels, train_ids, valid_ids, _ = load_oxford_flowers_metadata(cfg.dataset_dir)

    merged = np.concatenate([train_ids, valid_ids])
    rng = np.random.default_rng(cfg.seed)
    merged = merged[rng.permutation(len(merged))]

    n_monitor = max(1, int(round(len(merged) * val_fraction)))
    fit_ids = merged[n_monitor:]

    fit_labels = labels[fit_ids - 1]
    unique, counts = np.unique(fit_labels, return_counts=True)
    weights = len(fit_labels) / (len(unique) * counts)
    return {int(cls): float(w) for cls, w in zip(unique, weights)}
