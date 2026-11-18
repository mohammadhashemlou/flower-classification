"""Tests for the data pipeline.

Run with:  pytest tests/ -v
Tests that need the dataset are skipped automatically when it is absent.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data_loader import (
    compute_class_weights,
    compute_class_weights_trainval,
    create_data_pipeline,
    create_trainval_pipelines,
)
from src.utils import load_oxford_flowers_metadata, set_global_seed


@pytest.fixture(scope="module")
def cfg():
    return Config()


@pytest.fixture(scope="module")
def dataset_available(cfg):
    return (cfg.dataset_dir / "imagelabels.mat").exists()


def test_config_defaults(cfg):
    """Config should expose sane defaults and create its output dir."""
    assert cfg.num_classes == 102
    assert cfg.img_height == cfg.img_width == 224
    assert cfg.num_channels == 3
    assert cfg.models_dir.exists()


def test_metadata_shapes(cfg, dataset_available):
    """Oxford Flowers 102 ships 8,189 images in a 1020/1020/6149 split."""
    if not dataset_available:
        pytest.skip("dataset not found")

    labels, train_ids, valid_ids, test_ids = load_oxford_flowers_metadata(cfg.dataset_dir)

    assert labels.shape == (8189,)
    assert labels.min() == 0 and labels.max() == 101
    assert len(train_ids) == 1020
    assert len(valid_ids) == 1020
    assert len(test_ids) == 6149

    # Splits must not overlap
    assert not set(train_ids) & set(valid_ids)
    assert not set(train_ids) & set(test_ids)
    assert not set(valid_ids) & set(test_ids)


def test_pipeline_batch_shape(cfg, dataset_available):
    """Batches should be (B, 224, 224, 3) float32 in [0, 1] with int32 labels."""
    if not dataset_available:
        pytest.skip("dataset not found")

    set_global_seed(cfg.seed)
    ds = create_data_pipeline(cfg, split="val", cache=False)

    images, labels = next(iter(ds))

    assert images.shape == (cfg.batch_size, cfg.img_height, cfg.img_width, 3)
    assert images.dtype == np.float32
    assert float(images.numpy().min()) >= 0.0
    assert float(images.numpy().max()) <= 1.0
    assert labels.dtype == np.int32
    assert int(labels.numpy().min()) >= 0
    assert int(labels.numpy().max()) < cfg.num_classes


def test_unknown_split_raises(cfg):
    """An unknown split name must fail loudly, not silently."""
    with pytest.raises(ValueError, match="Unknown split"):
        create_data_pipeline(cfg, split="validation")


def test_trainval_split_is_disjoint(cfg, dataset_available):
    """The merged train+val mode must not leak monitor images into training."""
    if not dataset_available:
        pytest.skip("dataset not found")

    train_ds, monitor_ds, test_ds = create_trainval_pipelines(cfg, val_fraction=0.1)

    def collect_labels(ds):
        out = []
        for _, labels in ds:
            out.extend(labels.numpy().tolist())
        return out

    # Cardinality checks: 2,040 merged -> 90/10, test untouched
    assert len(collect_labels(train_ds)) == 1836
    assert len(collect_labels(monitor_ds)) == 204
    assert len(collect_labels(test_ds)) == 6149


def test_class_weights_normalised(cfg, dataset_available):
    """Inverse-frequency weights should average to ~1."""
    if not dataset_available:
        pytest.skip("dataset not found")

    weights = compute_class_weights(cfg)
    assert len(weights) == 102
    assert all(w > 0 for w in weights.values())
    assert 0.9 < float(np.mean(list(weights.values()))) < 1.1


def test_class_weights_trainval_match_pool_size(cfg, dataset_available):
    """Weights for the merged pool should cover every class in the fit split."""
    if not dataset_available:
        pytest.skip("dataset not found")

    weights = compute_class_weights_trainval(cfg, val_fraction=0.1)
    assert len(weights) == 102
    assert all(w > 0 for w in weights.values())
