"""Basic tests for the data loading pipeline."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.utils import set_global_seed, load_oxford_flowers_metadata


def test_config():
    """Config should create correct paths."""
    cfg = Config()
    assert cfg.num_classes == 102
    assert cfg.img_height == 224
    assert cfg.batch_size == 32
    assert cfg.models_dir.exists()
    print("✓ test_config passed")


def test_metadata_loading():
    """Oxford Flowers metadata should load with correct shapes."""
    cfg = Config()
    if not cfg.dataset_dir.exists():
        print("⚠ test_metadata_loading skipped (dataset not found)")
        return

    labels, train_ids, valid_ids, test_ids = load_oxford_flowers_metadata(cfg.dataset_dir)

    assert len(labels) == 8189, f"Expected 8189 labels, got {len(labels)}"
    assert labels.min() == 0, "Labels should be 0-based"
    assert labels.max() == 101, "Labels should max at 101"
    assert len(train_ids) == 1020, f"Expected 1020 train IDs, got {len(train_ids)}"
    assert len(valid_ids) == 1020, f"Expected 1020 val IDs, got {len(valid_ids)}"
    assert len(test_ids) == 6149, f"Expected 6149 test IDs, got {len(test_ids)}"
    print("✓ test_metadata_loading passed")


def test_seed():
    """set_global_seed should not raise."""
    set_global_seed(42)
    print("✓ test_seed passed")


if __name__ == "__main__":
    test_config()
    test_metadata_loading()
    test_seed()
    print("\nAll tests passed!")
