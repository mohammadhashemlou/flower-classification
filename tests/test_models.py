"""Tests for model construction, serialization, and transfer-learning plumbing.

Run with:  pytest tests/ -v
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tensorflow import keras

from src.config import Config
from src.models.baseline_cnn import build_baseline_cnn
from src.models.residual_cnn import build_residual_cnn
from src.models.transfer_learning import build_transfer_model, unfreeze_backbone
from src.training import (
    LabelSmoothingSparseCategoricalCrossentropy,
    build_cosine_lr_schedule,
    compile_model,
)


@pytest.fixture(scope="module")
def cfg():
    return Config()


def test_baseline_builds(cfg):
    model = build_baseline_cnn(cfg)
    assert model.output_shape == (None, cfg.num_classes)
    assert model.count_params() > 0


def test_residual_builds_with_and_without_augmentation(cfg):
    plain = build_residual_cnn(cfg, augment=False)
    augmented = build_residual_cnn(cfg, augment=True)

    assert plain.output_shape == (None, cfg.num_classes)
    assert augmented.count_params() == plain.count_params()
    # Augmentation layers carry no weights, so only the graph differs
    assert any(l.name == "data_augmentation" for l in augmented.layers)


def test_transfer_compact_head_is_smaller(cfg):
    """The default compact head should have far fewer params than a wide one."""
    compact = build_transfer_model(cfg, freeze_backbone=True, augment=False)
    wide = build_transfer_model(cfg, freeze_backbone=True, augment=False, head_units=512)

    assert compact.count_params() < wide.count_params()
    assert compact.output_shape == (None, cfg.num_classes)


def test_transfer_input_scaling_layer(cfg):
    """EfficientNet expects [0, 255]; the model must rescale from [0, 1]."""
    model = build_transfer_model(cfg, freeze_backbone=True, augment=False)
    names = [layer.name for layer in model.layers]
    assert "to_0_255" in names


def test_unknown_backbone_raises(cfg):
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_transfer_model(cfg, backbone="not_a_backbone")


def test_unfreeze_backbone_changes_trainable_count(cfg):
    model = build_transfer_model(cfg, freeze_backbone=True, augment=False)
    before = len(model.trainable_weights)

    unfreeze_backbone(model, num_layers_to_unfreeze=20, verbose=False)
    after = len(model.trainable_weights)

    assert after > before


def test_label_smoothing_loss_roundtrip(cfg):
    """The custom loss must survive .keras serialization without custom_objects."""
    model = build_transfer_model(cfg, freeze_backbone=True, augment=False)
    compile_model(model, cfg, label_smoothing=0.1)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.keras"
        model.save(path)
        restored = keras.models.load_model(path)  # no custom_objects on purpose

    assert isinstance(
        restored.loss, LabelSmoothingSparseCategoricalCrossentropy
    )
    assert restored.loss.smoothing == pytest.approx(0.1)
    assert restored.loss.num_classes == cfg.num_classes


def test_label_smoothing_loss_math(cfg):
    """Smoothing must add a penalty even for a perfectly confident prediction."""
    smoothed = LabelSmoothingSparseCategoricalCrossentropy(cfg.num_classes, 0.1)
    plain = LabelSmoothingSparseCategoricalCrossentropy(cfg.num_classes, 0.0)

    perfect = np.zeros((2, cfg.num_classes), dtype="float32")
    perfect[:, 0] = 1.0
    y = np.array([0, 0], dtype="int32")

    loss_smoothed = float(keras.backend.get_value(smoothed(y, perfect)))
    loss_plain = float(keras.backend.get_value(plain(y, perfect)))

    assert loss_plain == pytest.approx(0.0, abs=1e-6)
    assert loss_smoothed > loss_plain
    # ~ -101 * (0.1/102) * log(1e-7) with the 1e-7 clip
    assert 1.0 < loss_smoothed < 2.5


def test_cosine_schedule_decays(cfg):
    """The schedule must start at initial_lr and decay towards min_lr."""
    cfg.epochs = 10
    schedule = build_cosine_lr_schedule(cfg, steps_per_epoch=32)

    first = float(keras.backend.get_value(schedule(0)))
    last = float(keras.backend.get_value(schedule(10 * 32 - 1)))

    assert first == pytest.approx(cfg.initial_lr, rel=1e-3)
    assert last < first
    assert last >= cfg.min_lr * 0.5
