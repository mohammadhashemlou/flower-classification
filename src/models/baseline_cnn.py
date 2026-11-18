"""Baseline CNN — a simple from-scratch convolutional network.

Architecture:
    4 × [Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool → Dropout]
    → GlobalAveragePooling → Dense(256) → Dropout(0.5) → Dense(102, softmax)

This model serves as the starting point. It overfits on the small dataset
but establishes a performance floor to improve upon.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..config import Config


def _conv_block(
    x: tf.Tensor,
    filters: int,
    dropout_rate: float,
) -> tf.Tensor:
    """Two Conv2D + BatchNorm + ReLU layers followed by MaxPool and Dropout."""
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def build_baseline_cnn(cfg: Config) -> keras.Model:
    """Build the baseline CNN model.

    Args:
        cfg: Project configuration.

    Returns:
        A compiled keras.Model ready for training.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(cfg.img_height, cfg.img_width, cfg.num_channels)),

            # Block 1 — 32 filters
            _conv_block_inline(32, 0.2),

            # Block 2 — 64 filters
            _conv_block_inline(64, 0.25),

            # Block 3 — 128 filters
            _conv_block_inline(128, 0.3),

            # Block 4 — 256 filters
            _conv_block_inline(256, 0.35),

            # Classification head
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(cfg.num_classes, activation="softmax"),
        ],
        name="baseline_cnn",
    )
    return model


def _conv_block_inline(filters: int, dropout_rate: float):
    """Return a callable that adds a conv block to a Sequential model."""
    def _apply(x):
        x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)
        return x

    return _apply
