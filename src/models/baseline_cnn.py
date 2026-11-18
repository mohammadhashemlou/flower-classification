"""Baseline CNN trained from scratch.

Four conv blocks with batch norm and pooling, then global average pooling and
a small dense head. It overfits on this dataset but sets the floor to compare
the transfer learning models against.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from ..config import Config


def _conv_block(x, filters: int, dropout: float):
    """Two 3x3 convs with batch norm and ReLU, then pool and dropout."""
    for _ in range(2):
        x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return layers.Dropout(dropout)(x)


def build_baseline_cnn(cfg: Config) -> keras.Model:
    """Build the baseline CNN. Not compiled."""
    inputs = layers.Input(shape=(cfg.img_height, cfg.img_width, cfg.num_channels))

    x = _conv_block(inputs, 32, 0.20)
    x = _conv_block(x, 64, 0.25)
    x = _conv_block(x, 128, 0.30)
    x = _conv_block(x, 256, 0.35)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(cfg.num_classes, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")
