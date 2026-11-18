"""Residual CNN — a custom ResNet-style network built from scratch.

Uses residual blocks with projection shortcuts to handle dimension changes.
This architecture significantly improves gradient flow compared to the
baseline CNN, leading to better training stability and higher accuracy.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..config import Config
from ..data_loader import build_augmentation_layer


def residual_block(
    x: tf.Tensor,
    filters: int,
    strides: int = 1,
    dropout_rate: float = 0.0,
) -> tf.Tensor:
    """A standard residual block with two 3×3 convolutions and a skip connection.

    When the spatial dimensions or channel count change (strides > 1 or
    filter mismatch), a 1×1 projection convolution is applied to the shortcut.

    Args:
        x: Input tensor.
        filters: Number of output filters.
        strides: Convolution stride (downsampling when > 1).
        dropout_rate: Dropout applied after the residual addition.

    Returns:
        Output tensor after residual addition and ReLU.
    """
    shortcut = x

    # First conv: optional downsampling
    x = layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Second conv: keep spatial dimensions
    x = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Projection shortcut if dimensions don't match
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add skip connection
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    return x


def build_residual_cnn(cfg: Config, augment: bool = True) -> keras.Model:
    """Build the custom residual CNN model.

    Architecture:
        Input → [Augmentation] → Conv(32) → BN → ReLU
        → ResBlock(32,  s=1, p=0.05)
        → ResBlock(64,  s=2, p=0.10)
        → ResBlock(128, s=2, p=0.10)
        → ResBlock(256, s=2, p=0.15)
        → GAP → Dense(256) → BN → ReLU → Dropout(0.4) → Dense(102, softmax)

    Args:
        cfg: Project configuration.
        augment: Whether to include the augmentation layer in the model.

    Returns:
        A keras.Model (not yet compiled).
    """
    inputs = keras.Input(
        shape=(cfg.img_height, cfg.img_width, cfg.num_channels),
        name="input",
    )

    x = inputs
    if augment:
        x = build_augmentation_layer(cfg.img_height, cfg.img_width)(x)

    # Stem
    x = layers.Conv2D(32, 3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Residual stages
    x = residual_block(x, filters=32, strides=1, dropout_rate=0.05)
    x = residual_block(x, filters=64, strides=2, dropout_rate=0.10)
    x = residual_block(x, filters=128, strides=2, dropout_rate=0.10)
    x = residual_block(x, filters=256, strides=2, dropout_rate=0.15)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(cfg.num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="residual_cnn")
    return model
