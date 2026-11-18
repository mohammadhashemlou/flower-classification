"""A small ResNet-style CNN trained from scratch."""

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
    """Two 3x3 convs plus a skip connection.

    A 1x1 projection is added to the shortcut whenever the block changes the
    channel count or downsamples.
    """
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    return x


def build_residual_cnn(cfg: Config, augment: bool = True) -> keras.Model:
    """Build the residual CNN. Not compiled."""
    inputs = keras.Input(
        shape=(cfg.img_height, cfg.img_width, cfg.num_channels),
        name="input",
    )

    x = inputs
    if augment:
        x = build_augmentation_layer(cfg.img_height, cfg.img_width)(x)

    x = layers.Conv2D(32, 3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = residual_block(x, filters=32, strides=1, dropout_rate=0.05)
    x = residual_block(x, filters=64, strides=2, dropout_rate=0.10)
    x = residual_block(x, filters=128, strides=2, dropout_rate=0.10)
    x = residual_block(x, filters=256, strides=2, dropout_rate=0.15)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(cfg.num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="residual_cnn")
