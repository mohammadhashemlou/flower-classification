"""Baseline CNN — a simple from-scratch convolutional network.

Architecture:
    4 x [Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout]
    -> GlobalAveragePooling -> Dense(256) -> Dropout(0.5) -> Dense(102, softmax)

This model serves as the starting point. It overfits on the small dataset
but establishes a performance floor to improve upon.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from ..config import Config


def build_baseline_cnn(cfg: Config) -> keras.Model:
    """Build the baseline CNN model.

    Args:
        cfg: Project configuration.

    Returns:
        A keras.Model ready for compilation.
    """
    model = keras.Sequential(name="baseline_cnn")

    # Input
    model.add(layers.Input(shape=(cfg.img_height, cfg.img_width, cfg.num_channels)))

    # Block 1 - 32 filters
    model.add(layers.Conv2D(32, (3, 3), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(32, (3, 3), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Block 2 - 64 filters
    model.add(layers.Conv2D(64, (3, 3), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, (3, 3), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 3 - 128 filters
    model.add(layers.Conv2D(128, (3, 3), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(128, (3, 3), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Block 4 - 256 filters
    model.add(layers.Conv2D(256, (3, 3), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(256, (3, 3), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.35))

    # Classification head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(cfg.num_classes, activation="softmax"))

    return model
