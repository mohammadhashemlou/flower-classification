"""Transfer learning model using EfficientNetB0.

Instead of training a CNN from scratch (which struggles on the small
Oxford Flowers 102 dataset), we leverage ImageNet-pretrained features
from EfficientNetB0 and fine-tune a custom classification head.

This approach dramatically reduces overfitting and typically achieves
85-95% accuracy on this dataset.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

from ..config import Config
from ..data_loader import build_augmentation_layer


def build_transfer_model(
    cfg: Config,
    backbone: str = "efficientnetb0",
    freeze_backbone: bool = True,
    augment: bool = True,
    dropout_rate: float = 0.3,
    head_units: int = 512,
) -> keras.Model:
    """Build a transfer learning model on top of a pretrained backbone.

    Args:
        cfg: Project configuration.
        backbone: Name of the backbone ("efficientnetb0" or "resnet50").
        freeze_backbone: If True, backbone weights are frozen during initial training.
        augment: Whether to include the augmentation layer.
        dropout_rate: Dropout rate in the classification head.
        head_units: Number of units in the dense head layer.

    Returns:
        A keras.Model (not yet compiled).
    """
    inputs = keras.Input(
        shape=(cfg.img_height, cfg.img_width, cfg.num_channels),
        name="input",
    )

    x = inputs

    # Data augmentation (only active during training)
    if augment:
        x = build_augmentation_layer(cfg.img_height, cfg.img_width)(x)

    # Pretrained backbone
    if backbone == "efficientnetb0":
        base_model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_tensor=x,
        )
    elif backbone == "resnet50":
        from tensorflow.keras.applications import ResNet50
        base_model = ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=x,
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}")

    base_model.trainable = not freeze_backbone

    # Build classification head
    features = layers.GlobalAveragePooling2D()(base_model.output)
    features = layers.Dense(head_units)(features)
    features = layers.BatchNormalization()(features)
    features = layers.Activation("relu")(features)
    features = layers.Dropout(dropout_rate)(features)

    features = layers.Dense(head_units // 2)(features)
    features = layers.BatchNormalization()(features)
    features = layers.Activation("relu")(features)
    features = layers.Dropout(dropout_rate * 0.5)(features)

    outputs = layers.Dense(cfg.num_classes, activation="softmax", name="predictions")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"transfer_{backbone}")
    return model


def unfreeze_backbone(
    model: keras.Model,
    num_layers_to_unfreeze: int = 20,
) -> None:
    """Partially unfreeze the backbone for fine-tuning.

    Call this AFTER initial training with a frozen backbone. It unfreezes
    the top N layers of the backbone while keeping the rest frozen, allowing
    gentle fine-tuning at a lower learning rate.

    Args:
        model: The transfer learning model.
        num_layers_to_unfreeze: How many backbone layers to unfreeze from the top.
    """
    # Find the backbone sub-model
    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            backbone = layer
            break

    if backbone is None:
        print("Warning: Could not find backbone sub-model. Nothing to unfreeze.")
        return

    backbone.trainable = True
    for layer in backbone.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    print(f"Unfroze top {num_layers_to_unfreeze} layers of {backbone.name}.")
    trainable = sum(1 for l in backbone.layers if l.trainable)
    print(f"Trainable backbone layers: {trainable} / {len(backbone.layers)}")
