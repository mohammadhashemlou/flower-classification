"""Transfer learning models built on ImageNet-pretrained backbones."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..config import Config
from ..data_loader import build_augmentation_layer


def _attach_backbone(
    x: tf.Tensor,
    backbone: str,
    freeze_backbone: bool,
    input_shape: tuple,
) -> tf.Tensor:
    """Apply a pretrained backbone to an input tensor.

    The backbone is built standalone and then called on `x`. Building it with
    `input_tensor=` makes its internal Normalization layers hold EagerTensors,
    which breaks .keras serialization.
    """
    builders = {
        "efficientnetb0": keras.applications.EfficientNetB0,
        "efficientnetb3": keras.applications.EfficientNetB3,
        "resnet50": keras.applications.ResNet50,
        "mobilenetv3": keras.applications.MobileNetV3Large,
    }

    if backbone not in builders:
        raise ValueError(
            f"Unknown backbone: {backbone!r}. "
            f"Choose from: {', '.join(builders)}"
        )

    base = builders[backbone](
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base.trainable = not freeze_backbone

    # training=False keeps BatchNorm in inference mode, which matters when frozen
    return base(x, training=False)


def _coerce_rescaling_config(model: keras.Model) -> None:
    """Store Rescaling scale/offset as plain floats so .keras saving works.

    Keras 2.10's EfficientNet builds these layers with EagerTensor values and
    the config writer cannot serialize them. Numerics are unchanged.
    """
    stack = [model]
    while stack:
        current = stack.pop()
        for layer in current.layers:
            if isinstance(layer, layers.Rescaling):
                layer.scale = np.asarray(layer.scale).tolist()
                layer.offset = np.asarray(layer.offset).tolist()
            if hasattr(layer, "layers"):
                stack.append(layer)


def build_transfer_model(
    cfg: Config,
    backbone: str = "efficientnetb0",
    freeze_backbone: bool = True,
    augment: bool = True,
    dropout_rate: float = 0.4,
    head_units: int = 0,
) -> keras.Model:
    """Build an uncompiled transfer learning model.

    The head is deliberately small. With ~1020 training images a wide MLP head
    hits 100% train accuracy in a few epochs and then memorises, so the default
    is a single dropout-regularised linear layer over global average pooling.

    Args:
        cfg: Project configuration.
        backbone: Backbone name.
        freeze_backbone: Freeze backbone weights for feature extraction.
        augment: Include the augmentation layer.
        dropout_rate: Dropout rate in the head.
        head_units: Width of an extra hidden dense layer. 0 means no hidden layer.
    """
    inputs = keras.Input(
        shape=(cfg.img_height, cfg.img_width, cfg.num_channels),
        name="input",
    )

    x = inputs

    if augment:
        x = build_augmentation_layer(cfg.img_height, cfg.img_width)(x)

    # Pipeline emits [0, 1]; backbones expect [0, 255]
    x = layers.Rescaling(255.0, name="to_0_255")(x)

    if backbone == "resnet50":
        mean_bgr = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        x = layers.Lambda(
            lambda t: t - mean_bgr,
            name="resnet_preprocess",
        )(x)

    features = _attach_backbone(
        x,
        backbone,
        freeze_backbone,
        (cfg.img_height, cfg.img_width, cfg.num_channels),
    )

    features = layers.GlobalAveragePooling2D(name="gap")(features)
    features = layers.Dropout(dropout_rate, name="head_dropout")(features)

    if head_units > 0:
        features = layers.Dense(head_units, name="head_fc1")(features)
        features = layers.BatchNormalization(name="head_bn1")(features)
        features = layers.Activation("relu", name="head_relu1")(features)
        features = layers.Dropout(dropout_rate, name="head_drop2")(features)

    outputs = layers.Dense(
        cfg.num_classes,
        activation="softmax",
        name="predictions",
    )(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"transfer_{backbone}")

    _coerce_rescaling_config(model)

    return model


def unfreeze_backbone(
    model: keras.Model,
    num_layers_to_unfreeze: int = 20,
    verbose: bool = True,
) -> None:
    """Unfreeze the top N backbone layers for fine-tuning.

    Call this after training with a frozen backbone. BatchNorm layers stay
    frozen; they need larger batches than we use to give reliable statistics.

    Args:
        model: The transfer learning model.
        num_layers_to_unfreeze: How many layers to unfreeze from the top.
        verbose: Print a short summary.
    """
    # The augmentation Sequential is also a Model, so pick the one with the
    # most layers rather than the first match.
    nested_models = [
        layer
        for layer in model.layers
        if isinstance(layer, keras.Model) and layer.name != "data_augmentation"
    ]
    if not nested_models:
        raise RuntimeError("Could not find a backbone sub-model to unfreeze.")

    backbone = max(nested_models, key=lambda m: len(m.layers))

    backbone.trainable = True
    for layer in backbone.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    if verbose:
        trainable = sum(1 for layer in backbone.layers if layer.trainable)
        print(f"Unfroze top {num_layers_to_unfreeze} layers of '{backbone.name}'.")
        print(f"Trainable backbone layers: {trainable} / {len(backbone.layers)}")
        total_trainable = sum(
            tf.size(w).numpy() for w in model.trainable_weights
        )
        print(f"Total trainable parameters: {total_trainable:,}")
