"""Transfer learning model using ImageNet-pretrained backbones.

Instead of training a CNN from scratch (which struggles on the small
Oxford Flowers 102 dataset), we leverage ImageNet-pretrained features
and fine-tune a custom classification head.

This approach dramatically reduces overfitting and typically achieves
85-95% accuracy on this dataset.

IMPORTANT — input scaling:
    Our data pipeline produces images in the range [0, 1].
    Keras' EfficientNet includes its own Rescaling + Normalization layers
    and therefore expects raw pixel values in [0, 255]. We insert a
    Rescaling(255) layer before the backbone to bridge this gap.
    ResNet50 expects Caffe-style preprocessing, handled explicitly.
"""

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
    """Create a pretrained backbone and apply it to the input tensor.

    The backbone is built standalone (via `input_shape`) and then *called*
    on `x`. Building it with `input_tensor=` instead produces a model whose
    internal Normalization layers hold EagerTensors, which makes `.keras`
    serialization fail. Calling a standalone model avoids that entirely.

    Args:
        x: Input tensor already in [0, 255] range.
        backbone: Backbone identifier.
        freeze_backbone: Whether to freeze the backbone weights.
        input_shape: Shape tuple for the backbone input, e.g. (224, 224, 3).

    Returns:
        The backbone's output tensor.
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

    # training=False keeps BatchNorm in inference mode (critical when frozen)
    return base(x, training=False)


def _coerce_rescaling_config(model: keras.Model) -> None:
    """Make Rescaling layers JSON-serializable.

    Keras 2.10's EfficientNet builds its internal `Rescaling` layer with an
    EagerTensor `scale`, which makes `model.save(...)` fail when it tries to
    serialize the layer config. Converting these back to plain Python floats
    has no effect on numerics and lets `.keras` saving work.
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
    """Build a transfer learning model on top of a pretrained backbone.

    Head design is the single biggest lever against overfitting here.
    With ~1020 training images, a 512→256 MLP head (≈660k parameters)
    reaches 100% train accuracy within five epochs and then memorises.
    The default head is therefore a single dropout-regularised linear
    classifier on top of global average pooling (≈130k parameters), which
    keeps the train/val gap small while the backbone learns the features.

    Args:
        cfg: Project configuration.
        backbone: Name of the backbone architecture.
        freeze_backbone: If True, backbone weights are frozen (feature extraction).
        augment: Whether to include the augmentation layer.
        dropout_rate: Dropout rate in the classification head.
        head_units: If > 0, add a hidden dense layer of this width (wide
            head). If 0 (default), use a compact linear head.

    Returns:
        A keras.Model (not yet compiled).
    """
    inputs = keras.Input(
        shape=(cfg.img_height, cfg.img_width, cfg.num_channels),
        name="input",
    )

    x = inputs

    # Data augmentation (no-op at inference time)
    if augment:
        x = build_augmentation_layer(cfg.img_height, cfg.img_width)(x)

    # Our pipeline yields [0, 1]; backbones expect [0, 255].
    x = layers.Rescaling(255.0, name="to_0_255")(x)

    # Caffe-style preprocessing for ResNet50
    if backbone == "resnet50":
        mean_bgr = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        x = layers.Lambda(
            lambda t: t - mean_bgr,
            name="resnet_preprocess",
        )(x)

    # Pretrained backbone (applied as a callable, returns a tensor)
    features = _attach_backbone(
        x,
        backbone,
        freeze_backbone,
        (cfg.img_height, cfg.img_width, cfg.num_channels),
    )

    # ── Classification head ─────────────────────────────────────────
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

    # Allow full-model serialization to .keras
    _coerce_rescaling_config(model)

    return model


def unfreeze_backbone(
    model: keras.Model,
    num_layers_to_unfreeze: int = 20,
    verbose: bool = True,
) -> None:
    """Partially unfreeze the backbone for fine-tuning.

    Call this AFTER initial training with a frozen backbone. It unfreezes
    the top N layers of the backbone while keeping the rest frozen, allowing
    gentle fine-tuning at a lower learning rate.

    BatchNormalization layers inside the backbone are kept frozen — this is
    the standard recommendation for fine-tuning with small batch sizes.

    Args:
        model: The transfer learning model.
        num_layers_to_unfreeze: How many backbone layers to unfreeze from the top.
        verbose: Whether to print a summary of the unfreezing.
    """
    # Find the backbone sub-model: the largest nested model in the graph.
    # (The augmentation Sequential is also a Model, so we cannot just take
    # the first match — we pick the one with the most layers.)
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

    # Keep BatchNorm layers frozen (they need large batches to be reliable)
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    if verbose:
        trainable = sum(1 for l in backbone.layers if l.trainable)
        print(f"Unfroze top {num_layers_to_unfreeze} layers of '{backbone.name}'.")
        print(f"Trainable backbone layers: {trainable} / {len(backbone.layers)}")
        total_trainable = sum(
            tf.size(w).numpy() for w in model.trainable_weights
        )
        print(f"Total trainable parameters: {total_trainable:,}")
