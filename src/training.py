"""Training logic — compile, callbacks, and fit loop.

Provides a clean train_model() function that handles compilation,
callback setup (checkpoint, early stopping, LR scheduling, CSV logging),
and the training loop.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from typing import Optional, Dict

from .config import Config


def build_callbacks(
    cfg: Config,
    experiment_name: str,
) -> list:
    """Create the standard set of training callbacks.

    Args:
        cfg: Project configuration.
        experiment_name: Name used for the save directory.

    Returns:
        List of keras.callbacks.Callback instances.
    """
    exp_dir = cfg.experiment_dir(experiment_name)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(exp_dir / "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_lr,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            filename=str(exp_dir / "history.csv"),
            append=False,
        ),
    ]

    return callbacks


def build_cosine_lr_schedule(
    cfg: Config,
    steps_per_epoch: int,
) -> keras.optimizers.schedules.LearningRateSchedule:
    """Build a cosine-decay learning rate schedule with warmup.

    The first 5 epochs use linear warmup from 1e-6 to initial_lr,
    followed by cosine decay down to min_lr.

    Args:
        cfg: Project configuration.
        steps_per_epoch: Number of batches in one epoch.

    Returns:
        A LearningRateSchedule callable.
    """
    warmup_epochs = 5
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    # CosineDecay with warmup — falls back gracefully if warmup params
    # are not supported (TF < 2.14)
    try:
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg.initial_lr,
            decay_steps=total_steps - warmup_steps,
            alpha=cfg.min_lr / cfg.initial_lr,
            warmup_target=cfg.initial_lr,
            warmup_steps=warmup_steps,
        )
    except TypeError:
        # TF < 2.14: no warmup support — use plain cosine decay
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg.initial_lr,
            decay_steps=total_steps,
            alpha=cfg.min_lr / cfg.initial_lr,
        )


def sparse_ce_with_label_smoothing(
    num_classes: int,
    smoothing: float = 0.1,
):
    """Sparse categorical cross-entropy with label smoothing.

    Keras 2.10's built-in loss does not expose `label_smoothing`, so we
    implement it explicitly. Smoothing prevents the model from becoming
    overconfident, which measurably reduces overfitting on small datasets.

    Args:
        num_classes: Total number of classes.
        smoothing: Smoothing factor in [0, 1). 0 disables smoothing.

    Returns:
        A loss function with signature (y_true, y_pred).
    """

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

        y_one_hot = tf.one_hot(y_true, num_classes)
        y_smooth = y_one_hot * (1.0 - smoothing) + smoothing / num_classes

        return tf.reduce_mean(
            -tf.reduce_sum(y_smooth * tf.math.log(y_pred), axis=-1)
        )

    loss_fn.__name__ = f"sparse_ce_smooth_{smoothing}"
    return loss_fn


def compile_model(
    model: keras.Model,
    cfg: Config,
    learning_rate: Optional[float] = None,
    label_smoothing: float = 0.1,
) -> None:
    """Compile a model with standard settings.

    Args:
        model: The keras model to compile.
        cfg: Project configuration.
        learning_rate: Override the initial learning rate (None = use cfg).
        label_smoothing: Label smoothing factor (0 = disabled).
    """
    lr = learning_rate if learning_rate is not None else cfg.initial_lr

    if label_smoothing > 0:
        loss = sparse_ce_with_label_smoothing(cfg.num_classes, label_smoothing)
    else:
        loss = keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(name="top_5_accuracy"),
        ],
    )


def train_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    cfg: Config,
    experiment_name: str = "experiment",
    class_weight: Optional[Dict[int, float]] = None,
    use_cosine_schedule: bool = False,
    label_smoothing: float = 0.1,
) -> keras.callbacks.History:
    """Train a model end-to-end with callbacks and optional class weights.

    Args:
        model: A compiled (or uncompiled) keras model.
        train_ds: Training tf.data.Dataset.
        val_ds: Validation tf.data.Dataset.
        cfg: Project configuration.
        experiment_name: Name for checkpoint/log directory.
        class_weight: Optional dict of class weights for imbalanced data.
        use_cosine_schedule: If True, use cosine LR schedule instead of ReduceLROnPlateau.
        label_smoothing: Label smoothing factor for cross-entropy loss.

    Returns:
        The training History object.
    """
    # Compile if not already compiled
    if not model.optimizer:
        if use_cosine_schedule:
            steps = tf.data.experimental.cardinality(train_ds).numpy()
            lr = build_cosine_lr_schedule(cfg, steps)
            compile_model(model, cfg, learning_rate=lr, label_smoothing=label_smoothing)
        else:
            compile_model(model, cfg, label_smoothing=label_smoothing)

    callbacks = build_callbacks(cfg, experiment_name)

    print(f"\n{'=' * 60}")
    print(f"  Training: {experiment_name}")
    print(f"  Model: {model.name}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Initial LR: {cfg.initial_lr}")
    print(f"{'=' * 60}\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    return history
