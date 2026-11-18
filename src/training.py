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


class _NumericLrCallback(keras.callbacks.Callback):
    """Record a numeric learning rate in the logs.

    With a ``LearningRateSchedule`` the optimizer's ``learning_rate`` is a
    callable, so ``logs['lr']`` ends up holding an object and CSVLogger
    writes garbage. This callback resolves the schedule at the current step
    so the CSV always contains a real number.
    """

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "lr" not in logs:
            return
        lr = logs["lr"]
        if isinstance(lr, (int, float)):
            return

        schedule = self.model.optimizer.learning_rate
        step = self.model.optimizer.iterations
        value = schedule(step) if callable(schedule) else schedule
        logs["lr"] = float(keras.backend.get_value(value))


def build_callbacks(
    cfg: Config,
    experiment_name: str,
    use_schedule: bool = False,
) -> list:
    """Create the standard set of training callbacks.

    Args:
        cfg: Project configuration.
        experiment_name: Name used for the save directory.
        use_schedule: Set True when the optimizer uses a LearningRateSchedule.
            ``ReduceLROnPlateau`` cannot mutate a schedule (it expects a
            mutable scalar LR), so it is omitted in that case.

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
            monitor="val_accuracy",
            mode="max",
            patience=cfg.early_stopping_patience,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        # Runs before CSVLogger so the CSV records a float, not a schedule
        _NumericLrCallback(),
        keras.callbacks.CSVLogger(
            filename=str(exp_dir / "history.csv"),
            append=False,
        ),
    ]

    if not use_schedule:
        callbacks.insert(
            2,
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg.reduce_lr_factor,
                patience=cfg.reduce_lr_patience,
                min_lr=cfg.min_lr,
                verbose=1,
            ),
        )

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


@keras.utils.register_keras_serializable(package="flower")
class LabelSmoothingSparseCategoricalCrossentropy(keras.losses.Loss):
    """Sparse categorical cross-entropy with label smoothing.

    Keras 2.10's built-in ``SparseCategoricalCrossentropy`` has no
    ``label_smoothing`` argument, so we implement it as a proper ``Loss``
    subclass. Being a registered subclass (rather than a plain closure) means
    it is serialized into ``.keras`` files and restored automatically — no
    ``custom_objects`` needed when loading a checkpoint.

    Smoothing stops the model from becoming overconfident, which measurably
    reduces overfitting on small datasets such as Oxford Flowers 102.
    """

    def __init__(self, num_classes: int = 102, smoothing: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.smoothing = float(smoothing)

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

        y_one_hot = tf.one_hot(y_true, self.num_classes)
        y_smooth = y_one_hot * (1.0 - self.smoothing) + self.smoothing / self.num_classes

        return tf.reduce_mean(
            -tf.reduce_sum(y_smooth * tf.math.log(y_pred), axis=-1)
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "smoothing": self.smoothing,
            }
        )
        return config


def _build_loss(cfg: Config, label_smoothing: float):
    """Return the sparse categorical cross-entropy loss to use.

    Uses the registered smoothing loss when smoothing is requested, so the
    resulting checkpoint stays loadable without custom objects.
    """
    if label_smoothing <= 0:
        return keras.losses.SparseCategoricalCrossentropy()

    return LabelSmoothingSparseCategoricalCrossentropy(
        num_classes=cfg.num_classes,
        smoothing=label_smoothing,
    )


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

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=_build_loss(cfg, label_smoothing),
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

    callbacks = build_callbacks(cfg, experiment_name, use_schedule=use_cosine_schedule)

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
