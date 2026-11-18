"""Model compilation, callbacks and the training loop."""

from __future__ import annotations

from typing import Dict, Optional

import tensorflow as tf
from tensorflow import keras

from .config import Config


class _NumericLrCallback(keras.callbacks.Callback):
    """Log the learning rate as a number.

    When the optimizer uses a schedule, ``logs['lr']`` is the schedule object
    and CSVLogger writes that instead of a value.
    """

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "lr" not in logs or isinstance(logs["lr"], (int, float)):
            return

        schedule = self.model.optimizer.learning_rate
        step = self.model.optimizer.iterations
        value = schedule(step) if callable(schedule) else schedule
        logs["lr"] = float(keras.backend.get_value(value))


def build_cosine_lr_schedule(
    cfg: Config,
    steps_per_epoch: int,
) -> keras.optimizers.schedules.LearningRateSchedule:
    """Cosine decay from initial_lr down to min_lr."""
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch

    try:
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg.initial_lr,
            decay_steps=total_steps - warmup_steps,
            alpha=cfg.min_lr / cfg.initial_lr,
            warmup_target=cfg.initial_lr,
            warmup_steps=warmup_steps,
        )
    except TypeError:
        # warmup_target/warmup_steps only exist in newer TensorFlow builds
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg.initial_lr,
            decay_steps=total_steps,
            alpha=cfg.min_lr / cfg.initial_lr,
        )


@keras.utils.register_keras_serializable(package="flower")
class LabelSmoothingSparseCategoricalCrossentropy(keras.losses.Loss):
    """Cross-entropy with label smoothing.

    Keras 2.10's SparseCategoricalCrossentropy has no label_smoothing
    argument. Registering the class keeps it serialisable inside .keras
    files, so checkpoints load without custom_objects.
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

        return tf.reduce_mean(-tf.reduce_sum(y_smooth * tf.math.log(y_pred), axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes, "smoothing": self.smoothing})
        return config


def _build_loss(cfg: Config, label_smoothing: float):
    if label_smoothing <= 0:
        return keras.losses.SparseCategoricalCrossentropy()

    return LabelSmoothingSparseCategoricalCrossentropy(
        num_classes=cfg.num_classes,
        smoothing=label_smoothing,
    )


def build_callbacks(
    cfg: Config,
    experiment_name: str,
    use_schedule: bool = False,
) -> list:
    """Checkpoint, early stopping, optional ReduceLROnPlateau, CSV log.

    ReduceLROnPlateau is skipped when a schedule drives the optimizer,
    because it can only write to a scalar learning rate.
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


def compile_model(
    model: keras.Model,
    cfg: Config,
    learning_rate: Optional[float] = None,
    label_smoothing: float = 0.1,
) -> None:
    """Compile with Adam and top-1/top-5 accuracy metrics."""
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
    """Fit the model and return its history."""
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

    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
