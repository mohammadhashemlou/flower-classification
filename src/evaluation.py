"""Evaluation utilities — metrics, confusion matrix, visualizations."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from pathlib import Path
from typing import Tuple, Optional


def evaluate_model(
    model: keras.Model,
    test_ds: tf.data.Dataset,
) -> Tuple[float, float, float]:
    """Evaluate a trained model on the test set.

    Args:
        model: A trained keras model.
        test_ds: Test tf.data.Dataset.

    Returns:
        (test_loss, test_accuracy, test_top5_accuracy)
    """
    results = model.evaluate(test_ds, verbose=0)
    loss = results[0]
    acc = results[1]
    top5 = results[2] if len(results) > 2 else 0.0

    print(f"Test Loss:        {loss:.4f}")
    print(f"Test Accuracy:    {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"Test Top-5 Acc:   {top5:.4f}  ({top5 * 100:.2f}%)")

    return loss, acc, top5


def get_predictions(
    model: keras.Model,
    dataset: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a dataset and return predictions + true labels.

    Args:
        model: Trained keras model.
        dataset: tf.data.Dataset yielding (images, labels).

    Returns:
        (true_labels, predicted_labels) as 1-D numpy arrays.
    """
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(
    model: keras.Model,
    dataset: tf.data.Dataset,
    num_classes: int = 102,
    save_path: Optional[Path] = None,
    normalize: bool = True,
) -> None:
    """Plot a confusion matrix for model predictions.

    Args:
        model: Trained keras model.
        dataset: tf.data.Dataset yielding (images, labels).
        num_classes: Number of classes.
        save_path: Optional path to save the figure.
        normalize: If True, normalize rows to show percentages.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    y_true, y_pred = get_predictions(model, dataset)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""), fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    plt.close(fig)


def classification_report_csv(
    model: keras.Model,
    dataset: tf.data.Dataset,
    save_path: Path,
) -> None:
    """Generate a per-class classification report and save as CSV.

    Args:
        model: Trained keras model.
        dataset: tf.data.Dataset.
        save_path: Path to save the CSV file.
    """
    from sklearn.metrics import classification_report

    y_true, y_pred = get_predictions(model, dataset)
    report = classification_report(y_true, y_pred, output_dict=True)

    import pandas as pd
    df = pd.DataFrame(report).transpose()
    df.to_csv(save_path, index=True)
    print(f"Classification report saved to: {save_path}")
