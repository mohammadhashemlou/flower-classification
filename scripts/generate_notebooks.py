"""Generate clean Jupyter notebooks for the flower classification project."""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.split("\n")}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.split("\n")}


def make_notebook(cells, filename):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NOTEBOOKS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Created: {path}")


# ── 01: Data Exploration ──
make_notebook(
    [
        md("# 1 - Data Exploration\n\nExplore the Oxford Flowers 102 dataset: class distribution, image sizes, and sample visualizations."),
        code(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path.cwd().parent))\n"
            "\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from src.config import Config\n"
            "from src.utils import set_global_seed, load_oxford_flowers_metadata\n"
            "\n"
            "cfg = Config()\n"
            "set_global_seed(cfg.seed)\n"
            "\n"
            "labels, train_ids, valid_ids, test_ids = load_oxford_flowers_metadata(cfg.dataset_dir)\n"
            "\n"
            "print(f'Total images: {len(labels)}')\n"
            "print(f'Classes: {cfg.num_classes}')\n"
            "print(f'Train: {len(train_ids)} | Val: {len(valid_ids)} | Test: {len(test_ids)}')"
        ),
        md("## Class Distribution"),
        code(
            "train_labels = labels[train_ids - 1]\n"
            "val_labels = labels[valid_ids - 1]\n"
            "test_labels = labels[test_ids - 1]\n"
            "\n"
            "distribution = pd.DataFrame({\n"
            "    'class_id': range(cfg.num_classes),\n"
            "    'train': [sum(train_labels == i) for i in range(cfg.num_classes)],\n"
            "    'val': [sum(val_labels == i) for i in range(cfg.num_classes)],\n"
            "    'test': [sum(test_labels == i) for i in range(cfg.num_classes)],\n"
            "})\n"
            "distribution['total'] = distribution[['train', 'val', 'test']].sum(axis=1)\n"
            "\n"
            "print(f\"Min images per class: {distribution['total'].min()}\")\n"
            "print(f\"Max images per class: {distribution['total'].max()}\")\n"
            "print(f\"Avg images per class: {distribution['total'].mean():.1f}\")\n"
            "\n"
            "plt.figure(figsize=(18, 5))\n"
            "plt.bar(distribution['class_id'], distribution['total'])\n"
            "plt.title('Images per Class - Oxford Flowers 102')\n"
            "plt.xlabel('Class ID')\n"
            "plt.ylabel('Count')\n"
            "plt.xticks(range(0, 102, 5))\n"
            "plt.grid(axis='y', alpha=0.3)\n"
            "plt.show()"
        ),
        md("## Sample Images"),
        code(
            "import random\n"
            "from PIL import Image\n"
            "\n"
            "random.seed(cfg.seed)\n"
            "sample_ids = random.sample(list(train_ids), 6)\n"
            "\n"
            "fig, axes = plt.subplots(2, 3, figsize=(12, 8))\n"
            "for ax, img_id in zip(axes.flat, sample_ids):\n"
            "    img = Image.open(cfg.images_dir / f'image_{img_id:05d}.jpg').convert('RGB')\n"
            "    ax.imshow(img)\n"
            "    ax.set_title(f'ID: {img_id} | Class: {labels[img_id - 1]}')\n"
            "    ax.axis('off')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md("## Image Dimensions"),
        code(
            "from collections import Counter\n"
            "\n"
            "sizes = Counter()\n"
            "for img_path in cfg.images_dir.glob('*.jpg'):\n"
            "    with Image.open(img_path) as img:\n"
            "        sizes[img.size] += 1\n"
            "\n"
            "print(f'Unique image sizes: {len(sizes)}')\n"
            "print('\\nMost common dimensions:')\n"
            "for size, count in sizes.most_common(10):\n"
            "    print(f'  {size}: {count} images')"
        ),
    ],
    "01_data_exploration.ipynb",
)

# ── 02: Data Preparation ──
make_notebook(
    [
        md("# 2 - Data Preparation\n\nBuild the tf.data pipeline with preprocessing and augmentation."),
        code(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path.cwd().parent))\n"
            "\n"
            "import tensorflow as tf\n"
            "from src.config import Config\n"
            "from src.utils import set_global_seed\n"
            "from src.data_loader import create_all_pipelines, compute_class_weights\n"
            "\n"
            "cfg = Config()\n"
            "set_global_seed(cfg.seed)\n"
            "\n"
            "train_ds, val_ds, test_ds = create_all_pipelines(cfg, augment_train=True)\n"
            "\n"
            "print(f'Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}')\n"
            "print(f'Val batches: {tf.data.experimental.cardinality(val_ds).numpy()}')\n"
            "print(f'Test batches: {tf.data.experimental.cardinality(test_ds).numpy()}')"
        ),
        md("## Inspect a Batch"),
        code(
            "images, labels = next(iter(train_ds))\n"
            "print(f'Images shape: {images.shape}')\n"
            "print(f'Labels shape: {labels.shape}')\n"
            "print(f'Pixel range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]')\n"
            "print(f'Labels: {labels.numpy()}')"
        ),
        md("## Class Weights (for handling imbalance)"),
        code(
            "class_weights = compute_class_weights(cfg)\n"
            "print(f'Number of weighted classes: {len(class_weights)}')\n"
            "print(f'Min weight: {min(class_weights.values()):.3f}')\n"
            "print(f'Max weight: {max(class_weights.values()):.3f}')"
        ),
    ],
    "02_data_preparation.ipynb",
)

# ── 03: Baseline CNN ──
make_notebook(
    [
        md("# 3 - Baseline CNN\n\nA simple from-scratch CNN to establish a performance baseline."),
        code(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path.cwd().parent))\n"
            "\n"
            "from src.config import Config\n"
            "from src.utils import set_global_seed\n"
            "from src.data_loader import create_all_pipelines, compute_class_weights\n"
            "from src.models.baseline_cnn import build_baseline_cnn\n"
            "from src.training import train_model\n"
            "from src.evaluation import evaluate_model\n"
            "\n"
            "cfg = Config()\n"
            "set_global_seed(cfg.seed)\n"
            "\n"
            "train_ds, val_ds, test_ds = create_all_pipelines(cfg, augment_train=True)\n"
            "class_weights = compute_class_weights(cfg)"
        ),
        code("model = build_baseline_cnn(cfg)\nmodel.summary()"),
        code(
            "history = train_model(\n"
            "    model,\n"
            "    train_ds,\n"
            "    val_ds,\n"
            "    cfg,\n"
            "    experiment_name='baseline_cnn',\n"
            "    class_weight=class_weights,\n"
            ")"
        ),
        code(
            "from src.utils import plot_training_history\n"
            "\n"
            "plot_training_history(\n"
            "    cfg.experiment_dir('baseline_cnn') / 'history.csv',\n"
            "    title='Baseline CNN',\n"
            ")"
        ),
        code("loss, acc, top5 = evaluate_model(model, test_ds)"),
    ],
    "03_baseline_cnn.ipynb",
)

# ── 04: Residual CNN ──
make_notebook(
    [
        md("# 4 - Residual CNN\n\nA custom ResNet-style network with skip connections for better gradient flow."),
        code(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path.cwd().parent))\n"
            "\n"
            "from src.config import Config\n"
            "from src.utils import set_global_seed\n"
            "from src.data_loader import create_all_pipelines, compute_class_weights\n"
            "from src.models.residual_cnn import build_residual_cnn\n"
            "from src.training import train_model\n"
            "from src.evaluation import evaluate_model\n"
            "\n"
            "cfg = Config()\n"
            "cfg.initial_lr = 5e-4\n"
            "set_global_seed(cfg.seed)\n"
            "\n"
            "train_ds, val_ds, test_ds = create_all_pipelines(cfg, augment_train=True)\n"
            "class_weights = compute_class_weights(cfg)"
        ),
        code("model = build_residual_cnn(cfg, augment=True)\nmodel.summary()"),
        code(
            "history = train_model(\n"
            "    model,\n"
            "    train_ds,\n"
            "    val_ds,\n"
            "    cfg,\n"
            "    experiment_name='residual_cnn',\n"
            "    class_weight=class_weights,\n"
            ")"
        ),
        code(
            "from src.utils import plot_training_history\n"
            "\n"
            "plot_training_history(\n"
            "    cfg.experiment_dir('residual_cnn') / 'history.csv',\n"
            "    title='Residual CNN',\n"
            ")"
        ),
        code("loss, acc, top5 = evaluate_model(model, test_ds)"),
    ],
    "04_residual_cnn.ipynb",
)

# ── 05: Transfer Learning ──
make_notebook(
    [
        md("# 5 - Transfer Learning (EfficientNetB0)\n\n"
            "Leverage ImageNet-pretrained features for high accuracy with minimal overfitting.\n\n"
            "**Phase 1**: Train with frozen backbone (only classification head learns).\n"
            "**Phase 2**: Unfreeze top layers and fine-tune at a lower learning rate."),
        code(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path.cwd().parent))\n"
            "\n"
            "from src.config import Config\n"
            "from src.utils import set_global_seed\n"
            "from src.data_loader import create_all_pipelines, compute_class_weights\n"
            "from src.models.transfer_learning import build_transfer_model, unfreeze_backbone\n"
            "from src.training import train_model, compile_model\n"
            "from src.evaluation import evaluate_model\n"
            "\n"
            "cfg = Config()\n"
            "cfg.epochs = 30\n"
            "cfg.initial_lr = 1e-3\n"
            "set_global_seed(cfg.seed)\n"
            "\n"
            "train_ds, val_ds, test_ds = create_all_pipelines(cfg, augment_train=True)\n"
            "class_weights = compute_class_weights(cfg)"
        ),
        md("## Phase 1 - Frozen Backbone Training"),
        code("model = build_transfer_model(cfg, freeze_backbone=True, augment=True)\nmodel.summary()"),
        code(
            "history_phase1 = train_model(\n"
            "    model,\n"
            "    train_ds,\n"
            "    val_ds,\n"
            "    cfg,\n"
            "    experiment_name='transfer_phase1',\n"
            "    class_weight=class_weights,\n"
            ")"
        ),
        md("## Phase 2 - Fine-Tuning"),
        code(
            "# Unfreeze top 20 layers of the backbone\n"
            "unfreeze_backbone(model, num_layers_to_unfreeze=20)\n"
            "\n"
            "# Recompile with a lower learning rate for fine-tuning\n"
            "cfg.epochs = 20\n"
            "cfg.initial_lr = 1e-4\n"
            "compile_model(model, cfg, label_smoothing=0.1)\n"
            "\n"
            "history_phase2 = train_model(\n"
            "    model,\n"
            "    train_ds,\n"
            "    val_ds,\n"
            "    cfg,\n"
            "    experiment_name='transfer_phase2',\n"
            "    class_weight=class_weights,\n"
            ")"
        ),
        code(
            "from src.utils import plot_training_history\n"
            "\n"
            "plot_training_history(\n"
            "    cfg.experiment_dir('transfer_phase2') / 'history.csv',\n"
            "    title='Transfer Learning - Fine-Tuning',\n"
            ")"
        ),
        code("loss, acc, top5 = evaluate_model(model, test_ds)"),
    ],
    "05_transfer_learning.ipynb",
)

# ── 06: Results Comparison ──
make_notebook(
    [
        md("# 6 - Results Comparison\n\nCompare all models side-by-side."),
        code(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path.cwd().parent))\n"
            "\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from src.config import Config\n"
            "\n"
            "cfg = Config()\n"
            "\n"
            "experiments = {\n"
            "    'Baseline CNN': cfg.experiment_dir('baseline_cnn') / 'history.csv',\n"
            "    'Residual CNN': cfg.experiment_dir('residual_cnn') / 'history.csv',\n"
            "    'Transfer (Phase 1)': cfg.experiment_dir('transfer_phase1') / 'history.csv',\n"
            "    'Transfer (Phase 2)': cfg.experiment_dir('transfer_phase2') / 'history.csv',\n"
            "}\n"
            "\n"
            "results = []\n"
            "for name, path in experiments.items():\n"
            "    if path.exists():\n"
            "        df = pd.read_csv(path)\n"
            "        results.append({\n"
            "            'Model': name,\n"
            "            'Best Val Acc': df['val_accuracy'].max(),\n"
            "            'Best Val Loss': df['val_loss'].min(),\n"
            "            'Epochs': len(df),\n"
            "        })\n"
            "\n"
            "results_df = pd.DataFrame(results)\n"
            "display(results_df)\n"
            "\n"
            "# Plot comparison\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "for name, path in experiments.items():\n"
            "    if path.exists():\n"
            "        df = pd.read_csv(path)\n"
            "        ax.plot(df['val_accuracy'], label=name, linewidth=1.5)\n"
            "\n"
            "ax.set_title('Validation Accuracy - All Models')\n"
            "ax.set_xlabel('Epoch')\n"
            "ax.set_ylabel('Val Accuracy')\n"
            "ax.legend()\n"
            "ax.grid(True, alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    ],
    "06_results_comparison.ipynb",
)

print("\nAll notebooks created successfully!")
