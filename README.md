<div align="center">

# 🌸 Flower Classification

**Fine-grained recognition of 102 flower species using the Oxford Flowers 102 dataset**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-3DA639?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-2ea44f?style=flat-square)]()

*A from-scratch → residual → transfer-learning progression, built to be reproducible and easy to read.*

</div>

---

## 📖 Overview

Fine-grained flower classification is hard: 102 visually similar classes, only ~10 training images per class, and high intra-class variation. This project builds up the solution in stages, so every architectural decision is measurable:

| Stage | Approach | What it teaches |
| :--- | :--- | :--- |
| **1. Baseline CNN** | 4-block VGG-style CNN trained from scratch | Establishes the floor and exposes overfitting |
| **2. Residual CNN** | Custom ResNet with skip connections | Shows how gradient flow fixes slow convergence |
| **3. Transfer Learning** | ImageNet-pretrained EfficientNetB0 + custom head | Delivers production-grade accuracy |

The final model is trained in two phases — a frozen-backbone warm-up followed by selective fine-tuning — which is the standard recipe for small datasets.

---

## 📊 Results

<!-- RESULTS_TABLE_START -->
| Model | Val Accuracy | Test Accuracy | Overfitting Gap |
| :--- | :---: | :---: | :---: |
| Baseline CNN | 20.5% | — | Severe |
| Residual CNN | 35.4% | — | Moderate |
| **Transfer (EfficientNetB0)** | **TBD** | **TBD** | **Low** |
<!-- RESULTS_TABLE_END -->

> Trained on the official Oxford Flowers 102 split: 1,020 train / 1,020 val / 6,149 test images.

---

## 🗂 Project Structure

```
flower-classification/
├── src/
│   ├── config.py               # Single source of truth for hyperparameters & paths
│   ├── data_loader.py          # tf.data pipelines, augmentation, class weights
│   ├── training.py             # Callbacks, LR schedules, fit loop
│   ├── evaluation.py           # Metrics, confusion matrix, per-class report
│   ├── utils.py                # Seeding, dataset metadata, plotting
│   └── models/
│       ├── baseline_cnn.py     # VGG-style CNN from scratch
│       ├── residual_cnn.py     # Custom residual network
│       └── transfer_learning.py# EfficientNetB0 / ResNet50 / MobileNetV3 heads
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_baseline_cnn.ipynb
│   ├── 04_residual_cnn.ipynb
│   ├── 05_transfer_learning.ipynb
│   └── 06_results_comparison.ipynb
├── scripts/
│   └── train.py                # CLI training entry point
├── tests/
│   └── test_data_loader.py
├── models/                     # Checkpoints & history CSVs (weights gitignored)
├── requirements.txt
└── LICENSE
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/mohammadhashemlou/flower-classification.git
cd flower-classification

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset

Download [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) and lay it out as:

```
dataset/raw/oxford_flowers102/
├── jpg/                 # 8,191 images (image_00001.jpg …)
├── imagelabels.mat
└── setid.mat
```

### Train

```bash
# Transfer learning — phase 1: frozen backbone
python scripts/train.py --model transfer --backbone efficientnetb0 --epochs 20

# Transfer learning — phase 2: fine-tune top layers
python scripts/train.py --model transfer --backbone efficientnetb0 \
    --epochs 15 --fine-tune --unfreeze-layers 30 \
    --init-from models/transfer_efficientnetb0_frozen/best_model.keras

# From-scratch baselines
python scripts/train.py --model baseline --epochs 40
python scripts/train.py --model residual --epochs 60
```

### Evaluate a checkpoint

```python
from tensorflow import keras
from src.config import Config
from src.data_loader import create_data_pipeline
from src.evaluation import evaluate_model, plot_confusion_matrix

cfg = Config()
model = keras.models.load_model("models/transfer_efficientnetb0_frozen/best_model.keras")
test_ds = create_data_pipeline(cfg, split="test")

evaluate_model(model, test_ds)
plot_confusion_matrix(model, test_ds, num_classes=cfg.num_classes)
```

---

## 🧠 Design Decisions

**Why transfer learning?**
With ~10 images per class, a from-scratch CNN cannot learn 102 fine-grained categories. ImageNet features already encode edges, textures, and petal-like structures, so only a small head needs training.

**Why two-phase training?**
Unfreezing the whole backbone immediately destroys pretrained features with large early gradients. Phase 1 adapts the head; phase 2 gently fine-tunes the top blocks at a 10× lower learning rate.

**How is overfitting controlled?**
- On-the-fly augmentation (flip, rotation, zoom, contrast) inside the model
- Label smoothing (0.1) to soften overconfident targets
- Class weighting for the long-tailed class distribution
- Dropout + BatchNorm in the head
- Early stopping on `val_loss` with best-weight restoration
- `ReduceLROnPlateau` to escape plateaus without overshooting

**Why [0, 255] scaling for EfficientNet?**
Keras' EfficientNet bundles its own Rescaling + Normalization layers and expects raw pixels. Our pipeline emits `[0, 1]`, so a `Rescaling(255)` layer bridges the gap — a subtle bug that silently costs ~40% accuracy if missed.

---

## 🧪 Tests

```bash
python tests/test_data_loader.py
```

---

## 📄 License

Released under the [MIT License](LICENSE).

<div align="center">

**Built with TensorFlow & Keras**

</div>
