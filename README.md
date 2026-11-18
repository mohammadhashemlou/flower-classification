# Flower Classification — Oxford Flowers 102

A deep learning project for classifying 102 flower species using the Oxford Flowers 102 dataset. The project iterates through multiple model architectures — from a baseline CNN to residual networks with transfer learning — to achieve high accuracy while minimizing overfitting.

## Project Structure

```
flower-classification/
├── src/
│   ├── __init__.py
│   ├── config.py            # Central configuration (hyperparams, paths)
│   ├── data_loader.py       # Dataset loading, preprocessing, augmentation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_cnn.py  # Simple CNN from scratch
│   │   ├── residual_cnn.py  # Custom Residual CNN
│   │   └── transfer_learning.py  # Transfer learning (EfficientNet)
│   ├── training.py          # Training loop with callbacks
│   ├── evaluation.py        # Evaluation & metrics
│   └── utils.py             # Helpers (plotting, seeding, etc.)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_baseline_cnn.ipynb
│   ├── 04_residual_cnn.ipynb
│   ├── 05_transfer_learning.ipynb
│   └── 06_results_comparison.ipynb
├── models/                  # Saved weights & training history (gitignored)
├── tests/
│   └── test_data_loader.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Models & Results

| Model | Val Accuracy | Test Accuracy | Notes |
|---|---|---|---|
| Baseline CNN | ~20% | — | Overfits heavily |
| CNN + Augmentation | ~22% | — | Slight improvement |
| Residual CNN | ~35% | — | Skip connections help |
| Residual + Strong Aug | ~37% | — | Best from-scratch result |
| Transfer Learning (EfficientNet) | TBD | TBD | Expected: 85-95% |

## Setup

```bash
# Clone the repository
git clone https://github.com/mohammadhashemlou/flower-classification.git
cd flower-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dataset

Download the Oxford Flowers 102 dataset and place it under `dataset/raw/oxford_flowers102/`:

```
dataset/raw/oxford_flowers102/
├── jpg/            # 8,191 flower images
├── imagelabels.mat # Labels (1-102)
└── setid.mat       # Train/val/test splits
```

## Usage

### Train a model

```python
from src.config import Config
from src.data_loader import create_data_pipeline
from src.models.transfer_learning import build_transfer_model
from src.training import train_model

cfg = Config()

train_ds, val_ds, test_ds = create_data_pipeline(cfg)

model = build_transfer_model(cfg)
history = train_model(model, train_ds, val_ds, cfg)
```

### Evaluate

```python
from src.evaluation import evaluate_model

test_loss, test_acc = evaluate_model(model, test_ds)
print(f"Test Accuracy: {test_acc:.4f}")
```

## Tech Stack

- **TensorFlow / Keras** — Deep learning framework
- **tf.data** — Efficient data pipelines
- **EfficientNetB0** — Transfer learning backbone
- **Matplotlib / Seaborn** — Visualization
- **scipy.io** — MATLAB file parsing (dataset labels)

## Key Improvements

- **Transfer Learning**: Using pre-trained ImageNet weights for feature extraction
- **Data Augmentation**: Random flips, rotations, zoom, contrast, brightness
- **Class Weighting**: Handling class imbalance in the dataset
- **Cosine LR Schedule**: Smooth learning rate decay
- **Early Stopping**: Preventing unnecessary training epochs
- **Modular Architecture**: Clean separation of concerns for reproducibility
