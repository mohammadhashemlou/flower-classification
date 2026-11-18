<div align="center">

# 🌸 Flower Classification

**Fine-grained recognition of 102 flower species — from a from-scratch CNN to a fine-tuned EfficientNetB0**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-3DA639?style=flat-square)](LICENSE)
[![CI](https://github.com/mohammadhashemlou/flower-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/mohammadhashemlou/flower-classification/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/pytest-16%20tests-2ea44f?style=flat-square)](tests/)

*Six reproducible experiments, one honest benchmark: **88.66% top-1 / 97.07% top-5** on the official test split.*

</div>

---

## 📖 What this project is

[Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) is a hard fine-grained dataset: 102 visually similar species, and only **1,020 training images** — about 10 per class. A model can reach 100% training accuracy on it within a few epochs while still being wrong half the time on new images, so almost all of the engineering here is about **generalisation**, not raw capacity.

The repository builds the solution in measurable stages, so every decision has a number attached:

| Stage | Approach | What it isolates |
| :--- | :--- | :--- |
| `01` | VGG-style CNN from scratch | The floor — and proof that capacity is not the bottleneck |
| `02` | Custom residual CNN | Whether better gradient flow helps at this data scale |
| `03` | EfficientNetB0, frozen backbone | The value of ImageNet features |
| `04` | EfficientNetB0, fine-tuned | How much the backbone still needs to adapt |
| `05` | Same, trained on train+val | The effect of doubling the data |
| `06` | Same, fine-tuned | **Best model** |

---

## 📊 Results

All numbers below are produced by `scripts/benchmark.py` on the **official 6,149-image test split**, which is never used for training or model selection.

| # | Model | Train imgs | Monitor acc | Test top-1 | Test top-5 | Train−val gap |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: |
| 01 | Baseline CNN (from scratch) | 1,020 | 30.0% | 25.96% | 56.46% | +2.2 pp |
| 02 | Residual CNN (from scratch) | 1,020 | 34.4% | 29.89% | 55.83% | +22.1 pp |
| 03 | EfficientNetB0 frozen | 1,020 | 83.7% | 79.85% | 94.18% | +16.2 pp |
| 04 | EfficientNetB0 fine-tuned | 1,020 | 87.1% | 83.70% | 94.67% | +12.9 pp |
| 05 | EfficientNetB0 frozen (train+val) | 1,836 | 87.3% | 79.56% | 93.40% | −0.9 pp |
| 06 | **EfficientNetB0 fine-tuned (train+val)** | 1,836 | **95.1%** | **88.66%** | **97.07%** | +4.4 pp |

<details>
<summary><b>How to read this table</b></summary>

- **Monitor acc** is the validation accuracy used *during* training for checkpointing and early stopping. For stages 01–04 it is the official 1,020-image val split; for stages 05–06 it is a 204-image hold-out carved out of the merged train+val pool.
- **Train−val gap** is `final train accuracy − best monitor accuracy`. A large gap means memorisation; a small gap with low accuracy means underfitting.
- Stage 01 is a textbook underfit: 32% train, 30% monitor, 26% test. It is not overfitting — it simply cannot represent 102 fine-grained classes from 10 examples each.
- Stage 02 has the largest gap of all (56% train, 34% monitor) yet almost no accuracy gain over stage 01: skip connections let it memorise harder without learning anything more generalisable. Deeper is not automatically better at this data scale.
- Stage 03 shows the opposite failure: 100% train, 83.7% monitor. ImageNet features are good, but a frozen backbone plus a large head still memorises.
- Stage 05's monitor accuracy (87.3%) overstates its test accuracy (79.6%). The 204-image hold-out is small enough that ±3 pp of noise is expected — which is why the table reports test numbers rather than trusting the monitor column.
- Stage 06 is the only configuration where the gap stays small *and* accuracy is high, which is what "generalises" actually looks like.

</details>

![Validation accuracy across all six experiments](results/comparison.png)

Curves for every run: `models/<experiment>/training_curves.png` · regenerate the table and this plot with `python scripts/compare.py` · raw numbers in [`results/`](results/).

---

## 🗂 Project structure

```
flower-classification/
├── src/
│   ├── config.py                # Single source of truth: paths, hyperparameters
│   ├── data_loader.py           # tf.data pipelines, caching, augmentation, class weights
│   ├── training.py              # Callbacks, LR schedules, registered loss, fit loop
│   ├── evaluation.py            # Test metrics, confusion matrix, per-class report
│   ├── utils.py                 # Seeding, dataset metadata, plotting
│   └── models/
│       ├── baseline_cnn.py      # VGG-style CNN from scratch
│       ├── residual_cnn.py      # Custom residual network
│       └── transfer_learning.py # EfficientNetB0/B3, ResNet50, MobileNetV3 heads
├── scripts/
│   ├── train.py                 # CLI: train any stage, including fine-tuning
│   ├── evaluate.py              # CLI: score one checkpoint on a split
│   ├── benchmark.py             # CLI: score every checkpoint on the test split
│   ├── compare.py               # CLI: rank experiments, plot validation curves
│   ├── gpu_env.py               # Registers CUDA/cuDNN DLLs before TF imports
│   └── generate_notebooks.py    # Rebuilds notebooks/ from source
├── notebooks/                   # 01–06, thin wrappers over src/
├── tests/                       # 16 pytest tests (data, models, serialization)
├── models/                      # Per-experiment checkpoints, history, curves (gitignored)
└── requirements.txt
```

Notebooks are deliberately thin: they import from `src/` instead of re-implementing the pipeline, so there is exactly one copy of each piece of logic.

---

## 🚀 Quick start

```bash
git clone https://github.com/mohammadhashemlou/flower-classification.git
cd flower-classification
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset

Download [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) (`102flowers.tgz` + `imagelabels.mat` + `setid.mat`) and lay it out as:

```
dataset/raw/oxford_flowers102/
├── jpg/                 # 8,191 images: image_00001.jpg …
├── imagelabels.mat
└── setid.mat
```

### GPU notes (important on Windows)

TensorFlow **dropped native Windows GPU support after 2.10**, so the versions are platform-split in `requirements.txt`:

| Platform | TensorFlow | CUDA | cuDNN |
| :--- | :--- | :--- | :--- |
| Windows | 2.10.0 | 11.8 (`nvidia-*-cu11` wheels) | 8.9 |
| Linux | 2.16–2.18 | 12.x | 9.x |

`scripts/gpu_env.py` registers the CUDA/cuDNN directories on the DLL search path before TensorFlow is imported, and handles both wheel layouts (`bin/` and `lib/x64/`). Verify with:

```bash
python scripts/gpu_env.py     # prints the detected GPU
```

> Without this, TensorFlow silently falls back to CPU and training is ~25× slower.

### Train

```bash
# Stage 06 — the best configuration, in two phases
python scripts/train.py --model transfer --backbone efficientnetb0 \
    --train-on-trainval --head-units 512 --epochs 40 --lr 5e-4 \
    --experiment 05_efficientnet_frozen_x2

python scripts/train.py --model transfer --backbone efficientnetb0 \
    --train-on-trainval --epochs 30 --lr 1e-4 --cosine --fine-tune \
    --unfreeze-layers 100 \
    --init-from models/05_efficientnet_frozen_x2/best_model.keras \
    --experiment 06_efficientnet_finetuned_x2

# From-scratch baselines
python scripts/train.py --model baseline --epochs 40 --experiment 01_baseline_cnn
python scripts/train.py --model residual --epochs 60 --lr 5e-4 --cosine \
    --experiment 02_residual_cnn
```

### Evaluate and compare

```bash
python scripts/benchmark.py     # every checkpoint -> results/test_metrics.csv
python scripts/compare.py       # ranked table + results/comparison.png
python scripts/evaluate.py --checkpoint models/06_efficientnet_finetuned_x2/best_model.keras \
    --confusion-matrix --report
```

---

## 🧠 Design decisions

**Why transfer learning, and why two phases?**
With ~10 images per class, a from-scratch CNN plateaus around 26% test accuracy (stage 01). ImageNet features already encode the textures and shapes that separate flower species, so only a small head needs to learn. Unfreezing the whole backbone from the start destroys those features with large early gradients; phase 1 trains the head with the backbone frozen, phase 2 fine-tunes the top 100 layers at a 5× lower learning rate.

**Why cache decoded images in RAM?**
JPEG decoding dominated training: the pipeline managed 0.8 steps/s, so the GPU sat idle. Decoding once to `uint8`, caching, then casting to `float32` after the cache cut an epoch from **319 s to 12 s** — a 26× speedup, and it keeps per-epoch shuffling because the shuffle happens *after* the cache.

**Why train on train+val for the final model?**
1,020 images for 102 classes is the binding constraint. Merging the official val split doubles the data to 2,040; a 204-image slice of the merged pool is held out purely to drive early stopping. The test split is never touched. This is the single change that moved test accuracy from 83.7% to 88.66%.

**Why a compact head by default?**
A 512→256 MLP head has ~660k parameters — more than enough to memorise 1,020 images, which is exactly what stage 03 does. The default head is a single dropout-regularised linear classifier (~130k parameters). A wide head is still available via `--head-units 512`, and it wins once the dataset doubles (stage 05/06), which is why the final model uses it.

**How is overfitting controlled?**
- Augmentation *inside* the model (flip, rotation, zoom, translation, contrast) — no-ops at inference, so the exported model needs no separate preprocessing
- Label smoothing (0.1) via a registered `keras.losses.Loss` subclass
- Inverse-frequency class weights
- Dropout on pooled features, BatchNorm kept frozen inside the backbone during fine-tuning
- Early stopping on `val_accuracy` with best-weight restoration
- Cosine LR decay for the fine-tuning phase

**Why `Rescaling(255)` before EfficientNet?**
Keras' EfficientNet bundles its own normalisation and expects raw `[0, 255]` pixels, while this pipeline emits `[0, 1]`. Without the bridge layer the model trains but loses tens of accuracy points — a silent bug worth calling out.

---

## 🐛 Bugs this project fixed along the way

Each of these was a real failure encountered while building the pipeline, and each now has a regression test or a guard:

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| GPU never used, 0.8 steps/s | JPEG decode per epoch, not compute | `cache()` decoded `uint8` before shuffling |
| `Unknown loss function: sparse_ce_smooth_0.1` on load | Loss was a closure | Registered `Loss` subclass with `get_config()` |
| Training hung after the last epoch | `plt.show()` with a GUI backend in a headless run | Force `Agg`, always write a PNG, never `show()` |
| `CosineDecay * int` TypeError | `ReduceLROnPlateau` cannot mutate a schedule | Omit it when a schedule is active |
| CSV logged `lr` as an object | Optimizer LR was a callable | Resolve the schedule to a float before logging |
| `You must compile your model` on evaluate | Checkpoint loaded with `compile=False` | Recompile before `.evaluate()` |
| `.keras` save failed on EfficientNet | Internal `Rescaling` held EagerTensors | Coerce `scale`/`offset` to plain floats |
| Notebook cells rendered as one line | Source lines split without `\n` | `splitlines(keepends=True)` in the generator |

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

16 tests covering the pipeline (batch shapes, value range, split disjointness, class weights), the model builders (head sizes, scaling layer, backbone validation), serialization (`.keras` round-trip with no `custom_objects`) and the LR schedule. Tests that need the dataset skip themselves when it is absent, so CI runs without it.

---

## 📄 License

Released under the [MIT License](LICENSE).

<div align="center">

**Built with TensorFlow & Keras**

</div>
