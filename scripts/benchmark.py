"""Evaluate every saved checkpoint on the official test split.

Produces ``results/test_metrics.csv`` — one row per experiment with
top-1 and top-5 accuracy on the 6,149-image test split. This is the table
quoted in the README, and it is the only place where test numbers are
computed, so training-time model selection never sees the test set.

Usage
-----
    python scripts/benchmark.py
    python scripts/benchmark.py --only 06_efficientnet_frozen 07_efficientnet_finetuned
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_env import setup_cuda_dlls  # noqa: E402

setup_cuda_dlls(verbose=False)

import pandas as pd  # noqa: E402
from tensorflow import keras  # noqa: E402

from src.config import Config  # noqa: E402
from src.data_loader import create_data_pipeline  # noqa: E402
from src.training import compile_model  # noqa: E402
from src.utils import set_global_seed  # noqa: E402

LABELS = {
    "01_baseline_cnn": "Baseline CNN (from scratch)",
    "02_residual_cnn": "Residual CNN (from scratch)",
    "03_efficientnet_frozen": "EfficientNetB0 frozen",
    "04_efficientnet_finetuned": "EfficientNetB0 fine-tuned",
    "05_efficientnet_frozen_x2": "EfficientNetB0 frozen (train+val)",
    "06_efficientnet_finetuned_x2": "EfficientNetB0 fine-tuned (train+val)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark all checkpoints on the test split.")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Restrict to these experiment directory names.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "test_metrics.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config()
    set_global_seed(cfg.seed)

    print("Building test pipeline...")
    test_ds = create_data_pipeline(cfg, split="test", cache=True)

    candidates = sorted(p.parent for p in cfg.models_dir.glob("*/best_model.keras"))
    if args.only:
        wanted = set(args.only)
        candidates = [c for c in candidates if c.name in wanted]

    if not candidates:
        print("No checkpoints found under models/.")
        return

    rows = []
    for exp_dir in candidates:
        name = exp_dir.name
        label = LABELS.get(name, name)

        print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
        try:
            model = keras.models.load_model(str(exp_dir / "best_model.keras"), compile=False)
            # compile=False leaves no loss/metrics attached
            compile_model(model, cfg, learning_rate=cfg.initial_lr, label_smoothing=0.0)

            started = time.time()
            results = model.evaluate(test_ds, verbose=0)
            elapsed = time.time() - started

            loss, top1 = float(results[0]), float(results[1])
            top5 = float(results[2]) if len(results) > 2 else float("nan")

            print(f"  top-1: {top1 * 100:.2f}%   top-5: {top5 * 100:.2f}%   ({elapsed:.0f}s)")

            rows.append(
                {
                    "experiment": name,
                    "label": label,
                    "test_loss": round(loss, 4),
                    "test_top1": round(top1, 4),
                    "test_top5": round(top5, 4),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  ! failed: {type(exc).__name__}: {exc}")

    if not rows:
        print("\nNothing evaluated.")
        return

    table = pd.DataFrame(rows).sort_values("test_top1", ascending=False)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    print("\n" + "=" * 70)
    print("  Test-split results (6,149 images, 102 classes)")
    print("=" * 70)
    for _, r in table.iterrows():
        print(f"  {r['label']:<32}{r['test_top1'] * 100:>7.2f}%{r['test_top5'] * 100:>9.2f}%")
    print("=" * 70)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
