"""Compare every trained experiment side-by-side.

Reads ``history.csv`` from each run under ``models/``, prints a ranked
summary table, and writes a comparison plot to ``results/``.

Usage
-----
    python scripts/compare.py
    python scripts/compare.py --output results/comparison.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.config import Config  # noqa: E402

# Human-readable labels for experiment directories
PRETTY_NAMES = {
    "01_baseline_cnn": "Baseline CNN (from scratch)",
    "02_residual_cnn": "Residual CNN (from scratch)",
    "03_efficientnet_frozen": "EfficientNetB0 frozen",
    "04_efficientnet_finetuned": "EfficientNetB0 fine-tuned",
    "05_efficientnet_frozen_x2": "EfficientNetB0 frozen (train+val)",
    "06_efficientnet_finetuned_x2": "EfficientNetB0 fine-tuned (train+val)",
}


def collect(cfg: Config) -> pd.DataFrame:
    """Read every history.csv under models/ into one DataFrame."""
    rows = []
    for history in sorted(cfg.models_dir.glob("*/history.csv")):
        name = history.parent.name
        try:
            df = pd.read_csv(history)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  ! skipping {name}: {exc}")
            continue
        if df.empty or "val_accuracy" not in df.columns:
            continue

        best_idx = df["val_accuracy"].idxmax()
        rows.append(
            {
                "experiment": name,
                "label": PRETTY_NAMES.get(name, name),
                "epochs": len(df),
                "best_val_acc": float(df.loc[best_idx, "val_accuracy"]),
                "best_val_epoch": int(df.loc[best_idx, "epoch"]) + 1,
                "best_val_loss": float(df["val_loss"].min()),
                "final_train_acc": float(df["accuracy"].iloc[-1]),
                "overfit_gap": float(df["accuracy"].iloc[-1] - df.loc[best_idx, "val_accuracy"]),
            }
        )

    return pd.DataFrame(rows)


def plot_comparison(cfg: Config, out_path: Path) -> None:
    """Plot validation accuracy curves for every experiment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histories = sorted(cfg.models_dir.glob("*/history.csv"))
    if not histories:
        print("No histories found — nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for history in histories:
        name = history.parent.name
        df = pd.read_csv(history)
        if "val_accuracy" not in df.columns:
            continue
        ax.plot(
            df["epoch"] + 1,
            df["val_accuracy"] * 100,
            linewidth=1.8,
            label=PRETTY_NAMES.get(name, name),
        )

    ax.set_title("Validation accuracy — all experiments", fontsize=13, weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare training experiments.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "comparison.png"),
        help="Where to save the comparison plot.",
    )
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "results" / "comparison.csv"),
        help="Where to save the summary table.",
    )
    args = parser.parse_args()

    cfg = Config()
    table = collect(cfg)

    if table.empty:
        print("No experiment histories found under models/.")
        return

    table = table.sort_values("best_val_acc", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 92)
    print("  Experiment comparison")
    print("=" * 92)
    print(
        f"{'#':<3}{'Model':<32}{'Epochs':>7}{'Best Val':>10}{'@Ep':>5}"
        f"{'Val Loss':>10}{'Train':>9}{'Gap':>8}"
    )
    print("-" * 92)
    for i, row in table.iterrows():
        print(
            f"{i + 1:<3}{row['label']:<32}{row['epochs']:>7}"
            f"{row['best_val_acc'] * 100:>9.2f}%{row['best_val_epoch']:>5}"
            f"{row['best_val_loss']:>10.3f}{row['final_train_acc'] * 100:>8.2f}%"
            f"{row['overfit_gap'] * 100:>7.1f}%"
        )
    print("=" * 92 + "\n")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    print(f"Saved table: {csv_path}")

    plot_comparison(cfg, Path(args.output))


if __name__ == "__main__":
    main()
