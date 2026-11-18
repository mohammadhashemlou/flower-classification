"""Evaluate a saved checkpoint on the test split.

Usage
-----
    python scripts/evaluate.py --checkpoint models/transfer_efficientnetb0_frozen/best_model.keras
    python scripts/evaluate.py --checkpoint ... --confusion-matrix --report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_env import setup_cuda_dlls  # noqa: E402

setup_cuda_dlls(verbose=False)

from tensorflow import keras  # noqa: E402

from src.config import Config  # noqa: E402
from src.data_loader import create_data_pipeline  # noqa: E402
from src.evaluation import (  # noqa: E402
    classification_report_csv,
    evaluate_model,
    plot_confusion_matrix,
)
from src.utils import set_global_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a flower classifier checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .keras model file.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Also render a normalized 102x102 confusion matrix.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Also write a per-class precision/recall/F1 CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated artifacts (default: the checkpoint's folder).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    cfg = Config()
    set_global_seed(cfg.seed)

    print(f"Loading model: {ckpt}")
    # compile=False: we only run inference, and skipping compilation avoids
    # needing the exact training-time loss/optimizer to be deserializable.
    model = keras.models.load_model(str(ckpt), compile=False)

    print(f"Building '{args.split}' pipeline...")
    ds = create_data_pipeline(cfg, split=args.split)

    print("\n" + "=" * 56)
    evaluate_model(model, ds)
    print("=" * 56)

    out_dir = Path(args.output_dir) if args.output_dir else ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.confusion_matrix:
        print("\nRendering confusion matrix (this may take a minute)...")
        plot_confusion_matrix(
            model,
            ds,
            num_classes=cfg.num_classes,
            save_path=out_dir / f"confusion_matrix_{args.split}.png",
        )

    if args.report:
        print("\nGenerating per-class report...")
        classification_report_csv(
            model,
            ds,
            save_path=out_dir / f"classification_report_{args.split}.csv",
        )

    print(f"\nArtifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
