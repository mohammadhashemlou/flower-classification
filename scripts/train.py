"""Command-line training entry point.

Usage examples
--------------
    # Baseline CNN
    python scripts/train.py --model baseline --epochs 30

    # Residual CNN
    python scripts/train.py --model residual --epochs 60

    # Transfer learning — phase 1 (frozen backbone)
    python scripts/train.py --model transfer --backbone efficientnetb0 --epochs 20

    # Transfer learning — phase 2 (fine-tune)
    python scripts/train.py --model transfer --backbone efficientnetb0 \
        --epochs 15 --fine-tune --init-from models/transfer_phase1/best_model.keras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Register CUDA DLL directories BEFORE TensorFlow is imported
from gpu_env import setup_cuda_dlls  # noqa: E402

setup_cuda_dlls(verbose=False)

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from src.config import Config  # noqa: E402
from src.data_loader import (  # noqa: E402
    compute_class_weights,
    compute_class_weights_trainval,
    create_all_pipelines,
    create_trainval_pipelines,
)
from src.evaluation import evaluate_model  # noqa: E402
from src.models.baseline_cnn import build_baseline_cnn  # noqa: E402
from src.models.residual_cnn import build_residual_cnn  # noqa: E402
from src.models.transfer_learning import (  # noqa: E402
    build_transfer_model,
    unfreeze_backbone,
)
from src.training import (  # noqa: E402
    build_cosine_lr_schedule,
    compile_model,
    train_model,
)
from src.utils import configure_gpu, plot_training_history, set_global_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a flower classification model on Oxford Flowers 102."
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "residual", "transfer"],
        default="transfer",
        help="Which architecture to train (default: transfer).",
    )
    parser.add_argument(
        "--backbone",
        default="efficientnetb0",
        choices=["efficientnetb0", "efficientnetb3", "resnet50", "mobilenetv3"],
        help="Backbone for the transfer model (default: efficientnetb0).",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate.")
    parser.add_argument(
        "--experiment",
        default=None,
        help="Experiment name (default derived from --model).",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Unfreeze the top backbone layers for fine-tuning.",
    )
    parser.add_argument(
        "--unfreeze-layers",
        type=int,
        default=30,
        help="How many backbone layers to unfreeze when --fine-tune is set.",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="Path to a .keras file to initialise weights from (for fine-tuning).",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weighting.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor (default: 0.1).",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip the final test-set evaluation.",
    )
    parser.add_argument(
        "--cosine",
        action="store_true",
        help="Use a cosine LR schedule (with warmup) instead of ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--head-units",
        type=int,
        default=0,
        help="Hidden units in the classification head (0 = compact linear head).",
    )
    parser.add_argument(
        "--train-on-trainval",
        action="store_true",
        help="Merge the official val split into training (2x data) and hold "
        "out 10%% of the merged pool for monitoring. Test split stays untouched.",
    )
    return parser.parse_args()


def build_model(args: argparse.Namespace, cfg: Config) -> keras.Model:
    """Instantiate the requested architecture.

    When ``--init-from`` points at a saved ``.keras`` model, the whole model
    (architecture + weights) is restored instead of re-creating it and
    copying weights. This is required for fine-tuning: the restored model
    already contains the trained head, so unfreezing its backbone continues
    from the phase-1 optimum rather than from ImageNet weights.
    """
    if args.model == "baseline":
        return build_baseline_cnn(cfg)

    if args.model == "residual":
        return build_residual_cnn(cfg, augment=not args.no_augment)

    # ── transfer ────────────────────────────────────────────────────
    if args.init_from:
        init_path = Path(args.init_from)
        if not init_path.exists():
            raise FileNotFoundError(f"--init-from path not found: {init_path}")

        print(f"Restoring model from: {init_path}")
        model = keras.models.load_model(str(init_path), compile=False)

        if args.fine_tune:
            unfreeze_backbone(model, num_layers_to_unfreeze=args.unfreeze_layers)

        return model

    model = build_transfer_model(
        cfg,
        backbone=args.backbone,
        freeze_backbone=not args.fine_tune,
        augment=not args.no_augment,
        head_units=args.head_units,
    )

    if args.fine_tune:
        unfreeze_backbone(model, num_layers_to_unfreeze=args.unfreeze_layers)

    return model


def main() -> None:
    args = parse_args()

    cfg = Config()
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.initial_lr = args.lr

    set_global_seed(cfg.seed)
    gpus = configure_gpu()

    print("=" * 64)
    print("  Flower Classification — Training")
    print("=" * 64)
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  Keras      : {keras.__version__}")
    print(f"  GPUs       : {len(gpus)}")
    print(f"  Model      : {args.model}")
    if args.model == "transfer":
        print(f"  Backbone   : {args.backbone}")
        print(f"  Fine-tune  : {args.fine_tune}")
    print(f"  Epochs     : {cfg.epochs}")
    print(f"  Batch size : {cfg.batch_size}")
    print(f"  LR         : {cfg.initial_lr}")
    print("=" * 64 + "\n")

    # ── Data ────────────────────────────────────────────────────────
    if args.train_on_trainval:
        train_ds, val_ds, test_ds = create_trainval_pipelines(cfg, val_fraction=0.1)
        class_weight = (
            None if args.no_class_weights else compute_class_weights_trainval(cfg, 0.1)
        )
    else:
        train_ds, val_ds, test_ds = create_all_pipelines(cfg, augment_train=False)
        class_weight = None if args.no_class_weights else compute_class_weights(cfg)

    # ── Model ───────────────────────────────────────────────────────
    model = build_model(args, cfg)

    # Transfer learning needs a lower LR for fine-tuning
    lr = cfg.initial_lr
    if args.model == "transfer" and args.fine_tune:
        lr = min(lr, 1e-4)

    # Keep cfg in sync so the cosine schedule decays from the effective LR
    cfg.initial_lr = lr

    if args.cosine:
        steps_per_epoch = int(tf.data.experimental.cardinality(train_ds).numpy())
        schedule = build_cosine_lr_schedule(cfg, steps_per_epoch)
        compile_model(
            model, cfg, learning_rate=schedule, label_smoothing=args.label_smoothing
        )
    else:
        compile_model(model, cfg, learning_rate=lr, label_smoothing=args.label_smoothing)

    experiment = args.experiment or f"{args.model}"
    if args.model == "transfer":
        suffix = "finetune" if args.fine_tune else "frozen"
        experiment = args.experiment or f"transfer_{args.backbone}_{suffix}"

    # ── Train ───────────────────────────────────────────────────────
    train_model(
        model,
        train_ds,
        val_ds,
        cfg,
        experiment_name=experiment,
        class_weight=class_weight,
        use_cosine_schedule=args.cosine,
        label_smoothing=args.label_smoothing,
    )

    # ── Curves ──────────────────────────────────────────────────────
    exp_dir = cfg.experiment_dir(experiment)
    plot_training_history(exp_dir / "history.csv", title=experiment)

    # ── Evaluate ────────────────────────────────────────────────────
    if not args.no_evaluate:
        print("\n" + "=" * 64)
        print("  Test-set evaluation")
        print("=" * 64)
        best = exp_dir / "best_model.keras"
        if best.exists():
            eval_model = keras.models.load_model(best, compile=False)
            # A checkpoint loaded with compile=False has no optimizer/metrics,
            # so it must be compiled before .evaluate() can run.
            compile_model(
                eval_model, cfg, learning_rate=lr, label_smoothing=args.label_smoothing
            )
        else:
            eval_model = model
        evaluate_model(eval_model, test_ds)

    print(f"\nArtifacts saved to: {exp_dir}")


if __name__ == "__main__":
    main()
