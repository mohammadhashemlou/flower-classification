"""Central configuration for the flower classification project.

All hyperparameters, paths, and training settings live here so every
module imports from a single source of truth.
"""

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Config:
    """Project-wide configuration."""

    # ── Paths ────────────────────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parent.parent

    dataset_dir: Path = field(init=False)
    images_dir: Path = field(init=False)

    models_dir: Path = field(init=False)
    experiments_dir: Path = field(init=False)

    # ── Image ────────────────────────────────────────────────────────
    img_height: int = 224
    img_width: int = 224
    num_channels: int = 3

    # ── Dataset ──────────────────────────────────────────────────────
    num_classes: int = 102
    batch_size: int = 32

    # ── Training ─────────────────────────────────────────────────────
    epochs: int = 50
    initial_lr: float = 1e-3
    min_lr: float = 1e-6
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5

    # ── Reproducibility ──────────────────────────────────────────────
    seed: int = 42

    def __post_init__(self) -> None:
        self.dataset_dir = self.project_root / "dataset" / "raw" / "oxford_flowers102"
        self.images_dir = self.dataset_dir / "jpg"
        self.models_dir = self.project_root / "models"
        self.experiments_dir = self.models_dir

        # Ensure model directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def experiment_dir(self, name: str) -> Path:
        """Return (and create) a sub-directory for a named experiment."""
        path = self.models_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path
