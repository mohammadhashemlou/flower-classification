"""Project-wide configuration.

Everything tunable lives here so the rest of the code has one place to read from.
"""

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Config:
    """Paths and hyperparameters shared by the whole project."""

    project_root: Path = Path(__file__).resolve().parent.parent

    dataset_dir: Path = field(init=False)
    images_dir: Path = field(init=False)

    models_dir: Path = field(init=False)
    experiments_dir: Path = field(init=False)

    img_height: int = 224
    img_width: int = 224
    num_channels: int = 3

    num_classes: int = 102
    batch_size: int = 32

    epochs: int = 50
    initial_lr: float = 1e-3
    min_lr: float = 1e-6
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5

    seed: int = 42

    def __post_init__(self) -> None:
        self.dataset_dir = self.project_root / "dataset" / "raw" / "oxford_flowers102"
        self.images_dir = self.dataset_dir / "jpg"
        self.models_dir = self.project_root / "models"
        self.experiments_dir = self.models_dir

        self.models_dir.mkdir(parents=True, exist_ok=True)

    def experiment_dir(self, name: str) -> Path:
        """Return the output directory for an experiment, creating it if needed."""
        path = self.models_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path
