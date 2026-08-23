"""Central configuration for the independent U-Net research pipeline."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent

@dataclass
class TrainingConfig:
    dataset_dir: Path = ML_ROOT / "dataset"; image_dir_name: str = "images"; mask_dir_name: str = "masks"
    image_size: tuple[int, int] = (256, 256); in_channels: int = 1; out_channels: int = 1; base_channels: int = 32
    batch_size: int = 4; epochs: int = 50; learning_rate: float = 1e-3; weight_decay: float = 1e-5; loss_name: str = "bce_dice"; threshold: float = 0.5; seed: int = 42
    train_fraction: float = .70; validation_fraction: float = .15; test_fraction: float = .15; num_workers: int = 0; early_stopping_patience: int = 10; min_component_size: int = 20
    augment: bool = True; allow_horizontal_flip: bool = False; allow_vertical_flip: bool = False
    output_dir: Path = ML_ROOT / "outputs"; checkpoint_dir: Path = ML_ROOT / "checkpoints"
    patient_id_pattern: str = r"^(patient\d+|[A-Za-z0-9]+)(?:[_-](?:slice|image|img|mri)\d+)?$"
    def as_dict(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}

DEFAULT_CONFIG = TrainingConfig()
