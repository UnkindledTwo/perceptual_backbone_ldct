from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    train_dir: Path = PROJECT_ROOT / "dataset/train"
    val_dir: Path = PROJECT_ROOT / "dataset/val"
    test_dir: Path = PROJECT_ROOT / "dataset/test"
    output_dir: Path = PROJECT_ROOT / "outputs/unet"
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints/unet"
    noise_key: str = "noisy_clipped_100k"

    in_channels: int = 1
    out_channels: int = 1
    features: list[int] = None

    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-4
    patience: int = 10

    def __post_init__(self):
        if self.features is None:
            self.features = [64, 128, 256, 512]


CONFIG = Config()
