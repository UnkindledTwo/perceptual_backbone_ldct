from dataclasses import dataclass
from pathlib import Path

DOSE_KEYS: tuple[str, ...] = ("noisy_clipped_10k", "noisy_clipped_50k", "noisy_clipped_100k",
                              "noisy_clipped_500k", "noisy_clipped_1000k")


@dataclass(frozen=True)
class PerceptualConfig:
    run_name: str
    data_root: Path
    feature_net: str = "none"
    matching: str = "pointwise"
    loss_weight: float = 0.0
    output_dir: Path = Path("outputs/perceptual_backbone")
    input_key: str = "noisy_clipped_100k"
    target_key: str = "gt_clipped"
    crop_size: int = 224
    seed: int = 42
    batch_size: int = 8
    grad_accumulation: int = 1
    lr: float = 1e-4
    epochs: int = 30
    use_amp: bool = True

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name