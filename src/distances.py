import piq
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.feature_nets import FeatureNet


def normalize_channels(feature: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    # return feature / (feature.pow(2).sum(dim=1, keepdim=True)).sqrt() + eps)

    # Moved `eps` inside the square root to prevent sqrt(0) gradient issues
    return feature / (feature.pow(2).sum(dim=1, keepdim=True) + eps).sqrt()


def pointwise_blocks(predicted: list[torch.Tensor], reference: list[torch.Tensor]) -> list[torch.Tensor]:
    return [(normalize_channels(p) - normalize_channels(r)).pow(2).sum(dim=1).mean() for p, r in zip(predicted, reference)]


def local_stats(feature: torch.Tensor, window: int, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    mean = F.avg_pool2d(feature, window)
    variance = (F.avg_pool2d(feature.pow(2), window) - mean.pow(2)).clamp(min=0)
    # eps inside the root: clamp(min=eps) zeroes the gradient in flat windows, sqrt(0) gives NaN
    return mean, (variance + eps).sqrt()


def statistics_blocks(predicted: list[torch.Tensor], reference: list[torch.Tensor], window: int = 4,
                      c1: float = 1e-6, c2: float = 1e-6) -> list[torch.Tensor]:
    blocks = []
    for p, r in zip(predicted, reference):
        (mp, sp), (mr, sr) = local_stats(normalize_channels(p), window), local_stats(normalize_channels(r), window)
        similarity = (2 * mp * mr + c1) / (mp.pow(2) + mr.pow(2) + c1) * (2 * sp * sr + c2) / (sp.pow(2) + sr.pow(2) + c2)
        blocks.append((1 - similarity).mean())
    return blocks


class FeatureDistance(nn.Module):
    """Sum over feature blocks. block_scales (one per block) gives equal per-block weighting; None is a plain sum."""

    def __init__(self, net: FeatureNet, matching: str, block_scales: list[float] | None = None) -> None:
        super().__init__()
        self.net = net.eval().requires_grad_(False)
        self.blocks = pointwise_blocks if matching == "pointwise" else statistics_blocks
        self.register_buffer("block_scales", None if block_scales is None else torch.tensor(block_scales))

    def block_distances(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reference = self.net.features(target)
        return torch.stack(self.blocks(self.net.features(prediction), reference))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        blocks = self.block_distances(prediction, target)
        return (blocks if self.block_scales is None else blocks * self.block_scales).sum()


class PiqDistance(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.metric = (piq.LPIPS(reduction="mean") if name == "lpips_vgg" else piq.DISTS(reduction="mean")).eval()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.metric(prediction.clamp(0, 1).expand(-1, 3, -1, -1), target.expand(-1, 3, -1, -1))


def shared_crop(prediction: torch.Tensor, target: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    top = torch.randint(0, prediction.shape[-2] - size + 1, (1,)).item()
    left = torch.randint(0, prediction.shape[-1] - size + 1, (1,)).item()
    return prediction[..., top:top + size, left:left + size], target[..., top:top + size, left:left + size]


class PerceptualObjective(nn.Module):
    def __init__(self, distance: nn.Module | None, weight: float, crop_size: int = 224) -> None:
        super().__init__()
        self.distance = distance
        self.weight = weight
        self.crop_size = crop_size

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.l1_loss(prediction, target)
        if self.distance is None or self.weight == 0:
            return loss
        return loss + self.weight * self.distance(*shared_crop(prediction, target, self.crop_size))