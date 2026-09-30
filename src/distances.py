import piq
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.feature_nets import FeatureNet


def normalize_channels(feature: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return feature / (feature.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)


def pointwise_distance(predicted: list[torch.Tensor], reference: list[torch.Tensor]) -> torch.Tensor:
    return sum((normalize_channels(p) - normalize_channels(r)).pow(2).sum(dim=1).mean() for p, r in zip(predicted, reference))


def local_stats(feature: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    mean = F.avg_pool2d(feature, window)
    variance = (F.avg_pool2d(feature.pow(2), window) - mean.pow(2)).clamp(min=0)
    return mean, variance.sqrt()


def statistics_distance(predicted: list[torch.Tensor], reference: list[torch.Tensor], window: int = 4,
                        c1: float = 1e-6, c2: float = 1e-6) -> torch.Tensor:
    total = predicted[0].new_zeros(())
    for p, r in zip(predicted, reference):
        (mp, sp), (mr, sr) = local_stats(normalize_channels(p), window), local_stats(normalize_channels(r), window)
        similarity = (2 * mp * mr + c1) / (mp.pow(2) + mr.pow(2) + c1) * (2 * sp * sr + c2) / (sp.pow(2) + sr.pow(2) + c2)
        total = total + (1 - similarity).mean()
    return total


class FeatureDistance(nn.Module):
    def __init__(self, net: FeatureNet, matching: str) -> None:
        super().__init__()
        self.net = net.eval().requires_grad_(False)
        self.distance = pointwise_distance if matching == "pointwise" else statistics_distance

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reference = self.net.features(target)
        return self.distance(self.net.features(prediction), reference)


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