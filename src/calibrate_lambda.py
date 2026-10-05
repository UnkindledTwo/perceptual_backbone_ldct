# scripts/calibrate_lambda.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.distances import shared_crop


@torch.no_grad()
def calibrate_weight(model: nn.Module, distance: nn.Module, loader: DataLoader, device: torch.device,
                     target_influence: float = 0.95, crop_size: int = 224, seed: int = 42) -> float:
    torch.manual_seed(seed)
    model.eval()
    pixel_sum, feature_sum = 0.0, 0.0
    for noisy, clean, _ in tqdm(loader, desc="calibrate", leave=False):
        prediction, clean = model(noisy.to(device)), clean.to(device)
        pixel_sum += F.l1_loss(prediction, clean).item()
        feature_sum += distance(*shared_crop(prediction, clean, crop_size)).item()
    return target_influence / (1 - target_influence) * pixel_sum / feature_sum