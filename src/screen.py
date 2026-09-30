from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from scipy.stats import spearmanr
from tqdm import tqdm

from src.config import DOSE_KEYS
from src.data import load_slice
from src.distances import FeatureDistance

DOSES: tuple[int, ...] = (10_000, 50_000, 100_000, 500_000, 1_000_000)


def matched_blur(clean: torch.Tensor, noisy: torch.Tensor, steps: int = 20) -> torch.Tensor:
    target_mse, low, high = torch.mean((noisy - clean) ** 2).item(), 0.05, 8.0
    for _ in range(steps):
        sigma = (low + high) / 2
        blurred = TF.gaussian_blur(clean, kernel_size=2 * int(4 * sigma) + 1, sigma=sigma)
        low, high = (sigma, high) if torch.mean((blurred - clean) ** 2).item() < target_mse else (low, sigma)
    return blurred


@torch.no_grad()
def screen_network(distance: FeatureDistance, files: list[Path], device: torch.device,
                   crop: int = 224) -> pd.DataFrame:
    rows = []
    for path in tqdm(files, desc="screen", leave=False):
        clean = load_slice(path, "gt_clipped")[None, :, 144:144 + crop, 144:144 + crop].to(device)
        noisy = {dose: load_slice(path, key)[None, :, 144:144 + crop, 144:144 + crop].to(device)
                 for dose, key in zip(DOSES, DOSE_KEYS)}
        distances = [distance(noisy[dose], clean).item() for dose in DOSES]
        rows.append({
            "slice_id": path.stem,
            "ordering": spearmanr(distances, np.log(1 / np.array(DOSES))).statistic,
            **{f"d_{dose}": value for dose, value in zip(DOSES, distances)},
            "d_blur": distance(matched_blur(clean, noisy[100_000]), clean).item(),
        })
    frame = pd.DataFrame(rows)
    frame.attrs["blur_penalty"] = frame["d_blur"].sum() / frame["d_100000"].sum()
    return frame