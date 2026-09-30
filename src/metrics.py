from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import piq
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

LPIPS_ALEX = lpips.LPIPS(net="alex").eval()
DISTS_METRIC = piq.DISTS(reduction="none").eval()


def high_band_ratio(prediction: np.ndarray, target: np.ndarray, low: float = 0.25, high: float = 0.5) -> float:
    def band_power(image: np.ndarray) -> float:
        power = np.abs(np.fft.fft2(image - image.mean())) ** 2
        fy, fx = np.meshgrid(np.fft.fftfreq(image.shape[0]), np.fft.fftfreq(image.shape[1]), indexing="ij")
        radius = np.sqrt(fx ** 2 + fy ** 2)
        return float(power[(radius >= low) & (radius <= high)].sum())

    return band_power(prediction) / band_power(target)


@torch.no_grad()
def slice_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred, ref = (torch.from_numpy(z).float()[None, None].expand(1, 3, *z.shape) for z in (prediction, target))
    return {
        "psnr": float(peak_signal_noise_ratio(target, prediction, data_range=1.0)),
        "ssim": float(structural_similarity(target, prediction, data_range=1.0)),
        "lpips": float(LPIPS_ALEX(pred * 2 - 1, ref * 2 - 1)),
        "dists": float(DISTS_METRIC(pred.clamp(0, 1), ref.clamp(0, 1))),
        "r_h": high_band_ratio(prediction, target),
    }


def save_per_slice_metrics(rows: list[dict[str, float | str | int]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame