# src/dose_net_train.py
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import DOSE_KEYS

DOSE_CLASS_KEYS = (*DOSE_KEYS, "gt_clipped")
DOSE_CLASS_NAMES = ["10k", "50k", "100k", "500k", "1000k", "clean"]
# log10 of the photon count per class. The clean image has no dose, so it gets a fixed value above 1000k.
CLEAN_LOG_DOSE = 7.0
LOG_DOSE = torch.tensor([np.log10(d) for d in (1e4, 5e4, 1e5, 5e5, 1e6)] + [CLEAN_LOG_DOSE], dtype=torch.float32)
LOG_DOSE_MEAN, LOG_DOSE_STD = LOG_DOSE.mean().item(), LOG_DOSE.std().item()
TARGET_MODES = ("regression", "ordinal")


def outputs_for(mode: str) -> int:
    """Head width: one standardised log-dose, or one logit per threshold 'class > k'."""
    return 1 if mode == "regression" else len(DOSE_CLASS_NAMES) - 1


def body_mask(image: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Largest connected region above the threshold, with holes (lungs, bowel gas) filled."""
    labels, count = ndimage.label(image > threshold)
    if count == 0:
        return np.zeros(image.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())[1:]
    return ndimage.binary_fill_holes(labels == 1 + int(sizes.argmax()))


def body_fraction_map(mask: np.ndarray, patch: int) -> np.ndarray:
    """Share of body pixels in the patch at every top-left position (integral image)."""
    integral = np.pad(mask.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    inside = integral[patch:, patch:] - integral[:-patch, patch:] - integral[patch:, :-patch] + integral[:-patch, :-patch]
    return inside / patch ** 2


def sample_body_patches(mask: np.ndarray, patch: int, count: int, rng: np.random.Generator,
                        min_fraction: float = 0.9) -> list[tuple[int, int, float]]:
    """Top-left corners of `count` patches that lie inside the body mask.

    A position is valid when at least `min_fraction` of the patch is body. On a slice too small for
    that (neck, apex), the positions closest to the slice's best achievable fraction are used.
    """
    fraction = body_fraction_map(mask, patch)
    valid = fraction >= min_fraction
    if not valid.any():
        valid = fraction >= 0.95 * fraction.max()
    rows, cols = np.nonzero(valid)
    picks = rng.choice(len(rows), size=count, replace=len(rows) < count)
    return [(int(rows[i]), int(cols[i]), float(fraction[rows[i], cols[i]])) for i in picks]


class DosePatches(Dataset):
    """One patch position per slice and draw, shared by the six versions of the slice (one sample per class).

    Same sampling order as the v1 notebook cell, so v1 and v2 see identical patches for the same seed.
    """

    def __init__(self, files: list[Path], seed: int, patch: int = 128, per_slice: int = 4,
                 threshold: float = 0.05, min_fraction: float = 0.9) -> None:
        rng = np.random.default_rng(seed)
        self.patch, self.samples, self.fractions = patch, [], []
        for path in tqdm(files, desc="body masks", leave=False):
            with np.load(path) as sample:
                mask = body_mask(sample["gt_clipped"], threshold)
            for top, left, fraction in sample_body_patches(mask, patch, per_slice, rng, min_fraction):
                self.fractions.append(fraction)
                self.samples.extend((path, top, left, label) for label in range(len(DOSE_CLASS_KEYS)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, top, left, label = self.samples[index]
        with np.load(path) as sample:
            patch = sample[DOSE_CLASS_KEYS[label]][top:top + self.patch, left:left + self.patch].astype(np.float32)
        return torch.from_numpy(patch).unsqueeze(0), label


def dose_loss(outputs: torch.Tensor, labels: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "regression":
        target = (LOG_DOSE.to(outputs.device)[labels] - LOG_DOSE_MEAN) / LOG_DOSE_STD
        return F.smooth_l1_loss(outputs.squeeze(1), target)
    thresholds = torch.arange(outputs.shape[1], device=outputs.device)
    return F.binary_cross_entropy_with_logits(outputs, (labels[:, None] > thresholds).float())


def predict_classes(outputs: torch.Tensor, mode: str) -> torch.Tensor:
    """Class index per patch. Regression picks the nearest class value in log-dose; ordinal counts passed thresholds."""
    if mode == "regression":
        value = outputs.squeeze(1) * LOG_DOSE_STD + LOG_DOSE_MEAN
        return (value[:, None] - LOG_DOSE.to(outputs.device)).abs().argmin(1)
    return (outputs > 0).sum(1)


@torch.no_grad()
def evaluate_dose_net(network: nn.Module, loader: DataLoader, device: torch.device, mode: str) -> dict:
    """Validation loss, 6-class accuracy by decoded class (comparable to v1), log10-dose MAE and the confusion matrix."""
    network.eval()
    outputs, labels = [], []
    for patches, batch_labels in loader:
        outputs.append(network(patches.to(device)).float().cpu())
        labels.append(batch_labels)
    outputs, labels = torch.cat(outputs), torch.cat(labels)
    predictions = predict_classes(outputs, mode)
    if mode == "regression":
        value = outputs.squeeze(1) * LOG_DOSE_STD + LOG_DOSE_MEAN
    else:
        value = LOG_DOSE[predictions]
    hard = torch.isin(labels, torch.tensor([4, 5]))        # 1000k versus clean
    if mode == "regression":
        two_way = torch.where(value[hard] > (LOG_DOSE[4] + LOG_DOSE[5]) / 2, 5, 4)
    else:
        two_way = torch.where(outputs[hard][:, 4] > 0, 5, 4)
    classes = len(DOSE_CLASS_NAMES)
    return {
        "val_loss": dose_loss(outputs, labels, mode).item(),
        "val_accuracy": (predictions == labels).float().mean().item(),
        "val_accuracy_clean_vs_1000k": (two_way == labels[hard]).float().mean().item(),
        "val_mae_log10": (value - LOG_DOSE[labels]).abs().mean().item(),
        "confusion": np.bincount((labels * classes + predictions).numpy(), minlength=classes ** 2).reshape(classes, classes),
    }


def train_dose_net(network: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                   mode: str = "regression", epochs: int = 10, lr: float = 1e-3,
                   checkpoint: Path | None = None) -> list[dict]:
    """Adam, as in v1, but the epoch with the lowest validation loss is kept (and saved to `checkpoint`).

    The best state is loaded back into `network` before returning.
    """
    assert mode in TARGET_MODES, mode
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    best, best_state, log = float("inf"), None, []
    for epoch in range(epochs):
        network.train()
        running, seen = 0.0, 0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = dose_loss(network(patches), labels, mode)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
        report = evaluate_dose_net(network, val_loader, device, mode)
        log.append({"epoch": epoch + 1, "train_loss": running / seen,
                    **{k: v for k, v in report.items() if k != "confusion"}})
        print(f"epoch {epoch + 1:02d}/{epochs} | train {running / seen:.4f} | val {report['val_loss']:.4f} | "
              f"acc {report['val_accuracy']:.2%} | MAE {report['val_mae_log10']:.3f} log10")
        if report["val_loss"] < best:
            best, best_state = report["val_loss"], copy.deepcopy(network.state_dict())
            log[-1]["checkpoint"] = True
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint)
    network.load_state_dict(best_state)
    return log
