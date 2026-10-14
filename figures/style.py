# figures/style.py
"""Shared style of the data figures (Fig 2, Fig 4): inks, family colours, markers, deterministic export."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.record import FIGURES, guarded

INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
POINTWISE, STATISTICS = "#2a78d6", "#eb6834"
# family: colour, in the fixed order of the feature zoo (Fig 1)
FAMILIES = {
    "ImageNet CNN": "#2a78d6",
    "Calibrated IQA": "#eb6834",
    "Medical CNN": "#1baf7a",
    "Self-supervised ViT": "#eda100",
    "Medical vision-language": "#e87ba4",
    "Task-trained on CT (ours)": "#008300",
}
# configuration: short label, family, marker
CONFIGS = {
    "C1": ("C1 VGG-19, pointwise", "ImageNet CNN", "o"),
    "C2": ("C2 LPIPS-VGG, pointwise", "Calibrated IQA", "o"),
    "C3": ("C3 DISTS, statistics", "Calibrated IQA", "s"),
    "C4": ("C4 RadImageNet, pointwise", "Medical CNN", "o"),
    "C5": ("C5 DINOv2-S, pointwise", "Self-supervised ViT", "o"),
    "C6": ("C6 DINOv2-B, pointwise", "Self-supervised ViT", "^"),
    "C7": ("C7 MedSigLIP, pointwise", "Medical vision-language", "o"),
    "C8": ("C8 DINOv2-S, statistics", "Self-supervised ViT", "s"),
    "C9": ("C9 CT dose network v2, pointwise", "Task-trained on CT (ours)", "o"),
    "C10": ("C10 CT dose network v2, statistics", "Task-trained on CT (ours)", "s"),
}
SUFFIXES = ("pdf", "svg", "png")


def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "text.color": INK, "axes.labelcolor": SECONDARY,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "legend.frameon": False,
        "svg.fonttype": "none", "svg.hashsalt": "ldctbackbone", "pdf.fonttype": 42})


def save(figure: plt.Figure, name: str) -> list[Path]:
    """Writes name.pdf, .svg and .png to figures/ without timestamps, so a rerun gives the same bytes."""
    metadata = {"pdf": {"CreationDate": None}, "svg": {"Date": None}, "png": {"Software": None}}
    paths = [guarded(FIGURES / f"{name}.{suffix}") for suffix in SUFFIXES]
    for path, suffix in zip(paths, SUFFIXES):
        figure.savefig(path, dpi=300, metadata=metadata[suffix])
    plt.close(figure)
    return paths
