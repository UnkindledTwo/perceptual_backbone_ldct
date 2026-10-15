# figures/fig1_overview.py
"""Fig 1: overview of the benchmark (final protocol: CT dose network v2, three seeds). No data, no GPU.

    python -m figures.fig1_overview

Writes fig1_overview.svg (editable), .pdf and .png to figures/ without timestamps and records them in
results/week3/manifest.json.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figures.style import save
from src.record import record
from src.tables import FEATURE_EXTRACTOR_PARAMETERS

INK, MUTED, RULE, PANEL = "#111111", "#5c5b57", "#c9c8c2", "#f5f4f0"
BLUE, ORANGE, GREEN = "#2a78d6", "#d9622b", "#1b9a6c"
W, H = 20.0, 10.6

# family, networks, configurations
ZOO = [
    ("ImageNet CNN", "VGG-19 relu3_2", "C1"),
    ("Calibrated IQA", "LPIPS-VGG · DISTS", "C2 · C3"),
    ("Medical CNN", "RadImageNet ResNet-50", "C4"),
    ("Self-supervised ViT", "DINOv2-S · DINOv2-B", "C5, C8 · C6"),
    ("Medical vision-language", "MedSigLIP-448", "C7"),
    ("Task-trained on CT (ours)", f"CT dose network v2, {FEATURE_EXTRACTOR_PARAMETERS / 1e6:.2f} M", "C9, C10"),
]


def panel(ax: Axes, x: float, y: float, w: float, h: float, title: str, colour: str) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18", fc=PANEL, ec=RULE, lw=1))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.62), w, 0.62, boxstyle="round,pad=0,rounding_size=0.18", fc=colour, ec="none"))
    ax.text(x + 0.25, y + h - 0.31, title, fontsize=10.5, color="white", fontweight="bold", va="center")


def box(ax: Axes, x: float, y: float, w: float, h: float, text: str, size: float = 8.5, colour: str = INK,
        edge: str = RULE, weight: str = "normal") -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1", fc="white", ec=edge, lw=1))
    ax.text(x + w / 2, y + h / 2, text, fontsize=size, color=colour, ha="center", va="center", fontweight=weight,
            linespacing=1.3)


def arrow(ax: Axes, start: tuple[float, float], end: tuple[float, float], colour: str = INK, style: str = "-",
          curve: float = 0.0) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, lw=1.4, color=colour, linestyle=style,
                                 connectionstyle=f"arc3,rad={curve}", shrinkA=0, shrinkB=0))


def draw() -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "svg.fonttype": "none",
                         "svg.hashsalt": "ldctbackbone", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(W / 2, H / 2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # left: feature zoo
    panel(ax, 0.2, 2.6, 6.2, 7.8, r"Frozen feature network $\phi$", BLUE)
    for row, (family, networks, configs) in enumerate(ZOO):
        y = 8.55 - row * 0.98
        ours = "ours" in family
        ax.text(0.45, y + 0.6, family, fontsize=7.5, color=ORANGE if ours else MUTED, fontweight="bold" if ours else "normal")
        box(ax, 0.45, y - 0.12, 4.25, 0.62, networks, 8.5, edge=ORANGE if ours else RULE)
        ax.text(4.85, y + 0.19, configs, fontsize=8.5, color=BLUE, fontweight="bold", va="center")
    ax.text(0.45, 2.78, "Screen only: DINOv2-L, CLIP ViT-B/16.\nC0: L1 only, no feature network.", fontsize=7.2, color=MUTED,
            linespacing=1.35)

    # middle: training-free screen
    panel(ax, 6.9, 2.6, 5.0, 7.8, "Training-free screen", GREEN)
    box(ax, 7.15, 7.55, 4.5, 1.85, "Dose ordering  $M_\\phi$\n\nSpearman of $d_\\phi$(dose image, target)\nwith dose, 10k to 1000k", 8.5)
    box(ax, 7.15, 5.35, 4.5, 1.85, "Blur penalty  $B_\\phi$\n\n$d_\\phi$ of an equal-MSE blur\nrelative to the 100k image", 8.5)
    ax.text(9.4, 4.55, "Validation split, 182 slices.\nBoth matchings per network.", fontsize=8, color=INK, ha="center", va="center",
            linespacing=1.3)
    ax.text(9.4, 3.3, "A prediction, not a filter:\nevery C1 to C10 is trained.", fontsize=8, color=GREEN, ha="center", va="center",
            fontweight="bold", linespacing=1.3)

    # right: matched training
    panel(ax, 12.4, 2.6, 7.4, 7.8, r"$\Psi$-matched training of one fixed U-Net", ORANGE)
    box(ax, 12.65, 8.45, 1.85, 0.95, "LDCT\n100k", 8.5)
    box(ax, 15.1, 8.45, 1.9, 0.95, "U-Net\n(fixed)", 8.5, weight="bold")
    box(ax, 17.6, 8.45, 1.95, 0.95, "prediction\n$\\hat{x}$", 8.5)
    arrow(ax, (14.5, 8.92), (15.1, 8.92))
    arrow(ax, (17.0, 8.92), (17.6, 8.92))
    box(ax, 12.65, 6.75, 6.9, 1.15,
        "$\\mathcal{L}_\\phi = \\|\\hat{x} - x\\|_1 + \\lambda_\\phi\\, d_\\phi(\\mathrm{crop}(\\hat{x}),\\ \\mathrm{crop}(x))$", 10.5,
        edge=ORANGE)
    arrow(ax, (18.57, 8.45), (18.57, 7.9))
    box(ax, 12.65, 5.2, 3.3, 1.2, "Pointwise\nchannel-normalised\nfeatures, per position", 7.8)
    box(ax, 16.25, 5.2, 3.3, 1.2, "Local statistics\nmean and spread,\n4×4 windows", 7.8)
    arrow(ax, (14.3, 6.4), (14.3, 6.75))
    arrow(ax, (17.9, 6.4), (17.9, 6.75))
    ax.text(16.1, 5.8, "or", fontsize=8, color=MUTED, ha="center", va="center")
    ax.text(12.65, 4.45, "$\\lambda_\\phi$: perceptual share 95% at initialisation ($\\Psi^\\star = 0.95$)",
            fontsize=7.8, color=INK)
    ax.text(12.65, 3.85, "One shared random 224×224 crop per batch for $\\hat{x}$ and $x$", fontsize=7.8, color=INK)
    ax.text(12.65, 3.25, "AdamW, 30 epochs, lowest validation L1, seeds 42 / 123 / 456", fontsize=7.8, color=INK)

    # bottom: evaluation
    panel(ax, 0.2, 0.2, 19.6, 1.95, "Evaluation (test split, 220 slices, 31 patients, touched once per run)", INK)
    cells = ((0.45, 3.0, "Fidelity\nPSNR, SSIM"), (3.65, 3.0, "Perception\nLPIPS, DISTS"),
             (12.4, 3.5, "Texture\nhigh-band NPS ratio $R_H$"), (16.1, 3.45, "Paired Wilcoxon, Holm\nper slice and per patient"))
    for x, width, text in cells:
        box(ax, x, 0.38, width, 0.95, text, 8)
    box(ax, 6.9, 0.38, 5.0, 0.95, "Does the screen predict the outcome?\nSpearman of $M_\\phi$, $B_\\phi$ with test metrics", 7.6,
        colour=GREEN, edge=GREEN, weight="bold")

    arrow(ax, (6.4, 6.5), (6.9, 6.5))
    arrow(ax, (11.9, 6.5), (12.4, 6.5))
    arrow(ax, (16.1, 2.6), (16.1, 2.15))
    arrow(ax, (9.4, 2.6), (9.4, 2.15), GREEN, "--")
    return fig


if __name__ == "__main__":
    record("fig1_overview", "python -m figures.fig1_overview", [Path(__file__)], save(draw(), "fig1_overview"),
           {"feature_extractor_parameters": FEATURE_EXTRACTOR_PARAMETERS})
