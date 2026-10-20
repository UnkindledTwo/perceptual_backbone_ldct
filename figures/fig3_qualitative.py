# figures/fig3_qualitative.py
"""Fig 3: one test slice at seed 42 with a soft-tissue zoom. No data, no GPU.

    python -m figures.fig3_qualitative

Reads the predictions exported by the EXPORT_FIG3 cell of seeds_perceptual_backbone_ldct.ipynb (copied to
results/week3/fig3/seed{S}/). Rows: full slice, zoom, |prediction - NDCT|. Every image uses the same display range and
every error map the same scale. The slice is the first row of fig3_slices.csv (closest to the median seed-averaged C0
PSNR) and the zoom is chosen by `zoom_window` on the NDCT, neither by eye. Hashes and PSNR values are checked against the
export and the logged metrics. Seeds 123 and 456 were exported too but add nothing visible, so the figure shows seed 42.

fig3_qualitative_texture is the paper version: the same figure with the zoom row in the soft-tissue window, so the
grain is visible. fig3_qualitative is unchanged.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

from figures.style import INK, SECONDARY, STATISTICS, SURFACE, save, setup
from src.record import OUT, record, sha256

FIG3 = OUT / "fig3"
SEED = 42
CONFIGS = ["C0", "C1", "C3", "C5", "C7", "C8", "C10"]
TITLES = {"LDCT": "LDCT (100k)", "C0": "C0 L1", "C1": "C1 VGG-19\npointwise", "C3": "C3 DISTS\nstatistics",
          "C5": "C5 DINOv2-S\npointwise", "C7": "C7 MedSigLIP\npointwise", "C8": "C8 DINOv2-S\nstatistics",
          "C10": "C10 CT dose v2\nstatistics", "NDCT": "NDCT"}
COLUMNS = ["LDCT", *CONFIGS, "NDCT"]
DISPLAY = (0.0, 1.0)          # same grey range for every image (data are in [0, 1])
ERROR_MAX = 0.10              # same scale for every error map
PSNR_TOLERANCE = 1e-3
BODY_THRESHOLD = 0.05         # NDCT above this, opened, holes filled, largest component
SOFT_TISSUE = (0.35, 0.75)    # NDCT grey values of the soft-tissue peak (between fat and contrast or bone)
ZOOM_SIZE, ZOOM_STRIDE, ZOOM_MIN_SOFT = 96, 8, 0.9
ZOOM_DISPLAY = SOFT_TISSUE     # texture version: zoom row in the soft-tissue window


def slice_id() -> str:
    return pd.read_csv(OUT / "fig3_slices.csv")["slice_id"].iloc[0]


def load(seed: int, stem: str) -> tuple[dict[str, np.ndarray], dict]:
    """Arrays of one seed, after checking the file hash and the PSNR against the logged per-slice metrics."""
    folder = FIG3 / f"seed{seed}"
    export = json.loads((folder / "fig3_export.json").read_text())
    path = folder / f"{stem.removesuffix('.png')}.npz"
    assert export["seed"] == seed and sha256(path) == export["files_sha256"][path.name], f"seed {seed}: {path.name} is not the export"
    logged = pd.read_csv(OUT / "per_slice_metrics.csv")
    logged = logged[(logged["seed"] == seed) & (logged["slice_id"] == stem)].set_index("configuration")["psnr"]
    for config_id in CONFIGS:
        exported = export["psnr"][f"{config_id}/{stem}"]
        assert abs(exported - logged[config_id]) < PSNR_TOLERANCE, f"seed {seed} {config_id}: PSNR {exported:.4f}, logged {logged[config_id]:.4f}"
    with np.load(path) as arrays:
        return {name: arrays[name] for name in COLUMNS}, export


def zoom_window(ndct: np.ndarray) -> tuple[int, int]:
    """Top-left corner of the ZOOM_SIZE window (grid of ZOOM_STRIDE) with the median NDCT standard deviation among the
    windows that are at least ZOOM_MIN_SOFT soft tissue; ties go to the first window in row-major order."""
    body = ndimage.binary_fill_holes(ndimage.binary_opening(ndct > BODY_THRESHOLD, iterations=3))
    labels, _ = ndimage.label(body)
    body = labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1
    soft = body & (ndct >= SOFT_TISSUE[0]) & (ndct <= SOFT_TISSUE[1])
    shape = (ZOOM_SIZE, ZOOM_SIZE)
    fraction = sliding_window_view(soft, shape)[::ZOOM_STRIDE, ::ZOOM_STRIDE].mean(axis=(2, 3))
    spread = sliding_window_view(ndct.astype(np.float64), shape)[::ZOOM_STRIDE, ::ZOOM_STRIDE].std(axis=(2, 3))
    candidates = np.argwhere(fraction >= ZOOM_MIN_SOFT)
    order = np.argsort(spread[candidates[:, 0], candidates[:, 1]], kind="stable")
    row, column = candidates[order[(len(order) - 1) // 2]] * ZOOM_STRIDE
    return int(row), int(column)


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(10 * np.log10(1.0 / np.mean((prediction.astype(np.float64) - target) ** 2)))


def draw(stem: str, texture: bool = False) -> tuple[plt.Figure, tuple[int, int]]:
    setup()
    arrays = load(SEED, stem)[0]
    top, left = zoom_window(arrays["NDCT"])
    zoom = (slice(top, top + ZOOM_SIZE), slice(left, left + ZOOM_SIZE))
    zoom_range = ZOOM_DISPLAY if texture else DISPLAY
    figure, axes = plt.subplots(3, len(COLUMNS), figsize=(7.0, 2.85), gridspec_kw={"wspace": 0.03, "hspace": 0.06})
    figure.subplots_adjust(left=0.05, right=0.925, top=0.82, bottom=0.01)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    zoom_label = f"zoom\n{zoom_range[0]:g} to {zoom_range[1]:g}" if texture else f"zoom\n{ZOOM_SIZE} px"
    for row, label in enumerate(["image", zoom_label, "|error|"]):
        axes[row, 0].set_ylabel(label, fontsize=6.5, color=SECONDARY)
    for column, name in enumerate(COLUMNS):
        image, zoomed, error = axes[:, column]
        image.set_title(TITLES[name], fontsize=6, color=INK, pad=3)
        image.imshow(arrays[name], cmap="gray", vmin=DISPLAY[0], vmax=DISPLAY[1], interpolation="nearest")
        image.add_patch(Rectangle((left - 0.5, top - 0.5), ZOOM_SIZE, ZOOM_SIZE, fill=False, edgecolor=STATISTICS, linewidth=0.6))
        zoomed.imshow(arrays[name][zoom], cmap="gray", vmin=zoom_range[0], vmax=zoom_range[1], interpolation="nearest")
        if name == "NDCT":
            error.set_visible(False)
            continue
        image.text(0.03, 0.04, f"{psnr(arrays[name], arrays['NDCT']):.2f} dB", transform=image.transAxes, fontsize=5,
                   color=SURFACE, va="bottom", ha="left")
        error.imshow(np.abs(arrays[name] - arrays["NDCT"]), cmap="magma", vmin=0, vmax=ERROR_MAX, interpolation="nearest")
    bar = figure.add_axes([0.93, 0.02, 0.008, 0.25])
    figure.colorbar(ScalarMappable(Normalize(0, ERROR_MAX), "magma"), cax=bar).ax.tick_params(labelsize=5, length=2)
    zoom_text = f" Zoom grey range {zoom_range[0]:g} to {zoom_range[1]:g}." if texture else ""
    figure.suptitle(f"Test slice {stem} (closest to the median C0 PSNR), seed {SEED}. Grey range {DISPLAY[0]:g} to "
                    f"{DISPLAY[1]:g}, error scale 0 to {ERROR_MAX:g}.{zoom_text}\nZoom (box): the {ZOOM_SIZE} px window "
                    f"with the median NDCT std among windows at least {ZOOM_MIN_SOFT:.0%} soft tissue (NDCT "
                    f"{SOFT_TISSUE[0]:g} to {SOFT_TISSUE[1]:g} in the body)", fontsize=6, color=SECONDARY, y=0.99)
    return figure, (top, left)


if __name__ == "__main__":
    stem = slice_id()
    inputs = [OUT / "fig3_slices.csv", OUT / "per_slice_metrics.csv",
              *[FIG3 / f"seed{SEED}" / name for name in ("fig3_export.json", f"{stem.removesuffix('.png')}.npz")]]
    figure, (top, left) = draw(stem)
    settings = {"slice": stem, "seed": SEED, "configurations": CONFIGS, "display_range": DISPLAY, "error_max": ERROR_MAX,
                "zoom": {"top": top, "left": left, "size": ZOOM_SIZE, "stride": ZOOM_STRIDE,
                         "min_soft_tissue": ZOOM_MIN_SOFT, "soft_tissue_range": SOFT_TISSUE,
                         "body_threshold": BODY_THRESHOLD, "rule": "median NDCT std among soft-tissue windows"}}
    record("fig3_qualitative", "python -m figures.fig3_qualitative", inputs, save(figure, "fig3_qualitative"), settings)
    figure, _ = draw(stem, texture=True)
    record("fig3_qualitative_texture", "python -m figures.fig3_qualitative", inputs,
           save(figure, "fig3_qualitative_texture"), {**settings, "zoom_display_range": ZOOM_DISPLAY})
