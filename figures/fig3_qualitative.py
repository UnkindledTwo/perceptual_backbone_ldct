# figures/fig3_qualitative.py
"""Fig 3: one test slice at seeds 42, 123 and 456. No data, no GPU.

    python -m figures.fig3_qualitative

Reads the predictions exported by the EXPORT_FIG3 cell of seeds_perceptual_backbone_ldct.ipynb (copied to
results/week3/fig3/seed{S}/). Rows: one per seed, then |prediction - NDCT| at seed 42. Every image uses the same display
range and every error map the same scale. The slice is the first row of fig3_slices.csv (closest to the median
seed-averaged C0 PSNR), not chosen by eye. Hashes and PSNR values are checked against the export and the logged metrics.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from figures.style import INK, SECONDARY, SURFACE, save, setup
from src.record import OUT, record, sha256

FIG3 = OUT / "fig3"
SEEDS = (42, 123, 456)
CONFIGS = ["C0", "C1", "C3", "C5", "C7", "C8", "C10"]
TITLES = {"LDCT": "LDCT (100k)", "C0": "C0 L1", "C1": "C1 VGG-19\npointwise", "C3": "C3 DISTS\nstatistics",
          "C5": "C5 DINOv2-S\npointwise", "C7": "C7 MedSigLIP\npointwise", "C8": "C8 DINOv2-S\nstatistics",
          "C10": "C10 CT dose v2\nstatistics", "NDCT": "NDCT"}
COLUMNS = ["LDCT", *CONFIGS, "NDCT"]
DISPLAY = (0.0, 1.0)          # same grey range for every image (data are in [0, 1])
ERROR_MAX = 0.10              # same scale for every error map
ERROR_SEED = 42
PSNR_TOLERANCE = 1e-3


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


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(10 * np.log10(1.0 / np.mean((prediction.astype(np.float64) - target) ** 2)))


def draw(stem: str) -> plt.Figure:
    setup()
    images = {seed: load(seed, stem)[0] for seed in SEEDS}
    rows = len(SEEDS) + 1
    figure, axes = plt.subplots(rows, len(COLUMNS), figsize=(7.0, 0.70 * rows + 0.6), gridspec_kw={"wspace": 0.03, "hspace": 0.06})
    figure.subplots_adjust(left=0.05, right=0.925, top=0.85, bottom=0.01)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for column, name in enumerate(COLUMNS):
        axes[0, column].set_title(TITLES[name], fontsize=6, color=INK, pad=3)
    for row, seed in enumerate(SEEDS):
        arrays = images[seed]
        axes[row, 0].set_ylabel(f"seed {seed}", fontsize=6.5, color=SECONDARY)
        for column, name in enumerate(COLUMNS):
            ax = axes[row, column]
            ax.imshow(arrays[name], cmap="gray", vmin=DISPLAY[0], vmax=DISPLAY[1], interpolation="nearest")
            if name != "NDCT":
                ax.text(0.03, 0.04, f"{psnr(arrays[name], arrays['NDCT']):.2f} dB", transform=ax.transAxes, fontsize=5,
                        color=SURFACE, va="bottom", ha="left")
    error_row = axes[len(SEEDS)]
    error_row[0].set_ylabel(f"|error|\nseed {ERROR_SEED}", fontsize=6.5, color=SECONDARY)
    reference = images[ERROR_SEED]
    for column, name in enumerate(COLUMNS):
        if name == "NDCT":
            error_row[column].set_visible(False)
            continue
        error_row[column].imshow(np.abs(reference[name] - reference["NDCT"]), cmap="magma", vmin=0, vmax=ERROR_MAX, interpolation="nearest")
    bar = figure.add_axes([0.93, 0.02, 0.008, 0.19])
    figure.colorbar(ScalarMappable(Normalize(0, ERROR_MAX), "magma"), cax=bar).ax.tick_params(labelsize=5, length=2)
    figure.suptitle(f"Test slice {stem} (closest to the median C0 PSNR). Grey range {DISPLAY[0]:g} to {DISPLAY[1]:g}; "
                    f"error scale 0 to {ERROR_MAX:g}", fontsize=6, color=SECONDARY, y=0.985)
    return figure


if __name__ == "__main__":
    stem = slice_id()
    inputs = [OUT / "fig3_slices.csv", OUT / "per_slice_metrics.csv",
              *[FIG3 / f"seed{seed}" / name for seed in SEEDS for name in ("fig3_export.json", f"{stem.removesuffix('.png')}.npz")]]
    record("fig3_qualitative", "python -m figures.fig3_qualitative", inputs, save(draw(stem), "fig3_qualitative"),
           {"slice": stem, "seeds": SEEDS, "configurations": CONFIGS, "display_range": DISPLAY, "error_max": ERROR_MAX,
            "error_seed": ERROR_SEED})
