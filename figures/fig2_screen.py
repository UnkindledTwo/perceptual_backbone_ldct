# figures/fig2_screen.py
"""Fig 2: the training-free screen (validation split). No data, no GPU.

    python -m figures.fig2_screen

(a) distance to the clean image against dose, normalised by the distance at 10k, one line per trained
configuration, pixel L1 dashed. (b) blur penalty B per network, pointwise and statistics side by side.
fig2_screen is the full-width version; fig2_screen_column has the same panels at one-column width, with legends below.
The CT dose network rows are the adopted v2 (regression) network, from its per-slice validation screen.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figures.style import AXIS, CONFIGS, FAMILIES, INK, MUTED, POINTWISE, SECONDARY, STATISTICS, SURFACE, save, setup
from src.record import RESULTS, record

DOSES = [10_000, 50_000, 100_000, 500_000, 1_000_000]
DOSE_NET = "CT dose network v2 (regression)"
DOSE_SCREENS = {m: RESULTS / "screen" / f"ct_dose_net_{m}_v2_regression_val.csv" for m in ("pointwise", "statistics")}
INPUTS = [RESULTS / "zoo_screen_report.csv", RESULTS / "table2_feature_networks.csv", *DOSE_SCREENS.values()]
# bar order of panel (b): name in the screen report, tick label
NETWORKS = [("VGG-19 relu3_2", "VGG-19"), ("LPIPS-VGG", "LPIPS-\nVGG"), ("DISTS", "DISTS"), ("RadImageNet ResNet-50", "RadImage-\nNet"),
            ("DINOv2-S", "DINOv2\n-S"), ("DINOv2-B", "DINOv2\n-B"), ("DINOv2-L", "DINOv2\n-L"), ("CLIP ViT-B/16", "CLIP\nB/16"),
            ("MedSigLIP-448", "Med-\nSigLIP"), (DOSE_NET, "CT dose\nnet v2")]
COLUMN_TICKS = ["VGG\n-19", "LPIPS", "DISTS", "Rad-\nImgNet", "DINO\nv2-S", "DINO\nv2-B", "DINO\nv2-L", "CLIP", "Med-\nSigLIP", "CT\ndose"]


def screen_rows() -> pd.DataFrame:
    """Mean distances per network and matching; the v1 dose-network rows of the Week 1 report are replaced by v2."""
    zoo = pd.read_csv(INPUTS[0])
    zoo = zoo[zoo["Network"] != "CT dose network"][["Network", "Matching", *[f"D_{d}" for d in DOSES], "D_blur"]]
    rows = []
    for matching, path in DOSE_SCREENS.items():
        mean = pd.read_csv(path).mean(numeric_only=True)
        rows.append({"Network": DOSE_NET, "Matching": matching, **{f"D_{d}": mean[f"d_{d}"] for d in DOSES}, "D_blur": mean["d_blur"]})
    frame = pd.concat([zoo, pd.DataFrame(rows)], ignore_index=True)
    frame["B"] = frame["D_blur"] / frame["D_100000"]
    return frame.set_index(["Network", "Matching"])


def check(screen: pd.DataFrame, table2: pd.DataFrame) -> None:
    for _, row in table2.iterrows():
        assert np.isclose(screen.loc[(row["Network"], row["Matching"]), "B"], row["B_phi"], rtol=1e-6), f"{row['Configuration']}: B differs from Table 2"


def dose_panel(ax: Axes, screen: pd.DataFrame, table2: pd.DataFrame, column: bool = False) -> None:
    for _, row in table2.iterrows():
        label, family, marker = CONFIGS[row["Configuration"]]
        values = screen.loc[(row["Network"], row["Matching"]), [f"D_{d}" for d in DOSES]].to_numpy(dtype=float)
        ax.plot(DOSES, values / values[0], color=FAMILIES[family], lw=1.2, marker=marker, ms=3.6, mec=SURFACE, mew=0.5,
                ls="-" if row["Matching"] == "pointwise" else (0, (4, 1.5)), label=row["Configuration"] if column else label)
    pixel = screen.loc[("Pixel L1 (reference)", "-"), [f"D_{d}" for d in DOSES]].to_numpy(dtype=float)
    ax.plot(DOSES, pixel / pixel[0], color=INK, lw=1.0, ls=(0, (1.5, 1.5)), label="Pixel L1" if column else "Pixel L1 (reference)")
    ax.set_xscale("log")
    ax.set_xticks(DOSES, ["10k", "50k", "100k", "500k", "1000k"])
    ax.minorticks_off()
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Dose (photons per ray)")
    ax.set_ylabel("Distance, relative to 10k" if column else "Distance to clean, relative to 10k")
    ax.set_title("(a) Dose ordering: distance falls as dose rises", loc="left", color=INK)
    if column:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=6, handlelength=2.0, columnspacing=0.8,
                  labelspacing=0.3, borderaxespad=0.0, fontsize=5.5)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.02), handlelength=2.6, labelspacing=0.35, borderaxespad=0.2)


def blur_panel(ax: Axes, screen: pd.DataFrame, table2: pd.DataFrame, column: bool = False) -> None:
    trained = set(zip(table2["Network"], table2["Matching"]))
    width = 0.36
    for index, (network, _) in enumerate(NETWORKS):
        for offset, matching, colour in ((-width / 2, "pointwise", POINTWISE), (width / 2, "statistics", STATISTICS)):
            if (network, matching) not in screen.index:
                continue
            filled = (network, matching) in trained
            ax.bar(index + offset, screen.loc[(network, matching), "B"], width - 0.05, color=colour if filled else SURFACE,
                   edgecolor=colour, lw=0.8)
    ax.axhline(1.0, color=SECONDARY, lw=0.8)
    ax.axhline(screen.loc[("Pixel L1 (reference)", "-"), "B"], color=INK, lw=0.8, ls=(0, (1.5, 1.5)))
    if column:
        ax.set_xticks(range(len(NETWORKS)), COLUMN_TICKS, fontsize=5)
    else:
        ax.set_xticks(range(len(NETWORKS)), [tick for _, tick in NETWORKS])
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 2.0)
    ax.set_ylabel("Blur penalty B" if column else "Blur penalty B (blur / noise at equal MSE)")
    ax.set_title("(b) Blur penalty: above 1, blur costs more than 100k noise", loc="left", color=INK)
    handles = [Patch(fc=POINTWISE, ec=POINTWISE, label="Pointwise, trained"), Patch(fc=STATISTICS, ec=STATISTICS, label="Statistics, trained"),
               Patch(fc=SURFACE, ec=MUTED, label="Screen only (not trained)"),
               Line2D([], [], color=INK, lw=0.8, ls=(0, (1.5, 1.5)), label="Pixel L1 (reference)"),
               Line2D([], [], color=SECONDARY, lw=0.8, label="B = 1")]
    if column:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, columnspacing=0.8,
                  labelspacing=0.3, borderaxespad=0.0, fontsize=5.5)
    else:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.02), borderaxespad=0.2, labelspacing=0.35)


def draw(column: bool = False) -> plt.Figure:
    setup()
    if column:
        plt.rcParams.update({"font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5})
    screen, table2 = screen_rows(), pd.read_csv(INPUTS[1])
    check(screen, table2)
    figure, axes = plt.subplots(2, 1, figsize=(3.4, 3.6) if column else (7.0, 5.0), constrained_layout=True)
    dose_panel(axes[0], screen, table2, column)
    blur_panel(axes[1], screen, table2, column)
    for ax in axes:
        ax.spines["left"].set_color(AXIS)
    return figure


if __name__ == "__main__":
    record("fig2_screen", "python -m figures.fig2_screen", INPUTS,
           save(draw(), "fig2_screen") + save(draw(column=True), "fig2_screen_column"))
