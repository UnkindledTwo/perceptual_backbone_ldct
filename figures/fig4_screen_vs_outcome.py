# figures/fig4_screen_vs_outcome.py
"""Fig 4: screen score against trained outcome. No data, no GPU. Run src.seeds_tables first.

    python -m figures.fig4_screen_vs_outcome

Blur penalty B of each trained network against its trained |1 - R_H| (mean over seeds 42, 123, 456, bars: standard
deviation over seeds). Marker area follows log10 of the parameter count, colour the family; DINOv2-S and -B are joined.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figures.style import CONFIGS, FAMILIES, INK, MUTED, SECONDARY, SURFACE, save, setup
from src.record import OUT, RESULTS, record

INPUTS = [RESULTS / "table2_feature_networks.csv", OUT / "table1_three_seeds.csv", OUT / "screen_vs_outcome.csv"]
SCORE, OUTCOME = "B_phi", "r_h_error"
JOINED = ("C5", "C6")
# label offsets in points, set by hand where points sit close together
OFFSETS = {"C1": (6, 5), "C2": (-6, 7), "C3": (3, -10), "C4": (7, 0), "C5": (7, 4), "C6": (7, -6), "C7": (7, 0), "C8": (-8, 0),
           "C9": (7, 0), "C10": (7, 0)}


def marker_area(parameters: pd.Series) -> pd.Series:
    return 14 + 22 * (np.log10(parameters) - 5)


def draw() -> plt.Figure:
    setup()
    table2 = pd.read_csv(INPUTS[0]).set_index("Configuration")
    outcome = pd.read_csv(INPUTS[1]).set_index("config").loc[table2.index]
    tests = pd.read_csv(INPUTS[2])
    figure, ax = plt.subplots(figsize=(3.4, 3.1), constrained_layout=True)
    ax.plot(table2.loc[list(JOINED), SCORE], outcome.loc[list(JOINED), OUTCOME], color=FAMILIES["Self-supervised ViT"], lw=1.0, zorder=1)
    for config_id, row in table2.iterrows():
        _, family, _ = CONFIGS[config_id]
        x, y = row[SCORE], outcome.loc[config_id, OUTCOME]
        ax.errorbar(x, y, yerr=outcome.loc[config_id, f"{OUTCOME}_seed_std"], color=MUTED, lw=0.7, capsize=1.5, zorder=2)
        ax.scatter(x, y, s=marker_area(row["Parameters"]), marker="o" if row["Matching"] == "pointwise" else "s",
                   color=FAMILIES[family], edgecolor=SURFACE, lw=0.8, zorder=3)
        dx, dy = OFFSETS[config_id]
        ax.annotate(config_id, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color=INK,
                    ha="left" if dx > 0 else "right", va="center")
    ax.axvline(1.0, color=SECONDARY, lw=0.6, ls=(0, (1.5, 1.5)), zorder=0)
    text = []
    for subset, name in (("all trained networks", "all"), ("without C2 and C3", "no C2, C3")):
        row = tests[(tests["screen_score"] == SCORE) & (tests["outcome"] == OUTCOME) & (tests["subset"] == subset)].iloc[0]
        text.append(f"$\\rho$ = {row['spearman_rho']:.2f}, p = {row['permutation_p']:.3f} ({name}, n = {row['n']})")
    ax.text(0.03, 0.97, "\n".join(text), transform=ax.transAxes, fontsize=6, color=SECONDARY, va="top")
    ax.set_xlim(0.2, 2.0)
    ax.set_ylim(0.1, 0.6)
    ax.set_xlabel("Screen: blur penalty B (before training)")
    ax.set_ylabel("Trained |1 − R_H| (test, lower is better)")
    handles = [Line2D([], [], marker="o", ls="", color=colour, mec=SURFACE, ms=5, label=family) for family, colour in FAMILIES.items()]
    handles += [Line2D([], [], marker="o", ls="", color=MUTED, mec=SURFACE, ms=5, label="Pointwise"),
                Line2D([], [], marker="s", ls="", color=MUTED, mec=SURFACE, ms=5, label="Statistics")]
    figure.legend(handles=handles, loc="outside lower center", ncols=2, columnspacing=1.0, handletextpad=0.3)
    return figure


if __name__ == "__main__":
    record("fig4_screen_vs_outcome", "python -m figures.fig4_screen_vs_outcome", INPUTS, save(draw(), "fig4_screen_vs_outcome"),
           {"score": SCORE, "outcome": OUTCOME, "marker_area": "14 + 22 * (log10(parameters) - 5) points^2"})
