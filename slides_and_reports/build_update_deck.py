# slides_and_reports/build_update_deck.py
"""Corrected seed-42 slides for the Weeks 1 and 2 presentation, built from the files in results/ (no GPU, no data).

    python slides_and_reports/build_update_deck.py

Writes ldct_backbone_week_1_2_update.pdf next to this file. It replaces the result slides of
ldct_backboone_week_1_2.pdf (whose editable source no longer exists); the setup and Week 1 check slides stay valid.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "ldct_backbone_week_1_2_update.pdf"
INK, MUTED, RULE, PANEL, BLUE, ORANGE, GREEN = "#111111", "#6b6a66", "#d6d5cf", "#f5f4f0", "#2a78d6", "#d9622b", "#1b9a6c"
FOOT = "LDCT perceptual backbone | Weeks 1-2 closeout, corrected seed-42 slides"
METRICS = ["psnr", "ssim", "lpips", "dists", "r_h"]
PAIRS = [("C5", "C8", "DINOv2-S, pointwise → statistics"), ("C9", "C10", "CT dose network v2, pointwise → statistics"),
         ("C5", "C6", "pointwise, DINOv2-S → DINOv2-B")]
CURVES = [("train_C0", "C0 (L1)", MUTED), ("train_C8_varfix", "C8 (DINOv2-S, statistics)", BLUE),
          ("train_C9_v2", "C9 (CT dose network v2, pointwise)", GREEN), ("train_C10_v2", "C10 (CT dose network v2, statistics)", ORANGE)]


class Deck:
    def __init__(self, path: Path) -> None:
        self.pdf, self.page = PdfPages(path), 0

    def slide(self, title: str, tag: str = "") -> plt.Figure:
        self.page += 1
        fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
        fig.text(0.045, 0.9, title, fontsize=23, fontweight="bold", color=INK)
        fig.text(0.955, 0.915, tag, fontsize=11, color=BLUE, ha="right", fontweight="bold")
        fig.text(0.045, 0.035, FOOT, fontsize=9.5, color=MUTED)
        fig.text(0.955, 0.035, str(self.page), fontsize=9.5, color=MUTED, ha="right")
        return fig

    def save(self, fig: plt.Figure) -> None:
        self.pdf.savefig(fig)
        plt.close(fig)


def bullets(fig: plt.Figure, lines: list[str], x: float, y: float, size: float = 13, step: float = 0.047) -> None:
    for index, line in enumerate(lines):
        text = "   " + line.strip() if line.startswith(" ") else "• " + line       # leading space: continuation line
        fig.text(x, y - index * step, text, fontsize=size, color=INK, va="top")


def table(fig: plt.Figure, frame: pd.DataFrame, box: list[float], size: float = 11, widths: list[float] | None = None,
          bold_rows: tuple[int, ...] = ()) -> None:
    ax = fig.add_axes(box)
    ax.axis("off")
    drawn = ax.table(cellText=frame.values, colLabels=list(frame.columns), loc="upper center", cellLoc="center",
                     colWidths=widths)
    drawn.auto_set_font_size(False)
    drawn.set_fontsize(size)
    drawn.scale(1, 1.75)
    for (row, column), cell in drawn.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_facecolor(PANEL if row == 0 else "white")
        if column == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02
        if row == 0 or row - 1 in bold_rows:
            cell.set_text_props(fontweight="bold")


def result_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Configuration": frame["Configuration"],
        "PSNR ↑": frame["psnr"].map("{:.2f}".format), "SSIM ↑": frame["ssim"].map("{:.3f}".format),
        "LPIPS ↓": frame["lpips"].map("{:.3f}".format), "DISTS ↓": frame["dists"].map("{:.3f}".format),
        "R_H → 1": frame["r_h"].map("{:.3f}".format),
        "ms/step": frame["ms_per_step"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}"),
        "best epoch": frame["best_epoch"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}"),
    })


def build() -> None:
    final = pd.read_csv(RESULTS / "table1_seed42.csv")
    superseded = pd.read_csv(RESULTS / "table1_seed42_superseded.csv")
    table2 = pd.read_csv(RESULTS / "table2_feature_networks.csv")
    screen = pd.read_csv(RESULTS / "dose_net_validation_screen.csv")
    budget = pd.read_csv(RESULTS / "compute_budget.csv").set_index("Item")["Value"]
    by_id = {label.split(" ")[0]: row for label, row in final.set_index("Configuration").iterrows()}
    widths = [0.36, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.1]
    deck = Deck(OUT)

    fig = deck.slide("Weeks 1 and 2 closeout: corrected seed-42 slides")
    bullets(fig, [
        "Replaces the result slides of ldct_backboone_week_1_2.pdf (slides 6 to 12, 14 and 15).",
        "Final rows: C0 to C7 as trained in Week 2, C8 (DINOv2-S, statistics) after the variance fix,",
        "  C9 and C10 on the CT dose network v2 (log10-dose regression).",
        "Superseded runs (pre-fix C8, v1 C9 and C10) are shown once, labelled, for disclosure.",
        "Every number is read from results/ by build_update_deck.py. One seed; spreads are over test slices.",
        "Setup, configuration and Week 1 check slides of the original deck are unchanged.",
    ], 0.06, 0.74, 15, 0.062)
    fig.text(0.06, 0.2, "Arda Şahin | BIAI Lab | ISBI 2027 submission\nResults freeze Oct 18 · first draft Oct 23 · deadline Oct 26",
             fontsize=13, color=MUTED, linespacing=1.6)
    deck.save(fig)

    fig = deck.slide("Overview of the benchmark (Fig 1 draft)", "method")
    ax = fig.add_axes([0.045, 0.09, 0.91, 0.77])
    ax.imshow(plt.imread(ROOT / "figures" / "fig1_overview.png"))
    ax.axis("off")
    deck.save(fig)

    fig = deck.slide("Results, seed 42, test split (220 slices)", "week 2")
    table(fig, result_rows(final), [0.045, 0.2, 0.91, 0.66], 11.5, widths, bold_rows=(9,))
    c0, c8 = by_id["C0"], by_id["C8"]
    bullets(fig, [
        f"C8 (DINOv2-S, statistics) is the best trade-off: {c8.psnr - c0.psnr:+.2f} dB against C0 (L1) for the best LPIPS "
        f"({c8.lpips:.3f}) and DISTS ({c8.dists:.3f}).",
        "C9 and C10 (CT dose network v2) have the highest R_H but stay below C0 on PSNR and SSIM and are the weakest on DISTS.",
        "One seed. No significance claim; the spread over test slices is in table1_seed42.csv.",
    ], 0.05, 0.2, 12)
    deck.save(fig)

    fig = deck.slide("Superseded and ablation runs (disclosure only)", "week 2")
    table(fig, result_rows(superseded), [0.045, 0.45, 0.91, 0.4], 11, [0.44, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])
    c10_fix = superseded.set_index("run").loc["seed42/train_C10_varfix_v1"]
    c10_old = superseded.set_index("run").loc["seed42/train_C10_v1"]
    bullets(fig, [
        "Variance fix: eps inside the square root of the local spread, in place of a clamp that zeroed the gradient in flat windows.",
        f"C8 (DINOv2-S, statistics) is unchanged by the fix ({superseded.iloc[0].psnr:.2f} to {c8.psnr:.2f} dB, DISTS "
        f"{superseded.iloc[0].dists:.3f} to {c8.dists:.3f}).",
        f"C10 (CT dose network v1, statistics) moved from {c10_old.psnr:.2f} to {c10_fix.psnr:.2f} dB with the fix: one seed, "
        "of the order of run-to-run variation.",
        "v1 is the 6-way dose classifier. It is kept as an ablation, never in the final table.",
    ], 0.05, 0.47, 12.5)
    deck.save(fig)

    fig = deck.slide("CT dose network v2 and the adoption rule", "week 2")
    shown = screen.assign(**{"Dose network": screen["Dose network"].map({"v1": "v1 (6-way classifier)",
                                                                           "v2_regression": "v2 (log10-dose regression)"})})
    shown = pd.DataFrame({"Dose network": shown["Dose network"], "Matching": shown["Matching"],
                          "ordering M": shown["ordering_M"].map("{:.3f}".format),
                          "blur penalty B": shown["blur_penalty_B"].map("{:.3f}".format),
                          "dose ratio 10k/1000k": shown["dose_ratio_10k_1000k"].map("{:.2f}".format)})
    table(fig, shown, [0.045, 0.52, 0.91, 0.33], 12, [0.34, 0.16, 0.16, 0.16, 0.18])
    bullets(fig, [
        "Same three-block feature extractor (287,328 parameters); only the training target changed. Validation screen, 182 slices.",
        "Ordering M rose for both matchings. B stays below 1: an equal-MSE blur is still cheaper than the 100k noise.",
        "Disclosure: the adoption threshold was first 0.9 and was lowered to 0.8 after seeing this screen,",
        "  because statistics matching (0.852) missed 0.9. The rule is not prespecified (results/decision.json).",
        "The screen gain did not carry over: C9 and C10 on v2 do not improve uniformly on v1 (previous slide).",
    ], 0.05, 0.5, 12.5)
    deck.save(fig)

    fig = deck.slide("Planned pairs (descriptive)", "week 2")
    rows = []
    for first, second, label in PAIRS:
        a, b = by_id[first], by_id[second]
        rows.append({"Pair": f"{first} → {second} ({label})", "ΔPSNR": f"{b.psnr - a.psnr:+.2f}", "ΔSSIM": f"{b.ssim - a.ssim:+.3f}",
                     "ΔLPIPS": f"{b.lpips - a.lpips:+.3f}", "ΔDISTS": f"{b.dists - a.dists:+.3f}",
                     "R_H": f"{a.r_h:.3f} → {b.r_h:.3f}"})
    table(fig, pd.DataFrame(rows), [0.045, 0.52, 0.91, 0.3], 12, [0.46, 0.1, 0.1, 0.1, 0.1, 0.14])
    bullets(fig, [
        "Matching: statistics improve DINOv2-S on every metric.",
        "CT dose network v2: statistics raise R_H but cost PSNR, SSIM, LPIPS and DISTS. The sign differs from DINOv2-S.",
        "Size: DINOv2-B shows no benefit over DINOv2-S and is slower.",
        "No significance claim. Wilcoxon tests with Holm correction follow with three seeds.",
    ], 0.05, 0.47, 13)
    deck.save(fig)

    fig = deck.slide("Screen versus trained outcome", "preliminary")
    ax = fig.add_axes([0.07, 0.36, 0.4, 0.46])
    trained = table2.merge(final.assign(Configuration=final["Configuration"].str.split(" ").str[0])[["Configuration", "dists", "r_h"]],
                           on="Configuration")
    for _, row in trained.iterrows():
        dose = row["Network"].startswith("CT dose")
        ax.scatter(row["M_phi"], row["dists"], s=70, color=ORANGE if dose else BLUE,
                   marker="s" if row["Matching"] == "statistics" else "o", zorder=3)
        ax.annotate(row["Configuration"], (row["M_phi"], row["dists"]), textcoords="offset points", xytext=(6, 5), fontsize=10)
    ax.set_xlabel("screen ordering M (validation)", fontsize=11)
    ax.set_ylabel("DISTS after training (test)", fontsize=11)
    ax.grid(color=RULE, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_axes([0.55, 0.36, 0.4, 0.46])
    for _, row in trained.iterrows():
        dose = row["Network"].startswith("CT dose")
        ax.scatter(row["B_phi"], row["r_h"], s=70, color=ORANGE if dose else BLUE,
                   marker="s" if row["Matching"] == "statistics" else "o", zorder=3)
        ax.annotate(row["Configuration"], (row["B_phi"], row["r_h"]), textcoords="offset points", xytext=(6, 5), fontsize=10)
    ax.set_xlabel("screen blur penalty B (validation)", fontsize=11)
    ax.set_ylabel("R_H after training (test)", fontsize=11)
    ax.grid(color=RULE, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    bullets(fig, [
        "Circle: pointwise. Square: statistics. Orange: CT dose network v2. Ten trained networks, one seed.",
        "v2 raised M to 0.97 and 0.85, yet C9 and C10 keep the worst DISTS: M alone does not predict the outcome.",
        "B below 1 predicted smoothing; C9 and C10 have the highest R_H instead. The screen lacks a structure term.",
        "The Spearman tests of screen against outcome are a Week 3 deliverable (three seeds).",
    ], 0.05, 0.26, 12.5)
    deck.save(fig)

    fig = deck.slide("C10 validation instability", "diagnostics")
    ax = fig.add_axes([0.07, 0.42, 0.52, 0.36])
    for run, label, colour in CURVES:
        log = pd.read_csv(RESULTS / "seed42" / run / "train_log.csv")
        ax.plot(log["epoch"], log["val_l1"], color=colour, lw=2.2 if "C10" in label else 1.4, label=label, marker="o", ms=3)
    ax.set_yscale("log")
    ax.set_xlabel("epoch", fontsize=11)
    ax.set_ylabel("validation L1 per slice (sum over pixels)", fontsize=11)
    ax.grid(color=RULE, lw=0.6, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9.5, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax = fig.add_axes([0.68, 0.42, 0.27, 0.36])
    spikes = []
    for path in sorted((RESULTS / "seed42").glob("train_*/train_log.csv")):
        values = pd.read_csv(path)["val_l1"]
        spikes.append((path.parent.name.removeprefix("train_"), int((values > 2 * values.median()).sum())))
    spikes.sort(key=lambda item: item[1])
    ax.barh([name for name, _ in spikes], [count for _, count in spikes], color=[ORANGE if name == "C10_v2" else MUTED for name, _ in spikes])
    ax.set_xlabel("epochs with validation L1 above 2× the run median", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    bullets(fig, C10_BULLETS, 0.05, 0.31, 12.5)
    deck.save(fig)

    fig = deck.slide("Table 2 and compute", "resources")
    shown = pd.DataFrame({
        "Config": table2["Configuration"], "Network": table2["Network"], "Matching": table2["Matching"],
        "Parameters": table2["Parameters"].map(lambda v: f"{v / 1e6:.2f} M"),
        "λ": table2["lambda"].map("{:.3f}".format), "ms/step": table2["ms_per_step"].map("{:.0f}".format),
        "M": table2["M_phi"].map("{:.3f}".format), "B": table2["B_phi"].map("{:.3f}".format)})
    table(fig, shown, [0.045, 0.3, 0.91, 0.56], 11.5, [0.08, 0.3, 0.12, 0.12, 0.09, 0.09, 0.09, 0.09])
    bullets(fig, [
        "CT dose network: 287,328 parameters are the three blocks the loss reads; the whole network with its head has 1,173,857.",
        "λ and ms/step are those of the final runs. The Step 1.3 distance check was not rerun for the v2 dose network.",
        f"GPU hours: final seed-42 runs {budget['GPU hours of the final seed 42 U-Net runs (11 of 11)']}, superseded runs "
        f"{budget['GPU hours of the superseded seed 42 U-Net runs (4)']}, three-seed protocol about "
        f"{budget['Total GPU hours of the protocol (3 seeds of the final runs, plus the measured one-off parts)']} (A100 80 GB).",
    ], 0.05, 0.27, 12)
    deck.save(fig)

    fig = deck.slide("Takeaways")
    bullets(fig, [
        "Perceptual terms lower LPIPS by 20 to 53% and DISTS by 8 to 46% against C0 (L1), at a cost of 0.15 to 2.0 dB.",
        "C8 (DINOv2-S, statistics) is the practical candidate: best LPIPS and DISTS, 0.33 dB below C0.",
        "Statistics matching helps DINOv2-S on every metric. For the CT dose network v2 it raises R_H only.",
        "CT dose network v2 orders dose levels much better than v1, but C9 and C10 did not improve uniformly:",
        "  the screen's M does not predict the trained outcome on its own. This is the scientific question for Week 3.",
        "Reproducibility: per-run source revision and checkpoint hashes are in results/provenance.json;",
        "  seeds 123 and 456 run from seeds_perceptual_backbone_ldct.ipynb with frozen λ and the frozen v2 network.",
        "Status: one seed. Statistical tests, three-seed tables and final figures belong to the Oct 18 freeze.",
    ], 0.06, 0.78, 15, 0.068)
    deck.save(fig)
    deck.pdf.close()
    print("written:", OUT, f"({deck.page} slides)")


# Filled from slides_and_reports/c10_validation_instability.md.
C10_BULLETS = [
    "C10 (CT dose network v2, statistics): 8 of 30 epochs above twice the median validation L1 (largest 80,138 at epoch 3, "
    "30,478 at epoch 29).",
    "Training loss falls smoothly; 0 non-finite steps. Spikes also occur with LPIPS-VGG, MedSigLIP, RadImageNet and DISTS, never with L1.",
    "Checked on CPU: the saved checkpoint reproduces the logged best validation L1 (5,717 against 5,716); epoch 17 is the right choice.",
    "At its best epoch C10 looks like C0 (L1): no intensity offset, no failing slices, same sensitivity to the BatchNorm mode.",
    "No bug found. The cause of the spikes is open: the GPU replay with per-slice diagnostics has not been run yet.",
]

if __name__ == "__main__":
    build()
