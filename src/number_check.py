# src/number_check.py
"""Checks every number in the paper text against results/ and the code that produced it (no GPU, no data).

    python -m src.number_check

Each claim names a number as the LaTeX source prints it, the value it comes from and the rule (rounded value, bound
or exact protocol constant). Qualitative claims ("L1 keeps the best PSNR") are checked the same way. A coverage pass
then lists every number in the prose and the float captions; one that no claim and no ALLOWED name accounts for fails
the check. Table bodies are generated from results/ by src.seeds_tables and are not re-checked here. Writes
results/week3/number_check.csv and records the step in the manifest. Exit status 1 when anything fails.
"""
import json
import re
import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PerceptualConfig
from src.record import OUT, RESULTS, ROOT, guarded, record

PAPER = ROOT / "paper"
SECTIONS = ["abstract", "intro", "method", "experiments", "conclusion", "ethics"]
TABLES = ["table1_paper_patient", "table2_compact"]
SRC = ROOT / "src"
FIG3 = ROOT / "figures" / "fig3_qualitative.py"

# Numbers that are names or notation, not claims (literal LaTeX source, whitespace collapsed).
ALLOWED = [
    "VGG-19", "ResNet-50", "MedSigLIP-448", "ViT-B/16", "I2I 2025v1", "relu3\\_2", "\\log_{10}", "0.75\\linewidth",
    "\\frac{1}{H_l W_l}", "\\frac{1}{C_l \\lvert \\Omega_l \\rvert}", "\\rVert_2^2", "1 - s(", "(2ab + c) / (a^2 + b^2 + c)",
    "\\lvert 1 - R_H \\rvert", "$|1-R_H|$", "\\log(1/d)", "$B_\\phi < 1$ means", "B < 1",
]


@dataclass
class Claim:
    file: str
    needle: str          # LaTeX source around the number, whitespace collapsed
    printed: str         # the number as printed; "" for a qualitative claim
    value: object        # source value (number, string or bool)
    rule: str            # round, sci, ge, le, lt, eq, true
    source: str


def read(name: str) -> str:
    path = PAPER / "tables" / f"{name}.tex" if name in TABLES else PAPER / "sections" / f"{name}.tex"
    text = path.read_text()
    if name in TABLES:
        text = re.search(r"\\caption\{(.*?)\}\n\\label", text, re.S).group(1)
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("%"))
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    return re.sub(r"\s+", " ", text)


def half_up(value: float, places: int) -> Decimal:
    return Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def places(printed: str) -> int:
    return len(printed.split(".")[1]) if "." in printed else 0


def passes(claim: Claim) -> tuple[bool, str]:
    rule, printed, value = claim.rule, claim.printed, claim.value
    if rule == "true":
        return bool(value), str(bool(value))
    if rule == "eq":
        return str(value) == printed, str(value)
    if rule == "sci":
        mantissa, exponent = printed.split("e")
        shown = f"{float(value):.{places(mantissa)}e}"
        return float(shown) == float(printed), shown
    number = float(value)
    bound = float(printed)
    step = 10.0 ** -places(printed)
    if rule == "round":
        return half_up(number, places(printed)) == Decimal(printed), f"{number:.6g}"
    if rule == "ge":           # tight lower bound: printed <= value < printed + step
        return bound <= number < bound + step, f"{number:.6g}"
    if rule == "le":           # tight upper bound: printed - step < value <= printed
        return bound - step < number <= bound, f"{number:.6g}"
    if rule == "lt":           # strict bound, not tight
        return number < bound, f"{number:.3g}"
    raise ValueError(rule)


def source_has(path: Path, pattern: str) -> bool:
    return re.search(pattern, path.read_text(), re.M) is not None


def claims() -> list[Claim]:
    week3 = lambda name: pd.read_csv(OUT / name)
    table = week3("table1_three_seeds.csv").set_index("config")
    tests = week3("stats_families_slice_and_patient.csv")
    family = lambda name, metric: tests[(tests.family == name) & (tests.metric == metric)].set_index("comparison")
    a_psnr, a_lpips, a_rh = family("A", "psnr"), family("A", "lpips"), family("A", "r_h_error")
    b = lambda pair, metric: family("B", metric).loc[pair]
    zoo = pd.read_csv(RESULTS / "zoo_screen_report.csv")
    pretrained = zoo[~zoo.Network.str.startswith(("Pixel L1", "CT dose network"))]
    dose_screen = pd.read_csv(RESULTS / "dose_net_validation_screen.csv").set_index(["Dose network", "Matching"])
    medsiglip = zoo[zoo.Network == "MedSigLIP-448"].set_index("Matching").blur_penalty_B
    outcome = week3("screen_vs_outcome.csv").set_index(["screen_score", "outcome", "subset"]).spearman_rho
    outcome_p = week3("screen_vs_outcome.csv").set_index(["screen_score", "outcome", "subset"]).permutation_p
    exploratory = week3("screen_vs_outcome_exploratory.csv").set_index(["screen_score", "outcome"]).spearman_rho
    spikes = week3("validation_spikes.csv")
    networks = week3("table2_final.csv")
    checks = pd.read_csv(RESULTS / "step1_1_checks.csv").set_index("Check").Observed
    split_slices = checks["Train / val / test slices"].split(" / ")
    split_patients = checks["Train / val / test patients"].split(" / ")
    dose_report = json.loads((RESULTS / "seed42" / "ct_dose_net_v2_regression" / "report.json").read_text())
    decision = json.loads((RESULTS / "decision.json").read_text())
    config = PerceptualConfig(run_name="check", data_root=Path("."))
    fig3 = {name: value for name, value in re.findall(r"^([A-Z_][A-Z_, ]*?) = (.+?)\s*(?:#.*)?$", FIG3.read_text(), re.M)}
    seeds = sorted(pd.read_csv(OUT / "per_slice_metrics.csv").seed.unique())
    best_trade = ["C1", "C2", "C8"]
    top_group = ["C1", "C2", "C3", "C8"]
    perceptual = [f"C{k}" for k in range(1, 11)]
    psnr_cost = (table.loc["C0", "psnr"] - table.loc[best_trade, "psnr"])
    pairs = pd.read_csv(OUT / "per_slice_metrics.csv")
    c6_c5_seed_signs = np.sign(b("C6 vs C5", "r_h_error")[["mean_difference_seed42", "mean_difference_seed123",
                                                           "mean_difference_seed456"]].astype(float))
    t1, t2 = "table1_paper_patient", "table2_compact"

    rows = [
        # abstract
        ("abstract", "ten perceptual-loss configurations", "", len(networks) == 10, "true", "table2_final.csv rows"),
        ("abstract", "seven pretrained feature networks from five families", "",
         networks[~networks.Network.str.startswith("CT dose")].Network.nunique() == 7, "true",
         "table2_final.csv distinct pretrained networks (five families: Fig 1 / figures/style.py FAMILIES)"),
        ("abstract", "three seeds", "", len(seeds) == 3, "true", "per_slice_metrics.csv seeds"),
        ("abstract", "LPIPS (0.104 to", "0.104", table.loc["C0", "lpips"], "round", "table1_three_seeds C0 lpips"),
        ("abstract", "to about 0.058)", "0.058", table.loc[best_trade, "lpips"].mean(), "round",
         "table1_three_seeds mean lpips of C1, C2, C8"),
        ("abstract", "at 0.5 to 0.7~dB", "0.5", psnr_cost.min(), "round", "table1_three_seeds C0 minus C1/C2/C8 psnr, min"),
        ("abstract", "at 0.5 to 0.7~dB", "0.7", psnr_cost.max(), "round", "table1_three_seeds C0 minus C1/C2/C8 psnr, max"),
        ("abstract", "texture error from 0.333 to 0.195", "0.333", table.loc["C5", "r_h_error"], "round", "table1_three_seeds C5"),
        ("abstract", "texture error from 0.333 to 0.195", "0.195", table.loc["C8", "r_h_error"], "round", "table1_three_seeds C8"),
        ("abstract", "($p = 4.9 \\times 10^{-4}$, $r = -0.77$)", "4.9e-4", b("C8 vs C5", "r_h_error").patient_p_holm, "sci",
         "Family B C8 vs C5 |1-R_H| patient Holm p"),
        ("abstract", "($p = 4.9 \\times 10^{-4}$, $r = -0.77$)", "-0.77", b("C8 vs C5", "r_h_error").patient_effect_size_r,
         "round", "Family B C8 vs C5 |1-R_H| patient r"),
        ("abstract", "at no detectable PSNR cost", "", not b("C8 vs C5", "psnr").significant_patient, "true",
         "Family B C8 vs C5 PSNR not significant"),
        # method: Fig 3 caption
        ("method", "median C0 PSNR, seed 42.", "42", fig3["SEED"], "eq", "figures/fig3_qualitative.py SEED"),
        ("method", "image (grey range $[0, 1]$)", "(0.0, 1.0)", fig3["DISPLAY"], "eq", "figures/fig3_qualitative.py DISPLAY"),
        ("method", "window 0.35 to 0.75)", "(0.35, 0.75)", fig3["SOFT_TISSUE"], "eq", "figures/fig3_qualitative.py SOFT_TISSUE (= ZOOM_DISPLAY)"),
        ("method", "window 0.35 to 0.75)", "SOFT_TISSUE", fig3["ZOOM_DISPLAY"], "eq", "figures/fig3_qualitative.py ZOOM_DISPLAY"),
        ("method", "\\rvert$ (0 to 0.1).", "0.10", fig3["ERROR_MAX"], "eq", "figures/fig3_qualitative.py ERROR_MAX"),
        ("method", "the $96 \\times 96$ window", "96, 8, 0.9", fig3["ZOOM_SIZE, ZOOM_STRIDE, ZOOM_MIN_SOFT"], "eq",
         "figures/fig3_qualitative.py ZOOM_SIZE"),
        ("method", "at least 90\\% soft tissue", "96, 8, 0.9", fig3["ZOOM_SIZE, ZOOM_STRIDE, ZOOM_MIN_SOFT"], "eq",
         "figures/fig3_qualitative.py ZOOM_MIN_SOFT"),
        # method: objective, matching, networks
        ("method", "random $224 \\times 224$ crop", "224", config.crop_size, "eq", "src/config.py crop_size"),
        ("method", "each $4 \\times 4$ window", "", source_has(SRC / "distances.py", r"window: int = 4"), "true", "src/distances.py window"),
        ("method", "$c_1 = c_2 = 10^{-6}$", "", source_has(SRC / "distances.py", r"c1: float = 1e-6, c2: float = 1e-6"), "true",
         "src/distances.py c1, c2"),
        ("method", "VGG-19 at relu3\\_2", "", source_has(SRC / "feature_nets.py", r"layer_ends: tuple\[int, \.\.\.\] = \(13,\)"),
         "true", "src/feature_nets.py VGG layer 13 = relu3_2"),
        ("method", "layers 1 to 3", "", source_has(SRC / "feature_nets.py", r"return \[f1, f2, b\.layer3\(f2\)\]"), "true",
         "src/feature_nets.py ResNetFeatures"),
        ("method", "blocks 3, 6, 9", "", source_has(SRC / "feature_nets.py", r"else \(3, 6, 9\)"), "true", "src/feature_nets.py DINOv2 blocks"),
        ("method", "layers 7, 14, 20", "", source_has(SRC / "feature_nets.py", r"\(7, 14, 20\), has_cls=False"), "true",
         "src/feature_nets.py MedSigLIP layers"),
        ("method", "C8 and C10 use~(\\ref{eq:statistics}), C1, C4 to C7 and C9 use~(\\ref{eq:pointwise})", "",
         set(networks[networks.Matching == "statistics"].config) == {"C3", "C8", "C10"}
         and set(networks[networks.Matching == "pointwise"].config) == {"C1", "C2", "C4", "C5", "C6", "C7", "C9"}, "true",
         "table2_final.csv Matching (C2, C3 use LPIPS and DISTS with their own weights)"),
        # method: CT dose network
        ("method", "(32, 64, 128 channels; 287,328 parameters)", "",
         source_has(SRC / "ct_dose_net.py", r"conv_block\(1, 32, 1\), conv_block\(32, 64, 2\), conv_block\(64, 128, 2\)"),
         "true", "src/ct_dose_net.py blocks"),
        ("method", "(32, 64, 128 channels; 287,328 parameters)", "287328", dose_report["feature_extractor_parameters"], "eq",
         "ct_dose_net_v2_regression/report.json"),
        ("method", "$128 \\times 128$ training-split patches", "", source_has(SRC / "dose_net_train.py", r"patch: int = 128"),
         "true", "src/dose_net_train.py patch"),
        ("method", "(normal dose fixed at 7)", "", source_has(SRC / "dose_net_train.py", r"CLEAN_LOG_DOSE = 7\.0"), "true",
         "src/dose_net_train.py CLEAN_LOG_DOSE"),
        ("method", "smooth L1 for 10 epochs", "10", dose_report["epochs"], "eq", "ct_dose_net_v2_regression/report.json epochs"),
        ("method", "lowered from 0.9 to 0.8", "", "first set to 0.9 and lowered to 0.8" in decision["note"]
         and "> 0.8" in decision["rule"], "true", "results/decision.json"),
        # method: matched influence and screen
        ("method", "$\\lambda_\\phi = 19 \\, S_1 / S_\\phi$", "", source_has(SRC / "calibrate_lambda.py", r"target_influence: float = 0\.95")
         and source_has(SRC / "calibrate_lambda.py", r"target_influence / \(1 - target_influence\)"), "true",
         "src/calibrate_lambda.py 0.95 / 0.05 = 19"),
        ("method", "95\\% of the loss at initialization", "", source_has(SRC / "calibrate_lambda.py", r"target_influence: float = 0\.95"),
         "true", "src/calibrate_lambda.py"),
        ("method", "on the 182 validation slices", "182", split_slices[1], "eq", "step1_1_checks.csv"),
        ("method", "five photon counts $d$ from $10^4$ to $10^6$", "",
         source_has(SRC / "screen.py", r"DOSES: tuple\[int, \.\.\.\] = \(10_000, 50_000, 100_000, 500_000, 1_000_000\)"),
         "true", "src/screen.py DOSES"),
        ("method", "that of the $10^5$ noise", "", source_has(SRC / "screen.py", r"matched_blur\(clean, noisy\[100_000\]\)"),
         "true", "src/screen.py matched_blur"),
        # experiments: setup
        ("experiments", "($512 \\times 512$, $[0, 1]$)", "", np.load(OUT / "fig3" / "seed42" / "000039_03_01_061.npz")["NDCT"].shape
         == (512, 512) and checks["Input and target range"] == "[0.000, 1.000]", "true", "fig3 export shape, step1_1_checks.csv range"),
        ("experiments", "training input $10^5$ photons", "noisy_clipped_100k", config.input_key, "eq", "src/config.py input_key"),
        ("experiments", "split of 899, 182 and 220 slices (115, 34, 31 patients)", "899 / 182 / 220", " / ".join(split_slices),
         "eq", "step1_1_checks.csv"),
        ("experiments", "split of 899, 182 and 220 slices (115, 34, 31 patients)", "115 / 34 / 31", " / ".join(split_patients),
         "eq", "step1_1_checks.csv"),
        ("experiments", "All eleven configurations", "", len(networks) + 1 == 11, "true", "C0 plus table2_final.csv rows"),
        ("experiments", "AdamW ($10^{-4}$), batch 8, 30 epochs", "(0.0001, 8, 30)", (config.lr, config.batch_size, config.epochs),
         "eq", "src/config.py"),
        ("experiments", "Seeds 42, 123 and 456", "[42, 123, 456]", [int(s) for s in seeds], "eq", "per_slice_metrics.csv"),
        ("experiments", "(up to 8 of 30 epochs)", "8", int(spikes.spike_epochs.max()), "eq", "validation_spikes.csv max"),
        ("experiments", "(up to 8 of 30 epochs)", "30", config.epochs, "eq", "src/config.py epochs"),
        ("experiments", "one run (C8, seed 42) has no recorded code revision", "", source_has(OUT / "disclosures.md",
         r"Source revision of the final seed-42 C8"), "true", "results/week3/disclosures.md item 3"),
        ("experiments", "$R_H$ at 0.25 to 0.5 cycles per pixel", "", source_has(SRC / "metrics.py", r"low: float = 0\.25, high: float = 0\.5"),
         "true", "src/metrics.py high_band_ratio"),
        ("experiments", "(Holm over 30 and 9 tests)", "30", len(tests[tests.family == "A"]), "eq", "stats_families rows, Family A"),
        ("experiments", "(Holm over 30 and 9 tests)", "9", len(tests[tests.family == "B"]), "eq", "stats_families rows, Family B"),
        ("experiments", "over the 31 test patients", "31", int(tests.patient_n.max()), "eq", "stats_families patient_n"),
        ("experiments", "over the 31 test patients", "31", int(tests.patient_n.min()), "eq", "stats_families patient_n"),
        # experiments: screen
        ("experiments", "($M_\\phi$ 0.93 to 1.00;", "0.93", pretrained.ordering_M.min(), "round", "zoo_screen_report pretrained rows, min M"),
        ("experiments", "($M_\\phi$ 0.93 to 1.00;", "1.00", pretrained.ordering_M.max(), "round", "zoo_screen_report pretrained rows, max M"),
        ("experiments", "only the CT dose network has $B_\\phi < 1$", "", (pretrained.blur_penalty_B >= 1).all(), "true",
         "zoo_screen_report: every pretrained B >= 1"),
        ("experiments", "(0.34 pointwise, 0.86 statistics)", "0.34", dose_screen.loc[("v2_regression", "pointwise"), "blur_penalty_B"],
         "round", "dose_net_validation_screen v2"),
        ("experiments", "(0.34 pointwise, 0.86 statistics)", "0.86", dose_screen.loc[("v2_regression", "statistics"), "blur_penalty_B"],
         "round", "dose_net_validation_screen v2"),
        ("experiments", "except MedSigLIP (1.535 to 1.537)", "1.535", medsiglip["pointwise"], "round", "zoo_screen_report MedSigLIP"),
        ("experiments", "except MedSigLIP (1.535 to 1.537)", "1.537", medsiglip["statistics"], "round", "zoo_screen_report MedSigLIP"),
        ("experiments", "lowers $B_\\phi$ for every pretrained network except MedSigLIP", "",
         all(group.set_index("Matching").blur_penalty_B["statistics"] < group.set_index("Matching").blur_penalty_B["pointwise"]
             for name, group in pretrained.groupby("Network") if group.Matching.nunique() == 2 and name != "MedSigLIP-448"),
         "true", "zoo_screen_report"),
        ("experiments", "but raises it for the CT dose network", "", dose_screen.loc[("v2_regression", "statistics"), "blur_penalty_B"]
         > dose_screen.loc[("v2_regression", "pointwise"), "blur_penalty_B"], "true", "dose_net_validation_screen v2"),
        # experiments: Table 1 paragraph
        ("experiments", "(patient medians $-0.50$ to $-2.26$~dB", "-0.50", a_psnr.patient_median_difference.max(), "round",
         "Family A PSNR patient median difference, max"),
        ("experiments", "(patient medians $-0.50$ to $-2.26$~dB", "-2.26", a_psnr.patient_median_difference.min(), "round",
         "Family A PSNR patient median difference, min"),
        ("experiments", "Holm $p < 10^{-4}$", "1e-4", a_psnr.patient_p_holm.max(), "lt", "Family A PSNR patient Holm p, max"),
        ("experiments", "$\\lvert r \\rvert \\geq 0.93$", "0.93", a_psnr.patient_effect_size_r.abs().min(), "ge", "Family A PSNR |r|, min"),
        ("experiments", "seed spread 0.1 to 0.5~dB", "0.1", a_psnr.mean_difference_seed_std.min(), "round", "Family A PSNR seed std of the difference, min"),
        ("experiments", "seed spread 0.1 to 0.5~dB", "0.5", a_psnr.mean_difference_seed_std.max(), "round", "Family A PSNR seed std of the difference, max"),
        ("experiments", "every configuration loses PSNR", "", (a_psnr.patient_median_difference < 0).all() and a_psnr.significant_patient.all(),
         "true", "Family A PSNR"),
        ("experiments", "(all ten significant, Holm $p \\leq 0.021$)", "", (a_lpips.patient_median_difference < 0).all()
         and int(a_lpips.significant_patient.sum()) == 10, "true", "Family A LPIPS"),
        ("experiments", "(all ten significant, Holm $p \\leq 0.021$)", "0.021", a_lpips.patient_p_holm.max(), "le", "Family A LPIPS Holm p, max"),
        ("experiments", "L1 keeps the best PSNR and SSIM", "", table.drop("input").psnr.idxmax() == "C0"
         and table.drop("input").ssim.idxmax() == "C0", "true", "table1_three_seeds"),
        ("experiments", "form the best LPIPS and DISTS group", "", set(table.loc[perceptual].lpips.nsmallest(4).index) == set(top_group)
         and set(table.loc[perceptual].dists.nsmallest(4).index) == set(top_group), "true", "table1_three_seeds, four lowest"),
        ("experiments", "within one seed spread of each other", "", np.ptp(table.loc[top_group].dists) <= table.loc[top_group].dists_seed_std.max()
         and np.ptp(table.loc[top_group].lpips) <= table.loc[top_group].lpips_seed_std.max(), "true",
         "table1_three_seeds: range within the group <= largest seed std"),
        ("experiments", "(DISTS 0.065 to 0.067)", "0.065", table.loc[top_group].dists.min(), "round", "table1_three_seeds"),
        ("experiments", "(DISTS 0.065 to 0.067)", "0.067", table.loc[top_group].dists.max(), "round", "table1_three_seeds"),
        ("experiments", "has the best $\\lvert 1 - R_H \\rvert$ (0.167 against 0.352 for L1)", "", table.loc[perceptual].r_h_error.idxmin() == "C10",
         "true", "table1_three_seeds"),
        ("experiments", "(0.167 against 0.352 for L1)", "0.167", table.loc["C10", "r_h_error"], "round", "table1_three_seeds"),
        ("experiments", "(0.167 against 0.352 for L1)", "0.352", table.loc["C0", "r_h_error"], "round", "table1_three_seeds"),
        ("experiments", "has the worst PSNR and LPIPS of the perceptual losses", "", table.loc[perceptual].psnr.idxmin() == "C4"
         and table.loc[perceptual].lpips.idxmax() == "C4", "true", "table1_three_seeds"),
        ("experiments", "is the only configuration with worse texture than L1", "",
         list(table.loc[perceptual][table.loc[perceptual].r_h_error > table.loc["C0", "r_h_error"]].index) == ["C7"]
         and a_rh.loc["C7 vs C0"].significant_patient, "true", "table1_three_seeds and Family A"),
        ("experiments", "\\rvert$ 0.477, $R_H$ 0.53, smoother)", "0.477", table.loc["C7", "r_h_error"], "round", "table1_three_seeds"),
        ("experiments", "\\rvert$ 0.477, $R_H$ 0.53, smoother)", "0.53", table.loc["C7", "r_h"], "round", "table1_three_seeds"),
        ("experiments", "stays far behind on DISTS (0.097 to 0.098)", "0.097", table.loc[["C9", "C10"]].dists.min(), "round", "table1_three_seeds"),
        ("experiments", "stays far behind on DISTS (0.097 to 0.098)", "0.098", table.loc[["C9", "C10"]].dists.max(), "round", "table1_three_seeds"),
        # experiments: Fig 3 and Fig 4 paragraph
        ("experiments", "falls from 0.333 for", "0.333", table.loc["C5", "r_h_error"], "round", "table1_three_seeds C5"),
        ("experiments", "to 0.195 for C8", "0.195", table.loc["C8", "r_h_error"], "round", "table1_three_seeds C8"),
        ("experiments", "(Holm $p = 4.9 \\times 10^{-4}$, $r = -0.77$, seed spread 0.04)", "4.9e-4",
         b("C8 vs C5", "r_h_error").patient_p_holm, "sci", "Family B"),
        ("experiments", "(Holm $p = 4.9 \\times 10^{-4}$, $r = -0.77$, seed spread 0.04)", "-0.77",
         b("C8 vs C5", "r_h_error").patient_effect_size_r, "round", "Family B"),
        ("experiments", "(Holm $p = 4.9 \\times 10^{-4}$, $r = -0.77$, seed spread 0.04)", "0.04",
         b("C8 vs C5", "r_h_error").mean_difference_seed_std, "round", "Family B seed std of the difference"),
        ("experiments", "with no detectable PSNR or LPIPS change", "", not b("C8 vs C5", "psnr").significant_patient
         and not b("C8 vs C5", "lpips").significant_patient, "true", "Family B"),
        ("experiments", "at $\\rho = 0.84$ ($p = 0.004$; 0.71 without C2 and C3)", "0.84",
         outcome.loc[("B_phi", "r_h_error", "all trained networks")], "round", "screen_vs_outcome"),
        ("experiments", "at $\\rho = 0.84$ ($p = 0.004$; 0.71 without C2 and C3)", "0.004",
         outcome_p.loc[("B_phi", "r_h_error", "all trained networks")], "round", "screen_vs_outcome permutation p"),
        ("experiments", "at $\\rho = 0.84$ ($p = 0.004$; 0.71 without C2 and C3)", "0.71",
         outcome.loc[("B_phi", "r_h_error", "without C2 and C3")], "round", "screen_vs_outcome"),
        ("experiments", "DINOv2-B and DINOv2-S differ in neither PSNR nor LPIPS", "", not b("C6 vs C5", "psnr").significant_patient
         and not b("C6 vs C5", "lpips").significant_patient, "true", "Family B C6 vs C5"),
        ("experiments", "though significant when pooled, changes sign across seeds", "", bool(b("C6 vs C5", "r_h_error").significant_patient)
         and len(set(c6_c5_seed_signs)) > 1, "true", "Family B C6 vs C5 |1-R_H|, per-seed differences"),
        ("experiments", "worse texture ($\\rho = 0.89$, post hoc)", "0.89", exploratory.loc[("Parameters", "r_h_error")], "round",
         "screen_vs_outcome_exploratory, pointwise only"),
        ("experiments", "the trained result ($\\rho = 0.84$, exploratory)", "0.84", outcome.loc[("B_phi", "r_h_error", "all trained networks")],
         "round", "screen_vs_outcome (Fig 4 caption)"),
        # ethics: section numbers of the AI-use disclosure
        ("ethics", "drafts of Sections 1 to 4, check", "", section_titles()[:4] == ["Introduction", "Method", "Experiments and Results", "Conclusion"],
         "true", "\\section order in root.tex inputs"),
        ("ethics", "complete parts of Section 3,", "", section_titles()[2] == "Experiments and Results", "true", "\\section order"),
        # table captions (generated by src.seeds_tables)
        (t1, "Test split (220 slices, 31 patients, 100k)", "220 / 31", f"{split_slices[2]} / {split_patients[2]}", "eq", "step1_1_checks.csv"),
        (t1, "over seeds 42, 123 and 456", "[42, 123, 456]", [int(s) for s in seeds], "eq", "per_slice_metrics.csv"),
        (t1, "on the 31 patient means", "31", int(tests.patient_n.min()), "eq", "stats_families patient_n"),
        (t1, "Holm over 30 tests, p < 0.05", "30", len(tests[tests.family == "A"]), "eq", "stats_families rows, Family A"),
        (t1, "Holm over 30 tests, p < 0.05", "", source_has(SRC / "seeds_tables.py", r"^ALPHA = 0\.05$"), "true",
         "src/seeds_tables.py ALPHA"),
        (t2, "(M ≥ 0.95)", "0.95", networks[~networks.Network.str.startswith("CT dose")].M_phi.min(), "ge", "table2_final.csv pretrained M, min"),
        (t2, "only the CT dose network has B < 1", "", set(networks[networks.B_phi < 1].config) == {"C9", "C10"}, "true", "table2_final.csv"),
    ]
    return [Claim(*row) for row in rows]


def section_titles() -> list[str]:
    root = (PAPER / "root.tex").read_text()
    titles = []
    for name in re.findall(r"\\input\{sections/(\w+)\}", root):
        titles += re.findall(r"\\section\{([^}]*)\}", (PAPER / "sections" / f"{name}.tex").read_text())
    return titles


NUMBER = re.compile(r"10\^\{?-?\d+\}?|(?<![A-Za-z_\\\d.])-?\d+(?:,\d{3})*(?:\.\d+)?")


def coverage(texts: dict[str, str], checked: list[Claim]) -> list[dict]:
    """Numbers in the text that no claim needle and no ALLOWED entry covers."""
    rows = []
    for name, text in texts.items():
        spans = []
        for needle in [claim.needle for claim in checked if claim.file == name] + ALLOWED:
            spans += [(m.start(), m.end()) for m in re.finditer(re.escape(needle), text)]
        stripped = re.sub(r"\\(cite|ref|label|includegraphics|input|url|cfg)\{[^}]*\}", lambda m: " " * len(m.group(0)), text)
        for match in NUMBER.finditer(stripped):
            if not any(start <= match.start() and match.end() <= end for start, end in spans):
                rows.append({"file": name, "claim": text[max(0, match.start() - 40):match.end() + 20], "printed": match.group(0),
                             "source": "", "value": "", "rule": "coverage", "status": "UNCHECKED"})
    return rows


def main() -> int:
    texts = {name: read(name) for name in SECTIONS + TABLES}
    rows = []
    for claim in claims():
        found = claim.needle in texts[claim.file]
        ok, shown = passes(claim)
        rows.append({"file": claim.file, "claim": claim.needle, "printed": claim.printed, "source": claim.source,
                     "value": shown, "rule": claim.rule, "status": "PASS" if ok and found else ("NOT IN TEXT" if not found else "FAIL")})
    rows += coverage(texts, claims())
    report = pd.DataFrame(rows)
    path = guarded(OUT / "number_check.csv")
    report.to_csv(path, index=False)
    inputs = [PAPER / "sections" / f"{name}.tex" for name in SECTIONS] + [PAPER / "tables" / f"{name}.tex" for name in TABLES]
    record("number_check", "python -m src.number_check", inputs, [path],
           {"claims": int((report.rule != "coverage").sum()), "status": report.status.value_counts().to_dict()})
    print(report.status.value_counts().to_string())
    problems = report[report.status != "PASS"]
    if len(problems):
        print(problems.to_string(index=False))
    return 1 if len(problems) else 0


if __name__ == "__main__":
    sys.exit(main())
