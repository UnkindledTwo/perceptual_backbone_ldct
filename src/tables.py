# src/tables.py
"""Seed-42 tables of Weeks 1 and 2, rebuilt from the files in results/ (no GPU, no data).

    python -m src.tables

Final rows: C0 to C7 as trained in Week 2, C8 after the variance fix, C9 and C10 on the regression-v2
dose network. Superseded runs go to a separate, labelled table.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEED = 42
METRICS = ["psnr", "ssim", "lpips", "dists", "r_h"]
SEEDS_IN_PROTOCOL = 3
FEATURE_EXTRACTOR_PARAMETERS = 287_328      # the three conv blocks the loss reads, v1 and v2

# id: (run folder, label, feature network, matching, calibration file)
FINAL_RUNS = {
    "C0": ("train_C0", "C0 (L1)", "none", "-", None),
    "C1": ("train_C1", "C1 (VGG-19, pointwise)", "VGG-19 relu3_2", "pointwise", "calibration_weights.csv"),
    "C2": ("train_C2", "C2 (LPIPS-VGG, pointwise)", "LPIPS-VGG", "pointwise", "calibration_weights.csv"),
    "C3": ("train_C3", "C3 (DISTS, statistics)", "DISTS", "statistics", "calibration_weights.csv"),
    "C4": ("train_C4", "C4 (RadImageNet, pointwise)", "RadImageNet ResNet-50", "pointwise", "calibration_weights.csv"),
    "C5": ("train_C5", "C5 (DINOv2-S, pointwise)", "DINOv2-S", "pointwise", "calibration_weights.csv"),
    "C6": ("train_C6", "C6 (DINOv2-B, pointwise)", "DINOv2-B", "pointwise", "calibration_weights.csv"),
    "C7": ("train_C7", "C7 (MedSigLIP, pointwise)", "MedSigLIP-448", "pointwise", "calibration_weights.csv"),
    "C8": ("train_C8_varfix", "C8 (DINOv2-S, statistics), variance fix", "DINOv2-S", "statistics", "calibration_varfix.csv"),
    "C9": ("train_C9_v2", "C9 (CT dose network v2, pointwise)", "CT dose network v2 (regression)", "pointwise", "calibration_v2.csv"),
    "C10": ("train_C10_v2", "C10 (CT dose network v2, statistics)", "CT dose network v2 (regression)", "statistics", "calibration_v2.csv"),
}
# run folder: label. Kept for disclosure, never part of the final table.
SUPERSEDED_RUNS = {
    "train_C8": "C8 (DINOv2-S, statistics), before the variance fix",
    "train_C9_v1": "C9 (CT dose network v1, pointwise)",
    "train_C10_v1": "C10 (CT dose network v1, statistics), before the variance fix",
    "train_C10_varfix_v1": "C10 (CT dose network v1, statistics), variance fix",
}
DOSE_NETS = {"v1": "CT dose network v1", "v2_regression": "CT dose network v2 (regression)"}
DOSE_CLASSES = ["10k", "50k", "100k", "500k", "1000k", "clean"]


def read_json(*parts: str) -> dict:
    return json.loads(RESULTS.joinpath(*parts).read_text())


def to_markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    def cell(value) -> str:
        if value is None or value is pd.NA:
            return "-"
        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else "no"
        if isinstance(value, (float, np.floating)):
            return "-" if not np.isfinite(value) else f"{value:.{digits}f}"
        return str(value)

    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.astype(object).itertuples(index=False)]
    return "\n".join(lines)


def run_row(run: str, label: str) -> dict:
    """One Table 1 row from the per-slice metrics; the means must agree with the run summary."""
    summary = read_json(f"seed{SEED}", run, "run_summary.json")
    frame = pd.read_csv(RESULTS / f"seed{SEED}" / run / "metrics.csv")
    assert len(frame) == summary["n_test"] == 220, f"{run}: {len(frame)} test slices"
    for metric in METRICS:
        assert np.isclose(frame[metric].mean(), summary[f"{metric}_mean"], rtol=1e-9), f"{run}: {metric} differs from run_summary.json"
    log = pd.read_csv(RESULTS / f"seed{SEED}" / run / "train_log.csv")
    assert int(log.loc[log["val_l1"].idxmin(), "epoch"]) == summary["best_epoch"], f"{run}: best epoch is not the lowest validation L1"
    return {"Configuration": label, "run": f"seed{SEED}/{run}",
            **{m: frame[m].mean() for m in METRICS}, **{f"{m}_std": frame[m].std(ddof=1) for m in METRICS},
            "abs_1_minus_r_h": (1 - frame["r_h"]).abs().mean(), "ms_per_step": summary["ms_per_step"],
            "gpu_hours": summary["gpu_hours"], "peak_gpu_memory_gb": summary.get("peak_gpu_memory_gb", np.nan),
            "lambda": summary["loss_weight"], "best_epoch": summary["best_epoch"],
            "nonfinite_steps": summary["nonfinite_steps"], "n_test": len(frame)}


def table1() -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(RESULTS / "baseline_metrics.csv")
    input_row = {"Configuration": "LDCT input (100k)", "run": "baseline_metrics.csv",
                 **{m: baseline[m].mean() for m in METRICS}, **{f"{m}_std": baseline[m].std(ddof=1) for m in METRICS},
                 "n_test": len(baseline)}
    final = pd.DataFrame([input_row] + [run_row(run, label) for run, label, *_ in FINAL_RUNS.values()])
    superseded = pd.DataFrame([run_row(run, label) for run, label in SUPERSEDED_RUNS.items()])
    return final, superseded


def plan_table1(frame: pd.DataFrame) -> pd.DataFrame:
    def mean_std(row: pd.Series, metric: str, digits: int) -> str:
        return f"{row[metric]:.{digits}f} ± {row[f'{metric}_std']:.{digits}f}"

    return pd.DataFrame([{
        "Configuration (seed 42, test, 100k)": row["Configuration"],
        "PSNR (dB)": mean_std(row, "psnr", 2), "SSIM": mean_std(row, "ssim", 4), "LPIPS": mean_std(row, "lpips", 4),
        "DISTS": mean_std(row, "dists", 4), "R_H": mean_std(row, "r_h", 4),
        "ms per step": "-" if pd.isna(row["ms_per_step"]) else f"{row['ms_per_step']:.0f}",
        "best epoch": "-" if pd.isna(row["best_epoch"]) else f"{row['best_epoch']:.0f}",
    } for _, row in frame.iterrows()])


def dose_reports() -> dict[str, dict]:
    return {"v1": read_json(f"seed{SEED}", "ct_dose_net_v1", "report.json"),
            "v2_regression": read_json(f"seed{SEED}", "ct_dose_net_v2_regression", "report.json")}


def distance_checks() -> pd.DataFrame:
    """Step 1.3 checks. The dose-network rows are the v1 run unless zoo_checks_v2.csv (seeds notebook) exists."""
    checks = pd.read_csv(RESULTS / "zoo_checks.csv")
    checks["Network"] = checks["Network"].replace("CT dose network", "CT dose network v1 (check not rerun for v2)")
    path = RESULTS / "zoo_checks_v2.csv"
    return pd.concat([checks, pd.read_csv(path)], ignore_index=True) if path.exists() else checks


def screen_table() -> pd.DataFrame:
    """Step 1.4 screen: pretrained rows from Week 1, dose-network rows from the follow-up validation screen."""
    columns = ["Network", "Matching", "ordering_M", "dose_ratio_10k_1000k", "blur_penalty_B", "Parameters"]
    zoo = pd.read_csv(RESULTS / "zoo_screen_report.csv")
    zoo = zoo[zoo["Network"] != "CT dose network"][columns]
    dose = pd.read_csv(RESULTS / "dose_net_validation_screen.csv").rename(columns={"Dose network": "Network"})
    dose["Network"] = dose["Network"].map(DOSE_NETS)
    dose["Parameters"] = FEATURE_EXTRACTOR_PARAMETERS
    return pd.concat([zoo, dose[columns]], ignore_index=True)


def calibration_table() -> pd.DataFrame:
    rows = []
    for config_id, (run, label, _, _, source) in FINAL_RUNS.items():
        weight = read_json(f"seed{SEED}", run, "run_summary.json")["loss_weight"]
        if source is None:
            rows.append({"Configuration": label, "S_1": np.nan, "S_phi": np.nan, "lambda": 0.0, "Source": "-"})
            continue
        table = pd.read_csv(RESULTS / source)
        column = "lambda_new" if "lambda_new" in table.columns else "lambda"
        match = table[np.isclose(table[column], weight, rtol=1e-9)]
        assert len(match) == 1, f"{config_id}: lambda of the run is not in {source}"
        rows.append({"Configuration": label, "S_1": match["S_1"].iloc[0], "S_phi": match["S_phi"].iloc[0],
                     "lambda": weight, "Source": source})
    return pd.DataFrame(rows)


def table2(final: pd.DataFrame) -> pd.DataFrame:
    checks = distance_checks().set_index(["Network", "Matching"])
    screen = screen_table().set_index(["Network", "Matching"])
    total_v2 = dose_reports()["v2_regression"]["parameters"]
    rows = []
    for config_id, (_, label, network, matching, _) in FINAL_RUNS.items():
        if config_id == "C0":
            continue
        run = final.set_index("Configuration").loc[label]
        is_dose = network.startswith("CT dose network")
        key = (network, matching)
        rows.append({"Configuration": config_id, "Network": network, "Matching": matching,
                     "Parameters": FEATURE_EXTRACTOR_PARAMETERS if is_dose else int(checks.loc[key, "Parameters"]),
                     "Parameters_with_head": total_v2 if is_dose else None,
                     "lambda": run["lambda"], "ms_per_step": run["ms_per_step"],
                     "ms_per_batch_of_8": checks.loc[key, "ms_per_batch_of_8"] if key in checks.index else np.nan,
                     "M_phi": screen.loc[key, "ordering_M"], "B_phi": screen.loc[key, "blur_penalty_B"]})
    return pd.DataFrame(rows).astype({"Parameters_with_head": "Int64"})


def dose_net_table() -> pd.DataFrame:
    reports = dose_reports()
    screen = pd.read_csv(RESULTS / "dose_net_validation_screen.csv").set_index(["Dose network", "Matching"])

    def pair(tag: str, column: str) -> str:
        return " / ".join(f"{screen.loc[(tag, m), column]:.4f}" for m in ("pointwise", "statistics"))

    def value(tag: str, key: str, form: str) -> str:
        return format(reports[tag][key], form) if key in reports[tag] else "not recorded"

    rows = [("Training target", "6-way cross-entropy", "log10-dose regression, standardised, smooth L1, clean at log10 = 7"),
            ("Checkpoint", "last epoch", f"lowest validation loss (epoch {reports['v2_regression']['best_epoch']} of {reports['v2_regression']['epochs']})"),
            *[(name, value("v1", key, form), value("v2_regression", key, form)) for name, key, form in (
                ("Val accuracy, 6 classes (v2: decoded from the regression output)", "val_accuracy", ".2%"),
                ("Val accuracy, clean vs 1000k", "val_accuracy_clean_vs_1000k", ".2%"),
                ("Val MAE, log10 dose", "val_mae_log10", ".3f"),
                ("Parameters, whole network with head", "parameters", "d"),
                ("Training, GPU hours", "gpu_hours", ".3f"),
                ("Mean body fraction of a training patch", "mean_body_fraction_train", ".3f"))],
            ("Parameters, feature extractor (three blocks, used by the loss)", str(FEATURE_EXTRACTOR_PARAMETERS), str(FEATURE_EXTRACTOR_PARAMETERS)),
            ("Screen ordering M_phi (pointwise / statistics)", pair("v1", "ordering_M"), pair("v2_regression", "ordering_M")),
            ("Screen blur penalty B_phi (pointwise / statistics)", pair("v1", "blur_penalty_B"), pair("v2_regression", "blur_penalty_B"))]
    return pd.DataFrame(rows, columns=["Check", "v1 (ablation)", "v2 regression (adopted)"])


def per_class(tag: str) -> pd.DataFrame:
    folder = "ct_dose_net_v1" if tag == "v1" else "ct_dose_net_v2_regression"
    confusion = pd.read_csv(RESULTS / f"seed{SEED}" / folder / "confusion_matrix.csv", index_col=0).to_numpy()
    hits = np.diag(confusion)
    precision, recall = hits / confusion.sum(0), hits / confusion.sum(1)
    return pd.DataFrame({"class": DOSE_CLASSES, "precision": precision, "recall": recall,
                         "f1": 2 * precision * recall / (precision + recall), "support": confusion.sum(1)})


def compute_budget(final: pd.DataFrame, superseded: pd.DataFrame, environment: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = final[final["Configuration"] != "LDCT input (100k)"]
    timings, reports = read_json("timings.json"), dose_reports()
    unet_hours, old_hours = runs["gpu_hours"].sum(), superseded["gpu_hours"].sum()
    dose_hours = reports["v1"]["gpu_hours"] + reports["v2_regression"]["gpu_hours"]
    screen_hours = (timings["screen_pretrained"] + timings["screen_ct_dose_net"]) / 3600
    other_hours = dose_hours + screen_hours + timings["calibration"] / 3600
    hours = runs.set_index("Configuration")["gpu_hours"]
    budget = pd.DataFrame([
        ("Trained configurations", str(len(FINAL_RUNS))),
        ("Seeds", str(SEEDS_IN_PROTOCOL)),
        ("Full U-Net runs", str(len(FINAL_RUNS) * SEEDS_IN_PROTOCOL)),
        ("CT dose network pretraining, GPU hours (v1 / v2 regression)", f"{reports['v1']['gpu_hours']:.3f} / {reports['v2_regression']['gpu_hours']:.3f}"),
        ("Screen (Step 1.4), GPU hours (Week 1 run; the v2 validation screen was not timed)", f"{screen_hours:.3f}"),
        ("Calibration (Step 2.1), GPU hours (Week 2 run; the C8, C9, C10 recalibrations were not timed)", f"{timings['calibration'] / 3600:.3f}"),
        ("GPU hours per U-Net run, C0 (L1)", f"{hours[FINAL_RUNS['C0'][1]]:.3f}"),
        ("GPU hours per U-Net run, C7 (MedSigLIP, pointwise)", f"{hours[FINAL_RUNS['C7'][1]]:.3f}"),
        (f"GPU hours of the final seed {SEED} U-Net runs ({len(runs)} of {len(FINAL_RUNS)})", f"{unet_hours:.3f}"),
        (f"GPU hours of the superseded seed {SEED} U-Net runs ({len(superseded)})", f"{old_hours:.3f}"),
        ("GPU hours used in Weeks 1 and 2 (measured parts)", f"{unet_hours + old_hours + other_hours:.3f}"),
        ("Total GPU hours of the protocol (3 seeds of the final runs, plus the measured one-off parts)",
         f"{SEEDS_IN_PROTOCOL * unet_hours + old_hours + other_hours:.3f}"),
        ("GPU", f"{environment['gpu']} ({environment['gpu_memory_gb']} GB)"),
        ("Largest recorded peak GPU memory of a run, GB (not recorded for the final C8, C9, C10 runs)", f"{runs['peak_gpu_memory_gb'].max():.1f}"),
        ("GPU hours available in Weeks 1 to 3", "fill in by hand"),
    ], columns=["Item", "Value"])
    per_run = runs[["Configuration", "run", "ms_per_step", "gpu_hours", "peak_gpu_memory_gb"]]
    return budget, per_run


def provenance_tables() -> list[tuple[str, pd.DataFrame]]:
    path = RESULTS / "provenance.json"
    if not path.exists():
        return []
    record = json.loads(path.read_text())
    runs = pd.DataFrame(record["runs"])[["run", "status", "notebook", "src_revision", "uncommitted_at_run_time", "best_pth_sha256"]]
    weights = pd.DataFrame(record["dose_networks"])[["file", "status", "sha256", "link"]]
    runs["best_pth_sha256"] = runs["best_pth_sha256"].str[:16]
    return [("Source provenance per run (full record in `provenance.json`)", runs), ("Frozen dose-network weights", weights)]


def adoption_sentence() -> str:
    decision = read_json("decision.json")
    return (f"Adoption rule: {decision['rule']}. {decision.get('note', '')} "
            f"M(v2) is {decision['M_v2']['pointwise']:.3f} pointwise and {decision['M_v2']['statistics']:.3f} statistics.")


def main() -> None:
    environment = read_json("environment.json")
    final, superseded = table1()
    df_table2 = table2(final)
    budget, per_run = compute_budget(final, superseded, environment)
    final.to_csv(RESULTS / f"table1_seed{SEED}.csv", index=False)
    superseded.to_csv(RESULTS / f"table1_seed{SEED}_superseded.csv", index=False)
    df_table2.to_csv(RESULTS / "table2_feature_networks.csv", index=False)
    budget.to_csv(RESULTS / "compute_budget.csv", index=False)
    per_run.to_csv(RESULTS / "compute_per_run.csv", index=False)

    checks = distance_checks()[["Network", "Matching", "distance_at_equality", "gradient_reaches_prediction", "ms_per_batch_of_8"]]
    sections = [
        ("Environment", pd.DataFrame([(k, v) for k, v in environment.items() if not isinstance(v, dict)], columns=["Item", "Value"]), ""),
        ("Step 1.1 checks", pd.read_csv(RESULTS / "step1_1_checks.csv"), ""),
        ("Step 1.3 distance checks", checks,
         "" if (RESULTS / "zoo_checks_v2.csv").exists() else
         "The CT dose network rows are the v1 network. The check was not rerun for v2; the feature extractor has the same architecture."),
        ("Step 1.4 training-free screen (validation split)", screen_table(),
         "CT dose network rows: follow-up validation screen. Parameters of the CT dose network: feature extractor only."),
        ("Step 2.1 calibration (final lambda values)", calibration_table(), "S_1 is the same initial L1 term in every row (one initialisation, one crop sequence)."),
        ("Step 2.2 CT dose network checks", dose_net_table(), adoption_sentence()),
        ("Step 2.2 CT dose network v2 (regression), per class, decoded", per_class("v2_regression"), ""),
        ("Step 2.2 CT dose network v1, per class", per_class("v1"), ""),
        (f"Table 1: final configurations, seed {SEED}, test split (220 slices)", plan_table1(final),
         "± is the standard deviation over the 220 test slices of one seed. It is not a spread over seeds."),
        ("Table 1, superseded and ablation runs (disclosure only, not part of the final table)", plan_table1(superseded),
         "Same test slices and the same spread definition as above."),
        ("Table 2: values per trained feature network", df_table2,
         "Parameters: weights the loss reads. For the CT dose network this is the feature extractor (three conv blocks); "
         "Parameters_with_head adds the regression head, which the loss does not use. lambda and ms_per_step are those of the final runs. "
         "ms_per_batch_of_8 comes from the Step 1.3 check, which was not rerun for the v2 dose network."),
        ("Compute Budget", budget, ""),
        *[(title, frame, "") for title, frame in provenance_tables()],
    ]
    header = (f"Generated by `python -m src.tables` from the files in `results/`. Seed {SEED} only. Final configurations: C0 to C7 as trained in "
              "Week 2, C8 (DINOv2-S, statistics) after the variance fix, C9 and C10 on the CT dose network v2 (regression).")
    body = "\n\n".join(f"### {title}\n\n{to_markdown(frame)}" + (f"\n\n{note}" if note else "") for title, frame, note in sections)
    (RESULTS / "week1_week2_tables.md").write_text(f"{header}\n\n{body}\n")
    print(to_markdown(plan_table1(final)))


if __name__ == "__main__":
    main()
