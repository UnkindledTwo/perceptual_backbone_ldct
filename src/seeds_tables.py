# src/seeds_tables.py
"""Week 3 tables and statistics over seeds 42, 123 and 456, built from the files in results/ (no GPU, no data).

    python -m src.seeds_tables

Reads the per-run files and the Week 1 and 2 tables, writes only to results/week3/ and records every input
and output hash in results/week3/manifest.json. It does not call src.tables.main, so no older file changes.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import stats
from src.record import OUT, RESULTS, guarded, record, sha256
from src.latex import booktabs
from src.tables import FEATURE_EXTRACTOR_PARAMETERS, FINAL_RUNS, METRICS, run_row, to_markdown

SEEDS = (42, 123, 456)
COLUMNS = [*METRICS, "r_h_error"]
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "lpips": False, "dists": False, "r_h_error": False}
OWN_TRAINING_METRIC = {("C2", "lpips"), ("C3", "dists")}
RUN_FILES = ("metrics.csv", "train_log.csv", "run_summary.json")
SPIKE_FACTOR = 2.0
VGG_BASED = ("C1", "C2", "C3")
ALPHA = 0.05


def md_table(frame: pd.DataFrame) -> str:
    """to_markdown with the pipes of |1 − R_H| escaped, so the column count stays right on GitHub."""
    return to_markdown(frame).replace("|1 − R_H|", "\\|1 − R_H\\|")


def run_folder(seed: int, config_id: str) -> str:
    """Seed 42 keeps the folder names of the final runs; the seeds notebook writes train_C{k}."""
    return FINAL_RUNS[config_id][0] if seed == 42 else f"train_{config_id}"


def label(config_id: str) -> str:
    return FINAL_RUNS[config_id][1].removesuffix(", variance fix")


def run_dir(seed: int, config_id: str) -> Path:
    return RESULTS / f"seed{seed}" / run_folder(seed, config_id)


def input_files() -> list[Path]:
    runs = [run_dir(seed, c) / name for seed in SEEDS for c in FINAL_RUNS for name in RUN_FILES]
    return runs + [RESULTS / "baseline_metrics.csv", RESULTS / "table2_feature_networks.csv",
                   *[RESULTS / f"table1_seed{seed}.csv" for seed in SEEDS], RESULTS / "table1_seed42_superseded.csv",
                   RESULTS / "decision.json", RESULTS / "provenance.json"]


def per_slice() -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        for config_id in FINAL_RUNS:
            frame = pd.read_csv(run_dir(seed, config_id) / "metrics.csv", dtype={"patient_id": str})
            frames.append(frame.assign(configuration=config_id, seed=seed))
    result = pd.concat(frames, ignore_index=True)
    result["r_h_error"] = (1 - result["r_h"]).abs()
    slices = result.groupby(["configuration", "seed"])["slice_id"].apply(lambda ids: tuple(sorted(ids)))
    assert slices.nunique() == 1 and len(slices.iloc[0]) == 220, "runs do not share one set of 220 test slices"
    return result[["configuration", "seed", "slice_id", "patient_id", *COLUMNS]]


def per_seed_table() -> pd.DataFrame:
    """One row per run, with the integrity checks of src.tables.run_row."""
    rows = [{"config": c, "seed": seed, **run_row(run_folder(seed, c), label(c), seed)} for seed in SEEDS for c in FINAL_RUNS]
    return pd.DataFrame(rows).rename(columns={"abs_1_minus_r_h": "r_h_error"})


def check_against_existing(per_seed: pd.DataFrame) -> None:
    """The means must agree with the Table 1 files written at run time."""
    for seed in SEEDS:
        old = pd.read_csv(RESULTS / f"table1_seed{seed}.csv")
        old = old[old["Configuration"].str.startswith("C")].reset_index(drop=True)
        new = per_seed[per_seed["seed"] == seed].reset_index(drop=True)
        assert [name.split(" ")[0] for name in old["Configuration"]] == new["config"].tolist(), f"seed {seed}: row order"
        assert np.allclose(old[METRICS], new[METRICS], rtol=1e-9), f"seed {seed}: means differ from table1_seed{seed}.csv"


def three_seed_table(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over the three per-seed means."""
    baseline = pd.read_csv(RESULTS / "baseline_metrics.csv").assign(r_h_error=lambda f: (1 - f["r_h"]).abs())
    rows = [{"config": "input", "Configuration": "LDCT input (100k)", "n_seeds": 0, **{m: baseline[m].mean() for m in COLUMNS}}]
    for config_id in FINAL_RUNS:
        runs = per_seed[per_seed["config"] == config_id]
        rows.append({"config": config_id, "Configuration": label(config_id), "n_seeds": len(runs),
                     **{m: runs[m].mean() for m in COLUMNS}, **{f"{m}_seed_std": runs[m].std(ddof=1) for m in COLUMNS},
                     "gpu_hours_total": runs["gpu_hours"].sum()})
    return pd.DataFrame(rows)


def validation_spikes() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for config_id in FINAL_RUNS:
            val = pd.read_csv(run_dir(seed, config_id) / "train_log.csv")["val_l1"]
            rows.append({"config": config_id, "Configuration": label(config_id), "seed": seed,
                         "spike_epochs": int((val > SPIKE_FACTOR * val.median()).sum()),
                         "max_over_median": val.max() / val.median(), "best_epoch": int(val.idxmin()) + 1})
    return pd.DataFrame(rows)


def seed_runs_provenance() -> dict:
    """Source record of the runs made with the seeds notebook, taken from each run_summary.json."""
    runs, sources = [], {}
    for seed in SEEDS[1:]:
        for config_id in FINAL_RUNS:
            folder = run_dir(seed, config_id)
            summary = json.loads((folder / "run_summary.json").read_text())
            source, settings, weights = summary["source"], summary["settings"], folder / "best.pth"
            sources[seed] = source["src_sha256"]
            runs.append({"run": f"seed{seed}/{folder.name}", "configuration": label(config_id), "seed": seed,
                         "repo_commit": source["repo_commit"], "uncommitted_src_files": source["uncommitted_src_files"],
                         "notebook_version": source["notebook_version"], "loss_weight": summary["loss_weight"],
                         "dose_net_sha256": settings["dose_net_sha256"], "torch": settings["torch"], "gpu": settings["gpu"],
                         "best_epoch": summary["best_epoch"], "nonfinite_steps": summary["nonfinite_steps"],
                         "best_pth_sha256": summary["best_pth_sha256"],
                         "best_pth_matches_local_file": sha256(weights) == summary["best_pth_sha256"] if weights.exists() else None})
    first, second = (sources[seed] for seed in SEEDS[1:])
    return {"note": "Runs of seeds_perceptual_backbone_ldct.ipynb. Seed 42 and the dose networks are in results/provenance.json, "
                    "which is not changed. best_pth_matches_local_file is null where the gitignored checkpoint is not on this machine.",
            "src_files_differing_between_seed_123_and_456": sorted(name for name in first if first[name] != second.get(name)),
            "runs": runs}


def family_a_marks(family_a: pd.DataFrame) -> dict[tuple[str, str], str]:
    """† significantly better than C0, ‡ significantly worse (Holm p < ALPHA, side from the median difference)."""
    marks = {}
    for _, row in family_a[family_a["p_holm"] < ALPHA].iterrows():
        better = (row["median_difference"] > 0) == HIGHER_IS_BETTER[row["metric"]]
        marks[(row["comparison"].split(" ")[0], row["metric"])] = "†" if better else "‡"
    return marks


def formatted_table1(three_seeds: pd.DataFrame, family_a: pd.DataFrame) -> pd.DataFrame:
    trained = three_seeds[three_seeds["n_seeds"] > 0].set_index("config")
    best = {m: trained[m].idxmax() if higher else trained[m].idxmin() for m, higher in HIGHER_IS_BETTER.items()}
    marks = family_a_marks(family_a)

    def cell(row: pd.Series, metric: str, digits: int, key: str | None = None) -> str:
        key = key or metric
        if row["n_seeds"] == 0:
            return f"{row[metric]:.{digits}f}"
        text = f"{row[metric]:.{digits}f} ± {row[f'{metric}_seed_std']:.{digits}f}"
        text = f"**{text}**" if best.get(key) == row["config"] else text
        return text + marks.get((row["config"], key), "") + ("*" if (row["config"], key) in OWN_TRAINING_METRIC else "")

    return pd.DataFrame([{
        "Configuration": row["Configuration"], "PSNR (dB) ↑": cell(row, "psnr", 2), "SSIM ↑": cell(row, "ssim", 4),
        "LPIPS ↓": cell(row, "lpips", 4), "DISTS ↓": cell(row, "dists", 4), "R_H (→1)": cell(row, "r_h", 3),
        "|1 − R_H| ↓": cell(row, "r_h_error", 3)} for _, row in three_seeds.iterrows()])


def formatted_family(family: pd.DataFrame) -> pd.DataFrame:
    def names(comparison: str) -> str:
        first, second = comparison.split(" vs ")
        return f"{label(first)} vs {label(second)}"

    return pd.DataFrame({"Comparison": family["comparison"].map(names), "Metric": family["metric"], "n": family["n"],
                         "Median difference": family["median_difference"].map("{:.4f}".format),
                         "p (Holm)": family["p_holm"].map("{:.2e}".format),
                         "Effect size r": family["effect_size_r"].map("{:.3f}".format)})


def deltas_vs_c0(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Signed difference to C0 (L1) at each seed, mean and standard deviation over the seeds."""
    wide = per_seed.set_index(["config", "seed"])[COLUMNS]
    rows = []
    for config_id in list(FINAL_RUNS)[1:]:
        difference = wide.loc[config_id] - wide.loc["C0"]
        rows.append({"config": config_id, "Configuration": label(config_id),
                     **{f"{m}_delta": difference[m].mean() for m in COLUMNS},
                     **{f"{m}_delta_seed_std": difference[m].std(ddof=1) for m in COLUMNS}})
    return pd.DataFrame(rows)


def formatted_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    def cell(row: pd.Series, metric: str, digits: int) -> str:
        return f"{row[f'{metric}_delta']:+.{digits}f} ± {row[f'{metric}_delta_seed_std']:.{digits}f}"

    return pd.DataFrame([{"Configuration": row["Configuration"], "Δ PSNR (dB)": cell(row, "psnr", 2), "Δ SSIM": cell(row, "ssim", 4),
                          "Δ LPIPS": cell(row, "lpips", 4), "Δ DISTS": cell(row, "dists", 4), "Δ R_H": cell(row, "r_h", 3),
                          "Δ |1 − R_H|": cell(row, "r_h_error", 3)} for _, row in deltas.iterrows()])


def families_slice_and_patient(frame: pd.DataFrame, tests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Planned families at slice level (unchanged) next to the same tests on the 31 patient means."""
    result = pd.concat([stats.slice_and_patient(frame, "A", stats.FAMILY_A, ALPHA),
                        stats.slice_and_patient(frame, "B", stats.FAMILY_B, ALPHA)], ignore_index=True)
    for family in ("A", "B"):
        planned = tests[f"stats_family_{family.lower()}"]
        rebuilt = result[result["family"] == family].reset_index(drop=True)
        for column in ("n", "median_difference", "p_raw", "effect_size_r", "p_holm"):
            assert (rebuilt[f"slice_{column}"] == planned[column]).all(), f"family {family}: slice-level {column} differs"
    assert (result["patient_n"] == 31).all(), "patient-level tests must use the 31 test patients"
    return result


def formatted_levels(levels: pd.DataFrame) -> pd.DataFrame:
    def difference(value: float) -> str:
        return f"{value:+.4f}" if abs(value) < 0.1 else f"{value:+.3f}"

    return pd.DataFrame({
        "Comparison": levels["comparison"].map(lambda c: " vs ".join(c.split(" vs "))), "Metric": levels["metric"],
        "Slices: median Δ": levels["slice_median_difference"].map(difference),
        "Slices: p (Holm)": levels["slice_p_holm"].map("{:.1e}".format), "Slices: r": levels["slice_effect_size_r"].map("{:+.2f}".format),
        "Patients: median Δ": levels["patient_median_difference"].map(difference),
        "Patients: p (Holm)": levels["patient_p_holm"].map("{:.1e}".format),
        "Patients: r": levels["patient_effect_size_r"].map("{:+.2f}".format),
        **{f"Seed {seed}: mean Δ": levels[f"mean_difference_seed{seed}"].map(difference) for seed in SEEDS},
        "Seed std of mean Δ": levels["mean_difference_seed_std"].map(lambda v: f"{v:.4f}" if v < 0.1 else f"{v:.3f}")})


def screen_pointwise_only(frame: pd.DataFrame, table2: pd.DataFrame) -> pd.DataFrame:
    """Post hoc and exploratory: the screen against the outcome on the pointwise networks only (not a planned test)."""
    outcomes = stats.seed_averaged(frame, stats.OUTCOMES).groupby("configuration")[stats.OUTCOMES].mean()
    scores = table2[table2["Matching"] == "pointwise"].set_index("Configuration")[stats.SCREEN_SCORES]
    rows = []
    for score in stats.SCREEN_SCORES:
        for outcome in stats.OUTCOMES:
            rho, p_value = stats.screen_predicts_outcome(scores[score], outcomes.loc[scores.index, outcome])
            rows.append({"screen_score": score, "outcome": outcome, "subset": "pointwise only (" + ", ".join(scores.index) + ")",
                         "n": len(scores), "spearman_rho": rho, "permutation_p": p_value, "status": "post hoc, exploratory"})
    return pd.DataFrame(rows)


def shared_networks() -> list[str]:
    """Configurations of the screen that share a feature network, so the 10 points are not 10 independent networks."""
    groups = {}
    for config_id, (_, _, network, _, _) in list(FINAL_RUNS.items())[1:]:
        groups.setdefault(network, []).append(config_id)
    lines = [f"{' and '.join(ids)} share {network} (pointwise and statistics matching)" for network, ids in groups.items() if len(ids) > 1]
    return lines + [f"{', '.join(VGG_BASED)} are all VGG-based (VGG-19 relu3_2, LPIPS-VGG, DISTS on VGG-16 features)"]


def table2_final(per_seed: pd.DataFrame, table2: pd.DataFrame) -> pd.DataFrame:
    """Table 2 of the plan from results/table2_feature_networks.csv, with training time over the three seeds."""
    timing = per_seed.groupby("config")[["ms_per_step", "gpu_hours"]].agg(["mean", "std"])
    result = table2.rename(columns={"Configuration": "config"}).assign(
        Configuration=lambda f: f["config"].map(label),
        ms_per_step_three_seeds=lambda f: f["config"].map(timing[("ms_per_step", "mean")]),
        ms_per_step_seed_std=lambda f: f["config"].map(timing[("ms_per_step", "std")]),
        gpu_hours_per_run=lambda f: f["config"].map(timing[("gpu_hours", "mean")]))
    return result.rename(columns={"ms_per_step": "ms_per_step_seed42"})


def formatted_table2(table2: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Configuration": table2["Configuration"], "Network": table2["Network"], "Matching": table2["Matching"],
        "Parameters": table2["Parameters"].map(lambda v: f"{v:,}" + ("†" if v == FEATURE_EXTRACTOR_PARAMETERS else "")),
        "λ": table2["lambda"].map("{:.3f}".format), "Distance, ms per batch of 8": table2["ms_per_batch_of_8"].map("{:.1f}".format),
        "Training step, ms": table2["ms_per_step_three_seeds"].map("{:.0f}".format),
        "GPU h per run": table2["gpu_hours_per_run"].map("{:.3f}".format),
        "M ↑": table2["M_phi"].map("{:.3f}".format), "B": table2["B_phi"].map("{:.2f}".format)})


def ablation(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Superseded seed-42 runs next to the final seed-42 run of the same configuration (disclosure, not part of Table 1)."""
    superseded = pd.read_csv(RESULTS / "table1_seed42_superseded.csv").rename(columns={"abs_1_minus_r_h": "r_h_error"})
    superseded["config"] = superseded["run"].map(lambda run: run.split("_")[1])
    final = per_seed[(per_seed["seed"] == 42) & per_seed["config"].isin(superseded["config"])].copy()
    final["Configuration"] = final["config"].map(lambda c: FINAL_RUNS[c][1].replace(", variance fix", "") + " (final)")
    columns = ["config", "Configuration", "status", "run", *COLUMNS, "lambda", "best_epoch"]
    rows = pd.concat([final.assign(status="final"), superseded.assign(status="superseded")], ignore_index=True)[columns]
    order = {c: i for i, c in enumerate(FINAL_RUNS)}
    return rows.sort_values(["config", "status"], key=lambda s: s.map(order) if s.name == "config" else s).reset_index(drop=True)


def formatted_ablation(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"Configuration": rows["Configuration"], "Status": rows["status"],
                         "PSNR (dB) ↑": rows["psnr"].map("{:.2f}".format), "SSIM ↑": rows["ssim"].map("{:.4f}".format),
                         "LPIPS ↓": rows["lpips"].map("{:.4f}".format), "DISTS ↓": rows["dists"].map("{:.4f}".format),
                         "|1 − R_H| ↓": rows["r_h_error"].map("{:.3f}".format), "λ": rows["lambda"].map("{:.3f}".format),
                         "Best epoch": rows["best_epoch"].astype(int)})


def disclosures(per_seed: pd.DataFrame, spikes: pd.DataFrame, rows: pd.DataFrame, provenance_seeds: dict) -> str:
    """Disclosures for the paper, written from the result files (decision.json, provenance files, per-seed table)."""
    decision = json.loads((RESULTS / "decision.json").read_text())
    c8 = next(run for run in json.loads((RESULTS / "provenance.json").read_text())["runs"] if run["run"] == "seed42/train_C8_varfix")
    commits = {run["seed"]: (run["repo_commit"][:7], run["uncommitted_src_files"]) for run in provenance_seeds["runs"]
               if run["configuration"].startswith("C8 ")}
    c8_rows = per_seed[per_seed["config"] == "C8"].assign(
        source=lambda f: f["seed"].map(lambda s: "not recorded (code first committed in a7108c5)" if s == 42 else
                                       f"{commits[s][0]}, uncommitted src files: {len(commits[s][1])}"))
    c8_table = pd.DataFrame({"Seed": c8_rows["seed"], "PSNR (dB)": c8_rows["psnr"].map("{:.2f}".format),
                             "LPIPS": c8_rows["lpips"].map("{:.4f}".format), "DISTS": c8_rows["dists"].map("{:.4f}".format),
                             "|1 − R_H|": c8_rows["r_h_error"].map("{:.3f}".format), "Best epoch": c8_rows["best_epoch"],
                             "Source revision": c8_rows["source"]})
    spiking = spikes.pivot(index="config", columns="seed", values="spike_epochs").loc[list(FINAL_RUNS)]
    spiking = spiking[spiking.sum(axis=1) > 0]
    spike_text = "; ".join(f"{label(c)} {'/'.join(str(int(v)) for v in spiking.loc[c])}" for c in spiking.index)
    m_v1, m_v2 = decision["M_v1"], decision["M_v2"]
    sections = [
        ("1. Adoption threshold of the CT dose network v2 was changed after seeing the screen",
         f"Rule as applied: {decision['rule']}. Validation screen M (pointwise / statistics): v1 {m_v1['pointwise']:.3f} / "
         f"{m_v1['statistics']:.3f}, v2 {m_v2['pointwise']:.3f} / {m_v2['statistics']:.3f}. From `results/decision.json`: "
         f"\"{decision['note']}\" C9 and C10 therefore rest on a rule that was not prespecified; v1 is kept as an ablation (below)."),
        ("2. Superseded seed-42 runs (ablation, not part of Table 1)",
         md_table(formatted_ablation(rows)) + "\n\nThe pre-fix C8 and the v1 C10 ran before the variance clamp in "
         "`src/distances.py`; C9 and C10 v1 use the 6-way cross-entropy dose network. Per-run sources: `results/provenance.json`."),
        ("3. Source revision of the final seed-42 C8 (DINOv2-S, statistics) run: closed by disclosure, no replay",
         f"`results/provenance.json`: src revision \"{c8['src_revision']}\"; uncommitted at run time: \"{c8['uncommitted_at_run_time']}\"; "
         f"best.pth sha256 `{c8['best_pth_sha256'][:12]}…`. The revision cannot be recovered and was not replayed. "
         "Seeds 123 and 456 of the same configuration ran from the seeds notebook on recorded commits:\n\n" + md_table(c8_table) +
         "\n\nReported as a limitation: one of the 33 Table 1 runs has no recorded source revision."),
        ("4. Validation L1 is unstable in eval mode for some configurations",
         f"Epochs with validation L1 above {SPIKE_FACTOR:g} times the run median (seeds 42/123/456): {spike_text}. The GPU replay "
         "of C10 at seed 42 (`results/seed42/diag_C10_v2/`) traced the spikes to a BatchNorm train/eval mismatch on bright slices; "
         "the other configurations were not replayed. Checkpoints are selected on this validation L1 "
         "(`slides_and_reports/c10_validation_instability.md`)."),
    ]
    header = "Generated by `python -m src.seeds_tables` from the result files; every number below is read from them."
    return f"## Disclosures for the paper\n\n{header}\n\n" + "\n\n".join(f"### {title}\n\n{body}" for title, body in sections) + "\n"


def markdown(three_seeds: pd.DataFrame, spikes: pd.DataFrame, tests: dict[str, pd.DataFrame], freeze: dict[str, pd.DataFrame]) -> str:
    wide = spikes.pivot(index="config", columns="seed", values="spike_epochs").loc[list(FINAL_RUNS)]
    spike_table = pd.DataFrame({"Configuration": [label(c) for c in wide.index],
                                **{f"seed {seed}": wide[seed].to_numpy() for seed in SEEDS}})
    screen = tests["screen_vs_outcome"].assign(spearman_rho=lambda f: f["spearman_rho"].map("{:.3f}".format),
                                               permutation_p=lambda f: f["permutation_p"].map("{:.4f}".format))
    sections = [
        ("Table 1: test split (220 slices), 100k, seeds 42, 123, 456", formatted_table1(three_seeds, tests["stats_family_a"]),
         "Mean ± standard deviation over the three per-seed means. Bold: best trained configuration. "
         f"† significantly better and ‡ significantly worse than C0 (L1) in Family A (Holm-adjusted p < {ALPHA}, slice level; "
         "tested on PSNR, LPIPS and |1 − R_H| only). * scored with its own training metric. "
         "Bold on R_H follows |1 − R_H|, the mean of the per-slice absolute deviation."),
        (f"Family A: each configuration against C0 (L1), Holm over {len(tests['stats_family_a'])}", formatted_family(tests["stats_family_a"]),
         "Paired Wilcoxon signed-rank test on seed-averaged per-slice values. Difference is first minus second. "
         "Effect size: matched-pairs rank-biserial correlation."),
        (f"Family B: planned comparisons, Holm over {len(tests['stats_family_b'])}", formatted_family(tests["stats_family_b"]), ""),
        ("Screen score against trained outcome", screen,
         f"Spearman correlation over the trained networks, two-sided permutation p ({stats.PERMUTATIONS} permutations, seed "
         f"{stats.PERMUTATION_SEED}), not corrected for the number of rows. Outcomes are three-seed means. "
         "DISTS is circular for C3 (DISTS, statistics) and partly for C2 (LPIPS-VGG, pointwise), so each row is given with and without them."),
        ("Validation L1 spikes per run", spike_table,
         f"Number of the 30 epochs with validation L1 above {SPIKE_FACTOR:g} times the median of the run."),
        ("Table 1, signed difference to C0 (L1)", formatted_deltas(freeze["table1_deltas_vs_c0"]),
         "Per-seed mean minus the C0 mean of the same seed; mean ± standard deviation over the three seeds. "
         "Negative is better for LPIPS, DISTS and |1 − R_H|."),
        ("Families A and B at slice level (planned) and patient level (sensitivity check)",
         formatted_levels(freeze["stats_families_slice_and_patient"]),
         "Slice level: the planned tests (n = 220 seed-averaged slices), identical to the Family A and B tables above. Patient level: "
         "seeds averaged per slice, then slices averaged per patient (n = 31), same paired Wilcoxon test, same Holm families "
         "(30 and 9 tests). Δ is first minus second. Seed columns: mean difference over the 220 slices at each seed; the last "
         "column is its standard deviation over the seeds. The patient-level test is reported next to the planned families, not instead of them."),
        ("Screen against outcome: pointwise networks only (post hoc, exploratory)",
         freeze["screen_vs_outcome_exploratory"].drop(columns=["subset"]).assign(
             spearman_rho=lambda f: f["spearman_rho"].map("{:.3f}".format), permutation_p=lambda f: f["permutation_p"].map("{:.4f}".format)),
         "Not a planned test and not corrected. The ten trained networks are not ten independent networks: "
         + "; ".join(shared_networks()) + ". Restricting to pointwise matching removes the matching rule as a confounder of B."),
        ("Table 2: feature networks", formatted_table2(freeze["table2_final"]),
         f"† feature extractor only (the three conv blocks the loss reads); with the regression head the CT dose network v2 has "
         f"{int(freeze['table2_final']['Parameters_with_head'].dropna().iloc[0]):,} parameters. λ: frozen loss weight (95% perceptual share at "
         "initialisation, seed 42). Distance time: Step 1.3 check, one batch of 8. Training step and GPU hours: mean over the three seeds. "
         "M: dose ordering of the screen; B: equal-MSE blur penalty."),
    ]
    header = ("Generated by `python -m src.seeds_tables` from the files in `results/`. Inputs, outputs and their hashes: "
              "`manifest.json` in this folder. Seed 42 rows are the final runs (C8 after the variance fix, C9 and C10 on the CT dose network v2).")
    return header + "\n\n" + "\n\n".join(f"### {title}\n\n{md_table(frame)}" + (f"\n\n{note}" if note else "")
                                         for title, frame, note in sections) + "\n"


def fig3_slices(frame: pd.DataFrame, count: int = 2) -> pd.DataFrame:
    """Test slices closest to the median seed-averaged C0 PSNR (plan: not chosen by eye)."""
    psnr = frame[frame["configuration"] == "C0"].groupby(["slice_id", "patient_id"])["psnr"].mean().reset_index(name="c0_psnr_seed_mean")
    psnr["distance_to_median"] = (psnr["c0_psnr_seed_mean"] - psnr["c0_psnr_seed_mean"].median()).abs()
    return psnr.sort_values(["distance_to_median", "slice_id"]).head(count).reset_index(drop=True)


def latex_tables(three_seeds: pd.DataFrame, family_a: pd.DataFrame, freeze: dict[str, pd.DataFrame]) -> dict[str, str]:
    marks = (f"Mean $\\pm$ standard deviation over seeds 42, 123 and 456 (220 test slices, 31 patients, 100k). Bold: best trained "
             f"configuration. † significantly better and ‡ significantly worse than C0 (L1) (paired Wilcoxon on seed-averaged slices, "
             f"Holm over 30 tests, p < {ALPHA}; PSNR, LPIPS and |1 − R_H| only). * scored with its own training metric.")
    return {
        "table1_final": booktabs(formatted_table1(three_seeds, family_a), marks, "tab:results", "l" + "c" * 6),
        "table2_final": booktabs(formatted_table2(freeze["table2_final"]),
                                 "Feature networks. † feature extractor of the CT dose network v2 (three conv blocks). λ: frozen loss "
                                 "weight. Times: distance on a batch of 8, U-Net training step and GPU hours per run (mean over seeds). "
                                 "M: dose ordering, B: blur penalty of the training-free screen.", "tab:networks", "lllrrrrrrr"),
        "table_ablation_superseded": booktabs(formatted_ablation(freeze["table_ablation_superseded"]),
                                              "Superseded seed-42 runs next to the final run of the same configuration (ablation).",
                                              "tab:ablation", "llrrrrrrr"),
    }


def main() -> None:
    frame, per_seed = per_slice(), per_seed_table()
    check_against_existing(per_seed)
    three_seeds, spikes = three_seed_table(per_seed), validation_spikes()
    table2 = pd.read_csv(RESULTS / "table2_feature_networks.csv")
    tests = stats.run(frame, table2)
    provenance = seed_runs_provenance()
    freeze = {"table1_deltas_vs_c0": deltas_vs_c0(per_seed),
              "stats_families_slice_and_patient": families_slice_and_patient(frame, tests),
              "screen_vs_outcome_exploratory": screen_pointwise_only(frame, table2),
              "table2_final": table2_final(per_seed, table2), "table_ablation_superseded": ablation(per_seed)}
    tables = {"per_slice_metrics": frame, "table1_per_seed": per_seed, "table1_three_seeds": three_seeds,
              "validation_spikes": spikes, "fig3_slices": fig3_slices(frame), **tests,
              "table1_final": formatted_table1(three_seeds, tests["stats_family_a"]), **freeze}
    outputs = []
    for name, table in tables.items():
        outputs.append(guarded(OUT / f"{name}.csv"))
        table.to_csv(outputs[-1], index=False)
    texts = {"week3_tables.md": markdown(three_seeds, spikes, tests, freeze),
             "provenance_seeds.json": json.dumps(provenance, indent=1) + "\n",
             "disclosures.md": disclosures(per_seed, spikes, freeze["table_ablation_superseded"], provenance),
             **{f"{name}.tex": text for name, text in latex_tables(three_seeds, tests["stats_family_a"], freeze).items()}}
    for name, text in texts.items():
        outputs.append(guarded(OUT / name))
        outputs[-1].write_text(text)
    record("seeds_tables", "python -m src.seeds_tables", input_files(), outputs,
           {"seeds": SEEDS, "family_a": stats.FAMILY_A, "family_b": stats.FAMILY_B, "test_metrics": stats.TEST_METRICS,
            "permutations": stats.PERMUTATIONS, "permutation_seed": stats.PERMUTATION_SEED, "spike_factor": SPIKE_FACTOR,
            "alpha": ALPHA, "patient_level": "seeds averaged per slice, then slices per patient; same families and Holm",
            "exploratory": "screen_vs_outcome_exploratory.csv is post hoc and not part of the planned tests"})
    print(to_markdown(formatted_table1(three_seeds, tests["stats_family_a"])))

if __name__ == "__main__":
    main()
