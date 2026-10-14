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
from src.tables import FINAL_RUNS, METRICS, run_row, to_markdown

SEEDS = (42, 123, 456)
COLUMNS = [*METRICS, "r_h_error"]
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "lpips": False, "dists": False, "r_h_error": False}
OWN_TRAINING_METRIC = {("C2", "lpips"), ("C3", "dists")}
RUN_FILES = ("metrics.csv", "train_log.csv", "run_summary.json")
SPIKE_FACTOR = 2.0
ALPHA = 0.05


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
                   *[RESULTS / f"table1_seed{seed}.csv" for seed in SEEDS]]


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


def formatted_table1(three_seeds: pd.DataFrame, family_a: pd.DataFrame) -> pd.DataFrame:
    trained = three_seeds[three_seeds["n_seeds"] > 0].set_index("config")
    best = {m: trained[m].idxmax() if higher else trained[m].idxmin() for m, higher in HIGHER_IS_BETTER.items()}
    significant = {(row["comparison"].split(" ")[0], row["metric"]) for _, row in family_a.iterrows() if row["p_holm"] < ALPHA}

    def cell(row: pd.Series, metric: str, digits: int, key: str | None = None) -> str:
        key = key or metric
        if row["n_seeds"] == 0:
            return f"{row[metric]:.{digits}f}"
        text = f"{row[metric]:.{digits}f} ± {row[f'{metric}_seed_std']:.{digits}f}"
        text = f"**{text}**" if best.get(key) == row["config"] else text
        return text + ("†" if (row["config"], key) in significant else "") + ("*" if (row["config"], key) in OWN_TRAINING_METRIC else "")

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


def markdown(three_seeds: pd.DataFrame, spikes: pd.DataFrame, tests: dict[str, pd.DataFrame]) -> str:
    wide = spikes.pivot(index="config", columns="seed", values="spike_epochs").loc[list(FINAL_RUNS)]
    spike_table = pd.DataFrame({"Configuration": [label(c) for c in wide.index],
                                **{f"seed {seed}": wide[seed].to_numpy() for seed in SEEDS}})
    screen = tests["screen_vs_outcome"].assign(spearman_rho=lambda f: f["spearman_rho"].map("{:.3f}".format),
                                               permutation_p=lambda f: f["permutation_p"].map("{:.4f}".format))
    sections = [
        ("Table 1: test split (220 slices), 100k, seeds 42, 123, 456", formatted_table1(three_seeds, tests["stats_family_a"]),
         "Mean ± standard deviation over the three per-seed means. Bold: best trained configuration. "
         f"† Holm-adjusted p < {ALPHA} against C0 (L1) in Family A (tested on PSNR, LPIPS and |1 − R_H| only; the mark on R_H is the "
         "|1 − R_H| test and does not say which side is better). * scored with its own training metric. "
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
    ]
    header = ("Generated by `python -m src.seeds_tables` from the files in `results/`. Inputs, outputs and their hashes: "
              "`manifest.json` in this folder. Seed 42 rows are the final runs (C8 after the variance fix, C9 and C10 on the CT dose network v2).")
    return header + "\n\n" + "\n\n".join(f"### {title}\n\n{to_markdown(frame)}" + (f"\n\n{note}" if note else "")
                                         for title, frame, note in sections) + "\n"


def fig3_slices(frame: pd.DataFrame, count: int = 2) -> pd.DataFrame:
    """Test slices closest to the median seed-averaged C0 PSNR (plan: not chosen by eye)."""
    psnr = frame[frame["configuration"] == "C0"].groupby(["slice_id", "patient_id"])["psnr"].mean().reset_index(name="c0_psnr_seed_mean")
    psnr["distance_to_median"] = (psnr["c0_psnr_seed_mean"] - psnr["c0_psnr_seed_mean"].median()).abs()
    return psnr.sort_values(["distance_to_median", "slice_id"]).head(count).reset_index(drop=True)


def main() -> None:
    frame, per_seed = per_slice(), per_seed_table()
    check_against_existing(per_seed)
    three_seeds, spikes = three_seed_table(per_seed), validation_spikes()
    tests = stats.run(frame, pd.read_csv(RESULTS / "table2_feature_networks.csv"))
    tables = {"per_slice_metrics": frame, "table1_per_seed": per_seed, "table1_three_seeds": three_seeds,
              "validation_spikes": spikes, "fig3_slices": fig3_slices(frame), **tests}
    outputs = []
    for name, table in tables.items():
        outputs.append(guarded(OUT / f"{name}.csv"))
        table.to_csv(outputs[-1], index=False)
    outputs += [guarded(OUT / "week3_tables.md"), guarded(OUT / "provenance_seeds.json")]
    outputs[-2].write_text(markdown(three_seeds, spikes, tests))
    outputs[-1].write_text(json.dumps(seed_runs_provenance(), indent=1) + "\n")
    record("seeds_tables", "python -m src.seeds_tables", input_files(), outputs,
           {"seeds": SEEDS, "family_a": stats.FAMILY_A, "family_b": stats.FAMILY_B, "test_metrics": stats.TEST_METRICS,
            "permutations": stats.PERMUTATIONS, "permutation_seed": stats.PERMUTATION_SEED, "spike_factor": SPIKE_FACTOR,
            "alpha": ALPHA})
    print(to_markdown(formatted_table1(three_seeds, tests["stats_family_a"])))


if __name__ == "__main__":
    main()
