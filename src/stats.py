# src/stats.py
"""Step 3.2 statistics: paired Wilcoxon tests with Holm correction and the screen-versus-outcome test.

The families are fixed by paper_plan.md and are not changed after seeing p-values. Called by src.seeds_tables.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon

TEST_METRICS = ["psnr", "lpips", "r_h_error"]
FAMILY_A = [(f"C{k}", "C0") for k in range(1, 11)]
FAMILY_B = [("C8", "C5"), ("C10", "C9"), ("C6", "C5")]
SCREEN_SCORES = ["B_phi", "M_phi", "Parameters"]
OUTCOMES = ["dists", "lpips", "r_h_error"]
CIRCULAR = ["C2", "C3"]                     # DISTS is their own training metric (C3) or close to it (C2)
PERMUTATIONS, PERMUTATION_SEED = 10_000, 42


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running_max
    return adjusted.tolist()


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    ranks = rankdata(np.abs(nonzero))
    positive, negative = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative)) if len(nonzero) else 0.0


def seed_averaged(per_slice: pd.DataFrame, metrics: list[str], unit: str = "slice_id") -> pd.DataFrame:
    """Seeds averaged per slice; with unit="patient_id" the slices of each patient are averaged after that."""
    slices = per_slice.groupby(["configuration", "slice_id"])[metrics].mean().reset_index()
    if unit == "slice_id":
        return slices
    patients = per_slice[["slice_id", "patient_id"]].drop_duplicates()
    return slices.merge(patients, on="slice_id").groupby(["configuration", unit])[metrics].mean().reset_index()


def paired_family(per_slice: pd.DataFrame, pairs: list[tuple[str, str]], metrics: list[str],
                  unit: str = "slice_id") -> pd.DataFrame:
    averaged = seed_averaged(per_slice, metrics, unit)
    rows = []
    for metric in metrics:
        wide = averaged.pivot(index=unit, columns="configuration", values=metric)
        for first, second in pairs:
            differences = (wide[first] - wide[second]).dropna().to_numpy()
            rows.append({"comparison": f"{first} vs {second}", "metric": metric, "n": len(differences),
                         "median_difference": float(np.median(differences)),
                         "p_raw": float(wilcoxon(differences).pvalue), "effect_size_r": rank_biserial(differences)})
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_raw"].tolist())
    return result


def seed_spread(per_slice: pd.DataFrame, pairs: list[tuple[str, str]], metrics: list[str]) -> pd.DataFrame:
    """Mean difference over slices of each pair at each seed, and its standard deviation over the seeds."""
    means = per_slice.groupby(["configuration", "seed"])[metrics].mean()
    rows = []
    for metric in metrics:
        for first, second in pairs:
            per_seed = (means.loc[first, metric] - means.loc[second, metric]).sort_index()
            rows.append({"comparison": f"{first} vs {second}", "metric": metric,
                         **{f"mean_difference_seed{seed}": value for seed, value in per_seed.items()},
                         "mean_difference_seed_mean": per_seed.mean(), "mean_difference_seed_std": per_seed.std(ddof=1)})
    return pd.DataFrame(rows)


def slice_and_patient(per_slice: pd.DataFrame, family: str, pairs: list[tuple[str, str]], alpha: float) -> pd.DataFrame:
    """The planned slice-level family next to the same family on the patient means, with the seed spread of each difference."""
    keys = ["comparison", "metric"]
    levels = [paired_family(per_slice, pairs, TEST_METRICS, unit).set_index(keys).add_prefix(f"{level}_")
              for level, unit in (("slice", "slice_id"), ("patient", "patient_id"))]
    result = levels[0].join(levels[1]).join(seed_spread(per_slice, pairs, TEST_METRICS).set_index(keys)).reset_index()
    for level in ("slice", "patient"):
        result[f"significant_{level}"] = result[f"{level}_p_holm"] < alpha
    return result.assign(family=family)[["family", *result.columns]]


def screen_predicts_outcome(screen: pd.Series, outcome: pd.Series, permutations: int = PERMUTATIONS,
                            seed: int = PERMUTATION_SEED) -> tuple[float, float]:
    rho = spearmanr(screen, outcome).statistic
    rng = np.random.default_rng(seed)
    null = [spearmanr(screen, rng.permutation(outcome.to_numpy())).statistic for _ in range(permutations)]
    return float(rho), float(np.mean(np.abs(null) >= abs(rho)))


def screen_vs_outcome(per_slice: pd.DataFrame, table2: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation over the trained networks, with and without the configurations where DISTS is circular."""
    outcomes = seed_averaged(per_slice, OUTCOMES).groupby("configuration")[OUTCOMES].mean()
    scores = table2.set_index("Configuration")[SCREEN_SCORES]
    rows = []
    for subset, dropped in (("all trained networks", []), ("without C2 and C3", CIRCULAR)):
        names = [name for name in scores.index if name not in dropped]
        for score in SCREEN_SCORES:
            for outcome in OUTCOMES:
                rho, p_value = screen_predicts_outcome(scores.loc[names, score], outcomes.loc[names, outcome])
                rows.append({"screen_score": score, "outcome": outcome, "subset": subset, "n": len(names),
                             "spearman_rho": rho, "permutation_p": p_value})
    return pd.DataFrame(rows)


def run(per_slice: pd.DataFrame, table2: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"stats_family_a": paired_family(per_slice, FAMILY_A, TEST_METRICS),
            "stats_family_b": paired_family(per_slice, FAMILY_B, TEST_METRICS),
            "screen_vs_outcome": screen_vs_outcome(per_slice, table2)}
