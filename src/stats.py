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


def seed_averaged(per_slice: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    return per_slice.groupby(["configuration", "slice_id"])[metrics].mean().reset_index()


def paired_family(per_slice: pd.DataFrame, pairs: list[tuple[str, str]], metrics: list[str]) -> pd.DataFrame:
    averaged = seed_averaged(per_slice, metrics)
    rows = []
    for metric in metrics:
        wide = averaged.pivot(index="slice_id", columns="configuration", values=metric)
        for first, second in pairs:
            differences = (wide[first] - wide[second]).dropna().to_numpy()
            rows.append({"comparison": f"{first} vs {second}", "metric": metric, "n": len(differences),
                         "median_difference": float(np.median(differences)),
                         "p_raw": float(wilcoxon(differences).pvalue), "effect_size_r": rank_biserial(differences)})
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_raw"].tolist())
    return result


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
