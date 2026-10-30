"""Bootstrap confidence intervals and Wilcoxon signed-rank tests."""

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(values, num_resamples: int = 1000, alpha: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    resamples = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(num_resamples)
    ])
    lower = np.percentile(resamples, 100 * alpha / 2)
    upper = np.percentile(resamples, 100 * (1 - alpha / 2))
    return float(values.mean()), float(values.std(ddof=1)), float(lower), float(upper)


def paired_wilcoxon(aprs_values, baseline_values):
    aprs_values = np.asarray(aprs_values)
    baseline_values = np.asarray(baseline_values)
    if len(aprs_values) < 5:
        return {"statistic": None, "p_value": None, "note": "insufficient seeds"}
    stat, p = wilcoxon(aprs_values, baseline_values)
    return {"statistic": float(stat), "p_value": float(p)}