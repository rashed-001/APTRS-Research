"""Bootstrap CIs and paired tests. (Renamed from statistics.py, which shadowed the stdlib module.)"""
import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(values, n_resamples=10000, alpha=0.05, seed=42):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_resamples, len(v)))].mean(axis=1)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def min_attainable_p(n_pairs: int) -> float:
    """Smallest two-sided exact Wilcoxon p for n non-zero pairs."""
    return 2.0 ** (1 - n_pairs)


def paired_wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n_eff = int(np.count_nonzero(d))
    out = {"n_pairs": len(a), "n_nonzero": n_eff, "min_p": min_attainable_p(max(n_eff, 1)),
           "median_diff": float(np.median(d)), "frac_a_better": float(np.mean(d > 0))}
    if n_eff == 0:
        return {**out, "statistic": 0.0, "p_value": 1.0}
    res = wilcoxon(a, b, zero_method="wilcox", method="auto")
    return {**out, "statistic": float(res.statistic), "p_value": float(res.pvalue)}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj.tolist()
