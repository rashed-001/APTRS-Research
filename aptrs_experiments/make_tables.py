"""CSV -> LaTeX table bodies (results/tab_*.tex): removes manual transcription errors."""
import pandas as pd
R = "results"
f = lambda m, s, d=3: f"${m:.{d}f}\\pm{s:.{d}f}$"

def table(df, rowkey, cols, name):
    lines = []
    for r, g in df.groupby(rowkey, sort=False):
        cells = [f(g[c].mean(), g[c].std(ddof=1)) for c in cols]
        lines.append(" & ".join([str(r).replace("_", r"\_")] + cells) + r" \\")
    open(f"{R}/{name}.tex", "w").write("\n".join(lines) + "\n")

b = pd.read_csv(f"{R}/controlled_benchmark.csv")
table(b, "method", ["exact_acc", "precision", "recall", "f1"], "tab_path")
table(b, "method", ["reward", "steps", "invalid_rate", "latency_ms"], "tab_ops")
rec = pd.read_csv(f"{R}/recommendation.csv")
seedmean = rec.groupby(["budget", "method", "seed"], sort=False).mean(numeric_only=True).reset_index()
table(seedmean, ["budget", "method"], ["utility", "blocked_fraction", "within_budget"], "tab_rec")
for fn, key, cols in [("ablation", "variant", ["exact_acc", "reward_ratio"]),
                      ("nasim", "method", ["success_rate", "avg_steps", "invalid_rate"])]:
    try: table(pd.read_csv(f"{R}/{fn}.csv"), key, cols, f"tab_{fn}")
    except FileNotFoundError: print("missing", fn)
s = pd.read_csv(f"{R}/significance.csv")
open(f"{R}/tab_sig.tex", "w").write("\n".join(
    f"{r.metric} & {r.baseline.replace('_', chr(92)+'_')} & {r.p_value:.4f} & {r.p_holm:.4f} & {r.median_diff:.3f} \\\\"
    for r in s.itertuples()) + "\n")
