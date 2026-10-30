"""All methods x all seeds on held-out test graphs."""
import os, time
import numpy as np, pandas as pd
from tqdm import tqdm
from config import Config
from train import train_agent, train_ppo, evaluate_method
from stats_utils import bootstrap_ci, paired_wilcoxon, holm

METHODS = ["random", "shortest_path", "severity_greedy", "vanilla_dqn", "aptrs", "ppo", "maskable_ppo"]
METRICS = ["exact_acc", "precision", "recall", "f1", "reward", "steps", "invalid_rate",
           "goal_rate", "reward_ratio", "latency_ms"]


def evaluate_controlled(cfg, methods=METHODS):
    os.makedirs(cfg.results_dir, exist_ok=True)
    rows, curves = [], []
    for seed in tqdm(cfg.seeds, desc="Seeds"):
        for method in methods:
            t0, agent = time.time(), None
            try:
                if method in ("aptrs", "vanilla_dqn"):
                    agent, hist = train_agent(cfg, method, seed)
                    curves += [{**h, "method": method, "seed": seed} for h in hist]
                elif method in ("ppo", "maskable_ppo"):
                    agent = train_ppo(cfg, seed, masked=(method == "maskable_ppo"))
            except ImportError as e:
                print(f"[skip] {method}: {e}"); continue
            m = evaluate_method(cfg, method, agent, seed)
            m.update(method=method, seed=seed, train_time=time.time() - t0)
            rows.append(m)
            print(f"[seed {seed}] {method}: exact={m['exact_acc']:.3f} reward={m['reward']:.2f} "
                  f"invalid={m['invalid_rate']:.4f}")
    df = pd.DataFrame(rows); df.to_csv(f"{cfg.results_dir}/controlled_benchmark.csv", index=False)
    pd.DataFrame(curves).to_csv(f"{cfg.results_dir}/training_curves.csv", index=False)
    return df


def summarize(df, cfg):
    out = []
    for method in df["method"].unique():
        for k in METRICS:
            mean, sd, lo, hi = bootstrap_ci(df[df.method == method].sort_values("seed")[k].values)
            out.append(dict(method=method, metric=k, mean=mean, std=sd, ci_lower=lo, ci_upper=hi))
    s = pd.DataFrame(out); s.to_csv(f"{cfg.results_dir}/summary.csv", index=False); return s


def significance_tests(df, cfg, ref="aptrs"):
    rows = []
    for k in ["exact_acc", "reward"]:
        a = df[df.method == ref].sort_values("seed")[k].values
        part = []
        for b in df.method.unique():
            if b == ref: continue
            r = paired_wilcoxon(a, df[df.method == b].sort_values("seed")[k].values)
            part.append({"metric": k, "baseline": b, **r})
        for r, p in zip(part, holm([r["p_value"] for r in part])):
            r["p_holm"] = p
        rows += part
    s = pd.DataFrame(rows); s.to_csv(f"{cfg.results_dir}/significance.csv", index=False); return s


if __name__ == "__main__":
    cfg = Config(); df = evaluate_controlled(cfg)
    print(summarize(df, cfg)); print(significance_tests(df, cfg))
