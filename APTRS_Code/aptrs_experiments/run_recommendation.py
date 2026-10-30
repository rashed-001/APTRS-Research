import os, pandas as pd
from config import Config
from environment import make_env, REC
from recommendation import *

cfg = Config(); os.makedirs(cfg.results_dir, exist_ok=True)
METHODS = {"aptrs_payoff": payoff_guided_selection, "severity_greedy": severity_greedy_selection,
           "min_cost_hardening": min_cost_hardening, "ilp_optimum": ilp_optimum}
rows, skipped = [], 0
for seed in cfg.seeds:
    for i in range(cfg.rec_scenarios):
        env = make_env(cfg, seed, REC, i)
        if count_paths(env) > cfg.max_paths:
            skipped += 1; continue
        paths = enumerate_paths(env); w = path_weights(paths, env)
        for B in cfg.budgets:
            for name, fn in METHODS.items():
                m = evaluate_control_set(fn(paths, w, B, env), paths, w, B, env)
                rows.append({**m, "seed": seed, "scenario": i, "budget": B, "method": name,
                             "n_paths": len(paths)})
pd.DataFrame(rows).to_csv(f"{cfg.results_dir}/recommendation.csv", index=False)
print(f"saved; skipped {skipped} graphs with > {cfg.max_paths} paths (report this in the paper)")
