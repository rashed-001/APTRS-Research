"""Reward ablation: train under each variant, ALWAYS evaluate under the full reward so
returns are comparable (v1 compared returns across different reward scales)."""
import os, pandas as pd
from dataclasses import replace
from config import Config
from train import train_agent, evaluate_method

cfg = Config(); os.makedirs(cfg.results_dir, exist_ok=True)
V = {"full": {}, "no_start": {"start_reward": 0.0}, "no_cvss": {"cvss_divisor": float("inf")},
     "no_execution": {"execution_reward": 0.0}, "reduced_goal": {"goal_reward": 10.0}}
rows = []
for name, ov in V.items():
    for seed in cfg.seeds:
        agent, _ = train_agent(replace(cfg, **ov), "aptrs", seed)
        rows.append({**evaluate_method(cfg, "aptrs", agent, seed), "variant": name, "seed": seed})
pd.DataFrame(rows).to_csv(f"{cfg.results_dir}/ablation.csv", index=False)
