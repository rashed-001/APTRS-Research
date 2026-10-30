#!/bin/bash
set -e
echo "=== 1. Controlled benchmark (10 seeds, held-out test graphs) ===" && python evaluate.py
echo "=== 2. Recommendation layer ===" && python run_recommendation.py
echo "=== 3. Reward ablation ===" && python run_ablation.py
echo "=== 4. NASim confirmatory ===" && python -c "from config import Config; from train import run_nasim_experiments; run_nasim_experiments(Config())"
echo "=== 5. LaTeX tables ===" && python make_tables.py
