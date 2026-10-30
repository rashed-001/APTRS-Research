# APTRS Research

This repository contains the code and saved results for the APTRS research
experiments. The experiments compare graph-based planning and learning methods
for selecting security controls under a limited budget, including controlled
benchmarks, recommendation strategies, ablations, and NASim evaluations.

## Repository layout

```text
.
├── aptrs_experiments/              # Primary working copy of the experiment code
│   ├── results/                    # Saved benchmark, recommendation, and analysis CSVs
│   ├── config.py                   # Seeds, scenarios, model settings, and output paths
│   ├── evaluate.py                 # Controlled benchmark and statistical summaries
│   ├── run_recommendation.py       # Budgeted control recommendation experiments
│   ├── run_ablation.py             # Reward ablation experiments
│   ├── train.py                    # Training and evaluation routines
│   ├── requirements.txt            # Python dependencies
│   └── run_all.sh                  # Full experiment and table-generation sequence
├── APTRS_Code/
│   ├── aptrs_experiments/          # Matching copy of the experiment code and results
│   └── results/                    # Additional saved results
└── results/
    └── nasim.csv                   # Saved NASim result
```

`aptrs_experiments/` and `APTRS_Code/aptrs_experiments/` contain matching
copies. The root-level `aptrs_experiments/` is used in the commands below.
Existing result CSVs are included as experiment outputs; rerunning an
experiment can replace the corresponding CSV.

## Requirements

- Python 3.10 or later
- The packages listed in `aptrs_experiments/requirements.txt`
- Bash to run the complete `run_all.sh` sequence

Create and activate a virtual environment from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\aptrs_experiments\requirements.txt
```

## Running experiments

Run Python commands from the experiment directory so that its relative output
paths resolve as expected:

```powershell
Set-Location .\aptrs_experiments
python evaluate.py
python run_recommendation.py
python run_ablation.py
```

The full sequence, including the NASim confirmatory run and table generation,
can be run from that directory in Bash:

```bash
bash run_all.sh
```

Experiments may take a long time because they run over multiple seeds and
scenarios. Check each script's output and the configuration in `config.py`
before launching the full sequence. Results are written to the `results/`
directory relative to the experiment directory.



