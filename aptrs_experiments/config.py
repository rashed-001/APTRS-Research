"""Central configuration (v2). Seeds, split sizes and hyper-parameters live here."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Reproducibility: >=10 seeds so that a paired Wilcoxon test can reach p<0.05
    # (with n=5 pairs the smallest attainable two-sided p is 0.0625).
    seeds: List[int] = field(default_factory=lambda: list(range(10)))

    # Graph generation (graphs are padded to max_nodes so ONE network fits all)
    min_nodes: int = 8
    max_nodes: int = 26
    graph_family: str = "exposure_aware"

    # Scenario pools = number of DISTINCT graphs per split
    train_scenarios: int = 500
    val_scenarios: int = 250
    test_scenarios: int = 250
    rec_scenarios: int = 40           # graphs per seed for the recommendation study

    # Training
    episodes_per_epoch: int = 50
    val_episodes_per_epoch: int = 50
    max_epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 8e-4
    gamma: float = 0.95
    replay_size: int = 50000
    min_replay: int = 1000
    target_update_freq: int = 500
    early_stop_patience: int = 10

    # Network
    hidden1: int = 80
    hidden2: int = 40

    # PPO (budget matched to DQN in environment steps; tune lr/clip separately)
    ppo_timesteps: int = 60000
    ppo_lr: float = 3e-4
    ppo_n_steps: int = 256
    ppo_batch_size: int = 64
    ppo_n_epochs: int = 10
    ppo_clip: float = 0.2

    # Reward (Eq. 1)
    goal_reward: float = 100.0
    start_reward: float = 0.01
    execution_reward: float = 1.5
    invalid_reward: float = -1.0
    cvss_divisor: float = 10.0

    # Recommendation layer: U = alpha*blocked - beta*cost/B - gamma_d*sum(disruption)
    budgets: List[int] = field(default_factory=lambda: [2, 4, 6])
    alpha: float = 1.0
    beta: float = 1.0
    gamma_d: float = 0.05
    max_paths: int = 2000

    # NASim
    nasim_scenarios: List[str] = field(default_factory=lambda: ["small", "medium"])
    nasim_seeds: List[int] = field(default_factory=lambda: list(range(10)))

    results_dir: str = "results"
    figures_dir: str = "figures"
    models_dir: str = "models"
