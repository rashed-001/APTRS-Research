"""
APTRS (masked DQN), vanilla DQN, and a PPO wrapper.

The only difference between APTRS and vanilla DQN is the action mask
applied to the Q-values before the argmax during action selection and
during the Bellman target computation.
"""

import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import QNetwork


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d, m = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s_next, dtype=np.float32),
            np.array(d, dtype=np.float32),
            np.array(m, dtype=bool),
        )

    def __len__(self):
        return len(self.buffer)


class MaskedDQNAgent:
    """
    DQN agent. Set use_mask=True for APTRS, use_mask=False for vanilla DQN.
    """

    def __init__(self, state_dim, action_dim, cfg, use_mask: bool = True,
                 reward_overrides: Optional[dict] = None, device: str = "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = cfg
        self.use_mask = use_mask
        self.device = torch.device(device)
        self.reward_overrides = reward_overrides or {}

        self.q = QNetwork(state_dim, action_dim, cfg.hidden1, cfg.hidden2).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim, cfg.hidden1, cfg.hidden2).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=cfg.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay = ReplayBuffer(cfg.replay_size)
        self.steps = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, state: np.ndarray, valid_mask: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            valid_idx = np.where(valid_mask)[0]
            return int(np.random.choice(valid_idx))
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q(s).squeeze(0).cpu().numpy()
        if self.use_mask:
            q = np.where(valid_mask, q, -1e9)
        return int(np.argmax(q))

    def store(self, s, a, r, s_next, done, valid_next):
        self.replay.push(s, a, r, s_next, float(done), valid_next)

    def train_step(self):
        if len(self.replay) < self.cfg.min_replay:
            return None
        s, a, r, s_next, d, m = self.replay.sample(self.cfg.batch_size)

        s = torch.tensor(s, device=self.device)
        a = torch.tensor(a, device=self.device).unsqueeze(1)
        r = torch.tensor(r, device=self.device)
        s_next = torch.tensor(s_next, device=self.device)
        d = torch.tensor(d, device=self.device)
        m = torch.tensor(m, dtype=torch.bool, device=self.device)

        q_pred = self.q(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(s_next)
            if self.use_mask:
                q_next = q_next.masked_fill(~m, -1e9)
            q_next_max = q_next.max(dim=1).values
            target = r + self.cfg.gamma * (1 - d) * q_next_max

        loss = self.loss_fn(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.cfg.target_update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location=self.device))

    def snapshot(self):
        import copy
        return copy.deepcopy(self.q.state_dict())

    def restore(self, state):
        self.q.load_state_dict(state)


# ----------------------------------------------------------------------
class PPOPolicy:
    """Adapter giving SB3 PPO / sb3-contrib MaskablePPO the select_action() interface."""

    def __init__(self, model, masked: bool):
        self.model, self.masked = model, masked

    def select_action(self, state, valid_mask, greedy: bool = True) -> int:
        if self.masked:
            a, _ = self.model.predict(state, deterministic=greedy, action_masks=valid_mask)
        else:
            a, _ = self.model.predict(state, deterministic=greedy)
        return int(np.asarray(a).item())


def build_ppo(env, cfg, seed: int, masked: bool = False):
    """PPO (unmasked) or MaskablePPO on a pool-sampling env."""
    from stable_baselines3.common.monitor import Monitor
    if masked:
        from sb3_contrib import MaskablePPO as Algo
    else:
        from stable_baselines3 import PPO as Algo
    return Algo("MlpPolicy", Monitor(env), learning_rate=cfg.ppo_lr, n_steps=cfg.ppo_n_steps,
                batch_size=cfg.ppo_batch_size, n_epochs=cfg.ppo_n_epochs, gamma=cfg.gamma,
                clip_range=cfg.ppo_clip, verbose=0, seed=seed)
