"""
nasim_adapter.py

Wrapper that lets the APTRS agent, vanilla DQN, and PPO run on NASim
scenarios. NASim is a public testbed for autonomous pentesting agents; using
it provides a reproducibility signal that a self-generated benchmark cannot.

The adapter handles:
  * NASim version differences in reset() and step() return signatures,
  * conversion of NASim's observation to a float32 vector,
  * retrieval of the action mask at each step,
  * detection of successful episode termination,
  * training and evaluation loops for all baselines and APTRS.

Reference:
    Schwartz, J. and Kurniawati, H. (2020). NASim: Network Attack Simulator.
    AAMAS 2020, pp. 2118-2120.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import gymnasium as gym
from typing import Dict, List, Optional, Tuple

import nasim

from agents import MaskedDQNAgent


class _NASimActionWrapper(gym.ActionWrapper):
    def action(self, action):
        return int(np.asarray(action).item())


# =====================================================================
# Version-safe wrappers for NASim's env API
# =====================================================================

def _safe_reset(env, seed: Optional[int] = None):
    """
    NASim has used both `obs = env.reset()` and `obs, info = env.reset(seed=...)`
    across versions. This wrapper normalises both.
    """
    try:
        out = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}


def _safe_step(env, action: int):
    """
    NASim returns either 4-tuple (old gym) or 5-tuple (new gymnasium).
    This wrapper always returns the 5-tuple form.
    """
    out = env.step(action)
    if len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), False, info
    obs, reward, terminated, truncated, info = out
    return obs, float(reward), bool(terminated), bool(truncated), info


def _nasim_action_invalid(info) -> bool:
    if not isinstance(info, dict):
        return False
    return any(info.get(key, False) for key in ("connection_error", "permission_error"))


# =====================================================================
# Environment creation and observation handling
# =====================================================================

def make_nasim_env(scenario: str = "tiny", seed: int = 0):
    """
    Create and reset a NASim benchmark scenario.

    Parameters
    ----------
    scenario : str
        One of {'tiny', 'tiny-hard', 'small', 'small-hard', 'medium',
                'medium-hard', 'large'}.
    seed : int
        RNG seed for the environment.

    Returns
    -------
    env : NASim environment (already reset once)
    obs : initial observation
    info : auxiliary info dict
    """
    env = nasim.make_benchmark(scenario)
    obs, info = _safe_reset(env, seed=seed)
    return env, obs, info


def nasim_state_to_vector(state) -> np.ndarray:
    """
    Flatten any NASim observation into a fixed-length float32 vector.

    NASim's observations are already 1-D float arrays, but this wrapper
    guarantees shape and dtype consistency even if the API changes.
    """
    arr = np.asarray(state, dtype=np.float32).flatten()
    return arr


def nasim_valid_mask(env) -> np.ndarray:
    """
    Return the boolean action mask for the current state.

    Use NASim's mask API where available. NASim 0.12 calls
    `host_discovered` on the network instead of the current state, so its
    mask method raises; in that case mirror the transition preconditions.
    """
    state = getattr(env, "current_state", None)
    network = getattr(env, "network", None)
    get_action = getattr(env.action_space, "get_action", None)
    state_checks_available = (
        callable(get_action)
        and callable(getattr(state, "host_discovered", None))
        and callable(getattr(state, "host_reachable", None))
        and callable(getattr(state, "host_compromised", None))
        and callable(getattr(network, "has_required_remote_permission", None))
        and callable(getattr(network, "traffic_permitted", None))
    )
    if state_checks_available:
        mask = np.zeros(env.action_space.n, dtype=bool)
        for index in range(env.action_space.n):
            action = get_action(index)
            is_noop = getattr(action, "is_noop", None)
            if callable(is_noop) and is_noop():
                mask[index] = True
                continue

            target = getattr(action, "target", None)
            if target is None:
                raise RuntimeError("Cannot determine NASim action target")
            if not state.host_discovered(target) or not state.host_reachable(target):
                continue
            if action.is_remote() and not env.network.has_required_remote_permission(state, action):
                continue
            if action.is_exploit() and not env.network.traffic_permitted(state, target, action.service):
                continue
            if action.is_privilege_escalation() and not state.host_compromised(target):
                continue
            mask[index] = True
    else:
        mask = getattr(env, "action_mask", None)
        if mask is None:
            get_mask = getattr(env, "get_action_mask", None)
            if callable(get_mask):
                try:
                    mask = get_mask()
                except AttributeError:
                    mask = None
        if mask is None:
            raise RuntimeError("NASim does not expose a supported action-mask API")
    mask = np.asarray(mask, dtype=bool).copy()
    if mask.ndim != 1 or mask.shape[0] != env.action_space.n:
        raise RuntimeError("NASim action mask does not match the action space")
    return mask


def _require_action_mask(env) -> np.ndarray:
    mask = nasim_valid_mask(env)
    if not mask.any():
        raise RuntimeError("NASim returned no valid actions for a live state")
    return mask


def nasim_goal_reached(info: dict, terminated: bool, env) -> bool:
    """
    Determine whether the episode ended because the agent reached the goal.

    NASim's `info["success"]` describes whether an individual action worked,
    not whether the episode reached its goal. Episode termination is the
    reliable cross-version signal for goal completion.
    """
    if isinstance(info, dict):
        if info.get("goal_reached", False):
            return True
    return bool(terminated)


# =====================================================================
# Training loop for DQN-style agents (APTRS and vanilla DQN)
# =====================================================================

def train_dqn_on_nasim(
    agent: MaskedDQNAgent,
    scenario: str,
    seed: int,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 50,
    verbose: bool = False,
) -> List[float]:
    """
    Train a DQN or APTRS agent on a NASim scenario for a fixed number of
    episodes. Returns the per-episode reward list.
    """
    env, _, _ = make_nasim_env(scenario, seed=seed)
    rng = np.random.default_rng(seed)
    episode_rewards: List[float] = []

    for ep in range(num_episodes):
        obs, info = _safe_reset(env, seed=seed + ep)
        state = nasim_state_to_vector(obs)
        done = False
        ep_reward = 0.0
        steps = 0

        while not done and steps < max_steps_per_episode:
            valid = _require_action_mask(env)
            action = agent.select_action(state, valid)
            next_obs, reward, terminated, truncated, info = _safe_step(env, action)
            done = terminated or truncated
            next_state = nasim_state_to_vector(next_obs)
            next_valid = nasim_valid_mask(env)
            agent.store(state, action, reward, next_state, done, next_valid)
            agent.train_step()
            state = next_state
            ep_reward += reward
            steps += 1

        episode_rewards.append(ep_reward)
        if verbose and (ep + 1) % 100 == 0:
            print(f"[nasim {scenario} | seed {seed}] episode {ep+1}: "
                  f"avg reward (last 100) = {np.mean(episode_rewards[-100:]):.3f}")

    return episode_rewards


# =====================================================================
# Evaluation loops for every method on NASim
# =====================================================================

def evaluate_dqn_on_nasim(
    agent: MaskedDQNAgent,
    scenario: str,
    seed: int,
    num_eval: int = 100,
    max_steps_per_episode: int = 50,
) -> Dict[str, float]:
    """
    Evaluate a trained DQN/APTRS agent on NASim. Returns success rate,
    average steps, invalid-action rate, and average reward.
    """
    env, _, _ = make_nasim_env(scenario, seed=seed)
    successes, steps_list, invalid_list, rewards = [], [], [], []

    for ep in range(num_eval):
        obs, info = _safe_reset(env, seed=seed + 10000 + ep)
        state = nasim_state_to_vector(obs)
        done, steps, invalid_count = False, 0, 0
        ep_reward = 0.0
        last_info = info
        last_terminated = False

        while not done and steps < max_steps_per_episode:
            valid = _require_action_mask(env)
            action = agent.select_action(state, valid, greedy=True)
            next_obs, reward, terminated, truncated, last_info = _safe_step(env, action)
            if not valid[action] or _nasim_action_invalid(last_info):
                invalid_count += 1
            last_terminated = terminated
            done = terminated or truncated
            state = nasim_state_to_vector(next_obs)
            ep_reward += reward
            steps += 1

        successes.append(1.0 if nasim_goal_reached(last_info, last_terminated, env) else 0.0)
        steps_list.append(steps)
        invalid_list.append(invalid_count)
        rewards.append(ep_reward)

    return {
        "success_rate": float(np.mean(successes)),
        "avg_steps": float(np.mean(steps_list)),
        "invalid_rate": float(np.sum(invalid_list) / np.sum(steps_list)),
        "avg_reward": float(np.mean(rewards)),
    }


def evaluate_ppo_on_nasim(
    model,
    scenario: str,
    seed: int,
    num_eval: int = 100,
    max_steps_per_episode: int = 50,
) -> Dict[str, float]:
    """
    Evaluate a Stable-Baselines3 PPO model on NASim.
    """
    env, _, _ = make_nasim_env(scenario, seed=seed)
    successes, steps_list, invalid_list, rewards = [], [], [], []

    for ep in range(num_eval):
        obs, info = _safe_reset(env, seed=seed + 20000 + ep)
        done, steps, invalid_count = False, 0, 0
        ep_reward = 0.0
        last_info = info
        last_terminated = False

        while not done and steps < max_steps_per_episode:
            valid = _require_action_mask(env)
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).item())
            obs, reward, terminated, truncated, last_info = _safe_step(env, action)
            if not valid[action] or _nasim_action_invalid(last_info):
                invalid_count += 1
            last_terminated = terminated
            done = terminated or truncated
            ep_reward += reward
            steps += 1

        successes.append(1.0 if nasim_goal_reached(last_info, last_terminated, env) else 0.0)
        steps_list.append(steps)
        invalid_list.append(invalid_count)
        rewards.append(ep_reward)

    return {
        "success_rate": float(np.mean(successes)),
        "avg_steps": float(np.mean(steps_list)),
        "invalid_rate": float(np.sum(invalid_list) / np.sum(steps_list)),
        "avg_reward": float(np.mean(rewards)),
    }


def evaluate_baseline_on_nasim(
    method: str,
    scenario: str,
    seed: int,
    num_eval: int = 100,
    max_steps_per_episode: int = 50,
) -> Dict[str, float]:
    """
    Evaluate a non-learning baseline (random or shortest-path) on NASim.

    NASim does not expose a graph structure in the same way as our controlled
    benchmark, so `shortest_path` here is approximated by a deterministic
    policy that always picks the first valid action. A more principled
    shortest-path policy on NASim would require access to the internal
    topology, which we leave to future work. The `random` policy samples
    uniformly from valid actions.
    """
    env, _, _ = make_nasim_env(scenario, seed=seed)
    rng = np.random.default_rng(seed)
    successes, steps_list, invalid_list, rewards = [], [], [], []

    for ep in range(num_eval):
        obs, info = _safe_reset(env, seed=seed + 30000 + ep)
        done, steps, invalid_count = False, 0, 0
        ep_reward = 0.0
        last_info = info
        last_terminated = False

        while not done and steps < max_steps_per_episode:
            valid = _require_action_mask(env)
            valid_idx = np.where(valid)[0]

            if method == "random":
                action = int(rng.choice(valid_idx))
            elif method == "first_valid":
                action = int(valid_idx[0])
            else:
                raise ValueError(f"unknown baseline: {method}")

            obs, reward, terminated, truncated, last_info = _safe_step(env, action)
            if _nasim_action_invalid(last_info):
                invalid_count += 1
            last_terminated = terminated
            done = terminated or truncated
            ep_reward += reward
            steps += 1

        successes.append(1.0 if nasim_goal_reached(last_info, last_terminated, env) else 0.0)
        steps_list.append(steps)
        invalid_list.append(invalid_count)
        rewards.append(ep_reward)

    return {
        "success_rate": float(np.mean(successes)),
        "avg_steps": float(np.mean(steps_list)),
        "invalid_rate": float(np.sum(invalid_list) / np.sum(steps_list)),
        "avg_reward": float(np.mean(rewards)),
    }


# =====================================================================
# Top-level driver: run all methods on all NASim scenarios and seeds
# =====================================================================

def run_all_nasim(
    cfg,
    dqn_episodes: int = 1000,
    num_eval: int = 100,
    ppo_timesteps: int = 200_000,
) -> pd.DataFrame:
    """
    Run every method on every NASim scenario and every seed, then write
    results to `results/nasim.csv`.

    Parameters
    ----------
    cfg : Config
        The same Config object used elsewhere in the pipeline.
    """
    os.makedirs(cfg.results_dir, exist_ok=True)
    rows: List[Dict] = []

    for scenario in cfg.nasim_scenarios:
        for seed in cfg.nasim_seeds:
            # Random
            m = evaluate_baseline_on_nasim("random", scenario, seed, num_eval=num_eval)
            rows.append({"scenario": scenario, "seed": seed, "method": "random", **m})
            print(f"[NASim {scenario} | seed {seed}] random: {m}")

            # Shortest-path (deterministic-first-valid)
            m = evaluate_baseline_on_nasim("first_valid", scenario, seed, num_eval=num_eval)
            rows.append({"scenario": scenario, "seed": seed, "method": "first_valid", **m})
            print(f"[NASim {scenario} | seed {seed}] shortest_path: {m}")

            # Vanilla DQN
            env, obs, _ = make_nasim_env(scenario, seed=seed)
            state_dim = nasim_state_to_vector(obs).shape[0]
            action_dim = env.action_space.n
            dqn = MaskedDQNAgent(state_dim, action_dim, cfg, use_mask=False)
            train_dqn_on_nasim(dqn, scenario, seed, num_episodes=dqn_episodes)
            m = evaluate_dqn_on_nasim(dqn, scenario, seed, num_eval=num_eval)
            rows.append({"scenario": scenario, "seed": seed, "method": "vanilla_dqn", **m})
            print(f"[NASim {scenario} | seed {seed}] vanilla_dqn: {m}")

            # APTRS (masked DQN)
            env, obs, _ = make_nasim_env(scenario, seed=seed)
            state_dim = nasim_state_to_vector(obs).shape[0]
            action_dim = env.action_space.n
            aprs = MaskedDQNAgent(state_dim, action_dim, cfg, use_mask=True)
            train_dqn_on_nasim(aprs, scenario, seed, num_episodes=dqn_episodes)
            m = evaluate_dqn_on_nasim(aprs, scenario, seed, num_eval=num_eval)
            rows.append({"scenario": scenario, "seed": seed, "method": "aptrs", **m})
            print(f"[NASim {scenario} | seed {seed}] aptrs: {m}")

            # PPO
            try:
                from stable_baselines3 import PPO
                from stable_baselines3.common.monitor import Monitor
                env_ppo, _, _ = make_nasim_env(scenario, seed=seed)
                env_ppo = Monitor(_NASimActionWrapper(env_ppo))
                model = PPO(
                    "MlpPolicy",
                    env_ppo,
                    learning_rate=cfg.learning_rate,
                    n_steps=cfg.ppo_n_steps,
                    batch_size=cfg.batch_size,
                    n_epochs=cfg.ppo_n_epochs,
                    gamma=cfg.gamma,
                    clip_range=cfg.ppo_clip,
                    verbose=0,
                    seed=seed,
                )
                model.learn(total_timesteps=ppo_timesteps)
                m = evaluate_ppo_on_nasim(model, scenario, seed, num_eval=num_eval)
                rows.append({"scenario": scenario, "seed": seed, "method": "ppo", **m})
                print(f"[NASim {scenario} | seed {seed}] ppo: {m}")
            except ImportError as exc:
                print(f"[nasim_adapter] PPO skipped: {exc}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(cfg.results_dir, "nasim.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[nasim_adapter] results written to {out_path}")
    return df


# =====================================================================
# Standalone smoke test
# =====================================================================

if __name__ == "__main__":
    """
    Run this file directly to smoke-test the NASim adapter:
        python nasim_adapter.py
    It runs one scenario, one seed, all methods with small training budgets
    so you can verify that NASim is installed and the API is compatible.
    """
    from config import Config
    cfg = Config()
    cfg.nasim_scenarios = ["tiny"]
    cfg.nasim_seeds = [0]
    df = run_all_nasim(cfg, dqn_episodes=5, num_eval=2, ppo_timesteps=512)
    print("\n=== NASim smoke test complete ===")
    print(df)