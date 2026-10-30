"""Training / evaluation loops (v2): multi-graph training, held-out test graphs,
greedy evaluation, path metrics against the DP reference optimum."""
import random, time
import numpy as np
from environment import make_pool, TRAIN, VAL, TEST
from baselines import random_agent, shortest_path_agent, severity_greedy_agent

DQN_METHODS = ("aptrs", "vanilla_dqn")
MASKED = ("aptrs", "maskable_ppo")


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed)
    except ImportError:
        pass


def run_episode(env, agent, method, rng, train=True):
    state, _ = env.reset()
    done, total, steps, invalid, reached = False, 0.0, 0, 0, False
    path, losses, t0 = [env.current_node], [], time.perf_counter()
    while not done:
        valid = env.valid_actions()
        if method in MASKED:
            a = agent.select_action(state, valid, greedy=not train)
        elif method in ("vanilla_dqn", "ppo"):
            a = agent.select_action(state, np.ones_like(valid), greedy=not train)
        elif method == "random":
            a = random_agent(env, rng)
        elif method == "shortest_path":
            a = shortest_path_agent(env, rng)
        elif method == "severity_greedy":
            a = severity_greedy_agent(env, rng)
        else:
            raise ValueError(method)
        ns, r, term, trunc, info = env.step(a)
        done = term or trunc
        invalid += int(info["invalid"]); reached |= bool(info["reached_goal"])
        if not info["invalid"]:
            path.append(env.current_node)
        if train and method in DQN_METHODS:
            agent.store(state, a, r, ns, done, env.valid_actions())
            l = agent.train_step()
            if l is not None:
                losses.append(l)
        total += r; steps += 1; state = ns
    return {"reward": total, "steps": steps, "invalid": invalid, "reached_goal": reached,
            "path": path, "losses": losses, "latency_ms": (time.perf_counter() - t0) * 1e3}


def _evaluate_pool(method, agent, envs, rng):
    rows = []
    for env in envs:
        r = run_episode(env, agent, method, rng, train=False)
        opt, opt_ret = env.optimal_path()
        p, o = set(r["path"]), set(opt)
        prec, rec = len(p & o) / len(p), len(p & o) / len(o)
        rows.append({"reward": r["reward"], "steps": r["steps"], "invalid": r["invalid"],
                     "goal": float(r["reached_goal"]), "exact": float(r["path"] == opt),
                     "precision": prec, "recall": rec,
                     "f1": 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec),
                     "reward_ratio": r["reward"] / opt_ret, "latency_ms": r["latency_ms"]})
    m = lambda k: float(np.mean([x[k] for x in rows]))
    return {"reward": m("reward"), "reward_std": float(np.std([x["reward"] for x in rows], ddof=1)),
            "steps": m("steps"),
            "invalid_rate": sum(x["invalid"] for x in rows) / sum(x["steps"] for x in rows),
            "goal_rate": m("goal"), "exact_acc": m("exact"), "precision": m("precision"),
            "recall": m("recall"), "f1": m("f1"), "reward_ratio": m("reward_ratio"),
            "latency_ms": m("latency_ms")}


def evaluate_method(cfg, method, agent, seed):
    """Evaluate on the seed's held-out TEST graphs (same graphs for every method)."""
    envs = make_pool(cfg, seed, TEST, cfg.test_scenarios)
    return _evaluate_pool(method, agent, envs, np.random.default_rng([seed, 99]))


def train_agent(cfg, method, seed, device="cpu"):
    """Train aptrs / vanilla_dqn. Returns (agent, history); best-val weights restored."""
    from agents import MaskedDQNAgent
    seed_everything(seed)
    train_envs = make_pool(cfg, seed, TRAIN, cfg.train_scenarios)
    val_envs = make_pool(cfg, seed, VAL, cfg.val_scenarios)[: cfg.val_episodes_per_epoch]
    agent = MaskedDQNAgent(train_envs[0].state_dim, cfg.max_nodes, cfg,
                           use_mask=(method == "aptrs"), device=device)
    rng = np.random.default_rng([seed, 7])
    hist, best, best_state, wait = [], -np.inf, None, 0
    for epoch in range(cfg.max_epochs):
        ls = []
        for _ in range(cfg.episodes_per_epoch):
            env = train_envs[int(rng.integers(len(train_envs)))]
            ls += run_episode(env, agent, method, rng, train=True)["losses"]
        val = _evaluate_pool(method, agent, val_envs, rng)["reward"]
        hist.append({"epoch": epoch + 1, "train_loss": float(np.mean(ls)) if ls else np.nan,
                     "val_reward": val})
        if val > best:
            best, best_state, wait = val, agent.snapshot(), 0
        else:
            wait += 1
            if wait >= cfg.early_stop_patience:
                break
    agent.restore(best_state)
    return agent, hist


def train_ppo(cfg, seed, masked=False):
    """PPO / MaskablePPO trained on the same train pool, matched step budget."""
    import gymnasium as gym
    from agents import build_ppo, PPOPolicy
    seed_everything(seed)
    envs = make_pool(cfg, seed, TRAIN, cfg.train_scenarios)

    class PoolEnv(gym.Env):
        def __init__(s):
            s.rng = np.random.default_rng([seed, 11]); s.env = envs[0]
            s.observation_space, s.action_space = envs[0].observation_space, envs[0].action_space
        def reset(s, seed=None, options=None):
            s.env = envs[int(s.rng.integers(len(envs)))]; return s.env.reset()
        def step(s, a): return s.env.step(int(a))
        def action_masks(s): return s.env.valid_actions()

    model = build_ppo(PoolEnv(), cfg, seed, masked)
    model.learn(total_timesteps=cfg.ppo_timesteps)
    return PPOPolicy(model, masked)


def run_nasim_experiments(cfg):
    from nasim_adapter import run_all_nasim
    return run_all_nasim(cfg)
