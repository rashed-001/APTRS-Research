"""
Attack-graph MDP (v2).

Changes vs v1: (i) graphs are padded to cfg.max_nodes so a single Q-network can be
trained on many graphs and tested on unseen ones; (ii) the observation exposes the
goal, per-node severity/type and the current node's adjacency row; (iii) split-aware
seeding (train/val/test/rec never share a graph); (iv) a DP reference optimum.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple, List

TRAIN, VAL, TEST, REC = 0, 1, 2, 3


class AttackGraphEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, adjacency, cvss, start_node, goal_node, node_types, cfg):
        super().__init__()
        n, N = int(adjacency.shape[0]), int(cfg.max_nodes)
        assert n <= N
        self.cfg, self.num_nodes, self.N = cfg, n, N
        self.adj = np.zeros((N, N), np.float32); self.adj[:n, :n] = adjacency
        self.cvss = np.zeros(N, np.float32); self.cvss[:n] = cvss
        self.node_types = np.zeros(N, np.int64); self.node_types[:n] = node_types
        self.start_node, self.goal_node = int(start_node), int(goal_node)
        self.state_dim = 5 * N
        self.observation_space = spaces.Box(-1.0, 100.0, (self.state_dim,), np.float32)
        self.action_space = spaces.Discrete(N)
        self.current_node, self.steps, self.max_steps = self.start_node, 0, 2 * n

    # ------------------------------------------------------------------
    def _encode_state(self) -> np.ndarray:
        cur = np.zeros(self.N, np.float32); cur[self.current_node] = 1.0
        goal = np.zeros(self.N, np.float32); goal[self.goal_node] = 1.0
        return np.concatenate([cur, goal, self.cvss / 10.0,
                               self.node_types.astype(np.float32) / 2.0,
                               (self.adj[self.current_node] > 0).astype(np.float32)])

    def valid_actions(self) -> np.ndarray:
        return self.adj[self.current_node] > 0

    def transition_reward(self, u: int, v: int) -> float:
        """Eq. (1); cases evaluated in the order listed in the paper."""
        c = self.cfg
        if v == self.goal_node:
            return c.goal_reward
        if u == self.start_node:
            return c.start_reward
        d = max(0.0, float(self.cvss[v] - self.cvss[u]))
        if d > 0:
            return d / c.cvss_divisor
        if self.node_types[v] == 1:
            return c.execution_reward
        return 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.current_node, self.steps = self.start_node, 0
        return self._encode_state(), {}

    def step(self, action: int):
        self.steps += 1
        action = int(action)
        truncated = self.steps >= self.max_steps
        if not self.valid_actions()[action]:
            return (self._encode_state(), self.cfg.invalid_reward, False, truncated,
                    {"invalid": True, "reached_goal": False})
        r = self.transition_reward(self.current_node, action)
        self.current_node = action
        terminated = action == self.goal_node
        return (self._encode_state(), r, terminated, truncated,
                {"invalid": False, "reached_goal": terminated})

    # ------------------------------------------------------------------
    def optimal_path(self) -> Tuple[Optional[List[int]], float]:
        """Max-return start->goal path by DP (graph is a DAG in index order).
        Ties: fewer hops, then lower node index. This defines 'exact-path accuracy'."""
        n, g = self.num_nodes, self.goal_node
        assert not np.tril(self.adj[:n, :n]).any(), "graph must be index-ordered DAG"
        val = np.full(n, -np.inf); hops = np.zeros(n, int); nxt = np.full(n, -1)
        val[g] = 0.0
        for u in range(n - 1, -1, -1):
            if u == g:
                continue
            best = None
            for v in np.nonzero(self.adj[u, :n] > 0)[0]:
                if np.isinf(val[v]):
                    continue
                key = (round(self.transition_reward(u, int(v)) + val[v], 9), -(hops[v] + 1), -int(v))
                if best is None or key > best[0]:
                    best = (key, int(v))
            if best is not None:
                val[u], hops[u], nxt[u] = best[0][0], -best[0][1], best[1]
        if np.isinf(val[self.start_node]):
            return None, float("-inf")
        path, u = [self.start_node], self.start_node
        while u != g:
            u = int(nxt[u]); path.append(u)
        return path, float(val[self.start_node])


# ----------------------------------------------------------------------
def generate_scenario(rng: np.random.Generator, cfg) -> Dict:
    n = int(rng.integers(cfg.min_nodes, cfg.max_nodes + 1))
    adj = np.zeros((n, n), np.float32)
    layers = [0, n // 3, 2 * n // 3, n - 1]
    layer = lambda i: 0 if i < layers[1] else (1 if i < layers[2] else 2)
    for i in range(n):
        for j in range(n):
            if i != j and layer(j) == layer(i) + 1 and rng.random() < 0.35:
                adj[i, j] = 1.0
    adj[layers[2], n - 1] = 1.0
    for node in range(n - 1):
        if not adj[node].any():
            adj[node, int(rng.integers(node + 1, n))] = 1.0
    cvss = rng.uniform(0.0, 10.0, n).astype(np.float32); cvss[0] = 0.0
    types = rng.integers(0, 3, n).astype(np.int64); types[n - 1] = 2
    return {"adjacency": adj, "cvss": cvss, "start_node": 0, "goal_node": n - 1,
            "node_types": types, "num_nodes": n}


def make_env(cfg, seed: int, split: int = TEST, index: int = 0) -> AttackGraphEnv:
    """Graph (seed, split, index) is unique; splits never overlap."""
    sc = generate_scenario(np.random.default_rng([int(seed), int(split), int(index)]), cfg)
    return AttackGraphEnv(sc["adjacency"], sc["cvss"], sc["start_node"],
                          sc["goal_node"], sc["node_types"], cfg)


def make_pool(cfg, seed: int, split: int, count: int) -> List[AttackGraphEnv]:
    return [make_env(cfg, seed, split, i) for i in range(count)]
