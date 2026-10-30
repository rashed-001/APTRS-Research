"""Non-learning baselines: random, shortest-path (hop count), severity-greedy."""
import numpy as np
from collections import deque


def random_agent(env, rng):
    return int(rng.choice(np.where(env.valid_actions())[0]))


def shortest_path_agent(env, rng=None):
    """BFS (hop count) from current node to goal; returns the first hop."""
    adj, start, goal = env.adj, env.current_node, env.goal_node
    if start == goal:
        return goal
    prev, q = {start: None}, deque([start])
    while q:
        u = q.popleft()
        if u == goal:
            break
        for v in np.nonzero(adj[u] > 0)[0]:
            v = int(v)
            if v not in prev:
                prev[v] = u; q.append(v)
    if goal not in prev:
        return random_agent(env, rng or np.random.default_rng(0))
    node = goal
    while prev[node] != start:
        node = prev[node]
    return node


def severity_greedy_agent(env, rng=None):
    idx = np.where(env.valid_actions())[0]
    return int(idx[np.argmax(env.cvss[idx])])
