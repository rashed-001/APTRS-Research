"""Recommendation layer (v2). Exact ILP now optimises the SAME utility that is reported
(path-blocking variables), uses scipy.milp (no pulp), and greedy uses marginal gains."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def edges_of(p):
    return list(zip(p[:-1], p[1:]))


def count_paths(env) -> int:
    n = env.num_nodes; c = np.zeros(n); c[env.goal_node] = 1
    for u in range(n - 2, -1, -1):
        c[u] = sum(c[v] for v in np.nonzero(env.adj[u, :n] > 0)[0]) if u != env.goal_node else 1
    return int(c[env.start_node])


def enumerate_paths(env):
    paths, goal = [], env.goal_node
    def dfs(u, path):
        if u == goal:
            paths.append(list(path)); return
        for v in np.nonzero(env.adj[u] > 0)[0]:
            path.append(int(v)); dfs(int(v), path); path.pop()
    dfs(env.start_node, [env.start_node])
    return paths


def path_weights(paths, env):
    """Risk proxy: sum of CVSS along the path (NOT learned; see manuscript note)."""
    return [float(np.sum(env.cvss[p])) for p in paths]


def _controls(paths, env):
    ctrl = {}
    for p in paths:
        for e in edges_of(p):
            ctrl.setdefault(e, {"cost": float(env.cvss[e[1]]) + 1.0,
                                "disr": float(env.node_types[e[1]]) / 2.0})
    return ctrl


def evaluate_control_set(selected, paths, weights, budget, env):
    sel = set(selected); W = max(sum(weights), 1e-9)
    blocked = sum(w for p, w in zip(paths, weights) if sel & set(edges_of(p))) / W
    cost = sum(env.cvss[e[1]] + 1.0 for e in sel)
    disr = sum(env.node_types[e[1]] / 2.0 for e in sel)
    c = env.cfg
    return {"blocked_fraction": float(blocked), "cost": float(cost), "disruption": float(disr),
            "within_budget": bool(cost <= budget + 1e-9), "n_controls": len(sel),
            "utility": float(c.alpha * blocked - c.beta * cost / budget - c.gamma_d * disr)}


def payoff_guided_selection(paths, weights, budget, env):
    """Greedy on MARGINAL utility gain (already-blocked paths are not double counted)."""
    ctrl, c = _controls(paths, env), env.cfg
    W = max(sum(weights), 1e-9)
    cover = {e: [i for i, p in enumerate(paths) if e in set(edges_of(p))] for e in ctrl}
    blocked, sel, spent = set(), [], 0.0
    while True:
        best, best_g = None, 1e-12
        for e, d in ctrl.items():
            if e in sel or spent + d["cost"] > budget:
                continue
            g = (c.alpha * sum(weights[i] for i in cover[e] if i not in blocked) / W
                 - c.beta * d["cost"] / budget - c.gamma_d * d["disr"])
            if g > best_g:
                best, best_g = e, g
        if best is None:
            return sel
        sel.append(best); spent += ctrl[best]["cost"]; blocked.update(cover[best])


def severity_greedy_selection(paths, weights, budget, env):
    ctrl = _controls(paths, env)
    sel, spent = [], 0.0
    for e in sorted(ctrl, key=lambda e: -env.cvss[e[1]]):
        if spent + ctrl[e]["cost"] <= budget:
            sel.append(e); spent += ctrl[e]["cost"]
    return sel


def _solve(c, A, lb, ub, nvar):
    r = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=np.ones(nvar),
             bounds=Bounds(0, 1))
    if r.x is None:
        raise RuntimeError(r.message)
    return r.x


def ilp_optimum(paths, weights, budget, env):
    """Exact maximiser of the reported utility: x_e = control chosen, y_p = path blocked."""
    ctrl = _controls(paths, env); E = list(ctrl); P, W, c = len(paths), max(sum(weights), 1e-9), env.cfg
    nv = len(E) + P
    obj = np.zeros(nv)
    for j, e in enumerate(E):
        obj[j] = c.beta * ctrl[e]["cost"] / budget + c.gamma_d * ctrl[e]["disr"]
    for i in range(P):
        obj[len(E) + i] = -c.alpha * weights[i] / W
    idx = {e: j for j, e in enumerate(E)}
    A = np.zeros((P + 1, nv)); ub = np.zeros(P + 1)
    for i, p in enumerate(paths):
        A[i, len(E) + i] = 1
        for e in set(edges_of(p)):
            A[i, idx[e]] = -1
    for e, j in idx.items():
        A[P, j] = ctrl[e]["cost"]
    ub[P] = budget
    x = _solve(obj, A, -np.inf, ub, nv)
    return [e for e, j in idx.items() if x[j] > 0.5]


def min_cost_hardening(paths, weights, budget, env):
    """Noel et al. (2003): cheapest edge set blocking ALL paths. Budget is ignored by
    design; evaluate_control_set reports within_budget so infeasibility is visible."""
    ctrl = _controls(paths, env); E = list(ctrl); idx = {e: j for j, e in enumerate(E)}
    A = np.zeros((len(paths), len(E)))
    for i, p in enumerate(paths):
        for e in set(edges_of(p)):
            A[i, idx[e]] = 1
    x = _solve(np.array([ctrl[e]["cost"] for e in E]), A, 1, np.inf, len(E))
    return [e for e, j in idx.items() if x[j] > 0.5]
