"""Driver-set enumeration, sampling, and greedy extension."""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from .control import assemble_gramian, gramian_metrics


def generate_driver_sets(
    n_nodes: int,
    k: int,
    max_sets: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    """Enumerate exactly when feasible, otherwise sample unique sets without replacement."""
    total = comb(n_nodes, k)
    if total <= max_sets:
        sets = np.asarray(list(combinations(range(n_nodes), k)), dtype=int)
        return sets, True

    selected: set[tuple[int, ...]] = set()
    while len(selected) < max_sets:
        proposal = tuple(sorted(rng.choice(n_nodes, size=k, replace=False).tolist()))
        selected.add(proposal)
    sets = np.asarray(sorted(selected), dtype=int)
    return sets, False


def summarize_landscape(
    energies_linear: np.ndarray,
    thresholds: list[float],
) -> dict[str, float]:
    """Summarize the distribution of control energy across candidate sets."""
    best_energy = float(np.min(energies_linear))
    log10_energy = np.log10(energies_linear)
    finite_log10_energy = log10_energy[np.isfinite(log10_energy)]
    if finite_log10_energy.size >= 2:
        variance_log10_energy = float(np.var(finite_log10_energy, ddof=1))
    else:
        variance_log10_energy = np.nan
    if finite_log10_energy.size >= 1:
        quantiles = {
            "q05_log10_energy": float(np.quantile(finite_log10_energy, 0.05)),
            "q25_log10_energy": float(np.quantile(finite_log10_energy, 0.25)),
            "q50_log10_energy": float(np.quantile(finite_log10_energy, 0.50)),
            "q75_log10_energy": float(np.quantile(finite_log10_energy, 0.75)),
            "q95_log10_energy": float(np.quantile(finite_log10_energy, 0.95)),
        }
    else:
        quantiles = {
            "q05_log10_energy": np.nan,
            "q25_log10_energy": np.nan,
            "q50_log10_energy": np.nan,
            "q75_log10_energy": np.nan,
            "q95_log10_energy": np.nan,
        }
    summary = {
        "variance_log10_energy": variance_log10_energy,
        "best_energy": best_energy,
        **quantiles,
    }
    for threshold in thresholds:
        summary[f"near_optimal_fraction_{int(round(threshold * 100)):02d}pct"] = float(
            np.mean(energies_linear <= (1.0 + threshold) * best_energy)
        )
    return summary


def greedy_extension(
    scores: np.ndarray,
    node_gramians: np.ndarray,
    k: int,
    epsilon: float,
    candidate_pool: int,
    bottom_dim: int,
) -> tuple[np.ndarray, list[dict], dict]:
    """Seeded-greedy conditional extension from a ranking score."""
    order = np.argsort(-scores, kind="mergesort")
    selected = [int(order[0])]
    step_rows: list[dict] = []

    static_topk = order[:k]
    static_gramian = assemble_gramian(node_gramians, static_topk)
    static_metrics = gramian_metrics(static_gramian, epsilon=epsilon, bottom_dim=bottom_dim)

    current_energy = gramian_metrics(
        assemble_gramian(node_gramians, np.asarray(selected, dtype=int)),
        epsilon=epsilon,
        bottom_dim=bottom_dim,
    )["average_energy"]
    for step in range(2, k + 1):
        remaining = [node for node in order if node not in selected]
        candidates = remaining[:candidate_pool]
        candidate_rows = []
        for candidate in candidates:
            proposal = np.asarray(selected + [int(candidate)], dtype=int)
            metrics = gramian_metrics(assemble_gramian(node_gramians, proposal), epsilon=epsilon, bottom_dim=bottom_dim)
            candidate_rows.append((int(candidate), metrics["average_energy"], metrics["log10_average_energy"]))
        chosen_node, chosen_energy, chosen_log_energy = min(candidate_rows, key=lambda row: row[1])
        selected.append(chosen_node)
        step_rows.append(
            {
                "step": step,
                "added_node": chosen_node,
                "greedy_average_energy": chosen_energy,
                "greedy_log10_average_energy": chosen_log_energy,
                "marginal_gain_log10": float(np.log10(current_energy) - chosen_log_energy),
            }
        )
        current_energy = chosen_energy

    greedy_set = np.asarray(selected, dtype=int)
    greedy_metrics = gramian_metrics(
        assemble_gramian(node_gramians, greedy_set),
        epsilon=epsilon,
        bottom_dim=bottom_dim,
    )
    comparison = {
        "static_log10_average_energy": static_metrics["log10_average_energy"],
        "greedy_log10_average_energy": greedy_metrics["log10_average_energy"],
        "greedy_minus_static_log10": greedy_metrics["log10_average_energy"] - static_metrics["log10_average_energy"],
    }
    return greedy_set, step_rows, comparison
