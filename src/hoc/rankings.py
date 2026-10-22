"""Node-ranking heuristics for driver selection."""

from __future__ import annotations

import bct
import networkx as nx
import numpy as np
import pandas as pd

from .topology import (
    cycle_criticality,
    cycle_participation,
    degree_normalized_cycle_participation,
    simplicial_participation,
)


def stable_descending_ranks(scores: np.ndarray) -> np.ndarray:
    """Return 1-based ranks with stable tie handling."""
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def strength_scores(adjacency: np.ndarray) -> np.ndarray:
    """Weighted node strength."""
    return np.sum(adjacency, axis=1)


def betweenness_scores(adjacency: np.ndarray) -> np.ndarray:
    """Weighted betweenness with inverse-weight path lengths."""
    graph = nx.from_numpy_array(adjacency)
    for source, target, data in graph.edges(data=True):
        weight = float(data["weight"])
        data["length"] = 1.0 / weight if weight > 0 else np.inf
    scores = nx.betweenness_centrality(graph, normalized=True, weight="length")
    return np.asarray([scores[node] for node in range(adjacency.shape[0])], dtype=float)


def eigenvector_centrality_scores(adjacency: np.ndarray) -> np.ndarray:
    """Undirected weighted eigenvector centrality."""
    return np.asarray(bct.eigenvector_centrality_und(adjacency), dtype=float)


# Louvain is stochastic; a fixed seed makes the participation coefficient
# reproducible. The preprint results used an unseeded call (baseline/PROVENANCE.md).
LOUVAIN_SEED = 0


def participation_coefficient_scores(adjacency: np.ndarray, seed: int = LOUVAIN_SEED) -> np.ndarray:
    """Participation coefficient after (seeded) Louvain community detection."""
    communities, _ = bct.community_louvain(adjacency, seed=seed)
    return np.asarray(bct.participation_coef(adjacency, communities), dtype=float)


def compute_node_scores(
    adjacency: np.ndarray,
    methods: list[str],
) -> tuple[dict[str, np.ndarray], dict]:
    """Compute ranking scores requested for one subject-scale connectome."""
    outputs: dict[str, np.ndarray] = {}
    metadata: dict = {}

    strength = strength_scores(adjacency)
    cycle_scores, cycle_metadata = cycle_participation(adjacency)
    triangle_scores, tetra_scores = simplicial_participation(adjacency)

    metadata["cycle"] = cycle_metadata
    outputs["degree_strength"] = strength
    outputs["betweenness"] = betweenness_scores(adjacency)
    outputs["cycle_participation"] = cycle_scores

    if "eigenvector_centrality" in methods:
        outputs["eigenvector_centrality"] = eigenvector_centrality_scores(adjacency)
    if "participation_coefficient" in methods:
        outputs["participation_coefficient"] = participation_coefficient_scores(adjacency)
    if "degree_normalized_cycle_participation" in methods:
        outputs["degree_normalized_cycle_participation"] = degree_normalized_cycle_participation(cycle_scores, strength)
    if "cycle_criticality" in methods:
        outputs["cycle_criticality"] = cycle_criticality(adjacency)
    if "triangle_participation" in methods:
        outputs["triangle_participation"] = triangle_scores
    if "tetrahedron_participation" in methods:
        outputs["tetrahedron_participation"] = tetra_scores

    return outputs, metadata


def top_k_nodes(scores: np.ndarray, k: int) -> np.ndarray:
    """Select the top-k nodes using a stable descending order."""
    order = np.argsort(-scores, kind="mergesort")
    return np.asarray(order[:k], dtype=int)


def top_k_excluding_nodes(scores: np.ndarray, k: int, excluded: np.ndarray) -> np.ndarray:
    """Select top-k intact-ranking nodes after removing ineligible identities."""
    excluded = np.asarray(excluded, dtype=int)
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    eligible = order[~np.isin(order, excluded)]
    if eligible.shape[0] < k:
        raise ValueError("Fewer eligible nodes remain than the requested driver-set size.")
    return np.asarray(eligible[:k], dtype=int)


def random_top_k_nodes(n_nodes: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a random top-k set by shuffling node identities."""
    order = rng.permutation(n_nodes)
    return np.asarray(order[:k], dtype=int)


def scores_to_table(
    subject_id: str,
    scale: int,
    region_labels: pd.DataFrame,
    method: str,
    scores: np.ndarray,
) -> pd.DataFrame:
    """Convert node scores to a tidy table."""
    table = region_labels.copy()
    table["subject_id"] = subject_id
    table["method"] = method
    table["score"] = np.asarray(scores, dtype=float)
    table["rank"] = stable_descending_ranks(scores)
    return table[
        [
            "subject_id",
            "scale",
            "node_index",
            "region_label",
            "base_region",
            "hemi",
            "anatomical_group",
            "method",
            "score",
            "rank",
        ]
    ]
