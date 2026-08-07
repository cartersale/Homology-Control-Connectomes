"""Persistent-homology and simplicial topology scores."""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
from ripser import ripser


def adjacency_to_distance(adjacency: np.ndarray) -> np.ndarray:
    """Map normalized weights in [0, 1] to distances in [0, 1]."""
    adjacency = np.asarray(adjacency, dtype=float)
    distances = np.where(adjacency > 0, 1.0 - adjacency, 1.0)
    np.fill_diagonal(distances, 0.0)
    return distances


def cycle_participation(adjacency: np.ndarray) -> tuple[np.ndarray, dict]:
    """Compute node-wise H1 cycle participation scores from persistent homology."""
    distances = adjacency_to_distance(adjacency)
    result = ripser(distances, distance_matrix=True, maxdim=1, do_cocycles=True)
    diagram = result["dgms"][1]
    cocycles = result["cocycles"][1]

    scores = np.zeros(adjacency.shape[0], dtype=float)
    persistences = []

    for feature_index, birth_death in enumerate(diagram):
        birth, death = float(birth_death[0]), float(birth_death[1])
        if not np.isfinite(death) or death <= birth:
            continue
        persistence = death - birth
        persistences.append(persistence)
        cocycle = cocycles[feature_index] if feature_index < len(cocycles) else np.empty((0, 3))
        support_nodes = np.unique(cocycle[:, :2].astype(int).ravel()) if cocycle.size else np.array([], dtype=int)
        if support_nodes.size == 0:
            continue
        scores[support_nodes] += persistence

    normalized = min_max_normalize(scores)
    return normalized, {"diagram": diagram, "persistences": np.asarray(persistences, dtype=float)}


def total_h1_persistence(adjacency: np.ndarray) -> float:
    """Return the total finite H1 persistence for an adjacency matrix."""
    distances = adjacency_to_distance(adjacency)
    result = ripser(distances, distance_matrix=True, maxdim=1, do_cocycles=False)
    diagram = result["dgms"][1]
    persistences = diagram[:, 1] - diagram[:, 0]
    finite = persistences[np.isfinite(persistences)]
    return float(np.sum(finite[finite > 0]))


def degree_normalized_cycle_participation(cycle_scores: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """Normalize cycle participation by weighted degree strength."""
    return min_max_normalize(cycle_scores / np.maximum(strength, 1.0e-12))


def cycle_criticality(adjacency: np.ndarray) -> np.ndarray:
    """Measure how much total H1 persistence drops when a node is removed."""
    baseline = total_h1_persistence(adjacency)
    scores = np.zeros(adjacency.shape[0], dtype=float)
    for node_index in range(adjacency.shape[0]):
        keep = np.ones(adjacency.shape[0], dtype=bool)
        keep[node_index] = False
        scores[node_index] = baseline - total_h1_persistence(adjacency[np.ix_(keep, keep)])
    return min_max_normalize(scores)


def simplicial_participation(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count incident triangles and tetrahedra in the binary support graph."""
    graph = nx.from_numpy_array((adjacency > 0).astype(int))
    triangle_counts = np.zeros(adjacency.shape[0], dtype=float)
    tetra_counts = np.zeros(adjacency.shape[0], dtype=float)
    for clique in nx.enumerate_all_cliques(graph):
        size = len(clique)
        if size < 3:
            continue
        if size == 3:
            triangle_counts[clique] += 1.0
        elif size == 4:
            tetra_counts[clique] += 1.0
        elif size > 4:
            break
    return min_max_normalize(triangle_counts), min_max_normalize(tetra_counts)


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] inside one connectome."""
    values = np.asarray(values, dtype=float)
    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmax, vmin):
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)
