"""Ranking helpers: ties, seeds and Louvain determinism."""

from __future__ import annotations

import numpy as np

from hoc.rankings import (
    participation_coefficient_scores,
    random_top_k_nodes,
    stable_descending_ranks,
    top_k_nodes,
)
from hoc.utils import make_rng, subject_seed


def two_block_graph(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 16
    adjacency = rng.random((n, n)) * 0.05
    adjacency[:8, :8] += 0.9
    adjacency[8:, 8:] += 0.9
    adjacency = 0.5 * (adjacency + adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def test_ties_are_broken_by_node_index() -> None:
    scores = np.array([1.0, 2.0, 2.0, 0.0])
    np.testing.assert_array_equal(stable_descending_ranks(scores), [3, 1, 2, 4])
    np.testing.assert_array_equal(top_k_nodes(scores, 2), [1, 2])


def test_subject_seed_golden_values() -> None:
    # Every random set in the pipeline derives from these; a change here changes results.
    assert subject_seed(20260423) == 20260423
    assert subject_seed(20260423, 68, 0) == 1987241087427270781
    assert subject_seed(20260423, 219, 69, 5) == 4234411666706933530


def test_random_sets_are_reproducible() -> None:
    a = random_top_k_nodes(68, 5, make_rng(subject_seed(1, 68, 3)))
    b = random_top_k_nodes(68, 5, make_rng(subject_seed(1, 68, 3)))
    np.testing.assert_array_equal(a, b)


def test_seeded_louvain_is_deterministic() -> None:
    adjacency = two_block_graph()
    first = participation_coefficient_scores(adjacency)
    for _ in range(5):
        np.testing.assert_array_equal(participation_coefficient_scores(adjacency), first)
    assert np.all((first >= 0) & (first <= 1))
