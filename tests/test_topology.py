"""Persistent-homology and clique scores on graphs with known topology."""

from __future__ import annotations

import numpy as np
import pytest

from hoc.topology import (
    adjacency_to_distance,
    cycle_criticality,
    cycle_participation,
    min_max_normalize,
    simplicial_participation,
    total_h1_persistence,
)


def ring(n: int, weight: float = 0.5) -> np.ndarray:
    adjacency = np.zeros((n, n))
    for i in range(n):
        adjacency[i, (i + 1) % n] = adjacency[(i + 1) % n, i] = weight
    return adjacency


def test_distance_convention() -> None:
    adjacency = np.array([[0.0, 0.8, 0.0], [0.8, 0.0, 1.0], [0.0, 1.0, 0.0]])
    distances = adjacency_to_distance(adjacency)
    expected = np.array([[0.0, 0.2, 1.0], [0.2, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(distances, expected, atol=1e-15)


def test_ring_has_one_h1_bar() -> None:
    scores, meta = cycle_participation(ring(6, 0.5))
    # Ring edges enter at distance 0.5; everything else at 1.0, which fills the cycle.
    np.testing.assert_allclose(meta["persistences"], [0.5], atol=1e-12)
    assert total_h1_persistence(ring(6, 0.5)) == pytest.approx(0.5)
    assert scores.min() >= 0.0 and scores.max() == pytest.approx(1.0)


def test_tree_has_no_cycles() -> None:
    star = np.zeros((6, 6))
    star[0, 1:] = star[1:, 0] = 0.9
    scores, meta = cycle_participation(star)
    assert meta["persistences"].size == 0
    np.testing.assert_array_equal(scores, np.zeros(6))


def test_cycle_criticality_ring_is_uniform() -> None:
    # Removing any node of a ring destroys its only cycle: equal drops -> all zeros.
    np.testing.assert_array_equal(cycle_criticality(ring(7)), np.zeros(7))


def test_simplicial_counts() -> None:
    adjacency = np.ones((5, 5)) - np.eye(5)
    adjacency[4, 1:] = adjacency[1:, 4] = 0.0  # K4 on 0-3, pendant node 4 on node 0
    triangles, tetrahedra = simplicial_participation(adjacency)
    np.testing.assert_allclose(triangles, [1, 1, 1, 1, 0])
    np.testing.assert_allclose(tetrahedra, [1, 1, 1, 1, 0])


def test_min_max_normalize_edge_cases() -> None:
    np.testing.assert_array_equal(min_max_normalize(np.array([3.0, 3.0])), [0.0, 0.0])
    np.testing.assert_allclose(min_max_normalize(np.array([2.0, 4.0, 3.0])), [0.0, 1.0, 0.5])
