"""Driver-set enumeration, sampling and greedy extension."""

from __future__ import annotations

from math import comb

import numpy as np

from hoc.landscape import generate_driver_sets, greedy_extension, summarize_landscape


def test_exhaustive_enumeration() -> None:
    sets, exact = generate_driver_sets(7, 3, max_sets=1000, rng=np.random.default_rng(0))
    assert exact
    assert len(sets) == comb(7, 3)
    assert len({tuple(row) for row in sets}) == comb(7, 3)


def test_sampling_is_unique_sorted_and_seeded() -> None:
    a, exact = generate_driver_sets(30, 4, max_sets=200, rng=np.random.default_rng(1))
    b, _ = generate_driver_sets(30, 4, max_sets=200, rng=np.random.default_rng(1))
    assert not exact
    assert len(a) == 200 and len({tuple(row) for row in a}) == 200
    assert np.all(np.diff(a, axis=1) > 0)
    np.testing.assert_array_equal(a, b)


def test_summary_near_optimal_fraction() -> None:
    energies = np.array([1.0, 1.005, 1.08, 2.0])
    summary = summarize_landscape(energies, thresholds=[0.01, 0.10])
    assert summary["best_energy"] == 1.0
    assert summary["near_optimal_fraction_01pct"] == 0.5
    assert summary["near_optimal_fraction_10pct"] == 0.75


def test_greedy_on_diagonal_gramians_picks_strongest_nodes() -> None:
    # Diagonal system: node j's Gramian is e_j e_j^T / (2 a_j), so energy only
    # depends on which nodes are chosen and the best set holds the slowest nodes.
    rates = np.array([0.2, 3.0, 0.5, 2.5, 0.3, 4.0])
    gramians = np.zeros((6, 6, 6))
    for j, a in enumerate(rates):
        gramians[j, j, j] = 1.0 / (2.0 * a)
    scores = np.array([6.0, 5.0, 1.0, 4.0, 2.0, 3.0])  # top score is the slowest node
    greedy, steps, comparison = greedy_extension(scores, gramians, k=3, epsilon=1e-6, candidate_pool=6, bottom_dim=2)
    assert sorted(greedy.tolist()) == [0, 2, 4]
    assert [row["step"] for row in steps] == [2, 3]
    assert comparison["greedy_minus_static_log10"] < 0
