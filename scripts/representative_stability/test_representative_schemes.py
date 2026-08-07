"""Unit tests for the alternative H1 representative schemes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from representative_schemes import (
    adjacency_to_distance,
    compare_libraries,
    compute_persistence,
    cycle_participation_by_scheme,
    geodesic_representatives,
    gudhi_h1_generators,
    gudhi_representatives,
    min_max_normalize,
    ripser_representatives,
)


def ring_adjacency(n_nodes: int, weight: float = 0.5) -> np.ndarray:
    """A weighted cycle graph: exactly one long-lived H1 feature."""
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    for node in range(n_nodes):
        adjacency[node, (node + 1) % n_nodes] = weight
        adjacency[(node + 1) % n_nodes, node] = weight
    return adjacency


def two_ring_adjacency() -> np.ndarray:
    """Two disjoint rings joined by one weak bridge edge."""
    n_ring = 8
    adjacency = np.zeros((2 * n_ring, 2 * n_ring), dtype=float)
    for offset in (0, n_ring):
        block = ring_adjacency(n_ring, weight=0.6)
        adjacency[offset : offset + n_ring, offset : offset + n_ring] = block
    adjacency[0, n_ring] = 0.2
    adjacency[n_ring, 0] = 0.2
    return adjacency


def test_schemes_share_the_same_barcode():
    """Both schemes describe the same features: identical births, deaths, persistences."""
    distances = adjacency_to_distance(two_ring_adjacency())
    persistence = compute_persistence(distances)
    baseline = ripser_representatives(distances, persistence=persistence)
    geodesic = geodesic_representatives(distances, persistence=persistence)

    assert baseline.n_features == geodesic.n_features
    for left, right in zip(baseline.representatives, geodesic.representatives):
        assert left.feature_index == right.feature_index
        assert left.birth == pytest.approx(right.birth)
        assert left.death == pytest.approx(right.death)
        assert left.persistence == pytest.approx(right.persistence)


def test_geodesic_representative_recovers_the_ring():
    """On a clean ring the length-minimized representative is the ring itself."""
    n_nodes = 10
    distances = adjacency_to_distance(ring_adjacency(n_nodes))
    geodesic = geodesic_representatives(distances)

    assert geodesic.n_features == 1
    representative = geodesic.representatives[0]
    assert representative.nodes.size == n_nodes
    assert representative.n_edges == n_nodes
    assert not representative.fell_back_to_ripser
    assert representative.birth_edge_exact


def test_geodesic_support_is_a_closed_cycle_in_the_filtration():
    """Every non-fallback representative is a cycle whose edges exist below death."""
    adjacency = two_ring_adjacency()
    distances = adjacency_to_distance(adjacency)
    geodesic = geodesic_representatives(distances)

    for representative in geodesic.representatives:
        if representative.fell_back_to_ripser:
            continue
        assert representative.nodes.size >= 3
        assert representative.n_edges == representative.nodes.size
        source, target = representative.birth_edge
        assert distances[source, target] <= representative.death


def test_cocycle_support_and_geodesic_cycle_differ_on_a_ring():
    """The two schemes are genuinely different objects, not cosmetic variants.

    Ripser returns *cocycles*: their support is a sparse dual object (on a clean
    ring, a single edge), not the cycle itself. The geodesic scheme returns the
    actual cycle. This is the divergence the stability test is designed to probe.
    """
    n_nodes = 12
    distances = adjacency_to_distance(ring_adjacency(n_nodes))
    persistence = compute_persistence(distances)
    baseline = ripser_representatives(distances, persistence=persistence)
    geodesic = geodesic_representatives(distances, persistence=persistence)

    assert baseline.representatives[0].nodes.size < n_nodes
    assert geodesic.representatives[0].nodes.size == n_nodes


def test_hops_weighting_minimizes_edge_count():
    """The `hops` mode never returns more edges than the filtration-weighted mode."""
    rng = np.random.default_rng(3)
    adjacency = rng.random((25, 25))
    adjacency = 0.5 * (adjacency + adjacency.T)
    adjacency[adjacency < 0.55] = 0.0
    np.fill_diagonal(adjacency, 0.0)
    distances = adjacency_to_distance(adjacency)
    persistence = compute_persistence(distances)

    hops = geodesic_representatives(distances, persistence=persistence, weight_mode="hops")
    filtration = geodesic_representatives(distances, persistence=persistence, weight_mode="filtration")
    for left, right in zip(hops.representatives, filtration.representatives):
        if left.fell_back_to_ripser or right.fell_back_to_ripser:
            continue
        assert left.n_edges <= right.n_edges


def dense_random_adjacency(seed: int = 11, n_nodes: int = 40) -> np.ndarray:
    """A dense weighted graph with the many short-lived features real data shows."""
    rng = np.random.default_rng(seed)
    adjacency = rng.random((n_nodes, n_nodes))
    adjacency = 0.5 * (adjacency + adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def test_birth_thresholding_never_returns_a_bounding_triangle():
    """Searching at death fills cycles; searching at birth cannot.

    Vietoris-Rips is a flag complex, so a triangle whose diameter is below the search
    threshold is already filled and carries no homology. That is the failure mode the
    `birth` default exists to avoid.
    """
    distances = adjacency_to_distance(dense_random_adjacency())
    persistence = compute_persistence(distances)

    at_birth = geodesic_representatives(distances, persistence=persistence, threshold_mode="birth")
    at_death = geodesic_representatives(distances, persistence=persistence, threshold_mode="death")

    assert at_birth.diagnostics()["n_bounding_representatives"] == 0
    assert at_death.diagnostics()["n_bounding_representatives"] > 0


def test_birth_edges_are_identified_exactly_despite_single_precision():
    """Ripser reports births in float32; matching must absorb that rounding."""
    distances = adjacency_to_distance(dense_random_adjacency())
    geodesic = geodesic_representatives(distances)
    diagnostics = geodesic.diagnostics()

    assert diagnostics["n_features"] > 0
    assert diagnostics["n_birth_edge_inexact"] == 0
    assert diagnostics["n_fallback_to_ripser"] == 0


def test_participation_scores_are_persistence_weighted():
    """A node in no representative scores zero; a ring node carries the persistence."""
    n_nodes = 9
    distances = adjacency_to_distance(ring_adjacency(n_nodes))
    geodesic = geodesic_representatives(distances)
    raw = geodesic.participation_scores()

    persistence = geodesic.representatives[0].persistence
    assert np.allclose(raw, persistence)
    assert np.allclose(min_max_normalize(raw), 0.0)


def test_cycle_participation_by_scheme_returns_normalized_scores():
    """The public entry point returns one [0, 1] score vector per scheme."""
    adjacency = two_ring_adjacency()
    schemes = ("ripser", "geodesic", "gudhi", "gudhi_geodesic")
    scores, representative_sets, comparison = cycle_participation_by_scheme(adjacency, schemes=schemes)

    assert set(scores) == set(schemes)
    for scheme, vector in scores.items():
        assert vector.shape == (adjacency.shape[0],)
        assert vector.min() >= 0.0 and vector.max() <= 1.0
        assert representative_sets[scheme].diagnostics()["n_features"] >= 1
    assert comparison["n_matched"] >= 1


def test_gudhi_reproduces_the_same_barcode():
    """Scheme 3 is only a representative change if the features themselves match."""
    distances = adjacency_to_distance(dense_random_adjacency(seed=5))
    comparison = compare_libraries(distances)

    assert comparison["n_features_ripser"] == comparison["n_features_gudhi"]
    assert comparison["n_matched"] == comparison["n_features_ripser"]
    # Ripser reduces in single precision; that is the only permitted disagreement.
    assert comparison["max_barcode_offset"] < 1.0e-5


def test_gudhi_generators_index_their_own_diagram():
    """GUDHI's birth/death edges must reproduce its own birth/death values."""
    distances = adjacency_to_distance(dense_random_adjacency(seed=7))
    generators = gudhi_h1_generators(distances)
    diagram, births, deaths = generators["diagram"], generators["birth_edges"], generators["death_edges"]

    assert births.shape[0] == diagram.shape[0]
    assert np.allclose(distances[births[:, 0], births[:, 1]], diagram[:, 0])
    assert np.allclose(distances[deaths[:, 0], deaths[:, 1]], diagram[:, 1])


def test_gudhi_geodesic_matches_geodesic_when_birth_edges_agree():
    """Same birth edge plus same construction must give the same representative."""
    distances = adjacency_to_distance(dense_random_adjacency(seed=13))
    comparison = compare_libraries(distances)
    if comparison["fraction_same_birth_edge"] < 1.0:
        pytest.skip("libraries disagreed on at least one birth edge for this graph")

    from_ripser = geodesic_representatives(distances).participation_scores()
    from_gudhi = gudhi_representatives(distances, mode="geodesic").participation_scores()
    assert np.allclose(from_ripser, from_gudhi)


def test_zero_distance_edges_are_kept_in_the_graph():
    """Max-weight connectome edges map to distance 0 and must not be dropped."""
    adjacency = ring_adjacency(8, weight=1.0)
    distances = adjacency_to_distance(adjacency)
    assert np.count_nonzero(distances == 0.0) > adjacency.shape[0]

    geodesic = geodesic_representatives(distances)
    assert geodesic.n_features >= 1
    assert not any(r.fell_back_to_ripser for r in geodesic.representatives)
