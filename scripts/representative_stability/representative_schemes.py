"""Alternative H1 representative schemes for the representative-stability test.

A persistent H1 feature is a homology class: its birth, death, and persistence are
canonical, but the specific cycle chosen to represent the class is not. The main
pipeline derives node-wise cycle participation (`CP_i`) from Ripser's cocycle
supports. This module re-derives the same scores from a *different* representative
of the *same* features, so the downstream ranking and control-geometry results can
be recomputed under an alternative -- but equally valid -- representative choice.

Schemes implemented here:

- ``ripser``: the baseline, support of Ripser's returned H1 cocycles.
- ``geodesic``: for each feature, the shortest cycle through the feature's birth
  edge in the graph thresholded at the feature's birth. These are length-minimized
  ("geodesic-ish") representatives, localized where Ripser's cohomology
  representatives are typically not.
- ``gudhi``: the same filtration recomputed in GUDHI, scored from GUDHI's own
  birth/death pairing simplices. An independent implementation's representative
  choice, which is exactly what the "implementation dependence" limitation names.
- ``gudhi_geodesic``: the length-minimized cycle built on GUDHI's birth edge, which
  separates disagreement about the *birth simplex* from disagreement about the
  representative built on it.

This file deliberately does not import from ``hoc.topology`` beyond the shared
distance transform and normalizer, so the stability test stays decoupled from the
main analysis code. GUDHI is imported lazily: the Ripser-side schemes run without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
from ripser import ripser
from scipy.sparse.csgraph import csgraph_from_dense, dijkstra

SchemeName = Literal["ripser", "geodesic", "gudhi", "gudhi_geodesic"]
WeightMode = Literal["filtration", "hops"]
ThresholdMode = Literal["below_death", "death", "birth"]

# Ripser computes in single precision, so its birth/death values differ from the
# float64 distance matrix by roughly float32 epsilon. Comparisons of filtration
# values must absorb that, hence a tolerance far looser than float64 epsilon.
FILTRATION_TOLERANCE = 1.0e-6


def adjacency_to_distance(adjacency: np.ndarray) -> np.ndarray:
    """Map normalized weights in [0, 1] to distances in [0, 1].

    Mirrors ``hoc.topology.adjacency_to_distance`` so both schemes see exactly the
    same filtration.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    distances = np.where(adjacency > 0, 1.0 - adjacency, 1.0)
    np.fill_diagonal(distances, 0.0)
    return distances


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] inside one connectome."""
    values = np.asarray(values, dtype=float)
    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmax, vmin):
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


@dataclass
class Representative:
    """One representative cycle for one persistent H1 feature."""

    feature_index: int
    birth: float
    death: float
    persistence: float
    nodes: np.ndarray
    n_edges: int
    birth_edge: tuple[int, int] | None = None
    birth_edge_exact: bool = True
    fell_back_to_ripser: bool = False
    is_triangle: bool = False
    bounds_in_flag_complex: bool = False
    notes: str = ""


@dataclass
class RepresentativeSet:
    """All representatives for one connectome under one scheme."""

    scheme: str
    n_nodes: int
    representatives: list[Representative] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return len(self.representatives)

    def participation_scores(self) -> np.ndarray:
        """Sum persistence over features whose representative support contains node i."""
        raw = np.zeros(self.n_nodes, dtype=float)
        for representative in self.representatives:
            if representative.nodes.size:
                raw[representative.nodes] += representative.persistence
        return raw

    def diagnostics(self) -> dict:
        """Summarize representative geometry and fallback usage."""
        if not self.representatives:
            return {
                "scheme": self.scheme,
                "n_features": 0,
                "mean_support_size": np.nan,
                "median_support_size": np.nan,
                "mean_cycle_edges": np.nan,
                "n_birth_edge_inexact": 0,
                "n_fallback_to_ripser": 0,
                "n_triangle_cycles": 0,
                "n_bounding_representatives": 0,
                "total_persistence": 0.0,
            }
        support_sizes = np.array([r.nodes.size for r in self.representatives], dtype=float)
        cycle_edges = np.array([r.n_edges for r in self.representatives], dtype=float)
        return {
            "scheme": self.scheme,
            "n_features": len(self.representatives),
            "mean_support_size": float(np.mean(support_sizes)),
            "median_support_size": float(np.median(support_sizes)),
            "mean_cycle_edges": float(np.mean(cycle_edges)),
            "n_birth_edge_inexact": int(sum(not r.birth_edge_exact for r in self.representatives)),
            "n_fallback_to_ripser": int(sum(r.fell_back_to_ripser for r in self.representatives)),
            "n_triangle_cycles": int(sum(r.is_triangle for r in self.representatives)),
            "n_bounding_representatives": int(sum(r.bounds_in_flag_complex for r in self.representatives)),
            "total_persistence": float(sum(r.persistence for r in self.representatives)),
        }


def compute_persistence(distances: np.ndarray) -> dict:
    """Run the shared H1 computation once; both schemes reuse this output."""
    result = ripser(distances, distance_matrix=True, maxdim=1, do_cocycles=True)
    return {"diagram": np.asarray(result["dgms"][1], dtype=float), "cocycles": result["cocycles"][1]}


def _finite_features(diagram: np.ndarray) -> Iterable[tuple[int, float, float]]:
    """Yield (index, birth, death) for features with finite positive persistence."""
    for feature_index, birth_death in enumerate(diagram):
        birth, death = float(birth_death[0]), float(birth_death[1])
        if not np.isfinite(death) or death <= birth:
            continue
        yield feature_index, birth, death


def _cocycle_edges(cocycles: list, feature_index: int) -> np.ndarray:
    """Return the (n_edges, 2) integer edge support of one cocycle."""
    if feature_index >= len(cocycles):
        return np.empty((0, 2), dtype=int)
    cocycle = np.asarray(cocycles[feature_index])
    if cocycle.size == 0:
        return np.empty((0, 2), dtype=int)
    return cocycle[:, :2].astype(int)


def ripser_representatives(distances: np.ndarray, persistence: dict | None = None) -> RepresentativeSet:
    """Scheme 1: representatives are the supports of Ripser's H1 cocycles."""
    distances = np.asarray(distances, dtype=float)
    persistence = persistence if persistence is not None else compute_persistence(distances)
    diagram, cocycles = persistence["diagram"], persistence["cocycles"]

    representative_set = RepresentativeSet(scheme="ripser", n_nodes=distances.shape[0])
    for feature_index, birth, death in _finite_features(diagram):
        edges = _cocycle_edges(cocycles, feature_index)
        nodes = np.unique(edges.ravel()) if edges.size else np.array([], dtype=int)
        representative_set.representatives.append(
            Representative(
                feature_index=feature_index,
                birth=birth,
                death=death,
                persistence=death - birth,
                nodes=nodes,
                n_edges=int(edges.shape[0]),
                birth_edge=None,
                is_triangle=bool(nodes.size == 3),
            )
        )
    return representative_set


def _select_birth_edge(
    edges: np.ndarray,
    distances: np.ndarray,
    birth: float,
    tolerance: float,
) -> tuple[tuple[int, int] | None, bool]:
    """Pick the cocycle edge whose filtration value matches the feature's birth.

    Ripser's birth value is the diameter of the simplex that created the class, so
    the birth edge is the support edge with ``D_uv == birth``. When no support edge
    matches (which can happen for cocycle representatives that have been reduced
    away from the birth simplex), the closest one is used and flagged as inexact.
    """
    if edges.size == 0:
        return None, False
    edge_lengths = distances[edges[:, 0], edges[:, 1]]
    offsets = np.abs(edge_lengths - birth)
    best = int(np.argmin(offsets))
    edge = (int(edges[best, 0]), int(edges[best, 1]))
    return edge, bool(offsets[best] <= tolerance)


def _threshold_graph(
    distances: np.ndarray,
    threshold: float,
    weight_mode: WeightMode,
    strict: bool,
    excluded_edge: tuple[int, int] | None,
) -> np.ndarray:
    """Dense weight matrix for the graph at one filtration threshold.

    Non-edges are ``inf`` rather than ``0`` because a maximum-weight connectome edge
    maps to distance ``0``, which a sparsity-based encoding would silently drop.
    """
    slack = FILTRATION_TOLERANCE * max(1.0, abs(threshold))
    if strict:
        mask = distances < threshold - slack
    else:
        mask = distances <= threshold + slack
    np.fill_diagonal(mask, False)
    if excluded_edge is not None:
        source, target = excluded_edge
        mask[source, target] = False
        mask[target, source] = False

    weights = np.where(mask, distances if weight_mode == "filtration" else 1.0, np.inf)
    return weights


def _shortest_path(weights: np.ndarray, source: int, target: int) -> np.ndarray | None:
    """Dijkstra path from source to target, or None when disconnected."""
    graph = csgraph_from_dense(weights, null_value=np.inf)
    lengths, predecessors = dijkstra(graph, directed=False, indices=source, return_predecessors=True)
    if not np.isfinite(lengths[target]):
        return None
    path = [target]
    cursor = target
    while cursor != source:
        cursor = int(predecessors[cursor])
        if cursor < 0:
            return None
        path.append(cursor)
    return np.asarray(path[::-1], dtype=int)


def _bounds_in_flag_complex(distances: np.ndarray, nodes: np.ndarray, threshold: float) -> bool:
    """Detect the triangle case where the cycle is filled in the Rips complex.

    In a flag complex the 2-simplex on a triangle enters at the triangle's diameter,
    so a 3-node cycle whose diameter is at or below the threshold bounds and carries
    no homology. Longer cycles can bound too, but detecting that needs the full
    boundary matrix; this cheap check flags the dominant failure mode.
    """
    if nodes.size != 3:
        return False
    slack = FILTRATION_TOLERANCE * max(1.0, abs(threshold))
    submatrix = distances[np.ix_(nodes, nodes)]
    return bool(submatrix.max() <= threshold + slack)


def geodesic_representatives(
    distances: np.ndarray,
    persistence: dict | None = None,
    weight_mode: WeightMode = "filtration",
    threshold_mode: ThresholdMode = "birth",
) -> RepresentativeSet:
    """Scheme 2: length-minimized representatives.

    For each finite H1 feature, the birth edge ``(u, v)`` is identified from Ripser's
    cocycle support, that edge is deleted, and the shortest ``u -> v`` path is found
    in the thresholded graph. Path plus birth edge is the representative cycle: a
    length-minimized alternative to Ripser's representative for the same class.

    ``weight_mode`` selects the notion of length: ``filtration`` minimizes summed
    edge distance (geodesic), ``hops`` minimizes edge count (combinatorially
    shortest).

    ``threshold_mode`` selects the filtration stage the search runs in and is not a
    cosmetic choice. Vietoris-Rips is a *flag* complex, so any cycle whose edges are
    all present is filled as soon as its own diameter is reached: searching at (or
    just below) the death threshold can therefore return a shortest cycle that
    already bounds, i.e. is homologous to zero and not a representative of the class
    at all. At the birth threshold that failure is impossible -- the class is created
    by inserting ``(u, v)``, so every cycle through ``(u, v)`` in the graph at birth
    is nonzero in ``H_1`` there. ``birth`` is therefore the default. The ``death`` and
    ``below_death`` modes are kept for comparison and their bounding representatives
    are counted in the diagnostics.

    Features whose birth edge cannot be located, or whose endpoints are disconnected
    in the thresholded graph, fall back to Ripser's support and are flagged.
    """
    distances = np.asarray(distances, dtype=float)
    persistence = persistence if persistence is not None else compute_persistence(distances)
    diagram, cocycles = persistence["diagram"], persistence["cocycles"]

    representative_set = RepresentativeSet(scheme="geodesic", n_nodes=distances.shape[0])
    for feature_index, birth, death in _finite_features(diagram):
        edges = _cocycle_edges(cocycles, feature_index)
        ripser_nodes = np.unique(edges.ravel()) if edges.size else np.array([], dtype=int)
        tolerance = FILTRATION_TOLERANCE * max(1.0, abs(birth))
        birth_edge, birth_edge_exact = _select_birth_edge(edges, distances, birth, tolerance)

        def fallback(note: str) -> Representative:
            return Representative(
                feature_index=feature_index,
                birth=birth,
                death=death,
                persistence=death - birth,
                nodes=ripser_nodes,
                n_edges=int(edges.shape[0]),
                birth_edge=birth_edge,
                birth_edge_exact=birth_edge_exact,
                fell_back_to_ripser=True,
                is_triangle=bool(ripser_nodes.size == 3),
                notes=note,
            )

        if birth_edge is None:
            representative_set.representatives.append(fallback("no cocycle support"))
            continue

        cycle = _shortest_cycle_through(
            distances,
            birth_edge=birth_edge,
            birth=birth,
            death=death,
            weight_mode=weight_mode,
            threshold_mode=threshold_mode,
        )
        if cycle is None:
            representative_set.representatives.append(fallback("birth-edge endpoints disconnected"))
            continue

        nodes, bounds = cycle
        representative_set.representatives.append(
            Representative(
                feature_index=feature_index,
                birth=birth,
                death=death,
                persistence=death - birth,
                nodes=nodes,
                n_edges=int(nodes.size),
                birth_edge=birth_edge,
                birth_edge_exact=birth_edge_exact,
                is_triangle=bool(nodes.size == 3),
                bounds_in_flag_complex=bounds,
            )
        )
    return representative_set


def _shortest_cycle_through(
    distances: np.ndarray,
    birth_edge: tuple[int, int],
    birth: float,
    death: float,
    weight_mode: WeightMode,
    threshold_mode: ThresholdMode,
) -> tuple[np.ndarray, bool] | None:
    """Shortest cycle through one birth edge, or None if its endpoints disconnect."""
    if threshold_mode == "below_death":
        threshold, strict = death, True
    elif threshold_mode == "death":
        threshold, strict = death, False
    else:
        threshold, strict = birth, False

    weights = _threshold_graph(
        distances,
        threshold=threshold,
        weight_mode=weight_mode,
        strict=strict,
        excluded_edge=birth_edge,
    )
    path = _shortest_path(weights, source=birth_edge[0], target=birth_edge[1])
    if path is None:
        return None
    nodes = np.unique(path)
    return nodes, _bounds_in_flag_complex(distances, nodes, threshold)


def gudhi_h1_generators(distances: np.ndarray) -> dict:
    """Recompute the same Rips filtration in GUDHI and return its own H1 generators.

    GUDHI reduces the boundary matrix in homology, not cohomology, and pairs each H1
    class with the edge that creates it and the edge whose triangle destroys it. Those
    pairing simplices are an *independent* representative choice: nothing about them
    is inherited from Ripser or hand-crafted here.

    Returns the diagram plus the birth and death edges, one row per H1 feature.
    """
    import gudhi  # imported lazily so the other schemes work without GUDHI installed

    distances = np.asarray(distances, dtype=float)
    simplex_tree = gudhi.RipsComplex(distance_matrix=distances).create_simplex_tree(max_dimension=2)
    simplex_tree.compute_persistence()
    diagram = np.asarray(simplex_tree.persistence_intervals_in_dimension(1), dtype=float).reshape(-1, 2)
    generators = simplex_tree.flag_persistence_generators()[1]
    pairs = np.asarray(generators[0], dtype=int).reshape(-1, 4) if generators else np.empty((0, 4), dtype=int)
    if pairs.shape[0] != diagram.shape[0]:
        raise ValueError("GUDHI returned a generator count that does not match its H1 diagram.")
    return {"diagram": diagram, "birth_edges": pairs[:, :2], "death_edges": pairs[:, 2:]}


def gudhi_representatives(
    distances: np.ndarray,
    generators: dict | None = None,
    mode: Literal["pairing", "geodesic"] = "pairing",
    weight_mode: WeightMode = "filtration",
    threshold_mode: ThresholdMode = "birth",
) -> RepresentativeSet:
    """Scheme 3: representatives from a second library's own reduction.

    ``mode="pairing"`` scores the vertices of GUDHI's birth and death edges -- the
    support GUDHI itself hands back, with no post-processing. ``mode="geodesic"``
    instead builds the length-minimized cycle through GUDHI's birth edge, which
    isolates whether the two libraries disagree about the *birth simplex* from
    whether they disagree about the representative built on top of it.
    """
    distances = np.asarray(distances, dtype=float)
    generators = generators if generators is not None else gudhi_h1_generators(distances)
    diagram = generators["diagram"]
    birth_edges, death_edges = generators["birth_edges"], generators["death_edges"]

    scheme = "gudhi" if mode == "pairing" else "gudhi_geodesic"
    representative_set = RepresentativeSet(scheme=scheme, n_nodes=distances.shape[0])
    for feature_index, birth, death in _finite_features(diagram):
        birth_edge = (int(birth_edges[feature_index, 0]), int(birth_edges[feature_index, 1]))
        pairing_nodes = np.unique(np.concatenate([birth_edges[feature_index], death_edges[feature_index]]))

        if mode == "pairing":
            representative_set.representatives.append(
                Representative(
                    feature_index=feature_index,
                    birth=birth,
                    death=death,
                    persistence=death - birth,
                    nodes=pairing_nodes,
                    n_edges=2,
                    birth_edge=birth_edge,
                    is_triangle=bool(pairing_nodes.size == 3),
                )
            )
            continue

        cycle = _shortest_cycle_through(
            distances,
            birth_edge=birth_edge,
            birth=birth,
            death=death,
            weight_mode=weight_mode,
            threshold_mode=threshold_mode,
        )
        if cycle is None:
            representative_set.representatives.append(
                Representative(
                    feature_index=feature_index,
                    birth=birth,
                    death=death,
                    persistence=death - birth,
                    nodes=pairing_nodes,
                    n_edges=2,
                    birth_edge=birth_edge,
                    fell_back_to_ripser=True,
                    notes="birth-edge endpoints disconnected",
                )
            )
            continue

        nodes, bounds = cycle
        representative_set.representatives.append(
            Representative(
                feature_index=feature_index,
                birth=birth,
                death=death,
                persistence=death - birth,
                nodes=nodes,
                n_edges=int(nodes.size),
                birth_edge=birth_edge,
                is_triangle=bool(nodes.size == 3),
                bounds_in_flag_complex=bounds,
            )
        )
    return representative_set


def compare_libraries(distances: np.ndarray, persistence: dict | None = None, generators: dict | None = None) -> dict:
    """Do Ripser and GUDHI agree on the barcode, and on which edge births each class?

    Features are matched by their (birth, death) coordinates, which are canonical.
    Birth-edge agreement is then the sharpest single measure of the implementation
    dependence the limitation names: same class, same filtration, different library.
    """
    distances = np.asarray(distances, dtype=float)
    persistence = persistence if persistence is not None else compute_persistence(distances)
    generators = generators if generators is not None else gudhi_h1_generators(distances)

    ripser_features = list(_finite_features(persistence["diagram"]))
    gudhi_features = list(_finite_features(generators["diagram"]))

    ripser_order = sorted(range(len(ripser_features)), key=lambda i: ripser_features[i][1:])
    gudhi_order = sorted(range(len(gudhi_features)), key=lambda i: gudhi_features[i][1:])
    n_matched = min(len(ripser_order), len(gudhi_order))

    barcode_offsets = []
    n_same_birth_edge = 0
    n_shared_birth_vertex = 0
    for position in range(n_matched):
        _, ripser_birth, ripser_death = ripser_features[ripser_order[position]]
        gudhi_index, gudhi_birth, gudhi_death = gudhi_features[gudhi_order[position]]
        barcode_offsets.append(max(abs(ripser_birth - gudhi_birth), abs(ripser_death - gudhi_death)))

        cocycle_index = ripser_features[ripser_order[position]][0]
        edges = _cocycle_edges(persistence["cocycles"], cocycle_index)
        tolerance = FILTRATION_TOLERANCE * max(1.0, abs(ripser_birth))
        ripser_edge, _ = _select_birth_edge(edges, distances, ripser_birth, tolerance)
        gudhi_edge = (int(generators["birth_edges"][gudhi_index, 0]), int(generators["birth_edges"][gudhi_index, 1]))
        if ripser_edge is None:
            continue
        if set(ripser_edge) == set(gudhi_edge):
            n_same_birth_edge += 1
        if set(ripser_edge) & set(gudhi_edge):
            n_shared_birth_vertex += 1

    return {
        "n_features_ripser": len(ripser_features),
        "n_features_gudhi": len(gudhi_features),
        "n_matched": n_matched,
        "max_barcode_offset": float(np.max(barcode_offsets)) if barcode_offsets else 0.0,
        "n_same_birth_edge": n_same_birth_edge,
        "fraction_same_birth_edge": float(n_same_birth_edge / n_matched) if n_matched else np.nan,
        "fraction_shared_birth_vertex": float(n_shared_birth_vertex / n_matched) if n_matched else np.nan,
    }


def cycle_participation_by_scheme(
    adjacency: np.ndarray,
    schemes: Iterable[SchemeName] = ("ripser", "geodesic"),
    weight_mode: WeightMode = "filtration",
    threshold_mode: ThresholdMode = "birth",
) -> tuple[dict[str, np.ndarray], dict[str, RepresentativeSet], dict]:
    """Compute normalized `CP_i` for one connectome under several schemes.

    The Ripser-derived schemes share a single persistence computation, so their
    barcode is identical by construction and only the representative choice differs.
    The GUDHI schemes recompute the filtration independently; ``compare_libraries``
    checks that the two barcodes still agree, and the check is returned as the third
    element (empty when no GUDHI scheme was requested).
    """
    distances = adjacency_to_distance(adjacency)
    schemes = tuple(schemes)
    persistence = compute_persistence(distances)
    generators = gudhi_h1_generators(distances) if any(scheme.startswith("gudhi") for scheme in schemes) else None

    scores: dict[str, np.ndarray] = {}
    representative_sets: dict[str, RepresentativeSet] = {}
    for scheme in schemes:
        if scheme == "ripser":
            representative_set = ripser_representatives(distances, persistence=persistence)
        elif scheme == "geodesic":
            representative_set = geodesic_representatives(
                distances,
                persistence=persistence,
                weight_mode=weight_mode,
                threshold_mode=threshold_mode,
            )
        elif scheme in ("gudhi", "gudhi_geodesic"):
            representative_set = gudhi_representatives(
                distances,
                generators=generators,
                mode="pairing" if scheme == "gudhi" else "geodesic",
                weight_mode=weight_mode,
                threshold_mode=threshold_mode,
            )
        else:
            raise ValueError(f"Unknown representative scheme: {scheme}")
        representative_sets[scheme] = representative_set
        scores[scheme] = min_max_normalize(representative_set.participation_scores())

    comparison = (
        compare_libraries(distances, persistence=persistence, generators=generators)
        if generators is not None
        else {}
    )
    return scores, representative_sets, comparison
