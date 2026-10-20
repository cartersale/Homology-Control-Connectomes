"""Shared helpers for the network-null analysis (paper/new_idea.md).

Question: is the hub-concentration effect (degree-selected driver sets have a
Gramian dominated by one direction, q1 = lambda_1(W)/tr W, far above random sets)
a property of the brain's wiring, or does it follow from the strength sequence,
from spatial embedding, or from the weight distribution alone?

Nulls are applied to the *preprocessed* adjacency A0 (symmetrized, log1p,
max-normalized), which is the matrix whose strength defines the degree ranking and
whose spectrum defines the dynamics. All three nulls permute the existing weights,
so max(A0) = 1 and no renormalization is needed.

  strength   Maslov-Sneppen rewiring + simulated annealing on the weight assignment
             (Milisav & Misic; netneurotools.strength_preserving_rand_sa).
             Preserves binary degree exactly and strength approximately.
  geometric  Betzel & Bassett (2018) swaps within 10 Euclidean-distance bins of the
             Lausanne centroids (netneurotools.match_length_degree_distribution,
             with replacement, 10 x n_edges swaps). Preserves binary degree exactly
             and the edge-length distribution approximately; weights travel with edges.
  weights    Same binary topology, weights permuted across existing edges.

Fast Gramians (A0 symmetric): with A0 = U diag(l) U^T and mu = l - l_max - c,
the Gramian of driver set S in the eigenbasis is G_S = K o (U_S^T U_S),
K_ab = -1/(mu_a + mu_b), and eig(W_S) = eig(G_S). The Perron mode is the last
eigenvector, so W_perp = P W P with P = I - v1 v1^T has spectrum eig(G_S[:-1, :-1])
(plus a zero). Validated against hoc.control in 00_validate.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hoc.rankings import random_top_k_nodes, top_k_nodes  # noqa: E402
from hoc.utils import make_rng, subject_seed  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "revision" / "nulls"
BASE_SEED = 20260423
SCALES = (68, 114, 219)
K_VALUES = (1, 2, 3, 5)
N_SUBJECTS = 70
N_RANDOM = 100
N_NULLS = 50
EPSILON = 1.0e-5
C_VALUES = (0.01, 0.1, 1.0)
NULL_TYPES = ("strength", "geometric", "weights")
NULL_TAG = {"real": 0, "strength": 8101, "geometric": 8102, "weights": 8103}
CYCLE_SCALES = (114,)
HUB_K = 5  # hub set size for the hub-to-hub strength fraction
RC_PERCENTILE = 90  # rich-club degree level: 90th percentile of the real network's binary degree
GEO_NBINS = 10
GEO_SWAPS_PER_EDGE = 10


# ----------------------------------------------------------------------------- data


def load_stack(scale: int) -> np.ndarray:
    return np.load(REPO_ROOT / "results" / "processed" / f"connectomes_scale-{scale}.npz")["adjacency"]


def centroid_distances(scale: int) -> np.ndarray:
    xyz = pd.read_csv(REPO_ROOT / "results" / "revision" / "spectra" / f"centroids_scale-{scale}.csv")[["x", "y", "z"]].to_numpy()
    return cdist(xyz, xyz)


# ----------------------------------------------------------------------------- nulls


def null_seed(null: str, scale: int, subject_index: int, null_index: int) -> int:
    return subject_seed(BASE_SEED, NULL_TAG[null], scale, subject_index, null_index) % (2**32 - 1)


def strength_null(adjacency: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    from netneurotools.networks import strength_preserving_rand_sa

    np.random.seed(seed)
    randomized, energy = strength_preserving_rand_sa(adjacency, seed=seed)
    return _clean(randomized), float(energy)


def geometric_null(adjacency: np.ndarray, distances: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    from netneurotools.networks import match_length_degree_distribution

    n_edges = int(np.count_nonzero(np.triu(adjacency, 1)))
    np.random.seed(seed)
    _, randomized, n_rewired = match_length_degree_distribution(
        adjacency,
        distances,
        nbins=GEO_NBINS,
        nswap=GEO_SWAPS_PER_EDGE * n_edges,
        replacement=True,
        weighted=True,
        seed=seed,
    )
    return _clean(randomized), float(n_rewired)


def weight_shuffle_null(adjacency: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(adjacency.shape[0], 1)
    weights = adjacency[iu]
    edges = np.flatnonzero(weights > 0)
    shuffled = weights.copy()
    shuffled[edges] = weights[edges][rng.permutation(edges.size)]
    out = np.zeros_like(adjacency)
    out[iu] = shuffled
    return out + out.T, float("nan")


def make_null(null: str, adjacency: np.ndarray, distances: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    if null == "strength":
        return strength_null(adjacency, seed)
    if null == "geometric":
        return geometric_null(adjacency, distances, seed)
    if null == "weights":
        return weight_shuffle_null(adjacency, seed)
    raise ValueError(null)


def _clean(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


# ----------------------------------------------------------------------------- network statistics


def rich_club_level(real_adjacency: np.ndarray) -> int:
    degree = np.count_nonzero(real_adjacency, axis=0)
    return int(np.floor(np.percentile(degree, RC_PERCENTILE)))


def rich_club(adjacency: np.ndarray, level: int) -> tuple[float, float]:
    """Binary and weighted (Opsahl) rich-club coefficients among nodes with degree > level."""
    binary = (adjacency > 0).astype(float)
    degree = binary.sum(axis=0)
    club = degree > level
    n_club = int(club.sum())
    if n_club < 2:
        return np.nan, np.nan
    e_club = binary[np.ix_(club, club)].sum() / 2.0
    rc_bu = 2.0 * e_club / (n_club * (n_club - 1))
    w_club = adjacency[np.ix_(club, club)].sum() / 2.0
    ranked = np.sort(adjacency[np.triu_indices_from(adjacency, 1)])[::-1]
    rc_wu = w_club / ranked[: int(e_club)].sum() if e_club > 0 else np.nan
    return float(rc_bu), float(rc_wu)


def network_stats(adjacency: np.ndarray, spec: "Spectral", real_adjacency: np.ndarray, distances: np.ndarray, rc_level: int) -> dict:
    strength = adjacency.sum(axis=0)
    real_strength = real_adjacency.sum(axis=0)
    hubs = top_k_nodes(strength, HUB_K)
    iu = np.triu_indices_from(adjacency, 1)
    edges = adjacency[iu] > 0
    real_edges = real_adjacency[iu] > 0
    rc_bu, rc_wu = rich_club(adjacency, rc_level)
    return {
        "lambda1": spec.lam[-1],
        "lambda2": spec.lam[-2],
        "gap": spec.gap,
        "ipr_v1": float(np.sum(spec.perron**4)),
        "rc_level": rc_level,
        "rc_bu": rc_bu,
        "rc_wu": rc_wu,
        "hub_strength_frac": float(adjacency[np.ix_(hubs, hubs)].sum() / adjacency.sum()),
        "strength_r": float(np.corrcoef(strength, real_strength)[0, 1]),
        "strength_max_rel_err": float(np.max(np.abs(strength - real_strength)) / real_strength.mean()),
        "strength_top5_jaccard": _jaccard(hubs, top_k_nodes(real_strength, HUB_K)),
        "degree_preserved": bool(np.array_equal(np.count_nonzero(adjacency, 0), np.count_nonzero(real_adjacency, 0))),
        "edge_overlap": float(np.sum(edges & real_edges) / np.sum(real_edges)),
        "mean_edge_length": float(distances[iu][edges].mean()),
        "weighted_edge_length": float(np.sum(distances[iu] * adjacency[iu]) / adjacency[iu].sum()),
        "n_edges": int(edges.sum()),
        "max_weight": float(adjacency.max()),
    }


def _jaccard(a, b) -> float:
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / len(a | b)


# ----------------------------------------------------------------------------- Gramians


class Spectral:
    """Eigendecomposition of a symmetric A0; Gramian spectra for any driver set and c."""

    def __init__(self, adjacency: np.ndarray):
        lam, U = np.linalg.eigh(np.asarray(adjacency, dtype=float))
        self.lam = lam
        self.U = U
        self.lam_max = float(lam[-1])
        v1 = U[:, -1]
        self.perron = v1 * np.sign(v1.sum())
        self.perron_sq = v1**2
        self.gap = float(lam[-1] - lam[-2])

    def kernel(self, c: float) -> np.ndarray:
        mu = self.lam - self.lam_max - c
        return -1.0 / (mu[:, None] + mu[None, :])

    def set_spectra(self, kernel: np.ndarray, driver_sets) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(eig W, eig W_perp without its zero, Perron share) for each set; eigenvalues ascending."""
        n = self.U.shape[0]
        stack = np.empty((len(driver_sets), n, n))
        for j, s in enumerate(driver_sets):
            rows = self.U[np.asarray(s, dtype=int), :]
            stack[j] = kernel * (rows.T @ rows)
        traces = np.trace(stack, axis1=1, axis2=2)
        perron_share = stack[:, -1, -1] / traces
        w = np.linalg.eigvalsh(stack)
        w_perp = np.linalg.eigvalsh(stack[:, :-1, :-1])
        return w, w_perp, perron_share


def spectrum_metrics(w: np.ndarray, epsilon: float = EPSILON) -> dict[str, np.ndarray]:
    """q1 on raw W; ER and PR replicate hoc.control.gramian_metrics (W + eps I, floored at eps)."""
    w = np.clip(np.atleast_2d(w), 0.0, None)
    q1 = w[:, -1] / w.sum(axis=1)
    ev = np.clip(w + epsilon, epsilon, None)
    total = ev.sum(axis=1)
    p = ev / total[:, None]
    er = np.exp(-np.sum(p * np.log(p), axis=1))
    pr = total**2 / np.sum(ev**2, axis=1)
    return {"q1": q1, "er": er, "pr": pr}


def percentile_of(value: float, reference: np.ndarray) -> float:
    reference = np.asarray(reference)
    return float((np.sum(reference < value) + 0.5 * np.sum(reference == value)) / reference.size)


def random_driver_sets(null: str, null_index: int, scale: int, subject_index: int, k: int) -> list[np.ndarray]:
    """Real network: exactly the primary pipeline's random sets. Nulls: fresh draws per network."""
    if null == "real":
        seeds = [subject_seed(BASE_SEED, scale, subject_index, k, r) for r in range(N_RANDOM)]
    else:
        seeds = [subject_seed(BASE_SEED, NULL_TAG[null], null_index, scale, subject_index, k, r) for r in range(N_RANDOM)]
    return [random_top_k_nodes(scale, k, make_rng(s)) for s in seeds]


def set_level_rows(spec: Spectral, c: float, sets: dict[str, np.ndarray], random_sets: list[np.ndarray]) -> dict:
    """Headline set-level statistics for one network, one c, one k."""
    kernel = spec.kernel(c)
    names = list(sets)
    all_sets = [sets[m] for m in names] + list(random_sets)
    w, w_perp, pshare = spec.set_spectra(kernel, all_sets)
    met = spectrum_metrics(w)
    met_perp = spectrum_metrics(w_perp)
    loading = np.array([spec.perron_sq[np.asarray(s, dtype=int)].sum() for s in all_sets])
    n_named = len(names)
    rnd = slice(n_named, None)
    row: dict = {
        "rand_q1_mean": float(met["q1"][rnd].mean()),
        "rand_q1_sd": float(met["q1"][rnd].std(ddof=1)),
        "rand_er_mean": float(met["er"][rnd].mean()),
        "rand_loading_mean": float(loading[rnd].mean()),
        "rand_q1perp_mean": float(met_perp["q1"][rnd].mean()),
        "rho_loading_q1_all": _spearman(loading, met["q1"]),
        "rho_loading_q1_random": _spearman(loading[rnd], met["q1"][rnd]),
    }
    for j, m in enumerate(names):
        tag = m
        row[f"{tag}_q1"] = float(met["q1"][j])
        row[f"{tag}_er"] = float(met["er"][j])
        row[f"{tag}_pr"] = float(met["pr"][j])
        row[f"{tag}_loading"] = float(loading[j])
        row[f"{tag}_perron_share"] = float(pshare[j])
        row[f"{tag}_q1perp"] = float(met_perp["q1"][j])
        row[f"{tag}_q1_pct"] = percentile_of(met["q1"][j], met["q1"][rnd])
        row[f"{tag}_er_pct"] = percentile_of(met["er"][j], met["er"][rnd])
        row[f"{tag}_q1perp_pct"] = percentile_of(met_perp["q1"][j], met_perp["q1"][rnd])
        row[f"{tag}_loading_pct"] = percentile_of(loading[j], loading[rnd])
    return row


def _spearman(x, y) -> float:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


# ----------------------------------------------------------------------------- one network, all conditions


def analyze_network(
    adjacency: np.ndarray,
    real_adjacency: np.ndarray,
    real_gap: float,
    distances: np.ndarray,
    rc_level: int,
    null: str,
    null_index: int,
    scale: int,
    subject_index: int,
    with_cycle: bool,
    extra: dict,
) -> tuple[dict, list[dict]]:
    """Network-level row plus one set-level row per (c mode, c, k).

    c modes: 'fixed' uses c as given; 'gap' uses c * gap_null / gap_real, which makes the
    null's slowest-mode decay ratio c/(c + lambda1 - lambda2) equal to the real network's.
    Rankings (strength, and cycle participation if requested) are recomputed on this network.
    """
    spec = Spectral(adjacency)
    ids = {"null": null, "null_index": null_index, "scale": scale, "subject_index": subject_index, "subject_id": f"sub-{subject_index:03d}"}
    net = {**ids, **network_stats(adjacency, spec, real_adjacency, distances, rc_level), **extra}

    scores = {"degree": adjacency.sum(axis=0)}
    if with_cycle:
        from hoc.topology import cycle_participation

        scores["cycle"] = cycle_participation(adjacency)[0]

    conditions = []
    for c in C_VALUES:
        conditions.append(("fixed", c, c))
        if null != "real":
            conditions.append(("gap", c, c * spec.gap / real_gap))
    rows = []
    for k in K_VALUES:
        sets = {m: top_k_nodes(s, k) for m, s in scores.items()}
        rand = random_driver_sets(null, null_index, scale, subject_index, k)
        for mode, c_ref, c_used in conditions:
            row = {**ids, "c_mode": mode, "c_ref": c_ref, "c": c_used, "decay_ratio": c_used / (c_used + spec.gap), "k": k}
            row.update(set_level_rows(spec, c_used, sets, rand))
            rows.append(row)
            if null == "real":  # gap-matched == fixed for the real network
                rows.append({**row, "c_mode": "gap"})
    return net, rows
