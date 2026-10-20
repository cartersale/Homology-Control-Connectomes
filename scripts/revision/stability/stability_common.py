"""Shared helpers for the revision stability analyses (Reviewer 2, Major 1 / Minor 1).

Nothing here changes the pipeline's definitions; it re-implements two things for speed
and checks them against the library:

* ``EigenGramian`` -- for a symmetric stabilized system ``A = V diag(mu) V^T`` the
  infinite-horizon Gramian of input set S is ``V [K * (V_S^T V_S)] V^T`` with
  ``K_ab = -1/(mu_a + mu_b)``, so its spectrum is that of the inner matrix. This is
  algebraically identical to summing ``hoc.control.single_node_gramians`` (Lyapunov
  solves) and is validated against them in ``validate_fast_gramian``.
* effective rank / participation ratio exactly as ``hoc.control.gramian_metrics``
  (W + eps I, eigenvalues floored at eps).

Cycle-participation variants reuse ``scripts/representative_stability/representative_schemes``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for extra in (str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts" / "representative_stability")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from hoc.config import load_config  # noqa: E402
from hoc.preprocessing import stabilize_adjacency  # noqa: E402
from hoc.rankings import random_top_k_nodes, strength_scores, top_k_nodes  # noqa: E402
from hoc.utils import make_rng, subject_seed  # noqa: E402

import representative_schemes as rs  # noqa: E402

CONFIG = load_config(REPO_ROOT / "config" / "default.toml")
BASE_SEED = int(CONFIG["analysis"]["random_seed"])
PRIMARY_C = float(CONFIG["control"]["primary_c"])
PRIMARY_EPS = float(CONFIG["control"]["primary_epsilon"])
N_RANDOM = int(CONFIG["rankings"]["n_random_rankings"])
OUTPUT_DIR = REPO_ROOT / "results" / "revision" / "stability"

VARIANTS = ("ripser", "gudhi", "geodesic", "birth_edge")
VARIANT_LABELS = {
    "ripser": "Ripser cocycle support (primary)",
    "gudhi": "GUDHI persistence pairing (birth + death edge)",
    "geodesic": "Length-minimized cycle",
    "birth_edge": "Birth edge only (representative-free)",
}


def load_stack(scale: int) -> np.ndarray:
    return np.load(REPO_ROOT / "results" / "processed" / f"connectomes_scale-{scale}.npz")["adjacency"]


# --------------------------------------------------------------------------- control
class EigenGramian:
    """Spectra of set Gramians for a symmetric stabilized system."""

    def __init__(self, adjacency: np.ndarray, c: float = PRIMARY_C, epsilon: float = PRIMARY_EPS):
        system, _ = stabilize_adjacency(adjacency, c)
        system = 0.5 * (system + system.T)
        mu, vecs = np.linalg.eigh(system)
        self.kernel = -1.0 / (mu[:, None] + mu[None, :])
        self.vecs = vecs
        self.epsilon = float(epsilon)

    def spectra(self, sets: np.ndarray) -> np.ndarray:
        """Eigenvalues of W_S for a (B, k) array of driver sets."""
        sets = np.atleast_2d(np.asarray(sets, dtype=int))
        rows = self.vecs[sets]  # (B, k, n)
        inner = np.einsum("bka,bkc->bac", rows, rows) * self.kernel[None]
        inner = 0.5 * (inner + np.swapaxes(inner, 1, 2))
        return np.linalg.eigvalsh(inner)

    def metrics(self, sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Effective rank and participation ratio of W_S + eps I, floored at eps."""
        eig = np.clip(self.spectra(sets) + self.epsilon, self.epsilon, None)
        total = eig.sum(axis=1, keepdims=True)
        p = eig / total
        er = np.exp(-np.sum(p * np.log(p), axis=1))
        pr = total[:, 0] ** 2 / np.sum(eig**2, axis=1)
        return er, pr


def validate_fast_gramian(adjacency: np.ndarray, sets: list[np.ndarray]) -> float:
    """Max |ER_fast - ER_library| over the given sets."""
    from hoc.control import assemble_gramian, gramian_metrics, single_node_gramians

    system, _ = stabilize_adjacency(adjacency, PRIMARY_C)
    node_gramians = single_node_gramians(system)
    fast = EigenGramian(adjacency)
    worst = 0.0
    for selected in sets:
        lib = gramian_metrics(assemble_gramian(node_gramians, selected), epsilon=PRIMARY_EPS)
        er, pr = fast.metrics(np.asarray(selected)[None])
        worst = max(worst, abs(er[0] - lib["effective_rank"]), abs(pr[0] - lib["participation_ratio"]))
    return worst


# --------------------------------------------------------------------------- rankings
def variant_scores(adjacency: np.ndarray, variants=VARIANTS) -> dict[str, np.ndarray]:
    """Min-max normalized CP_i under each representative variant.

    ``birth_edge`` scores only the two vertices of each class's birth edge. The birth
    edge (like the whole persistence pairing used by ``gudhi``) is fixed by the
    filtration order, not chosen among homologous cycles, so it is representative-free.
    """
    distances = rs.adjacency_to_distance(adjacency)
    persistence = rs.compute_persistence(distances)
    need_gudhi = any(v in ("gudhi", "birth_edge") for v in variants)
    generators = rs.gudhi_h1_generators(distances) if need_gudhi else None
    n = adjacency.shape[0]
    out: dict[str, np.ndarray] = {}
    for variant in variants:
        if variant == "ripser":
            raw = rs.ripser_representatives(distances, persistence).participation_scores()
        elif variant == "geodesic":
            raw = rs.geodesic_representatives(distances, persistence).participation_scores()
        elif variant == "gudhi":
            raw = rs.gudhi_representatives(distances, generators, mode="pairing").participation_scores()
        elif variant == "birth_edge":
            raw = np.zeros(n)
            for idx, birth, death in rs._finite_features(generators["diagram"]):
                raw[generators["birth_edges"][idx]] += death - birth
        else:
            raise ValueError(variant)
        out[variant] = rs.min_max_normalize(raw)
    out["degree"] = strength_scores(adjacency)
    return out


def jaccard(a, b) -> float:
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / len(a | b) if (a or b) else np.nan


def spearman(a, b) -> float:
    return float(spearmanr(a, b).statistic)


# --------------------------------------------------------------------------- nulls
def uniform_random_sets(n: int, k: int, scale: int, subject_index: int) -> np.ndarray:
    """The exact 100 random sets of the primary pipeline (same seeds)."""
    return np.stack(
        [
            random_top_k_nodes(n, k, make_rng(subject_seed(BASE_SEED, scale, subject_index, k, sample)))
            for sample in range(N_RANDOM)
        ]
    )


def overlap_matched_sets(degree_set, n_shared: int, n: int, k: int, rng, n_sets: int = N_RANDOM) -> np.ndarray:
    """Random sets sharing exactly ``n_shared`` nodes with the degree top-k set."""
    degree_set = np.asarray(degree_set)
    others = np.setdiff1d(np.arange(n), degree_set)
    sets = np.empty((n_sets, k), dtype=int)
    for i in range(n_sets):
        sets[i, :n_shared] = rng.choice(degree_set, n_shared, replace=False)
        sets[i, n_shared:] = rng.choice(others, k - n_shared, replace=False)
    return sets


def strength_matched_sets(selected, strength_rank: np.ndarray, n: int, rng, window: int, n_sets: int = N_RANDOM):
    """Random sets whose members match the selected nodes' strength ranks within +/- window.

    ``strength_rank`` is 0-based descending rank. Each selected node is replaced by a
    node drawn uniformly from those with rank within ``window`` of its own (the
    node itself included), without repeats inside a set.
    """
    order = np.argsort(strength_rank)  # order[r] = node with rank r
    sets = np.empty((n_sets, len(selected)), dtype=int)
    for i in range(n_sets):
        chosen: list[int] = []
        for node in rng.permutation(np.asarray(selected)):
            r = strength_rank[node]
            pool = order[max(0, r - window) : min(n, r + window + 1)]
            pool = pool[~np.isin(pool, chosen)]
            if pool.size == 0:  # widen until something is free
                pool = np.setdiff1d(np.arange(n), chosen)
            chosen.append(int(rng.choice(pool)))
        sets[i] = chosen
    return sets


def percentile_vs(null: np.ndarray, value: float) -> float:
    """Percent of null values below ``value`` (ties count half)."""
    return float(100.0 * (np.mean(null < value) + 0.5 * np.mean(null == value)))


# --------------------------------------------------------------------------- stats
def rank_biserial(values: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (zeros dropped, as in Wilcoxon 'wilcox')."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & (v != 0)]
    if v.size == 0:
        return np.nan
    from scipy.stats import rankdata

    ranks = rankdata(np.abs(v))
    total = ranks.sum()
    return float((ranks[v > 0].sum() - ranks[v < 0].sum()) / total)


def paired_summary(values, rng, n_boot: int = 2000) -> dict:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    boot = rng.choice(v, size=(n_boot, v.size), replace=True)
    med_lo, med_hi = np.percentile(np.median(boot, axis=1), [2.5, 97.5])
    mean_lo, mean_hi = np.percentile(np.mean(boot, axis=1), [2.5, 97.5])
    if np.allclose(v, 0):
        p = 1.0
    else:
        p = float(wilcoxon(v, zero_method="wilcox", alternative="two-sided").pvalue)
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "mean_ci_low": float(mean_lo),
        "mean_ci_high": float(mean_hi),
        "median": float(np.median(v)),
        "median_ci_low": float(med_lo),
        "median_ci_high": float(med_hi),
        "n_positive": int(np.sum(v > 0)),
        "n_zero": int(np.sum(v == 0)),
        "n_negative": int(np.sum(v < 0)),
        "wilcoxon_p": p,
        "rank_biserial": rank_biserial(v),
    }
