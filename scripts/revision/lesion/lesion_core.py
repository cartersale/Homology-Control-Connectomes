"""Shared helpers for the reciprocal hub-lesion revision analysis.

The system matrix A = A0 - (lambda_max(A0) + c) I is symmetric, so the infinite-horizon
controllability Gramian of a driver set S has the closed form

    W_S = V (M_S o C) V^T,   M_S = V[S,:]^T V[S,:],   C_jl = -1 / (lambda_j + lambda_l),

with A = V diag(lambda) V^T.  The eigenvalues of W_S equal those of M_S o C, so metrics are
computed without solving Lyapunov equations.  ``validate_against_lyapunov`` checks this
against hoc.control (single_node_gramians / assemble_gramian / gramian_metrics).
Metric definitions replicate hoc.control.gramian_metrics exactly (W + eps I, floored at eps).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hoc.config import load_config  # noqa: E402
from hoc.preprocessing import stabilize_adjacency  # noqa: E402

CONFIG = load_config(REPO_ROOT / "config" / "default.toml")
C_VALUE = float(CONFIG["control"]["primary_c"])  # 0.1
EPSILON = float(CONFIG["control"]["primary_epsilon"])  # 1e-5
SCALES = [int(s) for s in CONFIG["analysis"]["parcellation_scales"]]
K_VALUES = [int(k) for k in CONFIG["analysis"]["top_k_values"]]
HUB_COUNTS = [int(h) for h in CONFIG["hub_lesion"]["hub_counts"]]
BASE_SEED = int(CONFIG["analysis"]["random_seed"])
OUT_DIR = REPO_ROOT / "results" / "revision" / "lesion"


def load_stack(scale: int) -> np.ndarray:
    data = np.load(REPO_ROOT / "results" / "processed" / f"connectomes_scale-{scale}.npz")
    return np.asarray(data["adjacency"], dtype=float)


class SpectralSystem:
    """Eigen-decomposed stabilized system for fast set-Gramian spectra."""

    def __init__(self, adjacency: np.ndarray, c: float = C_VALUE):
        system, _ = stabilize_adjacency(adjacency, c)  # identical stabilization to the pipeline
        system = 0.5 * (system + system.T)
        lam, vec = np.linalg.eigh(system)
        self.n = adjacency.shape[0]
        self.vec = vec
        self.cmat = -1.0 / (lam[:, None] + lam[None, :])

    def set_eigenvalues(self, nodes) -> np.ndarray:
        rows = self.vec[np.asarray(nodes, dtype=int), :]
        g = (rows.T @ rows) * self.cmat
        return np.linalg.eigvalsh(0.5 * (g + g.T))

    def batch_set_eigenvalues(self, node_sets: np.ndarray) -> np.ndarray:
        """node_sets: (B, k) integer array -> (B, n) eigenvalues."""
        rows = self.vec[node_sets]  # (B, k, n)
        g = np.einsum("bki,bkj->bij", rows, rows) * self.cmat[None]
        return np.linalg.eigvalsh(0.5 * (g + np.transpose(g, (0, 2, 1))))


def metrics_from_eigenvalues(raw_gramian_eigs: np.ndarray, epsilon: float = EPSILON) -> dict:
    """Replicates hoc.control.gramian_metrics on eigenvalues of W (vectorised over leading axes)."""
    raw = np.asarray(raw_gramian_eigs, dtype=float) + epsilon  # eigenvalues of W + eps I
    tol = max(np.finfo(float).eps, abs(epsilon) * 1.0e-8)
    at_floor = raw <= epsilon + tol
    ev = np.clip(raw, epsilon, None)
    s = ev.sum(axis=-1)
    p = ev / s[..., None]
    er = np.exp(-np.sum(p * np.log(p), axis=-1))
    pr = s**2 / np.sum(ev**2, axis=-1)
    energy = np.sum(1.0 / ev, axis=-1)
    return {
        "effective_rank": er,
        "participation_ratio": pr,
        "log10_average_energy": np.log10(energy),
        "fraction_at_floor": at_floor.mean(axis=-1),
        "n_at_floor": at_floor.sum(axis=-1),
        "log10_condition_number": np.log10(ev[..., -1] / ev[..., 0]),
        # energy carried by non-floored eigenvalues only (size-insensitive part of E_avg)
        "log10_nonfloor_energy": np.log10(np.sum(np.where(at_floor, 0.0, 1.0 / ev), axis=-1)),
    }


def top_k_excluding(scores: np.ndarray, k: int, excluded) -> np.ndarray:
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    eligible = order[~np.isin(order, np.asarray(excluded, dtype=int))]
    return eligible[:k]


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.asarray(scores), kind="mergesort")[:k]


def validate_against_lyapunov(adjacency: np.ndarray, node_sets: list) -> float:
    """Max abs difference in ER/PR/log10E between spectral and hoc.control paths."""
    from hoc.control import assemble_gramian, gramian_metrics, single_node_gramians

    system, _ = stabilize_adjacency(adjacency, C_VALUE)
    grams = single_node_gramians(system)
    spec = SpectralSystem(adjacency)
    worst = 0.0
    for nodes in node_sets:
        ref = gramian_metrics(assemble_gramian(grams, np.asarray(nodes)), epsilon=EPSILON)
        new = metrics_from_eigenvalues(spec.set_eigenvalues(nodes))
        for key in ("effective_rank", "participation_ratio", "log10_average_energy"):
            worst = max(worst, abs(float(ref[key]) - float(new[key])))
    return worst
