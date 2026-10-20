"""Shared helpers for the regime revision analyses (reviewer 2, Majors 2-4).

Fast Gramians: for symmetric A0 = U diag(l) U^T and A = A0 - (l_max + c) I with
eigenvalues mu_a = l_a - l_max - c, the single-input Lyapunov solution for
a driver set S is W_S = U (K o (U_S^T U_S)) U^T with
    K_ab = -1 / (mu_a + mu_b)                          (infinite horizon)
    K_ab = (1 - exp((mu_a + mu_b) T)) / -(mu_a + mu_b) (finite horizon T)
so eig(W_S) = eig(K o U_S^T U_S). This is validated against hoc.control in 00_validate.py.
Metrics replicate hoc.control.gramian_metrics exactly (W + eps I, eigenvalues floored at eps).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hoc.utils import make_rng, subject_seed  # noqa: E402
from hoc.rankings import random_top_k_nodes  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "revision" / "regime"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_SEED = 20260423
SCALES = (68, 114, 219)
K_VALUES = (1, 2, 3, 5)
N_SUBJECTS = 70
N_RANDOM = 100
METHODS = (
    "degree_strength",
    "betweenness",
    "cycle_participation",
    "eigenvector_centrality",
    "participation_coefficient",
    "degree_normalized_cycle_participation",
    "cycle_criticality",
    "triangle_participation",
    "tetrahedron_participation",
)
LARGE_K = {68: (1, 2, 3, 5, 7, 10, 14), 114: (1, 2, 3, 5, 6, 10, 11, 23), 219: (1, 2, 3, 5, 10, 11, 22, 44)}


def load_stack(scale: int) -> np.ndarray:
    return np.load(REPO_ROOT / "results" / "processed" / f"connectomes_scale-{scale}.npz")["adjacency"]


_RANKS: dict[int, pd.DataFrame] = {}


def load_ranks(scale: int) -> dict[tuple[str, str], np.ndarray]:
    """Return {(subject_id, method): node order (descending, stable)} from the saved primary node scores."""
    table = pd.read_csv(REPO_ROOT / "results" / "primary" / "node_scores.csv", usecols=["subject_id", "scale", "node_index", "method", "rank"])
    table = table[table["scale"] == scale]
    out = {}
    for (subject_id, method), sub in table.groupby(["subject_id", "method"]):
        order = sub.sort_values("rank")["node_index"].to_numpy(dtype=int)
        out[(subject_id, method)] = order
    return out


def random_sets(scale: int, subject_index: int, k: int, n_random: int = N_RANDOM) -> list[np.ndarray]:
    """Exactly the primary pipeline's random driver sets."""
    n_nodes = scale
    return [
        random_top_k_nodes(n_nodes, k, make_rng(subject_seed(BASE_SEED, scale, subject_index, k, r)))
        for r in range(n_random)
    ]


class Spectral:
    """Eigen-decomposition of one preprocessed adjacency (A0)."""

    def __init__(self, adjacency: np.ndarray):
        self.A0 = np.asarray(adjacency, dtype=float)
        lam, U = np.linalg.eigh(self.A0)
        self.lam = lam
        self.U = U
        # pipeline uses max real part of np.linalg.eigvals
        self.lam_max = float(np.max(np.real(np.linalg.eigvals(self.A0))))
        self.perron = U[:, np.argmax(lam)]
        self.perron = self.perron * np.sign(self.perron.sum())

    def mu(self, c: float) -> np.ndarray:
        return self.lam - self.lam_max - c

    def kernel(self, c: float, horizon: float | None = None) -> np.ndarray:
        mu = self.mu(c)
        s = mu[:, None] + mu[None, :]
        if horizon is None or not np.isfinite(horizon):
            return -1.0 / s
        return -(1.0 - np.exp(s * horizon)) / s

    def gramian_eigbasis(self, kernel: np.ndarray, driver_set: np.ndarray) -> np.ndarray:
        rows = self.U[np.asarray(driver_set, dtype=int), :]
        return kernel * (rows.T @ rows)

    def eigvals_batch(self, kernel: np.ndarray, driver_sets) -> np.ndarray:
        """Eigenvalues (ascending) of W_S for a list/array of driver sets -> (n_sets, N)."""
        driver_sets = [np.asarray(s, dtype=int) for s in driver_sets]
        n = self.U.shape[0]
        out = np.empty((len(driver_sets), n))
        batch = 256
        for start in range(0, len(driver_sets), batch):
            chunk = driver_sets[start : start + batch]
            stack = np.empty((len(chunk), n, n))
            for j, s in enumerate(chunk):
                rows = self.U[s, :]
                stack[j] = kernel * (rows.T @ rows)
            out[start : start + len(chunk)] = np.linalg.eigvalsh(stack)
        return out

    def full_gramian(self, kernel: np.ndarray, driver_set) -> np.ndarray:
        g = self.gramian_eigbasis(kernel, driver_set)
        w = self.U @ g @ self.U.T
        return 0.5 * (w + w.T)


def metrics_from_eigs(w: np.ndarray, epsilon: float) -> dict[str, np.ndarray]:
    """Vectorized replica of hoc.control.gramian_metrics on eigenvalues of W (rows = sets)."""
    w = np.atleast_2d(w)
    raw = w + epsilon
    tol = max(np.finfo(float).eps, abs(epsilon) * 1.0e-8)
    at_floor = raw <= epsilon + tol
    ev = np.clip(raw, epsilon, None)
    total = ev.sum(axis=1)
    p = ev / total[:, None]
    er = np.exp(-np.sum(p * np.log(p), axis=1))
    pr = total**2 / np.sum(ev**2, axis=1)
    e_avg = np.sum(1.0 / ev, axis=1)
    not_floor = ~at_floor
    e_nonfloor = np.sum(np.where(not_floor, 1.0 / ev, 0.0), axis=1)
    below_eps = np.mean(w < epsilon, axis=1)
    above = w > epsilon
    e_above = np.sum(np.where(above, 1.0 / ev, 0.0), axis=1)
    return {
        "effective_rank": er,
        "participation_ratio": pr,
        "log10_average_energy": np.log10(e_avg),
        "fraction_at_floor": at_floor.mean(axis=1),
        "fraction_below_eps": below_eps,
        "log10_energy_nonfloor": np.log10(np.where(e_nonfloor > 0, e_nonfloor, np.nan)),
        "log10_energy_above_eps": np.log10(np.where(e_above > 0, e_above, np.nan)),
        "n_above_eps": above.sum(axis=1),
        "log10_lambda_max": np.log10(ev[:, -1]),
    }


def paired_stats(diff: np.ndarray, rng_seed: int = BASE_SEED, n_boot: int = 5000) -> dict:
    """Mean/median paired difference, bootstrap CIs, n positive, Wilcoxon, rank-biserial."""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx]
    mean_ci = np.percentile(boots.mean(axis=1), [2.5, 97.5])
    med_ci = np.percentile(np.median(boots, axis=1), [2.5, 97.5])
    nz = d[d != 0]
    if nz.size == 0:
        p, rbc = 1.0, 0.0
    else:
        _, p = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        ranks = pd.Series(np.abs(nz)).rank().to_numpy()
        w_plus = ranks[nz > 0].sum()
        w_minus = ranks[nz < 0].sum()
        rbc = (w_plus - w_minus) / (w_plus + w_minus)
    return {
        "n": n,
        "mean_diff": float(d.mean()),
        "mean_ci_low": float(mean_ci[0]),
        "mean_ci_high": float(mean_ci[1]),
        "median_diff": float(np.median(d)),
        "median_ci_low": float(med_ci[0]),
        "median_ci_high": float(med_ci[1]),
        "n_positive": int(np.sum(d > 0)),
        "wilcoxon_p": float(p),
        "rank_biserial": float(rbc),
    }


def holm(frame: pd.DataFrame, family_cols: list[str] | None, p_col: str = "wilcoxon_p") -> pd.DataFrame:
    frame = frame.copy()
    frame["p_holm"] = np.nan
    groups = [(None, frame.index)] if not family_cols else frame.groupby(family_cols).groups.items()
    for _, idx in groups:
        frame.loc[idx, "p_holm"] = multipletests(frame.loc[idx, p_col].to_numpy(), method="holm")[1]
    return frame


def run_parallel(func, jobs, n_jobs: int = 12):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from joblib import Parallel, delayed

    return Parallel(n_jobs=n_jobs, verbose=0)(delayed(func)(*job) for job in jobs)
