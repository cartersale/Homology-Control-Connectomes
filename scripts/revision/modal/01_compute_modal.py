#!/usr/bin/env python
"""Set-level Gramian concentration vs single-node modal / average controllability.

For every subject (70), scale (68/114/219), c in {0.01, 0.1, 1.0}, k in {2, 3, 5}, and for
(i) the 6 primary static rankings, (ii) 4 counterfactual rankings (top-k Gu modal controllability,
top-k continuous-time modal controllability, top-k average controllability, bottom-k Perron loading)
and (iii) the 100 primary random sets, compute per set:

  q1, effective rank, participation ratio      (W_eps, eps = 1e-5, pipeline definitions)
  q1_raw = lambda_1(W)/tr W, perron_share = v1'Wv1/tr W = (L_S/2c)/tr W
  L_S = sum_{i in S} v1(i)^2                    (Perron loading)
  modal_gu   : mean_i phi_i, phi_i = sum_a (1 - xi_a^2) u_a(i)^2, xi = eig(A0)/(1+sigma_max(A0))
               (= 1 - (A0^2)_ii/(1+sigma_max)^2, Gu et al. 2015)
  modal_ct   : mean_i [1 - (e^{2A tau})_ii], tau = 1 (Gu formula applied to the exact
               discretisation e^{A tau} of our continuous-time A)
  avg_ctrl   : mean_i [(-2A)^{-1}]_ii = mean_i tr W_{i}  (continuous-time average controllability)
  red_G      : mean pairwise cosine of driver slow-mode profiles x_i = (u_a(i)/sqrt(-2 mu_a))_a
               = G_ij/sqrt(G_ii G_jj), G = B'(-2A)^{-1}B  (k x k "driver Gram")
  red_G_perp : same with the Perron mode removed from the profiles
  red_F      : mean pairwise Frobenius cosine <W_i, W_j>/(|W_i||W_j|) of single-node Gramians
  q1_G, er_G : lambda_1(G)/tr G and exp-entropy of eig(G) (k x k proxy of W's leading spectrum)
  er_raw     : exp-entropy of the unregularised spectrum of W (no eps floor); n_above_eps
  maj_violation: max_m [sum_{top m} eig(W) - sum_{top m} eig(G)] / tr W  (<= 0: eig(W) majorised by eig(G))
  pr_decomp_check: exact identity PR(W) = (sum_i tr W_i)^2 / sum_ij <W_i, W_j>

Writes runs/<name>/revision/modal/set_level_modal.csv.gz and node_level_modal.csv.gz
(choose the run with ``--run <name>``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REGIME_DIR = Path(__file__).resolve().parents[1] / "regime"  # shared helpers: regime/regime_common.py
if str(REGIME_DIR) not in sys.path:
    sys.path.insert(0, str(REGIME_DIR))
from regime_common import SCALES, Spectral, load_ranks, load_stack, metrics_from_eigs, random_sets, run_parallel  # noqa: E402
from hoc.runs import stage_from_argv  # noqa: E402
from hoc.utils import n_jobs  # noqa: E402

OUT: Path  # set in main(): <run>/revision/modal
C_VALUES = (0.01, 0.1, 1.0)
K_SET = (2, 3, 5)
EPS = 1.0e-5
TAU = 1.0
METHODS = (
    "degree_strength",
    "cycle_participation",
    "eigenvector_centrality",
    "betweenness",
    "participation_coefficient",
    "triangle_participation",
)


def top(score: np.ndarray, k: int, ascending: bool = False) -> np.ndarray:
    s = score if ascending else -score
    return np.argsort(s, kind="stable")[:k]


def mean_pairwise(mat: np.ndarray) -> float:
    k = mat.shape[0]
    iu = np.triu_indices(k, 1)
    return float(mat[iu].mean())


def subject_job(scale: int, si: int, subject_id: str, adjacency: np.ndarray, orders: dict) -> tuple[list, list]:
    spec = Spectral(adjacency)
    U, lam = spec.U, spec.lam
    p = int(np.argmax(lam))
    v1 = spec.perron
    sigma = np.linalg.svd(spec.A0, compute_uv=False)[0]
    xi = lam / (1.0 + sigma)
    modal_gu = (U**2) @ (1.0 - xi**2)
    rows, node_rows = [], []
    rand = {k: random_sets(scale, si, k) for k in K_SET}
    for c in C_VALUES:
        mu = spec.mu(c)
        K = spec.kernel(c)
        K2 = K**2
        avg = (U**2) @ (1.0 / (-2.0 * mu))
        modal_ct = (U**2) @ (1.0 - np.exp(2.0 * mu * TAU))
        X = U / np.sqrt(-2.0 * mu)[None, :]  # driver slow-mode profiles (rows)
        Xp = np.delete(X, p, axis=1)
        for i in range(U.shape[0]):
            node_rows.append({"subject_id": subject_id, "scale": scale, "c": c, "node": i, "modal_gu": modal_gu[i],
                              "modal_ct": modal_ct[i], "avg_ctrl": avg[i], "perron2": v1[i] ** 2})
        for k in K_SET:
            sets = []
            for m in METHODS:
                sets.append((m, -1, orders[m][:k]))
            sets.append(("top_modal_gu", -1, top(modal_gu, k)))
            sets.append(("top_modal_ct", -1, top(modal_ct, k)))
            sets.append(("top_avg_ctrl", -1, top(avg, k)))
            sets.append(("low_perron", -1, top(v1**2, k, ascending=True)))
            for r, s in enumerate(rand[k]):
                sets.append(("random", r, np.asarray(s, dtype=int)))
            eigs = spec.eigvals_batch(K, [s for _, _, s in sets])
            met = metrics_from_eigs(eigs, EPS)
            for j, (fam, r, s) in enumerate(sets):
                w = eigs[j]
                ev = np.clip(w + EPS, EPS, None)
                trw = w.sum()
                L = float(np.sum(v1[s] ** 2))
                Xs = X[s]
                G = Xs @ Xs.T
                dg = np.sqrt(np.diag(G))
                corrG = G / np.outer(dg, dg)
                Xps = Xp[s]
                Gp = Xps @ Xps.T
                dgp = np.sqrt(np.diag(Gp))
                corrGp = Gp / np.outer(dgp, dgp)
                Us = U[s]
                # Frobenius inner products <W_i, W_j> = (u_i o u_j)' K2 (u_i o u_j)
                prod = Us[:, None, :] * Us[None, :, :]  # k x k x N
                F = np.einsum("ija,ab,ijb->ij", prod, K2, prod)
                fd = np.sqrt(np.diag(F))
                corrF = F / np.outer(fd, fd)
                gev = np.clip(np.linalg.eigvalsh(G), 1e-300, None)
                gp = gev / gev.sum()
                wr = np.clip(w, 0, None)[::-1]
                pw = wr / wr.sum()
                pw = pw[pw > 0]
                gs = np.zeros_like(wr)
                gs[:k] = gev[::-1]
                maj = float(np.max(np.cumsum(wr) - np.cumsum(gs)) / trw)  # <= 0 iff eig(W) majorized by eig(G)
                rows.append({
                    "subject_id": subject_id, "scale": scale, "c": c, "k": k, "family": fam, "sample": r,
                    "nodes": " ".join(map(str, np.sort(s))),
                    "q1": ev[-1] / ev.sum(), "effective_rank": met["effective_rank"][j],
                    "participation_ratio": met["participation_ratio"][j],
                    "q1_raw": w[-1] / trw, "pr_raw": trw**2 / np.sum(w**2),
                    "perron_share": (L / (2 * c)) / trw, "trace_w": trw,
                    "perron_loading": L,
                    "modal_gu": float(modal_gu[s].mean()), "modal_ct": float(modal_ct[s].mean()),
                    "avg_ctrl": float(avg[s].mean()),
                    "red_G": mean_pairwise(corrG), "red_G_abs": mean_pairwise(np.abs(corrG)),
                    "red_G_perp": mean_pairwise(corrGp), "red_F": mean_pairwise(corrF),
                    "q1_G": gev[-1] / gev.sum(), "er_G": float(np.exp(-np.sum(gp * np.log(gp)))),
                    "lam1_W": w[-1], "lam1_G": gev[-1],
                    "pr_decomp": np.trace(G) ** 2 / F.sum(),
                    "er_raw": float(np.exp(-np.sum(pw * np.log(pw)))), "n_above_eps": int(met["n_above_eps"][j]),
                    "pr_G": gev.sum() ** 2 / np.sum(gev**2), "maj_violation": maj,
                })
    return rows, node_rows


def main() -> None:
    global OUT
    ctx = stage_from_argv("revision/modal/01_compute_modal")
    ctx.require_stage("00_prepare_data")
    ctx.require_stage("01_primary")
    ctx.begin()
    OUT = ctx.root / "revision" / "modal"
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for scale in SCALES:
        stack = load_stack(scale, ctx.root)
        ranks = load_ranks(scale, ctx.root)
        subjects = sorted({s for s, _ in ranks})
        for si, sid in enumerate(subjects):
            jobs.append((scale, si, sid, stack[si], {m: ranks[(sid, m)] for m in METHODS}))
    res = run_parallel(subject_job, jobs, n_jobs=n_jobs(12))
    df = pd.DataFrame([r for rr, _ in res for r in rr])
    nd = pd.DataFrame([r for _, nn in res for r in nn])
    print("trace identity (tr W = sum avg_ctrl) max rel err:",
          np.max(np.abs(df.trace_w - df.k * df.avg_ctrl) / df.trace_w))
    print("PR decomposition max rel err:", np.max(np.abs(df.pr_raw - df.pr_decomp) / df.pr_raw))
    df.to_csv(OUT / "set_level_modal.csv.gz", index=False)
    nd.to_csv(OUT / "node_level_modal.csv.gz", index=False)
    print(df.shape, nd.shape)
    ctx.finish()


if __name__ == "__main__":
    main()
