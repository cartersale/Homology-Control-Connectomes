"""Task 2: nctpy continuous normalization A/(rho+1) - I vs ours A0 - (lambda_max + 1) I (c = 1).

nctpy.utils.matrix_normalization(A, system='continuous', c=1) (nctpy not installed; reimplemented from its source):
    l = max |eig(A)|;  A_norm = A / (c + l) - I
Part A: for 3 subjects x 3 scales x {degree, cycle} x k in {1,2,3,5}, solve both Lyapunov equations
        independently with scipy (A W + W A^T + B B^T = 0) and compare W_nctpy with (rho+1) W_ours(c=1),
        plus effective rank at eps = 1e-5 (floor applied to each W as-is), eps = 0 (no floor), and with the
        floor rescaled (eps_nctpy = (rho+1) eps).
Part B: all 70 subjects, cycle - degree ER / PR under the nctpy normalization with eps = 1e-5 (k-averaged).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import solve_continuous_lyapunov

from _common import (K_VALUES, N_SUBJECTS, OUT_DIR, SCALES, Spectral, holm, load_ranks, load_stack, metrics_from_eigs,
                     paired_stats)
from hoc.control import gramian_metrics
from hoc.preprocessing import stabilize_adjacency


def nctpy_normalization(A, c=1.0):
    w = np.linalg.eigvals(A)
    rho = np.abs(w).max()
    return A / (c + rho) - np.eye(A.shape[0]), rho


def lyap(A, S):
    n = A.shape[0]
    B = np.zeros((n, len(S)))
    B[S, np.arange(len(S))] = 1.0
    W = solve_continuous_lyapunov(A, -B @ B.T)
    return 0.5 * (W + W.T)


rows = []
for scale in SCALES:
    stack = load_stack(scale)
    ranks = load_ranks(scale)
    for si in (0, 33, 69):
        A0 = stack[si]
        A_ours, lam = stabilize_adjacency(A0, 1.0)
        A_n, rho = nctpy_normalization(A0, 1.0)
        for m in ("degree_strength", "cycle_participation"):
            for k in K_VALUES:
                S = ranks[(f"sub-{si:03d}", m)][:k]
                W_o = lyap(A_ours, S)
                W_n = lyap(A_n, S)
                s = rho + 1.0
                g_o = gramian_metrics(W_o, 1e-5)
                g_n = gramian_metrics(W_n, 1e-5)
                g_n_scaled = gramian_metrics(W_n, 1e-5 * s)
                g_o0 = gramian_metrics(W_o, 0.0, apply_floor=False)
                g_n0 = gramian_metrics(W_n, 0.0, apply_floor=False)
                rows.append({
                    "scale": scale, "subject": si, "method": m, "k": k, "lambda_max": lam, "rho": rho, "scale_factor": s,
                    "max_abs_W_diff": float(np.max(np.abs(W_n - s * W_o))),
                    "rel_W_diff": float(np.max(np.abs(W_n - s * W_o)) / np.max(np.abs(W_n))),
                    "er_ours_eps1e-5": g_o["effective_rank"], "er_nctpy_eps1e-5": g_n["effective_rank"],
                    "d_er_eps1e-5": g_n["effective_rank"] - g_o["effective_rank"],
                    "d_er_eps_rescaled": g_n_scaled["effective_rank"] - g_o["effective_rank"],
                    "d_er_nofloor": g_n0["effective_rank"] - g_o0["effective_rank"],
                    "d_pr_eps1e-5": g_n["participation_ratio"] - g_o["participation_ratio"],
                })
eq = pd.DataFrame(rows)
eq.to_csv(OUT_DIR / "nctpy_equivalence_detail.csv", index=False)
print(eq[["max_abs_W_diff", "rel_W_diff", "d_er_eps1e-5", "d_er_eps_rescaled", "d_er_nofloor", "d_pr_eps1e-5"]].abs().max())
print("lambda_max vs rho max |diff|:", (eq["lambda_max"] - eq["rho"]).abs().max(), " scale factor range", eq["scale_factor"].min(), eq["scale_factor"].max())

# Part B: full cohort under nctpy normalization (eps = 1e-5) and ours at c = 1 for comparison
out = []
for scale in SCALES:
    stack = load_stack(scale)
    ranks = load_ranks(scale)
    for si in range(N_SUBJECTS):
        spec = Spectral(stack[si])
        rho = np.abs(np.linalg.eigvals(stack[si])).max()
        kern_ours = spec.kernel(1.0)
        for label, kern in (("nctpy_c1", kern_ours * (rho + 1.0)), ("ours_c1", kern_ours)):
            vals = {}
            for m in ("degree_strength", "cycle_participation"):
                sets = [ranks[(f"sub-{si:03d}", m)][:k] for k in K_VALUES]
                met = metrics_from_eigs(spec.eigvals_batch(kern, sets), 1e-5)
                vals[m] = (met["effective_rank"].mean(), met["participation_ratio"].mean())
            out.append({"scale": scale, "subject": si, "normalization": label,
                        "d_er": vals["cycle_participation"][0] - vals["degree_strength"][0],
                        "d_pr": vals["cycle_participation"][1] - vals["degree_strength"][1]})
coh = pd.DataFrame(out)
summ = []
for (norm, scale), g in coh.groupby(["normalization", "scale"]):
    for metric in ("d_er", "d_pr"):
        summ.append({"normalization": norm, "scale": scale, "metric": metric, **paired_stats(g[metric].to_numpy())})
summ = holm(pd.DataFrame(summ), ["normalization", "metric"])
summ.to_csv(OUT_DIR / "nctpy_cohort_summary.csv", index=False)
print(summ[["normalization", "scale", "metric", "mean_diff", "median_diff", "n_positive", "wilcoxon_p", "rank_biserial"]])
