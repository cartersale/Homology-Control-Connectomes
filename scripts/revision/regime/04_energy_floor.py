"""Task 6: energy floor (Reviewer 2 Major 4) at c = 0.1, infinite horizon.

Part A (driver sets): degree / cycle / eigenvector / betweenness top-k and the 100 random sets, k in {1,2,3,5},
  eps in {1e-8, 1e-5, 1e-2}: fraction of eigenvalues at the floor (pipeline definition: eig(W) <= ~1e-8 eps,
  i.e. numerically zero), fraction of eig(W) < eps, log10 E_avg vs log10(N/eps), cycle - degree
  Delta log10 E_avg (k-averaged; Holm family 3 scales x 3 eps).
Part B (landscape): the primary run's saved candidate sets (results/primary/landscape_samples, 5000 per
  subject/scale/k, or exhaustive when C(N,k) <= 5000). log10 E_avg per set at each eps; variance (ddof=1) and
  IQR across sets; same for log10 of the non-floor part (sum of 1/lambda over non-floored eigenvalues) and the
  above-eps part (sum over eig(W) > eps of 1/(lambda+eps)). Also checks recomputed eps=1e-5 energies against the
  saved average_energies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import (K_VALUES, N_SUBJECTS, OUT_DIR, REPO_ROOT, SCALES, Spectral, holm, load_ranks, load_stack,
                     metrics_from_eigs, paired_stats, random_sets, run_parallel)

C = 0.1
EPS = (1e-8, 1e-5, 1e-2)
METHODS_A = ("degree_strength", "cycle_participation", "eigenvector_centrality", "betweenness")
SAMPLES = REPO_ROOT / "results" / "primary" / "landscape_samples"


def iqr(x):
    q = np.nanquantile(x, [0.25, 0.75])
    return q[1] - q[0]


def job(scale, si, ranks, adjacency):
    spec = Spectral(adjacency)
    kern = spec.kernel(C)
    sid = f"sub-{si:03d}"
    rows_a, rows_b = [], []
    spectra = {}
    for k in K_VALUES:
        sets = [ranks[(sid, m)][:k] for m in METHODS_A] + random_sets(scale, si, k)
        labels = list(METHODS_A) + ["random"] * 100
        w = spec.eigvals_batch(kern, sets)
        if k == 5:
            spectra = {m: w[j] for j, m in enumerate(METHODS_A)}
            spectra["random0"] = w[len(METHODS_A)]
        for eps in EPS:
            met = metrics_from_eigs(w, eps)
            for j, m in enumerate(METHODS_A):
                rows_a.append({"subject_id": sid, "scale": scale, "k": k, "eps": eps, "method": m,
                               **{key: met[key][j] for key in ("fraction_at_floor", "fraction_below_eps", "log10_average_energy",
                                                               "log10_energy_nonfloor", "log10_energy_above_eps", "n_above_eps",
                                                               "effective_rank", "log10_lambda_max")}})
            nm = len(METHODS_A)
            rows_a.append({"subject_id": sid, "scale": scale, "k": k, "eps": eps, "method": "random_mean",
                           **{key: np.nanmean(met[key][nm:]) for key in ("fraction_at_floor", "fraction_below_eps", "log10_average_energy",
                                                                         "log10_energy_nonfloor", "log10_energy_above_eps", "n_above_eps",
                                                                         "effective_rank", "log10_lambda_max")}})
        # landscape
        data = np.load(SAMPLES / f"{sid}_scale-{scale}_k-{k}.npz")
        cand = data["candidate_sets"]
        saved = data["average_energies"]
        wl = spec.eigvals_batch(kern, list(cand))
        for eps in EPS:
            met = metrics_from_eigs(wl, eps)
            le = met["log10_average_energy"]
            nf = met["log10_energy_nonfloor"]
            ab = met["log10_energy_above_eps"]
            best = np.min(10**le)
            rows_b.append({"subject_id": sid, "scale": scale, "k": k, "eps": eps, "n_sets": len(cand),
                           "var_log10E": np.var(le, ddof=1), "iqr_log10E": iqr(le), "range_log10E": le.max() - le.min(),
                           "median_log10E": np.median(le), "log10_N_over_eps": np.log10(scale / eps),
                           "near_opt_01pct": np.mean(10**le <= 1.01 * best),
                           "var_log10E_nonfloor": np.nanvar(nf, ddof=1), "iqr_log10E_nonfloor": iqr(nf),
                           "var_log10E_above_eps": np.nanvar(ab, ddof=1), "iqr_log10E_above_eps": iqr(ab),
                           "frac_sets_no_above_eps": float(np.mean(met["n_above_eps"] == 0)),
                           "mean_fraction_at_floor": met["fraction_at_floor"].mean(),
                           "mean_fraction_below_eps": met["fraction_below_eps"].mean(),
                           "max_abs_dlog10E_vs_saved": float(np.max(np.abs(le - np.log10(saved)))) if eps == 1e-5 else np.nan})
    return rows_a, rows_b, spectra


if __name__ == "__main__":
    A, B, SPEC = [], [], {}
    for scale in SCALES:
        stack = load_stack(scale)
        ranks = load_ranks(scale)
        res = run_parallel(job, [(scale, i, {key: v for key, v in ranks.items() if key[0] == f"sub-{i:03d}"}, stack[i]) for i in range(N_SUBJECTS)], n_jobs=8)
        for a, b, sp in res:
            A.extend(a)
            B.extend(b)
        for m in res[0][2]:
            SPEC[f"scale{scale}_{m}"] = np.stack([r[2][m] for r in res])
        print("done", scale, flush=True)
    a = pd.DataFrame(A)
    a.to_csv(OUT_DIR / "task6_driver_sets.csv", index=False)
    b = pd.DataFrame(B)
    b.to_csv(OUT_DIR / "task6_landscape.csv", index=False)
    np.savez_compressed(OUT_DIR / "task6_spectra_k5_c0.1.npz", **SPEC)
    # paired Delta log10 E cycle - degree
    sub = a[a["method"].isin(["degree_strength", "cycle_participation"])].groupby(["scale", "eps", "subject_id", "method"], as_index=False)["log10_average_energy"].mean()
    wide = sub.pivot_table(index=["scale", "eps", "subject_id"], columns="method", values="log10_average_energy").reset_index()
    wide["d"] = wide["cycle_participation"] - wide["degree_strength"]
    rows = [{"scale": s, "eps": e, **paired_stats(g["d"].to_numpy())} for (s, e), g in wide.groupby(["scale", "eps"])]
    holm(pd.DataFrame(rows), None).to_csv(OUT_DIR / "task6_delta_log10E_summary.csv", index=False)
    print("max |log10E - saved| at eps=1e-5:", b["max_abs_dlog10E_vs_saved"].max())
