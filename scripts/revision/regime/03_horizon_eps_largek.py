"""Tasks 4 + 5: finite-horizon grid (corrects Table S2), epsilon sweep (Table S1) and larger k (Table S3).

Task 4: T in {1, 2.5, 5, inf} x c in {0.01, 0.1, 1.0} x eps in {1e-5, 1e-8}; degree vs cycle static top-k,
        subject difference = mean over k in {1,2,3,5} of (cycle - degree). Holm family: all T x c x scale cells
        within each eps and metric (36 tests).
Task 5 S1: c = 0.1 (and c = 0.01 for reproduction of the published table), infinite horizon,
        eps in {1e-8..1e-2}; k-averaged ER / PR differences. Holm family: eps x scale within c and metric (21).
Task 5 S3: c = 0.1 (and 0.01), eps = 1e-5, infinite horizon, per-k ER / PR differences (no k averaging).
        Holm family: all k x scale cells within c and metric (23).
Outputs go to runs/<name>/revision/regime/ (choose the run with ``--run <name>``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from regime_common import (K_VALUES, LARGE_K, N_SUBJECTS, SCALES, Spectral, holm, load_ranks, load_stack,
                           metrics_from_eigs, out_dir, paired_stats, run_parallel)
from hoc.runs import stage_from_argv
from hoc.utils import n_jobs

HORIZONS = (1.0, 2.5, 5.0, np.inf)
C_FH = (0.01, 0.1, 1.0)
EPS_FH = (1e-5, 1e-8)
EPS_S1 = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
C_S = (0.1, 0.01)


def job(scale, si, ranks, adjacency):
    spec = Spectral(adjacency)
    sid = f"sub-{si:03d}"
    rows = []
    for c in sorted(set(C_FH) | set(C_S)):
        for T in HORIZONS:
            if np.isfinite(T) and c not in C_FH:
                continue
            kern = spec.kernel(c, None if not np.isfinite(T) else T)
            ks = sorted(set(K_VALUES) | set(LARGE_K[scale])) if not np.isfinite(T) else K_VALUES
            for m in ("degree_strength", "cycle_participation"):
                w = spec.eigvals_batch(kern, [ranks[(sid, m)][:k] for k in ks])
                for eps in (EPS_S1 if not np.isfinite(T) else EPS_FH):
                    met = metrics_from_eigs(w, eps)
                    for j, k in enumerate(ks):
                        rows.append({"subject_id": sid, "scale": scale, "c": c, "T": T, "eps": eps, "k": k, "method": m,
                                     "effective_rank": met["effective_rank"][j], "participation_ratio": met["participation_ratio"][j],
                                     "log10_average_energy": met["log10_average_energy"][j],
                                     "fraction_at_floor": met["fraction_at_floor"][j]})
    return rows


def paired(frame, keys):
    wide = frame.pivot_table(index=keys + ["subject_id"], columns="method", values=["effective_rank", "participation_ratio", "log10_average_energy"])
    out = pd.DataFrame(index=wide.index)
    for metric in ("effective_rank", "participation_ratio", "log10_average_energy"):
        out[metric] = wide[(metric, "cycle_participation")] - wide[(metric, "degree_strength")]
    return out.reset_index()


def summarize(diff, keys, holm_family):
    rows = []
    for key, g in diff.groupby(keys):
        for metric in ("effective_rank", "participation_ratio", "log10_average_energy"):
            rows.append({**dict(zip(keys, key)), "metric": metric, **paired_stats(g[metric].to_numpy())})
    return holm(pd.DataFrame(rows), holm_family + ["metric"])


if __name__ == "__main__":
    ctx = stage_from_argv("revision/regime/03_horizon_eps_largek")
    ctx.require_stage("00_prepare_data")
    ctx.require_stage("01_primary")
    ctx.begin()
    ROOT = ctx.root
    OUT_DIR = out_dir(ROOT)
    allrows = []
    for scale in SCALES:
        stack = load_stack(scale, ROOT)
        ranks = load_ranks(scale, ROOT)
        res = run_parallel(job, [(scale, i, {key: v for key, v in ranks.items() if key[0] == f"sub-{i:03d}"}, stack[i]) for i in range(N_SUBJECTS)], n_jobs=n_jobs(8))
        for r in res:
            allrows.extend(r)
    tidy = pd.DataFrame(allrows)
    tidy.to_csv(OUT_DIR / "horizon_eps_largek_tidy.csv.gz", index=False)

    # Task 4
    fh = tidy[(tidy["c"].isin(C_FH)) & (tidy["eps"].isin(EPS_FH)) & (tidy["k"].isin(K_VALUES))]
    fh_sub = fh.groupby(["scale", "c", "T", "eps", "subject_id", "method"], as_index=False)[["effective_rank", "participation_ratio", "log10_average_energy", "fraction_at_floor"]].mean()
    d4 = paired(fh_sub, ["scale", "c", "T", "eps"])
    s4 = summarize(d4, ["eps", "c", "T", "scale"], ["eps"])
    s4.to_csv(OUT_DIR / "task4_finite_horizon_summary.csv", index=False)

    # Task 5 S1
    s1 = tidy[(tidy["c"].isin(C_S)) & (~np.isfinite(tidy["T"])) & (tidy["k"].isin(K_VALUES))]
    s1_sub = s1.groupby(["scale", "c", "eps", "subject_id", "method"], as_index=False)[["effective_rank", "participation_ratio", "log10_average_energy"]].mean()
    d1 = paired(s1_sub, ["scale", "c", "eps"])
    summarize(d1, ["c", "eps", "scale"], ["c"]).to_csv(OUT_DIR / "task5_S1_epsilon_summary.csv", index=False)

    # Task 5 S3
    s3 = tidy[(tidy["c"].isin(C_S)) & (~np.isfinite(tidy["T"])) & (tidy["eps"] == 1e-5)]
    d3 = paired(s3, ["scale", "c", "k"])
    summarize(d3, ["c", "scale", "k"], ["c"]).to_csv(OUT_DIR / "task5_S3_largek_summary.csv", index=False)
    print("done")
    ctx.finish()
