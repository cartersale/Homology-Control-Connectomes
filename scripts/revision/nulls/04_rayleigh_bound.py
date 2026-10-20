#!/usr/bin/env python
"""Exact accounting of the real-vs-null lambda1 excess by a strength-based Rayleigh bound.

For symmetric nonnegative W with strengths s = W1, the Rayleigh quotient with test vector s gives
    lambda1 >= R1 = s'Ws / s's = (1'W^3 1) / (1'W^2 1).
Under exact strength preservation, s's is fixed and the edge-weighted strength assortativity r_W
is affine in s'Ws, so R1 is the part of lambda1 that strength-to-strong wiring guarantees.
Power-iteration quotients R_m = 1'W^(2m+1)1 / 1'W^(2m)1 increase to lambda1 (R0 = mean strength).

Per network: lambda1, lambda2, gap, R0..R3, cos(s, v1), r_W (edge-weighted), r_bct (bct, binary
edges), s's, s'Ws. Per subject: real minus null-mean deltas, share of the lambda1 and gap excess
carried by R1, and within-subject Spearman rho across the 50 nulls.

Outputs (runs/<name>/revision/nulls/rayleigh/; choose the run with ``--run <name>``): networks.csv.gz, per_subject.csv, summary.csv
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import bct
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr, pearsonr

from nulls_common import N_SUBJECTS, NULL_TYPES, SCALES, load_stack, out_dir
from hoc.runs import stage_from_argv
from hoc.utils import n_jobs

ROOT: Path  # run directory (runs/<name>/), set in main()
OUT_DIR: Path  # <run>/revision/nulls, set in main()
DEST: Path  # OUT_DIR / "rayleigh", set in main()
N_QUOTIENTS = 4  # R0..R3


def edge_weighted_assortativity(A: np.ndarray, s: np.ndarray) -> float:
    i, j = np.triu_indices_from(A, 1)
    w = A[i, j]
    keep = w > 0
    i, j, w = i[keep], j[keep], w[keep]
    H = w.sum()
    si, sj = s[i], s[j]
    t1 = np.sum(w * si * sj) / H
    t2 = (np.sum(w * 0.5 * (si + sj)) / H) ** 2
    t3 = np.sum(w * 0.5 * (si**2 + sj**2)) / H
    return float((t1 - t2) / (t3 - t2))


def stats(A: np.ndarray) -> dict:
    evals, evecs = np.linalg.eigh(A)
    l1, l2 = evals[-1], evals[-2]
    v1 = np.abs(evecs[:, -1])
    s = A.sum(axis=1)
    walks = [float(A.shape[0])]  # 1'W^k 1, k = 0..2*N_QUOTIENTS-1
    x = np.ones(A.shape[0])
    for _ in range(2 * N_QUOTIENTS - 1):
        x = A @ x
        walks.append(float(x.sum()))
    out = {"lambda1": l1, "lambda2": l2, "gap": l1 - l2,
           "cos_s_v1": float(s @ v1 / np.linalg.norm(s)),
           "sTs": float(s @ s), "sTWs": float(s @ A @ s),
           "r_W": edge_weighted_assortativity(A, s),
           "r_bct": float(bct.assortativity_wei(A, flag=0))}
    for m in range(N_QUOTIENTS):
        out[f"R{m}"] = walks[2 * m + 1] / walks[2 * m]
    return out


def job(null: str, scale: int, subject: int) -> list[dict]:
    ids = {"null": null, "scale": scale, "subject": subject}
    if null == "real":
        return [{**ids, "null_index": -1, **stats(load_stack(scale, ROOT)[subject])}]
    triu = np.load(OUT_DIR / "networks" / null / f"scale-{scale}" / f"sub-{subject:03d}.npz")["triu"]
    iu = np.triu_indices(scale, 1)
    rows = []
    for k, vec in enumerate(triu):
        A = np.zeros((scale, scale))
        A[iu] = vec
        rows.append({**ids, "null_index": k, **stats(A + A.T)})
    return rows


def per_subject(df: pd.DataFrame) -> pd.DataFrame:
    real = df[df.null == "real"].set_index(["scale", "subject"])
    rows = []
    for (null, scale, subject), g in df[df.null != "real"].groupby(["null", "scale", "subject"]):
        r = real.loc[(scale, subject)]
        d = {c: r[c] - g[c].mean() for c in ["lambda1", "lambda2", "gap", "R0", "R1", "R2", "R3", "r_W", "r_bct", "cos_s_v1"]}
        row = {"null": null, "scale": scale, "subject": subject,
               **{f"d_{c}": v for c, v in d.items()},
               "share_l1_R0": d["R0"] / d["lambda1"],
               "share_l1_R1": d["R1"] / d["lambda1"],
               "share_gap_R1": d["R1"] / d["gap"],
               "slack_real": r["lambda1"] - r["R1"], "slack_null": (g["lambda1"] - g["R1"]).mean(),
               "rho_R1_l1": spearmanr(g["R1"], g["lambda1"]).statistic,
               "rho_rW_l1": spearmanr(g["r_W"], g["lambda1"]).statistic,
               "rho_rbct_l1": spearmanr(g["r_bct"], g["lambda1"]).statistic,
               "rho_rW_gap": spearmanr(g["r_W"], g["gap"]).statistic,
               "pearson_rW_sTWs": pearsonr(g["r_W"], g["sTWs"]).statistic,
               "sd_sTs_rel": g["sTs"].std() / g["sTs"].mean()}
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    global ROOT, OUT_DIR, DEST
    ctx = stage_from_argv("revision/nulls/04_rayleigh_bound")
    ctx.require_stage("00_prepare_data")
    ctx.require_stage("revision/nulls/01_run_nulls")
    ctx.begin()
    ROOT = ctx.root
    OUT_DIR = out_dir(ROOT)
    DEST = OUT_DIR / "rayleigh"
    DEST.mkdir(parents=True, exist_ok=True)
    tasks = [(n, sc, s) for sc in SCALES for s in range(N_SUBJECTS) for n in ("real",) + NULL_TYPES]
    rows = Parallel(n_jobs=n_jobs(-1), verbose=0)(delayed(job)(*t) for t in tasks)
    df = pd.DataFrame([r for chunk in rows for r in chunk])
    df.to_csv(DEST / "networks.csv.gz", index=False)
    ps = per_subject(df)
    ps.to_csv(DEST / "per_subject.csv", index=False)
    summary = ps.groupby(["null", "scale"]).median(numeric_only=True).drop(columns="subject")
    summary.to_csv(DEST / "summary.csv")
    means = df.groupby(["null", "scale"])[["lambda1", "R0", "R1", "R2", "R3", "gap", "r_W", "r_bct", "cos_s_v1"]].mean()
    pd.set_option("display.width", 200, "display.precision", 3)
    print(means, "\n")
    print(summary[["d_lambda1", "d_R0", "d_R1", "share_l1_R1", "share_gap_R1", "slack_real", "slack_null",
                   "rho_R1_l1", "rho_rW_l1", "rho_rbct_l1", "rho_rW_gap", "pearson_rW_sTWs", "sd_sTs_rel"]].T)
    ctx.finish()


if __name__ == "__main__":
    main()
