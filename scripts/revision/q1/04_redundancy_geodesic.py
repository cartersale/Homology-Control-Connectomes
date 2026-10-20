#!/usr/bin/env python
"""Is driver redundancy adjacency, or shared access to the slow mode?

For every real connectome, c in {0.01, 0.1, 1} and k in {2, 3, 5}, over the driver sets in
set_q1.csv.gz (all named criteria + the 100 primary random sets; matched sets excluded), compute:
  geo_inv   mean pairwise weighted shortest-path distance between drivers, edge length 1/w
  geo_log   same with edge length -log w (w = 1 gets length 1e-12 so it stays an edge)
  comm      driver-Gram communicability: mean pairwise cosine G_ij / sqrt(G_ii G_jj),
            G_S = B'(-2A)^{-1}B (c-dependent; the slow-mode profile overlap of the drivers)
Within each subject x scale x c x k, Spearman rho of each predictor with q1 across sets, and partial
Spearman rho given L_S (and given L_S plus the other predictor). Ranks are residualized on the
covariates' ranks (OLS) and the residuals correlated. Per subject the rho's are averaged over k.
Head-to-head: paired Wilcoxon across subjects of |rho_comm| - |rho_geo| (partial given L_S).

Outputs: results/revision/q1/redundancy_rho.csv (per subject x scale x c x k), redundancy_summary.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse.csgraph import shortest_path
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nulls"))
from _common import REPO_ROOT, Spectral, load_stack  # noqa: E402

OUT = REPO_ROOT / "results" / "revision" / "q1"
SCALES = (68, 114, 219)
C_VALUES = (0.01, 0.1, 1.0)
K_VALUES = (2, 3, 5)
PREDICTORS = ("geo_inv", "geo_log", "comm")


def mean_pairwise(M: np.ndarray, nodes: np.ndarray) -> float:
    sub = M[np.ix_(nodes, nodes)]
    iu = np.triu_indices(len(nodes), 1)
    vals = sub[iu]
    return float(vals.mean()) if np.all(np.isfinite(vals)) else np.nan


def rank_resid(y: np.ndarray, covs: list[np.ndarray]) -> np.ndarray:
    r = pd.Series(y).rank().to_numpy()
    Z = np.column_stack([np.ones_like(r)] + [pd.Series(c).rank().to_numpy() for c in covs])
    return r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]


def corr(a, b) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def job(scale: int, si: int, A: np.ndarray, sets: pd.DataFrame) -> list[dict]:
    spec = Spectral(A)
    w = A.copy()
    with np.errstate(divide="ignore"):
        L_inv = np.where(w > 0, 1.0 / w, 0.0)
        L_log = np.where(w > 0, np.maximum(-np.log(w), 1e-12), 0.0)
    D_inv = shortest_path(L_inv, method="D", directed=False)
    D_log = shortest_path(L_log, method="D", directed=False)
    rows = []
    for c in C_VALUES:
        mu = spec.lam - spec.lam_max - c
        R = (spec.U / (-2.0 * mu)) @ spec.U.T  # (-2A)^{-1}
        d = np.sqrt(np.diag(R))
        cos = R / np.outer(d, d)
        for k in K_VALUES:
            g = sets[(sets["c"] == c) & (sets["k"] == k)]
            node_lists = [np.array(n.split(";"), dtype=int) for n in g["nodes"]]
            x = {
                "geo_inv": np.array([mean_pairwise(D_inv, n) for n in node_lists]),
                "geo_log": np.array([mean_pairwise(D_log, n) for n in node_lists]),
                "comm": np.array([mean_pairwise(cos, n) for n in node_lists]),
            }
            q1 = g["q1"].to_numpy()
            L = g["loading"].to_numpy()
            ok = np.all([np.isfinite(v) for v in x.values()], axis=0)
            row = {"scale": scale, "subject_id": f"sub-{si:03d}", "c": c, "k": k, "n_sets": int(ok.sum())}
            rq = pd.Series(q1[ok]).rank().to_numpy()
            rq_L = rank_resid(q1[ok], [L[ok]])
            row["rho_L"] = corr(pd.Series(L[ok]).rank().to_numpy(), rq)
            for p in PREDICTORS:
                xp = x[p][ok]
                row[f"rho_{p}"] = corr(pd.Series(xp).rank().to_numpy(), rq)
                row[f"prho_{p}|L"] = corr(rank_resid(xp, [L[ok]]), rq_L)
            for geo in ("geo_inv", "geo_log"):
                row[f"prho_comm|L,{geo}"] = corr(rank_resid(x["comm"][ok], [L[ok], x[geo][ok]]), rank_resid(q1[ok], [L[ok], x[geo][ok]]))
                row[f"prho_{geo}|L,comm"] = corr(rank_resid(x[geo][ok], [L[ok], x["comm"][ok]]), rank_resid(q1[ok], [L[ok], x["comm"][ok]]))
            row["rho_geo_inv_comm"] = corr(pd.Series(x["geo_inv"][ok]).rank().to_numpy(), pd.Series(x["comm"][ok]).rank().to_numpy())
            rows.append(row)
    return rows


def main():
    df = pd.read_csv(OUT / "set_q1.csv.gz", usecols=["subject_id", "scale", "c", "k", "family", "nodes", "q1", "loading"])
    df = df[~df["family"].str.endswith("matched") & df["k"].isin(K_VALUES)]
    rows = []
    for scale in SCALES:
        stack = load_stack(scale)
        sub = df[df["scale"] == scale]
        groups = {sid: g for sid, g in sub.groupby("subject_id")}
        out = Parallel(n_jobs=12)(delayed(job)(scale, si, stack[si], groups[f"sub-{si:03d}"]) for si in range(stack.shape[0]))
        rows += [r for o in out for r in o]
        print(scale, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "redundancy_rho.csv", index=False)

    # per subject: average over k, then summarize across subjects
    cols = [c for c in res.columns if c.startswith(("rho", "prho"))]
    per = res.groupby(["scale", "subject_id", "c"])[cols].mean().reset_index()
    summ = []
    for (scale, c), g in per.groupby(["scale", "c"]):
        row = {"scale": scale, "c": c}
        for col in cols:
            row[col] = g[col].median()
        for geo in ("geo_inv", "geo_log"):
            for tag, a, b in (("raw", f"rho_comm", f"rho_{geo}"), ("given_L", f"prho_comm|L", f"prho_{geo}|L")):
                diff = g[a].abs() - g[b].abs()
                row[f"comm_minus_{geo}_{tag}"] = diff.median()
                row[f"n_comm_better_{geo}_{tag}"] = int((diff > 0).sum())
                row[f"p_comm_vs_{geo}_{tag}"] = wilcoxon(diff).pvalue
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "redundancy_summary.csv", index=False)
    show = ["scale", "c", "rho_L", "rho_geo_inv", "rho_geo_log", "rho_comm", "prho_geo_inv|L", "prho_geo_log|L", "prho_comm|L",
            "prho_comm|L,geo_inv", "prho_geo_inv|L,comm", "comm_minus_geo_inv_given_L", "n_comm_better_geo_inv_given_L", "p_comm_vs_geo_inv_given_L", "rho_geo_inv_comm"]
    print(summ[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
