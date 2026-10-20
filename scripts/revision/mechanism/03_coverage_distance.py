#!/usr/bin/env python
"""Coverage distance of driver sets on the real connectomes.

For a driver set S, coverage distance = mean over non-driver regions j of min_{s in S} d(j, s): how far,
on average, a region is from its closest driver. Distances: hop count on the binary graph, and weighted
shortest paths with edge length 1/w and -log w. Regions unreachable from every driver (the single isolated
region in one subject) are excluded from the mean.

Sets: all criteria and the 100 primary random sets from results/revision/q1/set_q1.csv.gz (k in {1,2,3,5}).
Outputs:
  coverage_sets.csv.gz        per set: coverage (3 distances), q1, L_S, at c in {0.01, 0.1, 1}
  coverage_by_criterion.csv   criterion minus random mean, percentile, n subjects below random (c = 0.1)
  coverage_rho.csv            within-subject Spearman rho(coverage, q1) and partial rho given L_S, per k and c
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nulls"))
from _common import REPO_ROOT, SCALES, load_stack  # noqa: E402

OUT = REPO_ROOT / "results" / "revision" / "mechanism"
Q1 = REPO_ROOT / "results" / "revision" / "q1" / "set_q1.csv.gz"
DISTS = ("hop", "inv", "log")


def coverage(D: np.ndarray, S: np.ndarray) -> float:
    rest = np.setdiff1d(np.arange(D.shape[0]), S)
    m = D[np.ix_(rest, S)].min(axis=1)
    m = m[np.isfinite(m)]
    return float(m.mean())


def rank_resid(y, covs):
    r = pd.Series(y).rank().to_numpy()
    Z = np.column_stack([np.ones_like(r)] + [pd.Series(c).rank().to_numpy() for c in covs])
    return r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]


def job(scale, si, A, sets):
    with np.errstate(divide="ignore"):
        D = {
            "hop": shortest_path((A > 0).astype(float), method="D", directed=False, unweighted=True),
            "inv": shortest_path(np.where(A > 0, 1.0 / np.where(A > 0, A, 1), 0.0), method="D", directed=False),
            "log": shortest_path(np.where(A > 0, np.maximum(-np.log(np.where(A > 0, A, 1)), 1e-12), 0.0), method="D", directed=False),
        }
    uniq = sets.drop_duplicates(["k", "family", "sample"])[["k", "family", "sample", "nodes"]]
    cov = {d: {} for d in DISTS}
    for _, r in uniq.iterrows():
        S = np.array(r["nodes"].split(";"), dtype=int)
        for d in DISTS:
            cov[d][(r["k"], r["family"], r["sample"])] = coverage(D[d], S)
    out = sets.copy()
    for d in DISTS:
        out[f"cover_{d}"] = [cov[d][(k, f, s)] for k, f, s in zip(out["k"], out["family"], out["sample"])]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Q1, usecols=["subject_id", "scale", "c", "k", "family", "sample", "nodes", "q1", "loading"])
    df = df[~df["family"].str.endswith("matched")]
    frames = []
    for scale in SCALES:
        stack = load_stack(scale)
        sub = df[df["scale"] == scale]
        groups = {sid: g for sid, g in sub.groupby("subject_id")}
        frames += Parallel(n_jobs=12)(delayed(job)(scale, si, stack[si], groups[f"sub-{si:03d}"]) for si in range(stack.shape[0]))
        print(scale, "done", flush=True)
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(OUT / "coverage_sets.csv.gz", index=False)

    # criteria vs random (coverage does not depend on c; use c = 0.1 rows)
    r1 = res[res["c"] == 0.1]
    rows = []
    for (scale, sid, k), g in r1.groupby(["scale", "subject_id", "k"]):
        rnd = g[g["family"] == "random"]
        for _, row in g[g["family"] != "random"].iterrows():
            out = {"scale": scale, "subject_id": sid, "k": k, "family": row["family"], "q1": row["q1"]}
            for d in DISTS:
                ref = rnd[f"cover_{d}"].to_numpy()
                out[f"d_{d}"] = row[f"cover_{d}"] - ref.mean()
                out[f"pct_{d}"] = (np.sum(ref < row[f"cover_{d}"]) + 0.5 * np.sum(ref == row[f"cover_{d}"])) / ref.size
            rows.append(out)
    crit = pd.DataFrame(rows)
    crit["kgrp"] = np.where(crit["k"] == 1, "1", "2+")
    per = crit.groupby(["scale", "subject_id", "family", "kgrp"])[[c for c in crit.columns if c.startswith(("d_", "pct_"))]].mean().reset_index()
    summ = per.groupby(["family", "kgrp", "scale"]).agg(
        **{f"mean_d_{d}": (f"d_{d}", "mean") for d in DISTS},
        **{f"pct_{d}": (f"pct_{d}", "mean") for d in DISTS},
        n_closer_inv=("d_inv", lambda x: int((x < 0).sum())),
    ).reset_index()
    summ.to_csv(OUT / "coverage_by_criterion.csv", index=False)

    # within-subject association with q1 across sets (k >= 2 and k = 1), raw and given L_S
    rows = []
    for (scale, sid, c, k), g in res.groupby(["scale", "subject_id", "c", "k"]):
        q = g["q1"].to_numpy()
        L = g["loading"].to_numpy()
        rq, rqL = pd.Series(q).rank().to_numpy(), rank_resid(q, [L])
        out = {"scale": scale, "subject_id": sid, "c": c, "k": k}
        for d in DISTS:
            x = g[f"cover_{d}"].to_numpy()
            out[f"rho_{d}"] = np.corrcoef(pd.Series(x).rank().to_numpy(), rq)[0, 1]
            out[f"prho_{d}|L"] = np.corrcoef(rank_resid(x, [L]), rqL)[0, 1]
        rows.append(out)
    rho = pd.DataFrame(rows)
    rho.to_csv(OUT / "coverage_rho.csv", index=False)

    pd.set_option("display.width", 250)
    show = summ[(summ["kgrp"] == "2+") & summ["family"].isin(["degree_strength", "eigenvector_centrality", "top_avg_ctrl", "betweenness",
                                                              "participation_coefficient", "cycle_participation", "cycle_resid", "top_modal_gu"])]
    print(show.pivot_table(index="family", columns="scale", values=["mean_d_hop", "mean_d_inv", "pct_inv", "n_closer_inv"]).round(3).to_string())
    rk = rho[rho["k"] > 1].groupby(["scale", "subject_id", "c"]).mean(numeric_only=True).reset_index()
    print(rk.groupby(["scale", "c"])[[c for c in rk.columns if c.startswith(("rho_", "prho_"))]].median().round(3).to_string())


if __name__ == "__main__":
    main()
