#!/usr/bin/env python
"""Perron loading vs effective rank at c = 0.01 and c = 0.1 (eps = 1e-5).

Sets: static top-k of the 9 saved rankings + the 100 primary random sets (same seeds).
For each set: ER, PR, Perron loading sum_{i in S} v_i^2 (v = leading eigvec of A0), and the
Perron share of the Gramian, v^T W v / tr W (fraction of controllability variance along the
slowest mode). Writes runs/<name>/revision/spectra/perron_c_check.csv and tables/t4b_perron_by_c.csv
(choose the run with ``--run <name>``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from hoc.config import load_config  # noqa: E402
from hoc.control import assemble_gramian, gramian_metrics, single_node_gramians  # noqa: E402
from hoc.preprocessing import stabilize_adjacency  # noqa: E402
from hoc.rankings import random_top_k_nodes, top_k_nodes  # noqa: E402
from hoc.runs import stage_from_argv  # noqa: E402
from hoc.utils import make_rng, n_jobs, subject_seed  # noqa: E402

STAGE = "revision/spectra/04_perron_c_check"
OUT: Path  # set in main(): <run>/revision/spectra
CFG = load_config(REPO_ROOT / "config/default.toml")
SEED = int(CFG["analysis"]["random_seed"])
EPS = float(CFG["control"]["primary_epsilon"])
K_VALUES = [int(k) for k in CFG["analysis"]["top_k_values"]]
C_VALUES = [0.01, 0.1]
METHODS = [
    "degree_strength", "eigenvector_centrality", "betweenness", "cycle_participation",
    "participation_coefficient", "triangle_participation", "tetrahedron_participation",
    "degree_normalized_cycle_participation", "cycle_criticality",
]


def run(scale, subj, adjacency, scores):
    n = adjacency.shape[0]
    v = np.linalg.eigh(adjacency)[1][:, -1]
    rows = []
    for c in C_VALUES:
        grams = single_node_gramians(stabilize_adjacency(adjacency, c)[0])
        for k in K_VALUES:
            sets = [(m, -1, top_k_nodes(s, k)) for m, s in scores.items()]
            sets += [("random", r, random_top_k_nodes(n, k, make_rng(subject_seed(SEED, scale, subj, k, r)))) for r in range(100)]
            for fam, r, nodes in sets:
                W = assemble_gramian(grams, nodes)
                met = gramian_metrics(W, epsilon=EPS)
                rows.append({
                    "c": c, "scale": scale, "subject_id": f"sub-{subj:03d}", "k": k, "family": fam, "sample": r,
                    "effective_rank": met["effective_rank"], "participation_ratio": met["participation_ratio"],
                    "perron_loading": float((v[nodes] ** 2).sum()),
                    "perron_share_W": float(v @ W @ v / np.trace(W)),
                })
    return rows


def main():
    global OUT
    ctx = stage_from_argv(STAGE)
    ctx.require_stage("00_prepare_data")
    ctx.require_stage("01_primary")
    ctx.begin()
    OUT = ctx.root / "revision" / "spectra"
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    ns = pd.read_csv(ctx.root / "primary/node_scores.csv", usecols=["subject_id", "scale", "node_index", "method", "score"])
    rows = []
    for scale in (68, 114, 219):
        stack = np.load(ctx.root / f"processed/connectomes_scale-{scale}.npz")["adjacency"]
        sub = ns[ns.scale == scale]
        maps = {}
        for (sid, m), g in sub.groupby(["subject_id", "method"]):
            maps.setdefault(sid, {})[m] = g.sort_values("node_index")["score"].to_numpy()
        res = Parallel(n_jobs=n_jobs(12))(delayed(run)(scale, s, stack[s], {m: maps[f"sub-{s:03d}"][m] for m in METHODS}) for s in range(stack.shape[0]))
        for r in res:
            rows.extend(r)
        print("done", scale, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "perron_c_check.csv", index=False)

    out = []
    for (c, sc), g in df.groupby(["c", "scale"]):
        rk = g[g["sample"] == -1]
        for x in ("perron_loading", "perron_share_W"):
            rec = {"c": c, "scale": sc, "x": x,
                   "rho_all_sets_all_k": spearmanr(g[x], g.effective_rank).statistic,
                   "rho_rankings_only_all_k": spearmanr(rk[x], rk.effective_rank).statistic}
            for k in K_VALUES:
                gk = g[g.k == k]
                rec[f"rho_all_k{k}"] = spearmanr(gk[x], gk.effective_rank).statistic
                rec[f"rho_within_subj_k{k}"] = np.nanmean([spearmanr(h[x], h.effective_rank).statistic for _, h in gk.groupby("subject_id")])
                rec[f"rho_rankings_only_k{k}"] = spearmanr(rk[rk.k == k][x], rk[rk.k == k].effective_rank).statistic
            out.append(rec)
    t = pd.DataFrame(out)
    t.to_csv(OUT / "tables" / "t4b_perron_by_c.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string())
    # binned relationship (all sets, k = 3)
    for c in C_VALUES:
        g = df[(df.c == c) & (df.k == 3)].copy()
        g["bin"] = pd.qcut(g.perron_loading, 10, labels=False, duplicates="drop")
        print("c =", c, "k=3 ER by Perron-loading decile (pooled scales):")
        print(g.groupby("bin").agg(loading=("perron_loading", "median"), er=("effective_rank", "mean")).round(3).T.to_string())
    m = df.groupby(["c", "scale", "family"])[["effective_rank", "perron_loading", "perron_share_W"]].mean().round(3)
    m.to_csv(OUT / "tables" / "t4b_perron_by_c_family_means.csv")
    print(m.loc[(slice(None), slice(None), ["degree_strength", "cycle_participation", "random"]), :].to_string())
    ctx.finish()


if __name__ == "__main__":
    main()
