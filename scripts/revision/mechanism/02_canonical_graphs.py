#!/usr/bin/env python
"""Does hub concentration appear in canonical random-graph models?

Models (N = 68 / 114 / 219, edge density matched to the cohort mean at each scale; 50 graphs each):
  ER    Erdos-Renyi G(N, p)
  BA    Barabasi-Albert, m = round(mean degree / 2)
  WS    Watts-Strogatz, ring lattice of even degree ~ mean degree, rewiring p = 0.1
  SBM   stochastic block model, 7 equal blocks, within-block probability 4x between-block
  RGG   random geometric graph, uniform points in the unit cube, radius set to match density
  CONN  the empirical connectome (one graph per subject; reference)
Weighting: 'binary' (all edge weights 1) and 'weighted' (edge weights drawn at random from the pooled
empirical A0 weights at that scale, then max-normalized). Graphs are regenerated until connected.

Per graph, following the main pipeline (A = A0 - (l1 + c) I; infinite-horizon Gramian; eps = 1e-5):
  network: l1, l2, gap, IPR of v1, cos(s, v1), rho(s_i, m_i) (hub-neighbour loading), strength assortativity
  driver sets: top-k strength (ties broken at random), and 100 uniform random sets, k in {1,2,3,5}
  per set: q1, effective rank, Perron loading L_S, coverage distance (mean over non-driver nodes of the
  distance to the closest driver; hop count for binary graphs, shortest path with length 1/w for weighted)
  at c in {0.01, 0.1, 1}: degree q1 minus random mean, percentile among random sets.
Outputs (runs/<name>/revision/mechanism/; choose the run with ``--run <name>``): canonical_graphs.csv.gz,
canonical_summary.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import bct
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

NULLS_DIR = Path(__file__).resolve().parents[1] / "nulls"  # shared helpers: nulls/nulls_common.py
if str(NULLS_DIR) not in sys.path:
    sys.path.insert(0, str(NULLS_DIR))
from nulls_common import SCALES, Spectral, load_stack, percentile_of, spectrum_metrics  # noqa: E402
from hoc.runs import stage_from_argv  # noqa: E402
from hoc.utils import n_jobs  # noqa: E402

ROOT: Path  # run directory (runs/<name>/), set in main()
OUT: Path  # <run>/revision/mechanism, set in main()
MODELS = ("ER", "BA", "WS", "SBM", "RGG")
WEIGHTINGS = ("binary", "weighted")
N_GRAPHS = 50
N_RANDOM = 100
K_VALUES = (1, 2, 3, 5)
C_VALUES = (0.01, 0.1, 1.0)
SEED = 20261001


def density(scale: int) -> float:
    A = load_stack(scale, ROOT)
    iu = np.triu_indices(scale, 1)
    return float(np.mean([(a[iu] > 0).mean() for a in A]))


def make_graph(model: str, n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    for _ in range(1000):
        seed = int(rng.integers(2**31))
        if model == "ER":
            G = nx.gnp_random_graph(n, p, seed=seed)
        elif model == "BA":
            G = nx.barabasi_albert_graph(n, max(1, round(p * (n - 1) / 2)), seed=seed)
        elif model == "WS":
            kk = int(round(p * (n - 1) / 2)) * 2
            G = nx.watts_strogatz_graph(n, kk, 0.1, seed=seed)
        elif model == "SBM":
            b = 7
            sizes = [n // b + (1 if i < n % b else 0) for i in range(b)]
            # p_in = 4 p_out, overall density p
            w_in = sum(s * (s - 1) / 2 for s in sizes)
            w_out = n * (n - 1) / 2 - w_in
            p_out = p * n * (n - 1) / 2 / (4 * w_in + w_out)
            probs = [[min(1.0, 4 * p_out) if i == j else p_out for j in range(b)] for i in range(b)]
            G = nx.stochastic_block_model(sizes, probs, seed=seed)
        elif model == "RGG":
            pts = np.random.default_rng(seed).random((n, 3))
            d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
            r = np.quantile(d[np.triu_indices(n, 1)], p)
            G = nx.from_numpy_array((d <= r).astype(float) - np.eye(n))
        else:
            raise ValueError(model)
        if nx.is_connected(G):
            return nx.to_numpy_array(G, nodelist=range(n))
    raise RuntimeError(f"could not draw a connected {model}")


def analyze(A: np.ndarray, binary: bool, rng: np.random.Generator) -> tuple[dict, list[dict]]:
    n = A.shape[0]
    spec = Spectral(A)
    v1 = spec.perron
    s = A.sum(axis=0)
    m = (A @ v1) / np.where(s > 0, s, 1)
    net = {"l1": spec.lam[-1], "l2": spec.lam[-2], "gap": spec.gap, "ipr_v1": float(np.sum(v1**4)),
           "cos_s_v1": float(s @ v1 / np.linalg.norm(s)), "rho_s_m": float(spearmanr(s, m).statistic),
           "strength_assort": float(bct.assortativity_wei(A, flag=0)), "strength_cv": float(s.std() / s.mean())}
    L = (A > 0).astype(float) if binary else np.where(A > 0, 1.0 / np.where(A > 0, A, 1), 0.0)
    D = shortest_path(L, method="D", directed=False, unweighted=binary)
    order = np.lexsort((rng.random(n), -s))  # descending strength, random tie-break
    rows = []
    for k in K_VALUES:
        sets = [order[:k]] + [rng.choice(n, k, replace=False) for _ in range(N_RANDOM)]
        cover = np.array([D[np.ix_(np.setdiff1d(np.arange(n), S), S)].min(axis=1).mean() for S in sets])
        loading = np.array([spec.perron_sq[S].sum() for S in sets])
        for c in C_VALUES:
            w, _, _ = spec.set_spectra(spec.kernel(c), sets)
            met = spectrum_metrics(w)
            q1, er = met["q1"], met["er"]
            rows.append({"k": k, "c": c,
                         "deg_q1": q1[0], "rand_q1": q1[1:].mean(), "deg_q1_pct": percentile_of(q1[0], q1[1:]),
                         "deg_er": er[0], "rand_er": er[1:].mean(),
                         "deg_L": loading[0], "rand_L": loading[1:].mean(),
                         "deg_cover": cover[0], "rand_cover": cover[1:].mean(),
                         "rho_cover_q1": float(spearmanr(cover, q1).statistic) if k > 1 else np.nan,
                         "rho_L_q1": float(spearmanr(loading, q1).statistic)})
    return net, rows


def job(model: str, weighting: str, scale: int, idx: int, p: float, weight_pool: np.ndarray) -> list[dict]:
    rng = np.random.default_rng([SEED, MODELS.index(model) if model in MODELS else 9, WEIGHTINGS.index(weighting), scale, idx])
    if model == "CONN":
        A = load_stack(scale, ROOT)[idx]
        if weighting == "binary":
            A = (A > 0).astype(float)
    else:
        A = make_graph(model, scale, p, rng)
        if weighting == "weighted":
            iu = np.triu_indices(scale, 1)
            e = np.flatnonzero(A[iu] > 0)
            W = np.zeros_like(A)
            vals = rng.choice(weight_pool, e.size, replace=True)
            W[iu[0][e], iu[1][e]] = vals / vals.max()
            A = W + W.T
    net, rows = analyze(A, weighting == "binary", rng)
    ids = {"model": model, "weighting": weighting, "scale": scale, "graph": idx}
    return [{**ids, **net, **r} for r in rows]


def main():
    global ROOT, OUT
    ctx = stage_from_argv("revision/mechanism/02_canonical_graphs")
    ctx.require_stage("00_prepare_data")
    ctx.begin()
    ROOT = ctx.root
    OUT = ROOT / "revision" / "mechanism"
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for scale in SCALES:
        p = density(scale)
        stack = load_stack(scale, ROOT)
        pool = np.concatenate([a[np.triu_indices(scale, 1)] for a in stack])
        pool = pool[pool > 0]
        for weighting in WEIGHTINGS:
            for model in MODELS:
                jobs += [(model, weighting, scale, i, p, pool) for i in range(N_GRAPHS)]
            jobs += [("CONN", weighting, scale, i, p, pool) for i in range(stack.shape[0]) if np.all(stack[i].sum(0) > 0)]
    out = Parallel(n_jobs=n_jobs(12))(delayed(job)(*j) for j in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(OUT / "canonical_graphs.csv.gz", index=False)

    df["deg_minus_rand_q1"] = df["deg_q1"] - df["rand_q1"]
    df["deg_minus_rand_er"] = df["deg_er"] - df["rand_er"]
    df["kgrp"] = np.where(df["k"] == 1, "1", "2+")
    summ = (df.groupby(["weighting", "model", "scale", "c", "kgrp"])
              [["gap", "ipr_v1", "cos_s_v1", "rho_s_m", "strength_assort", "strength_cv", "deg_q1", "rand_q1",
                "deg_minus_rand_q1", "deg_q1_pct", "deg_minus_rand_er", "deg_L", "rand_L", "deg_cover", "rand_cover",
                "rho_cover_q1", "rho_L_q1"]].mean().reset_index())
    summ.to_csv(OUT / "canonical_summary.csv", index=False)
    show = summ[(summ["c"] == 0.1) & (summ["kgrp"] == "2+")]
    pd.set_option("display.width", 250)
    for wt in WEIGHTINGS:
        print(f"== {wt}, c = 0.1, k >= 2")
        print(show[show["weighting"] == wt].drop(columns=["weighting", "c", "kgrp"]).round(3).to_string(index=False))
    ctx.finish()


if __name__ == "__main__":
    main()
