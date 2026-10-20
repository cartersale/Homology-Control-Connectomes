#!/usr/bin/env python
"""Validate the null-model machinery before the full run.

1. Fast eigenbasis Gramian == hoc.control (eigenvalues, ER, q1, Perron share, W_perp).
2. The real-network pass reproduces the saved primary numbers (spectra/set_metrics.csv:
   degree-set ER and top-1 share, random-set mean ER) at c = 0.1.
3. Each null preserves what it claims to preserve (subject 0, every scale), with timings.
Writes runs/<name>/revision/nulls/validation.json (choose the run with ``--run <name>``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nulls_common import (
    EPSILON,
    NULL_TYPES,
    SCALES,
    Spectral,
    analyze_network,
    centroid_distances,
    load_stack,
    make_null,
    null_seed,
    out_dir,
    random_driver_sets,
    rich_club_level,
    spectrum_metrics,
)
from hoc.control import assemble_gramian, gramian_metrics, single_node_gramians
from hoc.preprocessing import stabilize_adjacency
from hoc.runs import stage_from_argv

ROOT: Path  # run directory (runs/<name>/), set in main()
OUT_DIR: Path  # <run>/revision/nulls, set in main()
from hoc.rankings import top_k_nodes


def check_fast_gramian(adjacency: np.ndarray, c: float = 0.1) -> dict:
    spec = Spectral(adjacency)
    system, _ = stabilize_adjacency(adjacency, c)
    grams = single_node_gramians(system)
    rng = np.random.default_rng(0)
    sets = [top_k_nodes(adjacency.sum(0), 3)] + [rng.choice(adjacency.shape[0], 5, replace=False) for _ in range(4)]
    w, w_perp, pshare = spec.set_spectra(spec.kernel(c), sets)
    fast = spectrum_metrics(w)
    worst = {"eig_rel": 0.0, "er": 0.0, "q1": 0.0, "perron_share": 0.0, "q1perp": 0.0}
    v1 = spec.U[:, -1]
    projector = np.eye(len(v1)) - np.outer(v1, v1)
    for j, s in enumerate(sets):
        W = assemble_gramian(grams, s)
        ref = gramian_metrics(W, epsilon=EPSILON)
        ev = np.linalg.eigvalsh(W)
        worst["eig_rel"] = max(worst["eig_rel"], float(np.max(np.abs(ev - w[j])) / ev.max()))
        worst["er"] = max(worst["er"], abs(ref["effective_rank"] - fast["er"][j]))
        worst["q1"] = max(worst["q1"], abs(ev[-1] / ev.sum() - fast["q1"][j]))
        worst["perron_share"] = max(worst["perron_share"], abs(v1 @ W @ v1 / np.trace(W) - pshare[j]))
        ev_perp = np.linalg.eigvalsh(projector @ W @ projector)
        q1_perp = ev_perp[-1] / ev_perp.sum()
        worst["q1perp"] = max(worst["q1perp"], abs(q1_perp - spectrum_metrics(w_perp[j])["q1"][0]))
    return worst


def check_primary(scale: int, n_subjects: int = 3) -> dict:
    ref = pd.read_csv(ROOT / "revision" / "spectra" / "set_metrics.csv", usecols=["subject_id", "scale", "k", "family", "sample", "effective_rank", "top1_share"])
    ref = ref[ref["scale"] == scale]
    stack = load_stack(scale, ROOT)
    dist = centroid_distances(scale, ROOT)
    worst = {"degree_er": 0.0, "degree_top1": 0.0, "random_er_mean": 0.0}
    for subj in range(n_subjects):
        A = stack[subj]
        spec = Spectral(A)
        _, rows = analyze_network(A, A, spec.gap, dist, rich_club_level(A), "real", -1, scale, subj, False, {})
        sid = f"sub-{subj:03d}"
        for row in rows:
            if row["c_mode"] != "fixed" or row["c"] != 0.1:
                continue
            r = ref[(ref["subject_id"] == sid) & (ref["k"] == row["k"])]
            deg = r[r["family"] == "degree_strength"].iloc[0]
            worst["degree_er"] = max(worst["degree_er"], abs(deg["effective_rank"] - row["degree_er"]))
            # saved top1_share uses the eps-regularized spectrum; q1 uses raw W, so allow ~N*eps/trW
            worst["degree_top1"] = max(worst["degree_top1"], abs(deg["top1_share"] - row["degree_q1"]))
            worst["random_er_mean"] = max(worst["random_er_mean"], abs(r[r["family"] == "random"]["effective_rank"].mean() - row["rand_er_mean"]))
        # random sets must be the primary ones
        assert len(random_driver_sets("real", -1, scale, subj, 3)) == 100
    return worst


def check_nulls(scale: int) -> dict:
    A = load_stack(scale, ROOT)[0]
    dist = centroid_distances(scale, ROOT)
    spec_real = Spectral(A)
    level = rich_club_level(A)
    out = {}
    for null in NULL_TYPES:
        t0 = time.time()
        B, extra = make_null(null, A, dist, null_seed(null, scale, 0, 0))
        t_null = time.time() - t0
        t0 = time.time()
        net, rows = analyze_network(B, A, spec_real.gap, dist, level, null, 0, scale, 0, scale == 114, {"null_extra": extra})
        t_stats = time.time() - t0
        iu = np.triu_indices_from(A, 1)
        out[null] = {
            "seconds_null": round(t_null, 2),
            "seconds_stats": round(t_stats, 2),
            "symmetric": bool(np.allclose(B, B.T)),
            "weights_multiset_equal": bool(np.allclose(np.sort(A[iu]), np.sort(B[iu]))),
            **{key: net[key] for key in ("degree_preserved", "strength_r", "strength_max_rel_err", "strength_top5_jaccard", "edge_overlap", "mean_edge_length", "gap", "ipr_v1", "rc_bu")},
            "real_mean_edge_length": float(dist[iu][A[iu] > 0].mean()),
            "real_gap": spec_real.gap,
            "real_ipr_v1": float(np.sum(spec_real.perron**4)),
            "degree_q1_pct_k3_c0.1": next(r["degree_q1_pct"] for r in rows if r["k"] == 3 and r["c_mode"] == "fixed" and r["c"] == 0.1),
        }
        print(scale, null, out[null], flush=True)
    return out


def main():
    global ROOT, OUT_DIR
    ctx = stage_from_argv("revision/nulls/00_validate")
    ctx.require_stage("00_prepare_data")
    ctx.require_stage("revision/spectra/01_compute_spectra")
    ctx.begin()
    ROOT = ctx.root
    OUT_DIR = out_dir(ROOT)
    report = {"fast_gramian": {}, "primary_reproduction": {}, "nulls": {}}
    for scale in SCALES:
        A = load_stack(scale, ROOT)[0]
        report["fast_gramian"][scale] = check_fast_gramian(A)
        print(scale, "fast gramian", report["fast_gramian"][scale], flush=True)
        report["primary_reproduction"][scale] = check_primary(scale)
        print(scale, "primary", report["primary_reproduction"][scale], flush=True)
        report["nulls"][scale] = check_nulls(scale)
    (OUT_DIR / "validation.json").write_text(json.dumps(report, indent=2, default=float))
    ctx.finish()


if __name__ == "__main__":
    main()
