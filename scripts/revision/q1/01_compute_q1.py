#!/usr/bin/env python
"""P1-P3 of the revision plan: q1-led set tables and the W_perp test on the real connectomes.

For every subject (70), scale (68/114/219), c in {0.01, 0.1, 1}, k in {1, 2, 3, 5}, evaluate:
  - the 9 saved rankings (<run>/primary/node_scores.csv)
  - cycle_resid: top-k of cycle participation residualized on strength (OLS with intercept),
    as in scripts/revision/spectra/01_compute_spectra.py
  - top_avg_ctrl (continuous-time average controllability at this c), top_modal_gu (Gu 2015),
    top_modal_ct (tau = 1), low_perron (bottom-k Perron loading), as in scripts/revision/modal/
  - the 100 primary random sets (primary seeds)
  - the saved matched sets from <run>/revision/spectra/set_metrics.csv (degree_matched,
    spatial_matched, degree_spatial_matched; 100 each; node sets are c-independent)
Per set: q1 = lambda1/tr W (raw W), ER, PR (eps = 1e-5 pipeline definitions), the same three
for W_perp = P W P (Perron mode projected out), Perron loading L_S and Perron share.

Output: <run>/revision/q1/set_q1.csv.gz (one row per set) and validation.json, where
<run> = runs/<name>/ (choose it with ``--run <name>``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

NULLS_DIR = Path(__file__).resolve().parents[1] / "nulls"  # shared helpers: nulls/nulls_common.py
if str(NULLS_DIR) not in sys.path:
    sys.path.insert(0, str(NULLS_DIR))
from nulls_common import Spectral, load_stack, random_driver_sets, spectrum_metrics  # noqa: E402
from hoc.runs import stage_from_argv  # noqa: E402
from hoc.utils import n_jobs  # noqa: E402

ROOT: Path  # run directory (runs/<name>/), set in main()
OUT: Path  # <run>/revision/q1, set in main()
SCALES = (68, 114, 219)
C_VALUES = (0.01, 0.1, 1.0)
K_VALUES = (1, 2, 3, 5)
TAU = 1.0
MATCHED = ("degree_matched", "spatial_matched", "degree_spatial_matched")


def top(score: np.ndarray, k: int, ascending: bool = False) -> np.ndarray:
    return np.argsort(score if ascending else -score, kind="stable")[:k]


def load_orders(scale: int) -> dict:
    t = pd.read_csv(ROOT / "primary" / "node_scores.csv", usecols=["subject_id", "scale", "node_index", "method", "rank", "score"])
    t = t[t["scale"] == scale]
    orders, scores = {}, {}
    for (sid, m), g in t.groupby(["subject_id", "method"]):
        g = g.sort_values("rank")
        orders[(sid, m)] = g["node_index"].to_numpy(int)
        scores[(sid, m)] = g.sort_values("node_index")["score"].to_numpy(float)
    return orders, scores


def load_matched(scale: int) -> dict:
    t = pd.read_csv(ROOT / "revision" / "spectra" / "set_metrics.csv", usecols=["subject_id", "scale", "k", "family", "sample", "nodes"])
    t = t[(t["scale"] == scale) & t["family"].isin(MATCHED)]
    out = {}
    for (sid, k), g in t.groupby(["subject_id", "k"]):
        out[(sid, int(k))] = [(f, int(s), np.array(n.split(";"), dtype=int)) for f, s, n in zip(g["family"], g["sample"], g["nodes"])]
    return out


def job(scale: int, si: int, A: np.ndarray, orders: dict, scores: dict, matched: dict) -> list[dict]:
    sid = f"sub-{si:03d}"
    spec = Spectral(A)
    U, lam = spec.U, spec.lam
    sigma = np.linalg.svd(A, compute_uv=False)[0]
    modal_gu = (U**2) @ (1.0 - (lam / (1.0 + sigma)) ** 2)
    strength = scores[(sid, "degree_strength")]
    cp = scores[(sid, "cycle_participation")]
    X = np.column_stack([np.ones_like(strength), strength])
    resid = cp - X @ np.linalg.lstsq(X, cp, rcond=None)[0]
    methods = sorted(m for (s, m) in orders if s == sid)
    rows = []
    for c in C_VALUES:
        mu = lam - spec.lam_max - c
        avg = (U**2) @ (1.0 / (-2.0 * mu))
        modal_ct = (U**2) @ (1.0 - np.exp(2.0 * mu * TAU))
        kernel = spec.kernel(c)
        for k in K_VALUES:
            sets = [(m, -1, orders[(sid, m)][:k]) for m in methods]
            sets += [
                ("cycle_resid", -1, top(resid, k)),
                ("top_avg_ctrl", -1, top(avg, k)),
                ("top_modal_gu", -1, top(modal_gu, k)),
                ("top_modal_ct", -1, top(modal_ct, k)),
                ("low_perron", -1, top(spec.perron_sq, k, ascending=True)),
            ]
            sets += [("random", r, s) for r, s in enumerate(random_driver_sets("real", -1, scale, si, k))]
            sets += matched.get((sid, k), [])
            node_sets = [s for _, _, s in sets]
            w, w_perp, pshare = spec.set_spectra(kernel, node_sets)
            met, mp = spectrum_metrics(w), spectrum_metrics(w_perp)
            for j, (fam, r, s) in enumerate(sets):
                rows.append({
                    "subject_id": sid, "scale": scale, "c": c, "k": k, "family": fam, "sample": r,
                    "q1": met["q1"][j], "er": met["er"][j], "pr": met["pr"][j],
                    "q1_perp": mp["q1"][j], "er_perp": mp["er"][j], "pr_perp": mp["pr"][j],
                    "loading": float(spec.perron_sq[s].sum()), "perron_share": pshare[j],
                    "nodes": ";".join(map(str, s)),
                })
    return rows


def main():
    global ROOT, OUT
    ctx = stage_from_argv("revision/q1/01_compute_q1")
    ctx.require_stage("00_prepare_data")
    ctx.require_stage("01_primary")
    ctx.require_stage("revision/spectra/01_compute_spectra")
    ctx.require_stage("revision/modal/01_compute_modal")
    ctx.begin()
    ROOT = ctx.root
    OUT = ROOT / "revision" / "q1"
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for scale in SCALES:
        stack = load_stack(scale, ROOT)
        orders, scores = load_orders(scale)
        matched = load_matched(scale)
        out = Parallel(n_jobs=n_jobs(12))(delayed(job)(scale, si, stack[si], orders, scores, matched) for si in range(stack.shape[0]))
        frames.append(pd.DataFrame([r for rows in out for r in rows]))
        print(scale, "done", flush=True)
    df = pd.concat(frames, ignore_index=True)

    # validation against saved results at c = 0.1 (ER from spectra/, q1_raw from modal/)
    ref = pd.read_csv(ROOT / "revision" / "spectra" / "set_metrics.csv", usecols=["subject_id", "scale", "k", "family", "sample", "effective_rank"])
    mine = df[df["c"] == 0.1].merge(ref, on=["subject_id", "scale", "k", "family", "sample"])
    modal = pd.read_csv(ROOT / "revision" / "modal" / "set_level_modal.csv.gz", usecols=["subject_id", "scale", "c", "k", "family", "sample", "q1_raw"])
    mine2 = df.merge(modal, on=["subject_id", "scale", "c", "k", "family", "sample"])
    val = {
        "er_vs_spectra_max_abs": float((mine["er"] - mine["effective_rank"]).abs().max()), "er_vs_spectra_n": len(mine),
        "q1_vs_modal_max_abs": float((mine2["q1"] - mine2["q1_raw"]).abs().max()), "q1_vs_modal_n": len(mine2),
    }
    print(val)
    (OUT / "validation.json").write_text(json.dumps(val, indent=2))
    df.to_csv(OUT / "set_q1.csv.gz", index=False)
    ctx.finish()


if __name__ == "__main__":
    main()
