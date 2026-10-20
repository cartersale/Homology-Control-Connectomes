#!/usr/bin/env python
"""Why do degree-selected hubs load the leading eigenvector?

Exact identity (A0 v1 = l1 v1, nonnegative A0): for every region i with s_i > 0,
    v1(i) = s_i * m_i / l1,   m_i = sum_j A_ij v1(j) / s_i
i.e. a region's loading is its strength times the strength-weighted mean loading of its neighbours.
Strength is one power-iteration step from the uniform vector (A 1 = s); v1 is the limit.
If strong regions connect to strong regions (strength assortativity), m_i rises with s_i and the
leading eigenvector concentrates further on hubs.

Per network (real connectomes and every saved null network) this records:
  cos_s_v1        cosine between the strength vector and v1
  rho_s_v1        Spearman rho(s, v1)
  cos_As_v1       cosine of the second power-iteration step A s with v1
  rho_logs_logm   Spearman rho(s_i, m_i): do hubs' neighbours load more?
  var_share_s     share of var(log v1) carried by log s (var(log s) + cov(log s, log m)) / var(log v1)
  jac_top5        Jaccard of the top-5 regions by strength and by v1
  L_deg_k / L_max_k  Perron loading of the top-k strength set and the largest attainable loading (top-k by v1^2)
  l1, l2, gap, strength_assort
Outputs (runs/<name>/revision/mechanism/; choose the run with ``--run <name>``): hub_loading.csv.gz,
hub_loading_summary.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import bct
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr, wilcoxon

NULLS_DIR = Path(__file__).resolve().parents[1] / "nulls"  # shared helpers: nulls/nulls_common.py
if str(NULLS_DIR) not in sys.path:
    sys.path.insert(0, str(NULLS_DIR))
from nulls_common import NULL_TYPES, SCALES, load_stack, out_dir  # noqa: E402
from hoc.runs import stage_from_argv  # noqa: E402
from hoc.utils import n_jobs  # noqa: E402

ROOT: Path  # run directory (runs/<name>/), set in main()
NULL_DIR: Path  # <run>/revision/nulls, set in main()
OUT: Path  # <run>/revision/mechanism, set in main()
K_VALUES = (1, 2, 3, 5)


def stats(A: np.ndarray) -> dict:
    lam, U = np.linalg.eigh(A)
    v1 = U[:, -1] * np.sign(U[:, -1].sum())
    s = A.sum(axis=0)
    ok = s > 0
    m = np.zeros_like(s)
    m[ok] = (A @ v1)[ok] / s[ok]
    pos = ok & (v1 > 1e-15) & (m > 1e-15)
    ls, lm, lv = np.log(s[pos]), np.log(m[pos]), np.log(v1[pos])
    order_s = np.argsort(-s, kind="mergesort")
    order_v = np.argsort(-v1**2, kind="mergesort")
    out = {
        "l1": lam[-1], "l2": lam[-2], "gap": lam[-1] - lam[-2],
        "cos_s_v1": float(s @ v1 / np.linalg.norm(s)),
        "rho_s_v1": float(spearmanr(s, v1).statistic),
        "cos_As_v1": float((A @ s) @ v1 / np.linalg.norm(A @ s)),
        "rho_logs_logm": float(spearmanr(ls, lm).statistic),
        "var_share_s": float((ls.var() + np.cov(ls, lm)[0, 1]) / lv.var()),
        "jac_top5": len(set(order_s[:5]) & set(order_v[:5])) / len(set(order_s[:5]) | set(order_v[:5])),
        "strength_assort": float(bct.assortativity_wei(A, flag=0)),
        "max_identity_err": float(np.max(np.abs(v1[ok] - s[ok] * m[ok] / lam[-1]))),
    }
    for k in K_VALUES:
        out[f"L_deg_{k}"] = float((v1[order_s[:k]] ** 2).sum())
        out[f"L_max_{k}"] = float((v1[order_v[:k]] ** 2).sum())
        out[f"L_rand_{k}"] = k / A.shape[0]
    return out


def job(null: str, scale: int, si: int) -> list[dict]:
    ids = {"null": null, "scale": scale, "subject_id": f"sub-{si:03d}"}
    if null == "real":
        return [{**ids, "null_index": -1, **stats(load_stack(scale, ROOT)[si])}]
    triu = np.load(NULL_DIR / "networks" / null / f"scale-{scale}" / f"sub-{si:03d}.npz")["triu"]
    iu = np.triu_indices(scale, 1)
    rows = []
    for i, vec in enumerate(triu):
        A = np.zeros((scale, scale))
        A[iu] = vec
        rows.append({**ids, "null_index": i, **stats(A + A.T)})
    return rows


def main():
    global ROOT, NULL_DIR, OUT
    ctx = stage_from_argv("revision/mechanism/01_hub_loading")
    ctx.require_stage("00_prepare_data")
    ctx.require_stage("revision/nulls/01_run_nulls")
    ctx.begin()
    ROOT = ctx.root
    NULL_DIR = out_dir(ROOT)
    OUT = ROOT / "revision" / "mechanism"
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(n, s, i) for n in ("real",) + NULL_TYPES for s in SCALES for i in range(70)]
    out = Parallel(n_jobs=n_jobs(12))(delayed(job)(*j) for j in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(OUT / "hub_loading.csv.gz", index=False)
    print("identity max abs error:", df["max_identity_err"].max())

    cols = ["cos_s_v1", "rho_s_v1", "cos_As_v1", "rho_logs_logm", "var_share_s", "jac_top5", "strength_assort",
            "L_deg_2", "L_max_2", "L_deg_5", "L_max_5", "gap", "l1"]
    real = df[df["null"] == "real"]
    print("REAL (mean):\n", real.groupby("scale")[cols].mean().round(3).T.to_string())

    # real vs each null: per-subject real minus null mean, Wilcoxon
    rows = []
    for (null, scale), g in df[df["null"] != "real"].groupby(["null", "scale"]):
        nm = g.groupby("subject_id")[cols].mean()
        r = real[real["scale"] == scale].set_index("subject_id")[cols].loc[nm.index]
        for c in cols:
            d = (r[c] - nm[c]).to_numpy()
            rows.append({"null": null, "scale": scale, "stat": c, "real_mean": r[c].mean(), "null_mean": nm[c].mean(),
                         "n_real_higher": int((d > 0).sum()), "p": wilcoxon(d).pvalue if np.any(d != 0) else np.nan})
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "hub_loading_summary.csv", index=False)
    print(summ.pivot_table(index="stat", columns=["null", "scale"], values=["real_mean", "null_mean"]).round(3).to_string())

    # across real subjects: does assortativity (hub-hub wiring) predict loading alignment, L_S and the gap?
    acc = []
    for scale, g in real.groupby("scale"):
        for y in ("rho_logs_logm", "L_deg_5", "cos_s_v1", "gap", "l1"):
            r = spearmanr(g["strength_assort"], g[y])
            acc.append({"scale": scale, "y": y, "rho_with_assort": r.statistic, "p": r.pvalue})
    print(pd.DataFrame(acc).round(3).to_string(index=False))
    pd.DataFrame(acc).to_csv(OUT / "hub_loading_assort_corr.csv", index=False)
    ctx.finish()


if __name__ == "__main__":
    main()
