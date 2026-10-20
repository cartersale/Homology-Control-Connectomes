"""Diagnostic for Task 3: what drives ER among random sets, and does Perron loading predict ER once
driver strength is held fixed? Per subject x scale x k in {1,2,3,5} x c in {0.01, 0.1, 1.0}, across the 100
random sets + 9 ranking sets: Spearman(ER, summed driver strength), Spearman(ER, Perron loading), and the
partial Spearman(ER, Perron loading | summed strength) (rank-residualized). Output: task3_perron_diagnostic.csv
Outputs go to runs/<name>/revision/regime/ (choose the run with ``--run <name>``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from regime_common import (K_VALUES, METHODS, N_SUBJECTS, SCALES, Spectral, load_ranks, load_stack, metrics_from_eigs,
                           out_dir, random_sets, run_parallel)
from hoc.runs import stage_from_argv
from hoc.utils import n_jobs


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    bx = np.polyfit(rz, rx, 1); by = np.polyfit(rz, ry, 1)
    return np.corrcoef(rx - np.polyval(bx, rz), ry - np.polyval(by, rz))[0, 1]


def job(scale, si, ranks, A0):
    spec = Spectral(A0)
    strength = A0.sum(axis=1)
    v2 = spec.perron**2
    sid = f"sub-{si:03d}"
    rows = []
    for c in (0.01, 0.1, 1.0):
        kern = spec.kernel(c)
        for k in K_VALUES:
            sets = [ranks[(sid, m)][:k] for m in METHODS] + random_sets(scale, si, k)
            er = metrics_from_eigs(spec.eigvals_batch(kern, sets), 1e-5)["effective_rank"]
            pl = np.array([v2[s].sum() for s in sets]); st = np.array([strength[s].sum() for s in sets])
            rows.append({"scale": scale, "subject_id": sid, "c": c, "k": k,
                         "rho_er_strength": spearmanr(er, st).statistic, "rho_er_perron": spearmanr(er, pl).statistic,
                         "partial_rho_er_perron_given_strength": partial_spearman(er, pl, st),
                         "rho_perron_strength": spearmanr(pl, st).statistic})
    return rows


if __name__ == "__main__":
    ctx = stage_from_argv("revision/regime/06_perron_diagnostic")
    ctx.require_stage("00_prepare_data")
    ctx.require_stage("01_primary")
    ctx.begin()
    ROOT = ctx.root
    OUT_DIR = out_dir(ROOT)
    out = []
    for scale in SCALES:
        stack = load_stack(scale, ROOT); ranks = load_ranks(scale, ROOT)
        for r in run_parallel(job, [(scale, i, {kk: v for kk, v in ranks.items() if kk[0] == f"sub-{i:03d}"}, stack[i]) for i in range(N_SUBJECTS)], n_jobs=n_jobs(8)):
            out.extend(r)
    d = pd.DataFrame(out)
    d.to_csv(OUT_DIR / "task3_perron_diagnostic.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.groupby(["scale", "c", "k"])[["rho_er_strength", "rho_er_perron", "partial_rho_er_perron_given_strength", "rho_perron_strength"]].median().round(2))
    ctx.finish()
