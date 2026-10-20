"""Tasks 1 + 3: dense c sweep (eps = 1e-5, infinite horizon) with random baselines and Perron-mode mechanism.

Per subject x scale x c x k in {1,2,3,5}: 9 static top-k rankings (saved primary node ranks) and the
primary pipeline's 100 random sets (same seeds). Outputs
  csweep_sets_methods.csv      per subject/scale/c/k/method: ER, PR, log10E, perron loading, perron trace fraction,
                               random mean/sd, percentile vs random (ER, PR)
  csweep_random_raw.npz        per scale: random-set ER / PR / Perron loading arrays
  csweep_spearman.csv          per subject/scale/c/k: Spearman(Perron loading, ER) and (Perron trace frac, ER) across the 109 sets
Outputs go to runs/<name>/revision/regime/ (choose the run with ``--run <name>``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from regime_common import (K_VALUES, METHODS, N_RANDOM, N_SUBJECTS, SCALES, Spectral, load_ranks, load_stack,
                           metrics_from_eigs, out_dir, random_sets, run_parallel)
from hoc.runs import stage_from_argv
from hoc.utils import n_jobs

C_VALUES = (0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 2.0)
EPS = 1e-5


def pct_vs(value, null):
    return 100.0 * (np.mean(null < value) + 0.5 * np.mean(null == value))


def subject_job(scale, subject_index, ranks, adjacency):
    spec = Spectral(adjacency)
    sid = f"sub-{subject_index:03d}"
    v2 = spec.perron**2
    pidx = int(np.argmax(spec.lam))
    Usq = spec.U**2  # (node, mode)
    rows, sp_rows = [], []
    rnd = {k: random_sets(scale, subject_index, k) for k in K_VALUES}
    raw = np.zeros((len(C_VALUES), len(K_VALUES), N_RANDOM, 4))
    for ci, c in enumerate(C_VALUES):
        kern = spec.kernel(c)
        kdiag = np.diag(kern)
        for ki, k in enumerate(K_VALUES):
            sets = [ranks[(sid, m)][:k] for m in METHODS] + rnd[k]
            labels = list(METHODS) + ["random"] * N_RANDOM
            met = metrics_from_eigs(spec.eigvals_batch(kern, sets), EPS)
            perron_load = np.array([v2[s].sum() for s in sets])
            # v^T W v = K_pp * sum_{i in S} v_i^2 ; tr W = sum_a K_aa sum_{i in S} U_ia^2
            trace_w = np.array([Usq[s].sum(axis=0) @ kdiag for s in sets])
            perron_frac = kdiag[pidx] * perron_load / trace_w
            er, pr = met["effective_rank"], met["participation_ratio"]
            nm = len(METHODS)
            raw[ci, ki, :, 0] = er[nm:]
            raw[ci, ki, :, 1] = pr[nm:]
            raw[ci, ki, :, 2] = perron_load[nm:]
            raw[ci, ki, :, 3] = perron_frac[nm:]
            for j, m in enumerate(METHODS):
                rows.append({
                    "subject_id": sid, "scale": scale, "c": c, "k": k, "method": m,
                    "effective_rank": er[j], "participation_ratio": pr[j],
                    "log10_average_energy": met["log10_average_energy"][j],
                    "log10_lambda_max": met["log10_lambda_max"][j],
                    "perron_loading": perron_load[j], "perron_trace_fraction": perron_frac[j],
                    "random_mean_er": er[nm:].mean(), "random_sd_er": er[nm:].std(ddof=1),
                    "random_mean_pr": pr[nm:].mean(),
                    "er_minus_random": er[j] - er[nm:].mean(), "pr_minus_random": pr[j] - pr[nm:].mean(),
                    "er_percentile_vs_random": pct_vs(er[j], er[nm:]), "pr_percentile_vs_random": pct_vs(pr[j], pr[nm:]),
                })
            rows.append({
                "subject_id": sid, "scale": scale, "c": c, "k": k, "method": "random_mean",
                "effective_rank": er[nm:].mean(), "participation_ratio": pr[nm:].mean(),
                "log10_average_energy": met["log10_average_energy"][nm:].mean(),
                "log10_lambda_max": met["log10_lambda_max"][nm:].mean(),
                "perron_loading": perron_load[nm:].mean(), "perron_trace_fraction": perron_frac[nm:].mean(),
            })
            sp_rows.append({
                "subject_id": sid, "scale": scale, "c": c, "k": k,
                "rho_perronload_er": spearmanr(perron_load, er).statistic if k > 0 else np.nan,
                "rho_perronload_pr": spearmanr(perron_load, pr).statistic,
                "rho_perronfrac_er": spearmanr(perron_frac, er).statistic,
                "rho_perronload_er_rankings_only": spearmanr(perron_load[:nm], er[:nm]).statistic,
                "rho_perronload_er_random_only": spearmanr(perron_load[nm:], er[nm:]).statistic,
            })
    return rows, sp_rows, raw


if __name__ == "__main__":
    ctx = stage_from_argv("revision/regime/01_csweep_perron")
    ctx.require_stage("00_prepare_data")
    ctx.require_stage("01_primary")
    ctx.begin()
    ROOT = ctx.root
    OUT_DIR = out_dir(ROOT)
    all_rows, all_sp, raws = [], [], {}
    for scale in SCALES:
        stack = load_stack(scale, ROOT)
        ranks = load_ranks(scale, ROOT)
        jobs = [(scale, i, {key: v for key, v in ranks.items() if key[0] == f"sub-{i:03d}"}, stack[i]) for i in range(N_SUBJECTS)]
        res = run_parallel(subject_job, jobs, n_jobs=n_jobs(8))
        for r, s, raw in res:
            all_rows.extend(r)
            all_sp.extend(s)
        raws[f"scale_{scale}"] = np.stack([x[2] for x in res])  # subj, c, k, rand, [er, pr, perron_load, perron_frac]
        print("done scale", scale, flush=True)
    pd.DataFrame(all_rows).to_csv(OUT_DIR / "csweep_sets_methods.csv", index=False)
    pd.DataFrame(all_sp).to_csv(OUT_DIR / "csweep_spearman.csv", index=False)
    np.savez_compressed(OUT_DIR / "csweep_random_raw.npz", c_values=np.array(C_VALUES), k_values=np.array(K_VALUES), **raws)
    ctx.finish()
