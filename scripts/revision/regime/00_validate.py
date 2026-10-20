"""Validate the eigenbasis Gramian route against hoc.control and saved results.

Checks
1. W (eigenbasis) vs hoc.control.single_node_gramians/assemble_gramian (Lyapunov) for several
   subjects, c values and horizons.
2. Reproduce <run>/primary/geometry_metrics.csv (c=0.1, eps=1e-5) for all static rankings and
   all 100 random sets, all subjects/scales/k.
3. Reproduce <run>/robustness/sensitivity_summary.csv (c=0.01 and 1.0; eps sweep; finite horizons).
Writes <run>/revision/regime/validation.json, where <run> = runs/<name>/ (choose it with ``--run <name>``).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from regime_common import (K_VALUES, METHODS, N_SUBJECTS, SCALES, Spectral, load_ranks, load_stack,
                           metrics_from_eigs, out_dir, random_sets, run_parallel)
from hoc.control import assemble_gramian, gramian_metrics, single_node_gramians
from hoc.preprocessing import stabilize_adjacency
from hoc.runs import stage_from_argv

ctx = stage_from_argv("revision/regime/00_validate")
ctx.require_stage("00_prepare_data")
ctx.require_stage("01_primary")
ctx.require_stage("02_robustness")
ctx.begin()
ROOT = ctx.root
OUT_DIR = out_dir(ROOT)

report: dict = {}

# ---- 1. direct comparison with library ----
rows = []
for scale in SCALES:
    stack = load_stack(scale, ROOT)
    for subject_index in (0, 33, 69):
        spec = Spectral(stack[subject_index])
        for c in (0.01, 0.1, 1.0):
            system, _ = stabilize_adjacency(stack[subject_index], c)
            # library finite-horizon routine uses a naive O(N^4)-per-node einsum: only run it at 68
            for horizon in ((None, 1.0, 5.0) if scale == 68 else (None,)):
                lib = single_node_gramians(system, horizon=horizon)
                kern = spec.kernel(c, horizon)
                for s in (np.array([0]), np.array([3, 10, 20]), np.arange(5)):
                    w_lib = assemble_gramian(lib, s)
                    w_eig = spec.full_gramian(kern, s)
                    m_lib = gramian_metrics(w_lib, 1e-5)
                    m_eig = metrics_from_eigs(np.linalg.eigvalsh(w_eig), 1e-5)
                    rows.append({
                        "scale": scale, "subject": subject_index, "c": c, "T": horizon,
                        "max_abs_W": float(np.max(np.abs(w_lib - w_eig))),
                        "rel_W": float(np.max(np.abs(w_lib - w_eig)) / np.max(np.abs(w_lib))),
                        "d_er": abs(m_lib["effective_rank"] - m_eig["effective_rank"][0]),
                        "d_log10E": abs(m_lib["log10_average_energy"] - m_eig["log10_average_energy"][0]),
                    })
lib_cmp = pd.DataFrame(rows)
report["library_vs_eigbasis"] = {k: float(lib_cmp[k].max()) for k in ("max_abs_W", "rel_W", "d_er", "d_log10E")}
print("library vs eigbasis", report["library_vs_eigbasis"], flush=True)


# ---- 2. geometry_metrics.csv at c = 0.1 ----
def primary_subject(scale, subject_index, ranks, adjacency):
    spec = Spectral(adjacency)
    kern = spec.kernel(0.1)
    sid = f"sub-{subject_index:03d}"
    out = []
    for k in K_VALUES:
        sets = [ranks[(sid, m)][:k] for m in METHODS] + random_sets(scale, subject_index, k)
        labels = list(METHODS) + ["random"] * 100
        met = metrics_from_eigs(spec.eigvals_batch(kern, sets), 1e-5)
        for j, lab in enumerate(labels):
            out.append({"subject_id": sid, "scale": scale, "k": k, "method": lab, "rs": j - len(METHODS) if lab == "random" else -1,
                        "er_new": met["effective_rank"][j], "pr_new": met["participation_ratio"][j],
                        "le_new": met["log10_average_energy"][j]})
    return out


geo = pd.read_csv(ROOT / "primary/geometry_metrics.csv")
geo = geo[geo["selection_strategy"] == "static"].copy()
geo["rs"] = -1
rmask = geo["method"] == "random"
geo.loc[rmask, "rs"] = geo[rmask].groupby(["subject_id", "scale", "k"]).cumcount()
allrows = []
for scale in SCALES:
    stack = load_stack(scale, ROOT)
    ranks = load_ranks(scale, ROOT)
    res = run_parallel(primary_subject, [(scale, i, {key: v for key, v in ranks.items() if key[0] == f"sub-{i:03d}"}, stack[i]) for i in range(N_SUBJECTS)])
    for r in res:
        allrows.extend(r)
new = pd.DataFrame(allrows)
merged = geo.merge(new, on=["subject_id", "scale", "k", "method", "rs"], how="inner")
report["primary_geometry_rows_matched"] = int(len(merged))
report["primary_geometry_rows_saved"] = int(len(geo))
report["primary_max_abs_d_er"] = float(np.max(np.abs(merged["effective_rank"] - merged["er_new"])))
report["primary_max_abs_d_pr"] = float(np.max(np.abs(merged["participation_ratio"] - merged["pr_new"])))
report["primary_max_abs_d_log10E"] = float(np.max(np.abs(merged["log10_average_energy"] - merged["le_new"])))
print({k: v for k, v in report.items() if k.startswith("primary")}, flush=True)

# ---- 3. sensitivity_summary.csv ----
sens = pd.read_csv(ROOT / "robustness/sensitivity_summary.csv")
sens = sens[sens["k"].isin(K_VALUES + (7, 10, 14, 6, 11, 23, 22, 44))]


def sens_subject(scale, subject_index, ranks, adjacency):
    spec = Spectral(adjacency)
    sid = f"sub-{subject_index:03d}"
    out = []
    for c in (0.01, 1.0):
        for horizon in (np.nan, 1.0, 2.5, 5.0):
            kern = spec.kernel(c, None if np.isnan(horizon) else horizon)
            for k in sorted(set(K_VALUES) | set({68: (7, 10, 14), 114: (6, 10, 11, 23), 219: (10, 11, 22, 44)}[scale])):
                for m in ("degree_strength", "cycle_participation"):
                    w = spec.eigvals_batch(kern, [ranks[(sid, m)][:k]])
                    for eps in ((1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2) if np.isnan(horizon) else (1e-8, 1e-2)):
                        met = metrics_from_eigs(w, eps)
                        out.append({"subject_id": sid, "scale": scale, "k": k, "c_value": c, "horizon": horizon, "epsilon": eps,
                                    "method": m, "er_new": met["effective_rank"][0], "le_new": met["log10_average_energy"][0],
                                    "ff_new": met["fraction_at_floor"][0]})
    return out


allrows = []
for scale in SCALES:
    stack = load_stack(scale, ROOT)
    ranks = load_ranks(scale, ROOT)
    res = run_parallel(sens_subject, [(scale, i, {key: v for key, v in ranks.items() if key[0] == f"sub-{i:03d}"}, stack[i]) for i in range(N_SUBJECTS)])
    for r in res:
        allrows.extend(r)
new = pd.DataFrame(allrows)
sens_f = sens[sens["epsilon_case"] == "epsilon_floor"].copy()
sens_f["horizon"] = sens_f["horizon"].fillna(-1)
new["horizon"] = new["horizon"].fillna(-1)
m = sens_f.merge(new, on=["subject_id", "scale", "k", "c_value", "horizon", "epsilon", "method"], how="inner")
report["sensitivity_rows_matched"] = int(len(m))
report["sensitivity_rows_saved_floor"] = int(len(sens_f))
report["sensitivity_max_abs_d_er"] = float(np.max(np.abs(m["effective_rank"] - m["er_new"])))
report["sensitivity_max_abs_d_log10E"] = float(np.max(np.abs(m["log10_average_energy"] - m["le_new"])))
report["sensitivity_max_abs_d_fraction_floor"] = float(np.max(np.abs(m["fraction_eigenvalues_at_floor"] - m["ff_new"])))
report["sensitivity_mean_abs_d_fraction_floor"] = float(np.mean(np.abs(m["fraction_eigenvalues_at_floor"] - m["ff_new"])))
bad = m.assign(d=np.abs(m["effective_rank"] - m["er_new"])).sort_values("d", ascending=False).head(5)
print(bad[["scale", "k", "c_value", "horizon", "epsilon", "method", "effective_rank", "er_new"]])
print({k: v for k, v in report.items() if k.startswith("sens")})
(OUT_DIR / "validation.json").write_text(json.dumps(report, indent=2))
ctx.finish()
