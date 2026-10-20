#!/usr/bin/env python
"""A1-A3 of the revision plan: what makes the real spectral gap large, and what mediates it.

A1  lambda1 vs lambda2 split. Per subject: real minus null-mean of lambda1, lambda2 and the gap,
    z-scored against the subject's own nulls; share of the gap excess carried by lambda1
    (d_gap = d_lambda1 - d_lambda2).
A2  Mediators that do not reuse the degree driver set (top-k strength nodes). Computed on the
    real connectomes and on every saved null network:
      wdeg_rho        Spearman rho between edge weight and the product of endpoint *binary*
                      degrees (binary degree is preserved by all three nulls, so this measures
                      only where strong weights sit on the topology)
      wdeg_top_share  share of total weight on edges whose endpoints are both in the top 20% of
                      binary degree (a broader, degree-defined class, not the strength top-k)
      strength_assort weighted strength assortativity (bct.assortativity_wei)
      rc_wu_p{75,85,95}  weighted (Opsahl) rich club at those binary-degree percentiles
    plus the existing rc_bu, rc_wu (p90), hub_strength_frac (circular: top-5 strength), ipr_v1.
    Mediation as in 02_analyze (subject fixed effects; prop = 1 - beta_direct / beta_total), for
    two outcomes: degree-set q1 (fixed c = 0.1; k = avg and each k) and the spectral gap itself.
A3  Summary numbers for the Abstract and Fig 5 -> gap_summary.md.

Outputs (runs/<name>/revision/nulls/; choose the run with ``--run <name>``): gap_split.csv, mediators_network.csv.gz,
mediation_noncircular.csv, gap_summary.md
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import bct
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from scipy.stats import spearmanr, wilcoxon

from nulls_common import NULL_TYPES, SCALES, load_stack, out_dir, rich_club
from hoc.runs import stage_from_argv
from hoc.utils import n_jobs

ROOT: Path  # run directory (runs/<name>/), set in main()
OUT_DIR: Path  # <run>/revision/nulls, set in main()

TOP_DEGREE_FRAC = 0.20
RC_PERCENTILES = (75, 85, 95)
NEW_MEDIATORS = ["wdeg_rho", "wdeg_top_share", "strength_assort"] + [f"rc_wu_p{p}" for p in RC_PERCENTILES]
OLD_MEDIATORS = ["rc_bu", "rc_wu", "hub_strength_frac", "ipr_v1"]
CIRCULAR = {"hub_strength_frac"}


def mediator_stats(A: np.ndarray, real_degree: np.ndarray) -> dict:
    iu = np.triu_indices_from(A, 1)
    w = A[iu]
    edges = w > 0
    degree = np.count_nonzero(A, axis=0)
    deg_prod = (degree[:, None] * degree[None, :])[iu]
    n_top = int(np.ceil(TOP_DEGREE_FRAC * A.shape[0]))
    top = np.zeros(A.shape[0], bool)
    top[np.argsort(-degree, kind="mergesort")[:n_top]] = True
    both_top = (top[:, None] & top[None, :])[iu]
    out = {
        "wdeg_rho": float(spearmanr(w[edges], deg_prod[edges]).statistic),
        "wdeg_top_share": float(w[both_top].sum() / w.sum()),
        "strength_assort": float(bct.assortativity_wei(A, flag=0)),
    }
    for p in RC_PERCENTILES:  # levels from the real network's degree, so identical across its nulls
        out[f"rc_wu_p{p}"] = rich_club(A, int(np.floor(np.percentile(real_degree, p))))[1]
    return out


def job(null: str, scale: int, subject: int) -> list[dict]:
    real = load_stack(scale, ROOT)[subject]
    real_degree = np.count_nonzero(real, axis=0)
    ids = {"null": null, "scale": scale, "subject_id": f"sub-{subject:03d}"}
    if null == "real":
        return [{**ids, "null_index": -1, **mediator_stats(real, real_degree)}]
    triu = np.load(OUT_DIR / "networks" / null / f"scale-{scale}" / f"sub-{subject:03d}.npz")["triu"]
    iu = np.triu_indices(scale, 1)
    rows = []
    for i, vec in enumerate(triu):
        A = np.zeros((scale, scale))
        A[iu] = vec
        A = A + A.T
        rows.append({**ids, "null_index": i, **mediator_stats(A, real_degree)})
    return rows


def per_subject_contrast(real: pd.Series, nulls: pd.DataFrame, stat: str) -> pd.DataFrame:
    rows = []
    for sid, g in nulls.groupby("subject_id"):
        x = g[stat].to_numpy(float)
        r = float(real.loc[sid])
        sd = x.std(ddof=1)
        sd = sd if sd > 1e-12 * max(1.0, abs(x.mean())) else 0.0  # identical values: avoid float-noise z
        sd = sd if sd > 1e-12 * max(1.0, abs(x.mean())) else 0.0  # identical values: avoid float-noise z
        rows.append({"subject_id": sid, "real": r, "null_mean": x.mean(), "delta": r - x.mean(),
                     "z": (r - x.mean()) / sd if sd > 0 else np.nan,
                     "above95": r > np.percentile(x, 95), "below5": r < np.percentile(x, 5)})
    return pd.DataFrame(rows)


def summarize(d: pd.DataFrame) -> dict:
    z = d["z"].dropna()
    return {"real_mean": d["real"].mean(), "null_mean": d["null_mean"].mean(), "delta_mean": d["delta"].mean(),
            "z_median": z.median() if len(z) else np.nan,
            "wilcoxon_p": wilcoxon(z).pvalue if len(z) >= 5 and (z != 0).any() else np.nan,
            "n_above95": int(d["above95"].sum()), "n_below5": int(d["below5"].sum()), "n": len(d)}


def gap_split(net: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (null, scale), g in net[net["null"] != "real"].groupby(["null", "scale"]):
        real = net[(net["null"] == "real") & (net["scale"] == scale)].set_index("subject_id")
        per = {s: per_subject_contrast(real[s], g, s) for s in ("lambda1", "lambda2", "gap")}
        for s, d in per.items():
            rows.append({"null": null, "scale": scale, "stat": s, **summarize(d)})
        share = per["lambda1"]["delta"] / per["gap"]["delta"]
        rows.append({"null": null, "scale": scale, "stat": "lambda1_share_of_gap_excess",
                     "real_mean": np.nan, "null_mean": np.nan, "delta_mean": float(share.median()),
                     "z_median": np.nan, "wilcoxon_p": np.nan,
                     "n_above95": int((share > 0.5).sum()), "n_below5": int((share < 0).sum()), "n": len(share)})
    return pd.DataFrame(rows)


def mediation(frame: pd.DataFrame, outcome: str, mediators: list[str], keys: dict) -> list[dict]:
    rows = []
    frame = frame.assign(is_real=(frame["null"] == "real").astype(float)).dropna(subset=[outcome])
    base = smf.ols(f"{outcome} ~ is_real + C(subject_id)", data=frame).fit()
    b0 = base.params["is_real"]
    nulls_only = frame[frame["null"] != "real"]
    for m in mediators:
        sub = frame.dropna(subset=[m])
        if sub[m].std() == 0 or sub.groupby("subject_id")[m].nunique().max() <= 1:
            continue
        full = smf.ols(f"{outcome} ~ is_real + {m} + C(subject_id)", data=sub).fit()
        rhos = [spearmanr(g[m], g[outcome]).statistic for _, g in nulls_only.groupby("subject_id")
                if g[m].nunique() > 1 and g[outcome].nunique() > 1]
        rows.append({**keys, "outcome": outcome, "mediator": m, "circular": m in CIRCULAR,
                     "beta_real_total": b0, "beta_real_direct": full.params["is_real"],
                     "prop_mediated": 1 - full.params["is_real"] / b0 if b0 != 0 else np.nan,
                     "p_mediator": full.pvalues[m],
                     "median_within_subject_rho": float(np.nanmedian(rhos)) if rhos else np.nan})
    return rows


def main():
    global ROOT, OUT_DIR
    ctx = stage_from_argv("revision/nulls/03_gap_mediators")
    ctx.require_stage("00_prepare_data")
    ctx.require_stage("revision/nulls/01_run_nulls")
    ctx.require_stage("revision/nulls/02_analyze")
    ctx.begin()
    ROOT = ctx.root
    OUT_DIR = out_dir(ROOT)
    net = pd.read_csv(OUT_DIR / "networks.csv.gz")

    # A1
    split = gap_split(net)
    split.to_csv(OUT_DIR / "gap_split.csv", index=False)
    print(split.round(3).to_string(index=False), flush=True)

    # A2: new mediators on all networks
    path = OUT_DIR / "mediators_network.csv.gz"
    if path.exists():  # cache: reuse mediators computed by an earlier run in this run directory
        med_net = pd.read_csv(path)
    else:
        jobs = [(n, s, i) for n in ("real",) + NULL_TYPES for s in SCALES for i in range(70)]
        out = Parallel(n_jobs=n_jobs(12))(delayed(job)(*j) for j in jobs)
        med_net = pd.DataFrame([r for rows in out for r in rows])
        med_net.to_csv(path, index=False)
    full = net.merge(med_net, on=["null", "null_index", "scale", "subject_id"])
    assert len(full) == len(net), (len(full), len(net))

    # real vs null for each mediator: a usable mediator must differ in every null
    contrast = []
    for (null, scale), g in full[full["null"] != "real"].groupby(["null", "scale"]):
        real = full[(full["null"] == "real") & (full["scale"] == scale)].set_index("subject_id")
        for m in NEW_MEDIATORS + OLD_MEDIATORS:
            contrast.append({"null": null, "scale": scale, "stat": m, **summarize(per_subject_contrast(real[m], g, m))})
    contrast = pd.DataFrame(contrast)

    sets = pd.read_csv(OUT_DIR / "sets.csv.gz", usecols=["null", "null_index", "scale", "subject_id", "c_mode", "c_ref", "k", "degree_q1"])
    sets = sets[(sets["c_mode"] == "fixed") & (sets["c_ref"] == 0.1)]
    rows = []
    mediators = NEW_MEDIATORS + OLD_MEDIATORS
    for null in NULL_TYPES:
        for scale in SCALES:
            nets = full[(full["scale"] == scale) & full["null"].isin(["real", null])]
            rows += mediation(nets, "gap", mediators, {"null": null, "scale": scale, "k": "-"})
            for k in ("avg", "1", "2", "3", "5"):
                s = sets[(sets["scale"] == scale) & (sets["k"].astype(str) == k) & sets["null"].isin(["real", null])]
                rows += mediation(nets.merge(s, on=["null", "null_index", "scale", "subject_id"]), "degree_q1", mediators, {"null": null, "scale": scale, "k": k})
    med = pd.DataFrame(rows)
    med.to_csv(OUT_DIR / "mediation_noncircular.csv", index=False)
    contrast.to_csv(OUT_DIR / "mediator_contrasts.csv", index=False)

    # A3 summary
    lines = ["# Gap summary (A1-A3; auto-generated by 03_gap_mediators.py)", "",
             "## A1: lambda1 vs lambda2 (real vs null mean; median z; subjects >95th / <5th null pct)", ""]
    for stat in ("lambda1", "lambda2", "gap", "lambda1_share_of_gap_excess"):
        lines.append(f"### {stat}")
        lines.append(split[split["stat"] == stat].drop(columns="stat").round(3).to_markdown(index=False))
        lines.append("")
    lines += ["## A2: mediator candidates, real vs null", "", contrast.round(3).to_markdown(index=False), "",
              "## A2: proportion of the real-null difference mediated (k = avg for q1; c = 0.1 fixed)", ""]
    piv = med[med["k"].isin(["avg", "-"])].pivot_table(index=["outcome", "mediator"], columns=["null", "scale"], values="prop_mediated")
    lines += [piv.round(2).to_markdown(), ""]
    (OUT_DIR / "gap_summary.md").write_text("\n".join(lines))
    print(piv.round(2).to_string())
    print(contrast.pivot_table(index="stat", columns=["null", "scale"], values="z_median").round(2).to_string())
    ctx.finish()


if __name__ == "__main__":
    main()
