#!/usr/bin/env python
"""Statistics for the modal/average-controllability comparison (reads 01 output)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "regime"))
from _common import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "results" / "revision" / "modal"
d = pd.read_csv(OUT / "set_level_modal.csv.gz")
PRED = ["perron_loading", "perron_share", "modal_gu", "modal_ct", "avg_ctrl", "red_G", "red_G_perp", "red_F", "q1_G"]
TARGETS = ["q1", "effective_rank", "er_raw"]
NODE = ["perron_loading", "modal_gu", "avg_ctrl"]


def z(x):
    r = rankdata(x)
    s = r.std()
    return (r - r.mean()) / s if s > 0 else r * 0


def r2(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ beta
    return 1 - res.var() / y.var(), beta[1:]


def partial_rho(y, x, Z):
    ry, rx = z(y), z(x)
    Zr = np.column_stack([np.ones(len(y))] + [z(c) for c in Z])
    ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]
    ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]
    return float(np.corrcoef(ey, ex)[0, 1])


def summ(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {"mean": v.mean(), "median": np.median(v), "n_pos": int((v > 0).sum()), "n": v.size}


# ---------- A/C/D: within-subject over the 100 random sets ----------
rows = []
rnd = d[d.family == "random"]
for (scale, k, c, sid), g in rnd.groupby(["scale", "k", "c", "subject_id"]):
    rec = {"scale": scale, "k": k, "c": c, "subject_id": sid}
    for t in TARGETS:
        y = g[t].to_numpy()
        for p in PRED:
            rec[f"rho_{t}_{p}"] = spearmanr(g[p], y)[0]
        rec[f"prho_{t}_redGperp|node"] = partial_rho(y, g.red_G_perp.to_numpy(), [g[n].to_numpy() for n in NODE])
        rec[f"prho_{t}_redF|node"] = partial_rho(y, g.red_F.to_numpy(), [g[n].to_numpy() for n in NODE])
        rec[f"prho_{t}_modalgu|L,avg"] = partial_rho(y, g.modal_gu.to_numpy(), [g.perron_loading.to_numpy(), g.avg_ctrl.to_numpy()])
        rec[f"prho_{t}_L|modalgu,avg"] = partial_rho(y, g.perron_loading.to_numpy(), [g.modal_gu.to_numpy(), g.avg_ctrl.to_numpy()])
        rec[f"prho_{t}_avg|L,modalgu"] = partial_rho(y, g.avg_ctrl.to_numpy(), [g.perron_loading.to_numpy(), g.modal_gu.to_numpy()])
        yz = z(y)
        models = {
            "L": ["perron_loading"], "modal_gu": ["modal_gu"], "modal_ct": ["modal_ct"], "avg": ["avg_ctrl"],
            "pshare": ["perron_share"],
            "node3": NODE, "node3+pshare": NODE + ["perron_share"],
            "node3+redGperp": NODE + ["red_G_perp"], "node3+pshare+redGperp": NODE + ["perron_share", "red_G_perp"],
            "node3+redF": NODE + ["red_F"],
        }
        for name, cols in models.items():
            R2, beta = r2(yz, np.column_stack([z(g[cc].to_numpy()) for cc in cols]))
            rec[f"R2_{t}_{name}"] = R2
            if name in ("node3+redGperp",):
                for cc, b in zip(cols, beta):
                    rec[f"beta_{t}_{name}_{cc}"] = b
    rows.append(rec)
ws = pd.DataFrame(rows)
ws.to_csv(OUT / "within_subject_stats.csv", index=False)

long = ws.melt(id_vars=["scale", "k", "c", "subject_id"])
summary = long.groupby(["variable", "scale", "k", "c"])["value"].apply(lambda v: pd.Series(summ(v))).unstack().reset_index()
summary.to_csv(OUT / "within_subject_summary.csv", index=False)

# ---------- B: pooled Spearman across all sets (random + ranked + counterfactual) ----------
prow = []
for (scale, k, c), g in d.groupby(["scale", "k", "c"]):
    for t in TARGETS:
        for p in PRED:
            prow.append({"scale": scale, "k": k, "c": c, "target": t, "predictor": p, "rho": spearmanr(g[p], g[t])[0]})
        prow.append({"scale": scale, "k": k, "c": c, "target": t, "predictor": "partial redGperp|node",
                     "rho": partial_rho(g[t].to_numpy(), g.red_G_perp.to_numpy(), [g[n].to_numpy() for n in NODE])})
pooled = pd.DataFrame(prow)
pooled.to_csv(OUT / "pooled_spearman.csv", index=False)

# ---------- E: counterfactual / ranked sets vs the subject's 100 random sets ----------
crow = []
fams = [f for f in d.family.unique() if f != "random"]
for (scale, k, c, sid), g in d.groupby(["scale", "k", "c", "subject_id"]):
    r = g[g.family == "random"]
    for f in fams:
        row = g[g.family == f].iloc[0]
        rec = {"scale": scale, "k": k, "c": c, "subject_id": sid, "family": f}
        for t in TARGETS + ["perron_loading", "red_G_perp", "modal_gu", "avg_ctrl"]:
            rv = r[t].to_numpy()
            rec[f"{t}_diff"] = row[t] - np.median(rv)
            rec[f"{t}_pct"] = (np.sum(rv < row[t]) + 0.5 * np.sum(rv == row[t])) / rv.size
        crow.append(rec)
cf = pd.DataFrame(crow)
cf.to_csv(OUT / "sets_vs_random.csv", index=False)
cfl = cf.melt(id_vars=["scale", "k", "c", "subject_id", "family"])
cfs = cfl.groupby(["family", "variable", "scale", "k", "c"])["value"].apply(lambda v: pd.Series(summ(v))).unstack().reset_index()
cfs.to_csv(OUT / "sets_vs_random_summary.csv", index=False)

# cycle vs top-modal paired difference
pr = []
for (scale, k, c), g in cf.groupby(["scale", "k", "c"]):
    a = g[g.family == "cycle_participation"].set_index("subject_id")
    for f in ["top_modal_gu", "top_modal_ct", "top_avg_ctrl", "low_perron", "degree_strength"]:
        b = g[g.family == f].set_index("subject_id").loc[a.index]
        for t in TARGETS:
            pr.append({"scale": scale, "k": k, "c": c, "contrast": f"cycle - {f}", "target": t,
                       **summ(a[f"{t}_diff"] - b[f"{t}_diff"])})
pd.DataFrame(pr).to_csv(OUT / "cycle_vs_counterfactual.csv", index=False)

# ---------- F: node-level correspondence of modal/avg controllability with the rankings ----------
nd = pd.read_csv(OUT / "node_level_modal.csv.gz")
ns = pd.read_csv(REPO_ROOT / "results" / "primary" / "node_scores.csv", usecols=["subject_id", "scale", "node_index", "method", "score"])
ns = ns.pivot_table(index=["subject_id", "scale", "node_index"], columns="method", values="score").reset_index()
nrow = []
for c in [0.01, 0.1, 1.0]:
    m = nd[nd.c == c].merge(ns, left_on=["subject_id", "scale", "node"], right_on=["subject_id", "scale", "node_index"])
    for (scale, sid), g in m.groupby(["scale", "subject_id"]):
        for a in ["modal_gu", "modal_ct", "avg_ctrl", "perron2"]:
            for b in ["degree_strength", "cycle_participation", "eigenvector_centrality", "betweenness", "participation_coefficient", "triangle_participation"]:
                nrow.append({"c": c, "scale": scale, "subject_id": sid, "node_metric": a, "ranking": b, "rho": spearmanr(g[a], g[b])[0]})
nn = pd.DataFrame(nrow)
nsum = nn.groupby(["node_metric", "ranking", "scale", "c"])["rho"].apply(lambda v: pd.Series(summ(v))).unstack().reset_index()
nsum.to_csv(OUT / "node_level_correlations.csv", index=False)

# overlap of top-k modal sets with ranked sets
orow = []
for (scale, k, c, sid), g in d[d.family != "random"].groupby(["scale", "k", "c", "subject_id"]):
    sets = {f: set(map(int, s.split())) for f, s in zip(g.family, g.nodes)}
    for a in ["top_modal_gu", "top_modal_ct", "top_avg_ctrl", "low_perron"]:
        for b in ["cycle_participation", "degree_strength", "eigenvector_centrality", "participation_coefficient"]:
            orow.append({"scale": scale, "k": k, "c": c, "a": a, "b": b, "jaccard": len(sets[a] & sets[b]) / len(sets[a] | sets[b])})
pd.DataFrame(orow).groupby(["a", "b", "scale", "k", "c"]).jaccard.mean().reset_index().to_csv(OUT / "set_overlap.csv", index=False)

# ---------- G: k x k driver Gram G as proxy for W's leading spectrum ----------
grow = []
for (scale, k, c), g in d.groupby(["scale", "k", "c"]):
    grow.append({"scale": scale, "k": k, "c": c,
                 "rho_q1G_q1raw": spearmanr(g.q1_G, g.q1_raw)[0],
                 "rho_q1G_q1": spearmanr(g.q1_G, g.q1)[0],
                 "rho_erG_er": spearmanr(g.er_G, g.effective_rank)[0],
                 "median_lam1G_over_lam1W": np.median(g.lam1_G / g.lam1_W),
                 "median_q1G_minus_q1raw": np.median(g.q1_G - g.q1_raw),
                 "median_q1raw_minus_pshare": np.median(g.q1_raw - g.perron_share),
                 "rho_pshare_q1": spearmanr(g.perron_share, g.q1)[0]})
pd.DataFrame(grow).to_csv(OUT / "driver_gram_proxy.csv", index=False)

# ---------- H: out-of-sample prediction of ranked sets from models fit on the 100 random sets ----------
MODELS = {"node3": NODE, "node3+redGperp": NODE + ["red_G_perp"], "q1G": ["q1_G"], "pshare": ["perron_share"]}
hrow = []
for (scale, k, c, sid), g in d.groupby(["scale", "k", "c", "subject_id"]):
    r = g[g.family == "random"]
    for t in ["q1", "effective_rank"]:
        preds = {}
        for name, cols in MODELS.items():
            Xr = r[cols].to_numpy()
            mu_, sd_ = Xr.mean(0), Xr.std(0)
            X1 = np.column_stack([np.ones(len(r)), (Xr - mu_) / sd_])
            beta = np.linalg.lstsq(X1, r[t].to_numpy(), rcond=None)[0]
            for f in ["cycle_participation", "degree_strength", "top_modal_gu", "eigenvector_centrality"]:
                row = g[g.family == f]
                preds[(name, f)] = float(np.column_stack([np.ones(1), (row[cols].to_numpy() - mu_) / sd_]) @ beta)
        obs = {f: float(g[g.family == f][t].iloc[0]) for f in ["cycle_participation", "degree_strength", "top_modal_gu", "eigenvector_centrality"]}
        for name in MODELS:
            hrow.append({"scale": scale, "k": k, "c": c, "subject_id": sid, "target": t, "model": name,
                         "obs_cyc_minus_deg": obs["cycle_participation"] - obs["degree_strength"],
                         "pred_cyc_minus_deg": preds[(name, "cycle_participation")] - preds[(name, "degree_strength")],
                         "obs_cyc_minus_modal": obs["cycle_participation"] - obs["top_modal_gu"],
                         "pred_cyc_minus_modal": preds[(name, "cycle_participation")] - preds[(name, "top_modal_gu")]})
h = pd.DataFrame(hrow)
h.to_csv(OUT / "ranked_set_prediction.csv", index=False)
hs = []
for (scale, k, c, t, m), g in h.groupby(["scale", "k", "c", "target", "model"]):
    hs.append({"scale": scale, "k": k, "c": c, "target": t, "model": m,
               "obs_median": g.obs_cyc_minus_deg.median(), "pred_median": g.pred_cyc_minus_deg.median(),
               "frac_explained_median": np.median(g.pred_cyc_minus_deg) / np.median(g.obs_cyc_minus_deg),
               "rho_obs_pred_across_subjects": spearmanr(g.obs_cyc_minus_deg, g.pred_cyc_minus_deg)[0],
               "n_obs_pos": int((g.obs_cyc_minus_deg > 0).sum()), "n_pred_pos": int((g.pred_cyc_minus_deg > 0).sum())})
pd.DataFrame(hs).to_csv(OUT / "ranked_set_prediction_summary.csv", index=False)
print("done")

# ---------- I: across subjects, does the cycle - degree contrast track node-level or pairwise differences? ----------
rows = []
for (scale, k, c), g in d[d.family.isin(["cycle_participation", "degree_strength"])].groupby(["scale", "k", "c"]):
    a = g[g.family == "cycle_participation"].set_index("subject_id")
    b = g[g.family == "degree_strength"].set_index("subject_id").loc[a.index]
    D = a.select_dtypes("number") - b.select_dtypes("number")
    r = {"scale": scale, "k": k, "c": c}
    for t in ["effective_rank", "q1"]:
        for p in ["perron_loading", "avg_ctrl", "modal_gu", "red_G_perp", "red_G", "q1_G"]:
            r[f"{t}~{p}"] = spearmanr(D[t], D[p])[0]
    r.update({"dER_med": D.effective_rank.median(), "dER_npos": int((D.effective_rank > 0).sum()),
              "dq1_med": D.q1.median(), "dq1_npos": int((D.q1 > 0).sum()), "dL_med": D.perron_loading.median(),
              "dL_npos": int((D.perron_loading > 0).sum()), "dAC_npos": int((D.avg_ctrl > 0).sum()),
              "dmodal_npos": int((D.modal_gu > 0).sum())})
    rows.append(r)
pd.DataFrame(rows).to_csv(OUT / "cycle_minus_degree_node_vs_set.csv", index=False)
print("done I")
