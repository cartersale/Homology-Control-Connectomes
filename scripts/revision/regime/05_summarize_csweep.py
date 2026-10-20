"""Summaries + draft figure for Tasks 1 and 3 (reads outputs of 01_csweep_perron.py).

Subject-level quantities are averaged over k in {1,2,3,5} first, then summarized across the 70 subjects.
Outputs
  task1_csweep_cycle_minus_degree.csv  (ER, PR; Holm family = 13 c x 3 scales per metric)
  task1_csweep_vs_random.csv           (method - random-mean ER/PR, percentile; Holm family = 13 c x 3 scales per method x metric)
  task1_signflip.csv                   (interpolated zero crossing of mean / median diff, and of n_positive = 35)
  task3_perron.csv                     (Perron loading, Perron trace fraction per method; Spearman summaries)
  fig_csweep_draft.png
Outputs go to runs/<name>/revision/regime/ (choose the run with ``--run <name>``).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from regime_common import holm, out_dir, paired_stats
from hoc.runs import stage_from_argv

ctx = stage_from_argv("revision/regime/05_summarize_csweep")
ctx.require_stage("revision/regime/01_csweep_perron")
ctx.begin()
ROOT = ctx.root
OUT_DIR = out_dir(ROOT)

m = pd.read_csv(OUT_DIR / "csweep_sets_methods.csv")
sp = pd.read_csv(OUT_DIR / "csweep_spearman.csv")
SCALES = (68, 114, 219)

# ---------- Task 1: cycle - degree ----------
sub = m.groupby(["scale", "c", "subject_id", "method"], as_index=False).mean(numeric_only=True)
wide = sub.pivot_table(index=["scale", "c", "subject_id"], columns="method", values=["effective_rank", "participation_ratio", "perron_loading", "perron_trace_fraction"])
diff = pd.DataFrame({
    "d_er": wide[("effective_rank", "cycle_participation")] - wide[("effective_rank", "degree_strength")],
    "d_pr": wide[("participation_ratio", "cycle_participation")] - wide[("participation_ratio", "degree_strength")],
    "d_perron_load": wide[("perron_loading", "degree_strength")] - wide[("perron_loading", "cycle_participation")],
}).reset_index()
rows = []
for (scale, c), g in diff.groupby(["scale", "c"]):
    for metric in ("d_er", "d_pr"):
        rows.append({"scale": scale, "c": c, "metric": {"d_er": "effective_rank", "d_pr": "participation_ratio"}[metric], **paired_stats(g[metric].to_numpy())})
t1 = holm(pd.DataFrame(rows), ["metric"])
t1.to_csv(OUT_DIR / "task1_csweep_cycle_minus_degree.csv", index=False)

# ---------- sign flip ----------
def crossing(cs, ys, level=0.0):
    x = np.log10(np.asarray(cs, dtype=float))
    y = np.asarray(ys) - level
    for i in range(len(y) - 1):
        if y[i] > 0 and y[i + 1] <= 0:
            xf = x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1])
            return 10**xf
    return np.nan

flip = []
for (metric, scale), g in t1.groupby(["metric", "scale"]):
    g = g.sort_values("c")
    flip.append({"metric": metric, "scale": scale,
                 "c_flip_mean": crossing(g["c"], g["mean_diff"]),
                 "c_flip_median": crossing(g["c"], g["median_diff"]),
                 "c_npos_below_half": crossing(g["c"], g["n_positive"], 35),
                 "c_last_holm_sig_positive": g[(g["p_holm"] < 0.05) & (g["mean_diff"] > 0)]["c"].max(),
                 "c_first_holm_sig_negative": g[(g["p_holm"] < 0.05) & (g["mean_diff"] < 0)]["c"].min()})
flip = pd.DataFrame(flip)
flip.to_csv(OUT_DIR / "task1_signflip.csv", index=False)

# ---------- Task 1: vs random ----------
rows = []
meth = m[m["method"] != "random_mean"]
subm = meth.groupby(["scale", "c", "subject_id", "method"], as_index=False)[["er_minus_random", "pr_minus_random", "er_percentile_vs_random", "pr_percentile_vs_random"]].mean()
for (method, scale, c), g in subm.groupby(["method", "scale", "c"]):
    for metric, pct in (("er_minus_random", "er_percentile_vs_random"), ("pr_minus_random", "pr_percentile_vs_random")):
        rows.append({"method": method, "scale": scale, "c": c, "metric": metric, "mean_percentile": g[pct].mean(),
                     "median_percentile": g[pct].median(), **paired_stats(g[metric].to_numpy())})
t1r = holm(pd.DataFrame(rows), ["method", "metric"])
t1r.to_csv(OUT_DIR / "task1_csweep_vs_random.csv", index=False)

# ---------- Task 3 ----------
rows = []
for (method, scale, c), g in sub[sub["method"].isin(["degree_strength", "cycle_participation", "eigenvector_centrality", "betweenness",
                                                      "participation_coefficient", "triangle_participation", "random_mean"])].groupby(["method", "scale", "c"]):
    rows.append({"method": method, "scale": scale, "c": c, "perron_loading": g["perron_loading"].mean(),
                 "perron_trace_fraction": g["perron_trace_fraction"].mean(), "effective_rank": g["effective_rank"].mean(),
                 "log10_lambda_max": g["log10_lambda_max"].mean()})
t3m = pd.DataFrame(rows)
t3m.to_csv(OUT_DIR / "task3_perron_by_method.csv", index=False)
rows = []
for (scale, c), g in sp.groupby(["scale", "c"]):
    d = diff[(diff["scale"] == scale) & (diff["c"] == c)]
    rows.append({"scale": scale, "c": c,
                 "median_rho_within_subject_k": g["rho_perronload_er"].median(),
                 "q25_rho": g["rho_perronload_er"].quantile(0.25), "q75_rho": g["rho_perronload_er"].quantile(0.75),
                 "frac_rho_negative": (g["rho_perronload_er"] < 0).mean(),
                 "median_rho_rankings_only": g["rho_perronload_er_rankings_only"].median(),
                 "median_rho_random_only": g["rho_perronload_er_random_only"].median(),
                 "median_rho_perronfrac_er": g["rho_perronfrac_er"].median(),
                 "rho_across_subjects_dPerron_dER": spearmanr(d["d_perron_load"], d["d_er"]).statistic,
                 "p_across_subjects": spearmanr(d["d_perron_load"], d["d_er"]).pvalue})
t3s = pd.DataFrame(rows)
t3s.to_csv(OUT_DIR / "task3_perron_spearman.csv", index=False)

# ---------- figure ----------
COL = {68: "#2a78d6", 114: "#eb6834", 219: "#1baf7a"}
MCOL = {"cycle_participation": "#2a78d6", "degree_strength": "#eb6834", "eigenvector_centrality": "#1baf7a"}
MLAB = {"cycle_participation": "Cycle participation", "degree_strength": "Degree", "eigenvector_centrality": "Eigenvector centrality"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
for ax, metric, lab in ((axes[0, 0], "effective_rank", "Cycle − degree effective rank"), (axes[0, 1], "participation_ratio", "Cycle − degree participation ratio")):
    for scale in SCALES:
        g = t1[(t1["metric"] == metric) & (t1["scale"] == scale)].sort_values("c")
        ax.fill_between(g["c"], g["mean_ci_low"], g["mean_ci_high"], color=COL[scale], alpha=0.18, lw=0)
        ax.plot(g["c"], g["mean_diff"], "-o", color=COL[scale], lw=2, ms=4, label=f"{scale} regions")
    ax.axhline(0, color="#8a8984", lw=1)
    ax.axvline(0.1, color="#c3c2b7", lw=1, ls="--")
    ax.set_xscale("log"); ax.set_xlabel("stabilization margin c"); ax.set_ylabel(lab)
axes[0, 0].legend(frameon=False)
axes[0, 0].set_title("A  Cycle − degree ER (mean, 95% bootstrap CI)", loc="left", fontsize=9)
axes[0, 1].set_title("B  Cycle − degree PR", loc="left", fontsize=9)
ax = axes[0, 2]
for scale in SCALES:
    g = t1[(t1["metric"] == "effective_rank") & (t1["scale"] == scale)].sort_values("c")
    ax.plot(g["c"], g["n_positive"], "-o", color=COL[scale], lw=2, ms=4, label=f"{scale}")
ax.axhline(35, color="#8a8984", lw=1); ax.set_xscale("log"); ax.set_ylim(0, 71)
ax.set_xlabel("stabilization margin c"); ax.set_ylabel("subjects with cycle ER > degree ER (of 70)")
ax.set_title("C  n positive (ER)", loc="left", fontsize=9)
for j, scale in enumerate(SCALES):
    ax = axes[1, j]
    for method in MCOL:
        g = t1r[(t1r["method"] == method) & (t1r["scale"] == scale) & (t1r["metric"] == "er_minus_random")].sort_values("c")
        ax.fill_between(g["c"], g["mean_ci_low"], g["mean_ci_high"], color=MCOL[method], alpha=0.18, lw=0)
        ax.plot(g["c"], g["mean_diff"], "-o", color=MCOL[method], lw=2, ms=4, label=MLAB[method])
    ax.axhline(0, color="#8a8984", lw=1); ax.set_xscale("log")
    ax.set_xlabel("stabilization margin c"); ax.set_ylabel("ER − mean of 100 random sets")
    ax.set_title(f"{'DEF'[j]}  vs random, {scale} regions", loc="left", fontsize=9)
axes[1, 0].legend(frameon=False)
fig.suptitle("Draft: c sweep, ε = 1e-5, infinite horizon, k ∈ {1,2,3,5} averaged within subject (n = 70)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_csweep_draft.png", dpi=200)

# Perron figure
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
ax = axes[0]
for scale in SCALES:
    g = t3s[t3s["scale"] == scale].sort_values("c")
    ax.plot(g["c"], g["median_rho_within_subject_k"], "-o", color=COL[scale], lw=2, ms=4, label=f"{scale}")
    ax.fill_between(g["c"], g["q25_rho"], g["q75_rho"], color=COL[scale], alpha=0.15, lw=0)
ax.axhline(0, color="#8a8984", lw=1); ax.set_xscale("log"); ax.legend(frameon=False, title="regions")
ax.set_xlabel("c"); ax.set_ylabel("Spearman(Perron loading, ER)\nacross 109 sets (median, IQR)")
ax.set_title("A  Perron loading vs ER", loc="left", fontsize=9)
for j, scale in enumerate((68, 219)):
    ax = axes[1 + j]
    for method in list(MCOL) + ["random_mean"]:
        g = t3m[(t3m["method"] == method) & (t3m["scale"] == scale)].sort_values("c")
        ax.plot(g["c"], g["perron_trace_fraction"], "-o", color=MCOL.get(method, "#8a8984"), lw=2, ms=4, label=MLAB.get(method, "Random (mean)"))
    ax.set_xscale("log"); ax.set_xlabel("c"); ax.set_ylabel("vᵀWv / tr W")
    ax.set_title(f"{'BC'[j]}  Perron share of tr W, {scale} regions", loc="left", fontsize=9)
axes[1].legend(frameon=False, loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_perron_draft.png", dpi=200)

pd.set_option("display.width", 200)
print(t1[t1.metric == "effective_rank"][["scale", "c", "mean_diff", "median_diff", "n_positive", "p_holm", "rank_biserial"]].to_string())
print(flip.to_string())
print(t3s.to_string())
ctx.finish()
