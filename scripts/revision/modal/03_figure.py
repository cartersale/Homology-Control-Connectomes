#!/usr/bin/env python
"""Draft figure: node-level controllability vs set-level Gramian concentration."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[3] / "results" / "revision" / "modal"
d = pd.read_csv(OUT / "set_level_modal.csv.gz")
cf = pd.read_csv(OUT / "sets_vs_random_summary.csv")
ws = pd.read_csv(OUT / "within_subject_summary.csv")
INK, MUTED = "#0b0b0b", "#52514e"
COL = {"cycle_participation": "#2a78d6", "degree_strength": "#eb6834", "top_modal_gu": "#1baf7a",
       "top_avg_ctrl": "#eda100", "eigenvector_centrality": "#e87ba4", "low_perron": "#4a3aa7"}
LAB = {"cycle_participation": "cycle participation", "degree_strength": "degree", "top_modal_gu": "top modal ctrl (Gu)",
       "top_avg_ctrl": "top average ctrl", "eigenvector_centrality": "eigenvector", "low_perron": "lowest Perron loading"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)

# A: sandwich perron_share <= q1_raw <= q1(G)
g = d[(d.scale == 114) & (d.k == 3) & (d.c == 0.1)]
a = ax[0]
a.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
a.scatter(g.q1_raw, g.q1_G, s=3, color="#2a78d6", alpha=0.25, lw=0, label=r"upper: $\lambda_1(G_S)/\mathrm{tr}\,G_S$ (pairwise)")
a.scatter(g.q1_raw, g.perron_share, s=3, color="#eb6834", alpha=0.25, lw=0, label=r"lower: Perron share (node-additive)")
a.set_xlabel(r"$q_1=\lambda_1(W)/\mathrm{tr}\,W$"); a.set_ylabel("bound")
a.set_title("A  Bounds on q1 (114 nodes, k=3, c=0.1)", loc="left", color=INK)
a.legend(frameon=False, markerscale=4, loc="upper left", fontsize=7)

# B: sets vs random percentiles
b = ax[1]
cs = [0.01, 0.1, 1.0]
for f, col in COL.items():
    for var, mk in [("q1_pct", "o"), ("effective_rank_pct", "s")]:
        x = cf[(cf.family == f) & (cf.variable == var) & (cf.k == 3) & (cf.scale == 114)].set_index("c").loc[cs, "median"]
        b.plot(np.log10(cs), x.values, marker=mk, color=col, lw=1.5 if var == "q1_pct" else 1, ls="-" if var == "q1_pct" else ":", ms=5,
               label=LAB[f] if var == "q1_pct" else None, mfc=col if var == "q1_pct" else "white")
b.axhline(0.5, color=MUTED, lw=0.8)
b.set_xticks(np.log10(cs)); b.set_xticklabels(cs); b.set_xlabel("c"); b.set_ylabel("median percentile vs 100 random sets")
b.set_title("B  Ranked sets vs random\n(filled: q1; open: eff. rank; degree, eigvec, top-avg overlap at q1=1)", loc="left", color=INK)
b.legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5))

# C: within-subject rho with q1
cc = ax[2]
P = {"rho_q1_perron_loading": ("Perron loading $L_S$", "#eb6834"), "rho_q1_avg_ctrl": ("mean avg ctrl", "#eda100"),
     "rho_q1_modal_gu": ("mean modal ctrl (Gu), sign-flipped", "#1baf7a"), "rho_q1_red_G": ("driver redundancy $\\bar r_G$", "#4a3aa7"),
     "rho_q1_q1_G": ("$q_1(G_S)$", "#2a78d6")}
for v, (lab, col) in P.items():
    s = ws[(ws.variable == v) & (ws.k == 3) & (ws.scale == 114)].set_index("c").loc[cs, "median"]
    sgn = -1 if "modal" in v else 1
    cc.plot(np.log10(cs), sgn * s.values, marker="o", color=col, lw=1.5, ms=5, label=lab)
cc.set_xticks(np.log10(cs)); cc.set_xticklabels(cs); cc.set_xlabel("c"); cc.set_ylabel(r"median within-subject Spearman $\rho$ with $q_1$")
cc.set_ylim(0, 1); cc.set_title("C  Predicting q1 within subject", loc="left", color=INK)
cc.legend(frameon=False, fontsize=6.5, loc="lower left")
fig.savefig(OUT / "fig_modal_draft.png", dpi=200)
print("saved")
