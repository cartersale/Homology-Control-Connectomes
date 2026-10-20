#!/usr/bin/env python
"""Draft revision figure for the target-state analyses -> results/revision/targets/fig_targets_draft.png"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
SCALE_COL = {68: "#2a78d6", 114: "#eb6834", 219: "#1baf7a"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e8e6", "grid.linewidth": 0.6})

named = pd.read_csv(C.OUT / "named_target_summary.csv")
ov = pd.read_csv(C.OUT / "driver_target_overlap.csv")
sens = pd.read_csv(C.OUT / "sensitivity_summary.csv")
spin = pd.read_csv(C.OUT / "spin_tests.csv")
maps = pd.read_csv(C.OUT / "parcel_maps_for_spin.csv")
sp = pd.read_csv(C.OUT / "single_parcel_tests.csv")
assign = pd.read_csv(C.OUT / "parcel_network_assignment.csv")

TARGETS = [("legacy_occipital_module", "Occipital (legacy mask)"), ("kw_occipital_module", "Occipital (corrected)"),
           ("net_Visual", "Yeo Visual"), ("net_SomMot", "Yeo SomMot"), ("net_DorsAttn", "Yeo DorsAttn"),
           ("net_VentAttn", "Yeo VentAttn"), ("net_Limbic", "Yeo Limbic"), ("net_FrontoParietal", "Yeo FrontoParietal"),
           ("net_Default", "Yeo Default"), ("net_unimodal_VisSomMot", "Unimodal (Vis+SomMot)"),
           ("net_transmodal_DefFP", "Transmodal (Def+FP)"), ("kw_association_module", "Association (keyword)"),
           ("principal_gradient", "Principal gradient (registered)"), ("uniform_cortical", "Uniform cortical")]

fig = plt.figure(figsize=(13, 9.5))
gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1], hspace=0.38, wspace=0.42)

# A: forest, full vs driver-excluded
ax = fig.add_subplot(gs[:, 0])
for i, (t, lbl) in enumerate(TARGETS):
    for j, s in enumerate(C.SCALES):
        for variant, mk, off in (("full", "o", -0.1), ("drivers_excluded", "D", 0.1)):
            r = named[(named.target_name == t) & (named.scale == s) & (named.variant == variant)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = i + (j - 1) * 0.26 + off * 0.8
            face = SCALE_COL[s] if r.p_holm < 0.05 else "white"
            ax.plot([r.ci_low, r.ci_high], [y, y], color=SCALE_COL[s], lw=1.1)
            ax.plot(r.median_pct, y, mk, ms=4.5, mfc=face, mec=SCALE_COL[s], mew=1.0)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(range(len(TARGETS)))
ax.set_yticklabels([l for _, l in TARGETS])
ax.invert_yaxis()
ax.set_xlabel("Median % difference, 100·(E_cycle − E_degree)/E_degree\n(< 0: cycle drivers cheaper)")
ax.set_title("A  Target energy: full (●) vs drivers excluded (◆)\nfilled = Holm p < .05; colour = scale", loc="left", fontsize=9)
for s in C.SCALES:
    ax.plot([], [], "o", color=SCALE_COL[s], label=f"{s} regions")
ax.legend(loc="lower right", frameon=False)
ax.grid(axis="y", visible=False)

# B: overlap (fraction of subject x k cases with >= 1 driver inside network), 114 regions
ax = fig.add_subplot(gs[0, 1])
nets = ["net_Visual", "net_SomMot", "net_DorsAttn", "net_VentAttn", "net_Limbic", "net_FrontoParietal", "net_Default"]
x = np.arange(len(nets))
for s, off in zip(C.SCALES, (-0.27, 0, 0.27)):
    o = ov[(ov.scale == s)].set_index("target_name").loc[nets]
    ax.bar(x + off - 0.0, o.frac_any_driver_in_cyc, 0.12, color=ORANGE, alpha=[1, 0.7, 0.45][C.SCALES.index(s)], label=f"cycle {s}")
    ax.bar(x + off + 0.12, o.frac_any_driver_in_deg, 0.12, color=BLUE, alpha=[1, 0.7, 0.45][C.SCALES.index(s)], label=f"degree {s}")
ax.set_xticks(x + 0.06)
ax.set_xticklabels([n.replace("net_", "").replace("FrontoParietal", "FP") for n in nets], rotation=35, ha="right")
ax.set_ylabel("Fraction of subject × k with ≥1 driver in network")
ax.set_title("B  Driver–target overlap\n(groups: 68 / 114 / 219)", loc="left", fontsize=9)
ax.legend(frameon=False, fontsize=6, ncol=2)

# C: single-parcel pct vs coupling difference (114)
ax = fig.add_subplot(gs[0, 2])
p = sp[(sp.variant == "drivers_excluded") & (sp.scale == 114)].copy()
p["node_index"] = p.target_name.str[-3:].astype(int)
p = p.merge(assign[assign.scale == 114][["node_index", "yeo7_name"]], on="node_index")
netcol = {"Visual": "#4a3aa7", "SomMot": "#1baf7a", "DorsAttn": "#008300", "VentAttn": "#e87ba4", "Limbic": "#eda100",
          "FrontoParietal": "#eb6834", "Default": "#e34948"}
for n, g in p.groupby("yeo7_name"):
    ax.scatter(g.mean_d_coupling, g.mean_pct, s=14, color=netcol.get(n, GRAY), label=n, edgecolor="white", linewidth=0.4)
row = pd.read_csv(C.OUT / "single_parcel_summary.csv").query("variant=='drivers_excluded' and scale==114").iloc[0]
ax.axhline(0, color=INK, lw=0.7)
ax.axvline(0, color=INK, lw=0.7)
ax.set_xlabel("Δ structural coupling to drivers (cycle − degree)")
ax.set_ylabel("Parcel mean % difference (symlog)"); ax.set_yscale("symlog", linthresh=5)
ax.set_title(f"C  Single-parcel targets, 114, drivers excluded\nSpearman ρ = {row.rho_parcels_pct_vs_dcoupling:.2f} across parcels", loc="left", fontsize=9)
ax.legend(frameon=False, fontsize=6, ncol=2)

# D: eps sensitivity
ax = fig.add_subplot(gs[1, 1])
sel = [("net_Visual", "#4a3aa7"), ("net_DorsAttn", "#008300"), ("net_VentAttn", "#e87ba4"), ("net_Default", "#e34948"), ("uniform_cortical", GRAY)]
for t, col in sel:
    for s, ls in zip(C.SCALES, (":", "-", "--")):
        r = sens[(sens.variant == "full") & (sens.c == 0.1) & (sens.target_name == t) & (sens.scale == s)].sort_values("eps")
        ax.plot(np.log10(r.eps), r.median_pct, ls, color=col, marker="o", ms=3, lw=1.2, label=t.replace("net_", "") if s == 114 else None)
ax.axhline(0, color=INK, lw=0.7)
ax.set_xticks([-8, -5, -2])
ax.set_xticklabels(["1e-8", "1e-5", "1e-2"])
ax.set_xlabel("ε (c = 0.1)   line style: 68 dotted, 114 solid, 219 dashed")
ax.set_ylabel("Median % difference")
ax.set_title("D  ε sensitivity of network targets", loc="left", fontsize=9)
ax.legend(frameon=False, fontsize=6)

# E: rank divergence vs G1, legacy vs registered (114)
ax = fig.add_subplot(gs[1, 2])
m = maps[maps.scale == 114]
for col, name, c in (("G1_legacy_projection", "legacy projection", GRAY), ("G1_registered", "registered", BLUE)):
    z = (m[col] - m[col].mean()) / m[col].std()
    r = spin[(spin.scale == 114) & (spin.x == "rank_divergence") & (spin.y == col)].iloc[0]
    ax.scatter(z, m.rank_divergence, s=12, color=c, alpha=0.75, edgecolor="white", linewidth=0.3,
               label=f"{name}: ρ = {r.spearman_rho:.2f}, p_spin = {r.p_spin_vasa:.3f}")
ax.axhline(0, color=INK, lw=0.7)
ax.set_xlabel("Principal gradient G1 (z; transmodal +)")
ax.set_ylabel("Rank divergence (degree − cycle rank)")
ax.set_title("E  Rank divergence vs G1, 114 regions", loc="left", fontsize=9)
ax.legend(frameon=False, fontsize=6, loc="lower left")

fig.savefig(C.OUT / "fig_targets_draft.png", dpi=200, bbox_inches="tight", facecolor="white")
print("saved")
