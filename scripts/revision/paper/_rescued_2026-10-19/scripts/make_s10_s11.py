"""Read-only generation of S10/S11 LaTeX tables from saved results."""
import re
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R = "/Users/cartersale/Research/projects/higher_order_connectomes/results/revision/"
OUT = "/private/tmp/claude-501/-Users-cartersale-Research-projects-higher-order-connectomes/cea7c7b4-c39c-4a27-8881-388281d3a80d/scratchpad/tables/"
SC = [68, 114, 219]


def pf(p):
    """p-value formatting: exact value, 2 significant digits."""
    if pd.isna(p):
        return "--"
    if p >= 0.01:
        return f"{p:.2g}" if p < 0.1 else f"{p:.2f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def f2(x, sign=True):
    if pd.isna(x):
        return "--"
    s = f"{x:+.2f}" if sign else f"{x:.2f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def tex_esc(s):
    return s.replace("_", "\\_")


LABEL = OrderedDict([
    ("net_Visual", "Yeo Visual"), ("net_SomMot", "Yeo SomMot"), ("net_DorsAttn", "Yeo DorsAttn"),
    ("net_VentAttn", "Yeo VentAttn"), ("net_Limbic", "Yeo Limbic"), ("net_FrontoParietal", "Yeo FrontoParietal"),
    ("net_Default", "Yeo Default"), ("net_unimodal_VisSomMot", "Unimodal (Vis+SomMot)"),
    ("net_transmodal_DefFP", "Transmodal (Def+FP)"), ("net_transmodal_vs_unimodal", "Transmodal $-$ unimodal"),
    ("kw_occipital_module", "Occipital module (corrected)"), ("kw_association_module", "Association module"),
    ("kw_association_vs_occipital", "Association $-$ occipital (corr.)"),
    ("legacy_occipital_module", "Occipital module (legacy, +precuneus)"),
    ("legacy_association_vs_occipital", "Association $-$ occipital (legacy)"),
    ("landmark_v1", "V1 (pericalcarine)"), ("landmark_m1", "M1 (precentral)"),
    ("landmark_pcc_precuneus", "PCC/precuneus"), ("landmark_lpfc", "lPFC"),
    ("principal_gradient", "G1 (registered)"), ("bipolar_gradient", "G1 bipolar (registered)"),
    ("legacy_principal_gradient", "G1 (legacy, misregistered)"),
    ("second_gradient", "G2 (registered)"), ("uniform_cortical", "Uniform cortical"),
])
GROUPS = [
    ("Yeo-7 networks", ["net_Visual", "net_SomMot", "net_DorsAttn", "net_VentAttn", "net_Limbic",
                        "net_FrontoParietal", "net_Default", "net_unimodal_VisSomMot", "net_transmodal_DefFP",
                        "net_transmodal_vs_unimodal"]),
    ("Keyword modules", ["kw_occipital_module", "kw_association_module", "kw_association_vs_occipital",
                         "legacy_occipital_module", "legacy_association_vs_occipital"]),
    ("Landmarks", ["landmark_v1", "landmark_m1", "landmark_pcc_precuneus", "landmark_lpfc"]),
    ("Gradients / distributed", ["principal_gradient", "bipolar_gradient", "legacy_principal_gradient",
                                 "second_gradient", "uniform_cortical"]),
]

# ---------------------------------------------------------------- S10
T = R + "targets/"
nts = pd.read_csv(T + "named_target_summary.csv")
out = []


def named_table(variant, caption):
    d = nts[nts.variant == variant].set_index(["target_name", "scale"])
    L = [f"% {caption}", "\\begin{tabular}{l" + "rcr" * 3 + "}", "\\toprule",
         " & \\multicolumn{3}{c}{68} & \\multicolumn{3}{c}{114} & \\multicolumn{3}{c}{219} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
         "Target & Median \\% & $n_-/n_+$ & $p_H$ & Median \\% & $n_-/n_+$ & $p_H$ & Median \\% & $n_-/n_+$ & $p_H$ \\\\",
         "\\midrule"]
    for gi, (g, tg) in enumerate(GROUPS):
        if gi:
            L.append("\\midrule")
        L.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{g}}}}} \\\\")
        for t in tg:
            cells = []
            for s in SC:
                r = d.loc[(t, s)]
                cells += [f2(r.median_pct), f"{int(r.n_negative)}/{int(r.n_positive)}", pf(r.p_holm)]
            L.append(LABEL[t] + " & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(L)


out.append("% ===== Table S10.1: named targets, full targets =====")
out.append(named_table("full", "Named targets, full. pct = 100(E_cyc-E_deg)/E_deg, k-averaged; c=0.1, eps=1e-5"))
out.append("% ===== Table S10.2: named targets, drivers excluded =====")
out.append(named_table("drivers_excluded", "Named targets, union of both methods' drivers zeroed and target renormalized"))

# overlap
ov = pd.read_csv(T + "driver_target_overlap.csv").set_index(["target_name", "scale"])
OVT = ["net_Visual", "net_SomMot", "net_DorsAttn", "net_VentAttn", "net_Limbic", "net_FrontoParietal",
       "net_Default", "kw_occipital_module", "kw_association_module", "landmark_v1", "landmark_m1",
       "landmark_pcc_precuneus", "landmark_lpfc"]
L = ["% ===== Table S10.3: driver-target overlap (fraction of subject x k cases with >=1 driver inside target; cycle / degree) =====",
     "\\begin{tabular}{lccc}", "\\toprule", "Target & 68 & 114 & 219 \\\\", "\\midrule"]
for t in OVT:
    L.append(LABEL[t] + " & " + " & ".join(
        f"{ov.loc[(t, s)].frac_any_driver_in_cyc:.2f} / {ov.loc[(t, s)].frac_any_driver_in_deg:.2f}" for s in SC) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", ""]
out.append("\n".join(L))

# distance / coupling / proximity controls
pc = pd.read_csv(T + "proximity_controls.csv").set_index(["target_name", "scale"])
fu = nts[nts.variant == "full"].set_index(["target_name", "scale"])
DT = ["net_Visual", "kw_occipital_module", "net_DorsAttn", "net_VentAttn", "net_Default", "net_SomMot",
      "kw_association_module", "landmark_v1", "principal_gradient", "second_gradient"]
L = ["% ===== Table S10.4: proximity. dDist = cyc-deg target-mass-weighted distance to nearest driver (mm), k-avg; rho across 70 subjects (Spearman); intercept = median-regression pct at dDist=0 [bootstrap 95% CI]; no-overlap = subset of (subject,k) with no driver of either method in target =====",
     "\\begin{tabular}{llrcrrcc}", "\\toprule",
     "Target & Scale & Med.\\ $\\Delta$dist (mm) & Cyc.\\ closer & $\\rho$(pct,$\\Delta$dist) & $\\rho$(pct,$\\Delta$coupl.) & Equal-dist.\\ intercept \\% & No-overlap median \\% ($n$; $p_H$) \\\\",
     "\\midrule"]
for i, t in enumerate(DT):
    if i:
        L.append("\\addlinespace")
    for s in SC:
        r = fu.loc[(t, s)]
        q = pc.loc[(t, s)] if (t, s) in pc.index else pd.Series(dict(noov_median_pct=np.nan, eqdist_intercept=np.nan, eqdist_ci_low=np.nan, eqdist_ci_high=np.nan, missing=True))
        if q.get('missing', False):
            no = "not computed"
        elif pd.isna(q.noov_median_pct):
            no = "n/a ($<$20 subj.)"
        else:
            no = f"{f2(q.noov_median_pct)} ({int(q.noov_n)}; {pf(q.noov_p_holm)})"
        L.append(f"{LABEL[t] if s == 68 else ''} & {s} & {f2(r.median_d_dist_mm)} & {int(r.n_subj_cycle_closer)}/70 & "
                 f"{f2(r.rho_pct_vs_ddist)} & {f2(r.rho_pct_vs_dcoupling)} & "
                 f"{f2(q.eqdist_intercept)} [{f2(q.eqdist_ci_low)}, {f2(q.eqdist_ci_high)}] & {no} \\\\")
L += ["\\bottomrule", "\\end{tabular}", ""]
out.append("\n".join(L))

# single parcel
sp = pd.read_csv(T + "single_parcel_summary.csv")
L = ["% ===== Table S10.5: single-parcel targets =====",
     "\\begin{tabular}{llrcccc}", "\\toprule",
     "Variant & Scale & Median parcel \\% [95\\% CI] & Sig.\\ neg./pos.\\ (BH) & $\\rho$(pct,$\\Delta$dist) & $\\rho$(pct,$\\Delta$coupl.) & Within-subj.\\ $\\rho$(dist) \\\\",
     "\\midrule"]
for _, r in sp.iterrows():
    L.append(f"{'Full' if r.variant == 'full' else 'Drivers excl.'} & {r.scale} & {f2(r.median_parcel_mean_pct)} [{f2(r.ci_low)}, {f2(r.ci_high)}] & "
             f"{int(r.n_sig_negative)}/{int(r.n_sig_positive)} of {int(r.n_parcels)} & {f2(r.rho_parcels_pct_vs_ddist)} & "
             f"{f2(r.rho_parcels_pct_vs_dcoupling)} & {f2(r.median_within_subject_rho_ddist)} ({int(r.n_subj_within_rho_ddist_positive)}/70 $>0$) \\\\")
L += ["\\bottomrule", "\\end{tabular}", ""]
out.append("\n".join(L))

# sensitivity
ss = pd.read_csv(T + "sensitivity_summary.csv")
CONDS = [(0.1, 1e-8), (0.1, 1e-5), (0.1, 1e-2), (0.01, 1e-5), (1.0, 1e-5)]
ST = ["net_Visual", "kw_occipital_module", "net_DorsAttn", "net_VentAttn", "net_Default", "net_SomMot",
      "kw_association_module", "kw_association_vs_occipital", "net_transmodal_vs_unimodal", "landmark_v1",
      "principal_gradient", "second_gradient", "uniform_cortical"]


def sens_table(variant):
    d = ss[ss.variant == variant]
    L = [f"% ===== Table S10.6{'a' if variant == 'full' else 'b'}: eps/c sensitivity, {variant}; median pct, * = Holm p<0.05 (Holm within scale x family x condition) =====",
         "\\begin{tabular}{llrrrrr}", "\\toprule",
         " & & \\multicolumn{3}{c}{$c=0.1$} & \\multicolumn{2}{c}{$\\varepsilon=10^{-5}$} \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}",
         "Target & Scale & $\\varepsilon=10^{-8}$ & $10^{-5}$ & $10^{-2}$ & $c=0.01$ & $c=1$ \\\\", "\\midrule"]
    for i, t in enumerate(ST):
        if i:
            L.append("\\addlinespace")
        for s in SC:
            cells = []
            for c, e in CONDS:
                r = d[(d.target_name == t) & (d.scale == s) & np.isclose(d.c, c) & np.isclose(d.eps, e)].iloc[0]
                cells.append(f2(r.median_pct) + ("$^*$" if r.p_holm < 0.05 else ""))
            L.append(f"{LABEL[t] if s == 68 else ''} & {s} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(L)


out.append(sens_table("full"))
out.append(sens_table("drivers_excluded"))

# floor fraction (values transcribed from REPORT.md section 5 table, generated by 08 script)
FF = {(68, "1e-08"): (0.927, 0.934), (68, "1e-05"): (0.930, 0.931), (68, "1e-02"): (0.929, 0.922),
      (114, "1e-08"): (0.959, 0.957), (114, "1e-05"): (0.960, 0.954), (114, "1e-02"): (0.954, 0.950),
      (219, "1e-08"): (0.977, 0.974), (219, "1e-05"): (0.976, 0.972), (219, "1e-02"): (0.964, 0.962)}
sl = pd.read_csv(T + "sensitivity_long.csv.gz")
sl = sl[(sl.c == 0.1) & (sl.eps == 1e-5)]
tmed = sl.groupby(["scale", "target_name"]).floor_frac.median().groupby("scale").agg(["min", "max"])
L = ["% ===== Table S10.7: fraction of target energy carried by eigen-directions at the eps floor (c = 0.1) =====",
     "\\begin{tabular}{llccc}", "\\toprule",
     "Scale & $\\varepsilon$ & Degree sets & Cycle sets & Range of per-target medians ($\\varepsilon=10^{-5}$) \\\\", "\\midrule"]
for s in SC:
    for e in ["1e-08", "1e-05", "1e-02"]:
        dg, cy = FF[(s, e)]
        rng = f"{tmed.loc[s, 'min']:.2f}--{tmed.loc[s, 'max']:.2f}" if e == "1e-05" else ""
        L.append(f"{s if e == '1e-08' else ''} & $10^{{{int(e.split('e')[1])}}}$ & {dg:.2f} & {cy:.2f} & {rng} \\\\")
L += ["\\bottomrule", "\\end{tabular}", ""]
out.append("\n".join(L))

# spin tests
sp = pd.read_csv(T + "spin_tests.csv")
YN = {"rank_divergence": "Rank divergence", "degree_strength": "Degree strength",
      "cycle_participation": "Cycle participation", "G1_legacy_projection": "G1 (legacy, misregistered)",
      "G1_registered": "G1 (registered)", "G2_registered": "G2 (registered)"}
L = ["% ===== Table S10.8: spin tests (10,000 rotations; Vasa primary, Alexander-Bloch secondary); two-sided p =====",
     "\\begin{tabular}{lllrccc}", "\\toprule",
     "Scale & $x$ & $y$ & Spearman $\\rho$ & $p_{\\mathrm{param}}$ & $p_{\\mathrm{spin}}$ (Vasa) & $p_{\\mathrm{spin}}$ (A--B) \\\\", "\\midrule"]
for s in SC:
    if s != 68:
        L.append("\\addlinespace")
    for _, r in sp[sp.scale == s].iterrows():
        L.append(f"{s} & {YN[r.x]} & {YN[r.y]} & {f2(r.spearman_rho)} & {pf(r.parametric_p)} & {r.p_spin_vasa:.4f} & {r.p_spin_original:.4f} \\\\")
L += ["\\bottomrule", "\\end{tabular}", ""]
out.append("\n".join(L))

# rank divergence by network
rd = pd.read_csv(T + "rank_divergence_network_enrichment.csv").set_index(["network", "scale"])
L = ["% ===== Table S10.9: mean rank divergence (degree rank - cycle rank; + = cycle priority) by Yeo-7 network; n parcels; Vasa spin p (uncorrected) =====",
     "\\begin{tabular}{lccc}", "\\toprule", "Network & 68 & 114 & 219 \\\\", "\\midrule"]
for n in ["Visual", "SomMot", "DorsAttn", "VentAttn", "Limbic", "FrontoParietal", "Default"]:
    L.append(n + " & " + " & ".join(
        f"{f2(rd.loc[(n, s)].mean_rank_divergence)} ($n$={int(rd.loc[(n, s)].n_parcels)}; $p$={rd.loc[(n, s)].p_spin_vasa:.2f})" for s in SC) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", ""]
out.append("\n".join(L))

# parcel lists (compact)
mp = pd.read_csv(T + "target_mask_parcels.csv")


def compact(lst):
    if not isinstance(lst, str) or not lst:
        return ""
    groups = OrderedDict()
    for p in lst.split(";"):
        m = re.match(r"([LR])_([a-z]+)(?:_(\d+))?$", p)
        h, base, idx = m.group(1), m.group(2), m.group(3)
        groups.setdefault(base, {"L": [], "R": []})[h].append(idx)
    parts = []
    for base, hh in groups.items():
        segs = []
        for h in "LR":
            ids = hh[h]
            if not ids:
                continue
            if ids == [None]:
                segs.append(h)
            else:
                segs.append(h + ",".join(ids))
        parts.append(f"{base} ({'; '.join(segs)})")
    return ", ".join(parts)


MT = ["kw_occipital_module", "legacy_occipital_module", "kw_association_module", "net_Visual", "net_SomMot",
      "net_DorsAttn", "net_VentAttn", "net_Limbic", "net_FrontoParietal", "net_Default", "landmark_v1",
      "landmark_m1", "landmark_pcc_precuneus", "landmark_lpfc"]
L = ["% ===== Table S10.10: exact parcels per mask. Format: base region (L idx; R idx); 'L'/'R' alone = unsubdivided parcel at scale 68 =====",
     "\\begin{tabular}{llr p{12cm}}", "\\toprule", "Target & Scale & $n$ & Parcels \\\\", "\\midrule"]
for i, t in enumerate(MT):
    if i:
        L.append("\\addlinespace")
    for s in SC:
        r = mp[(mp.target_name == t) & (mp.scale == s)].iloc[0]
        L.append(f"{LABEL[t] if s == 68 else ''} & {s} & {int(r.n_positive)} & {tex_esc(compact(r.positive_parcels))} \\\\")
L += ["\\bottomrule", "\\end{tabular}", ""]
out.append("\n".join(L))
open(OUT + "S10_targets.tex", "w").write("\n".join(out))

# ---------------------------------------------------------------- S11
LE = R + "lesion/"
s32 = pd.read_csv(LE + "summary_3x2.csv")
cs = pd.read_csv(LE + "crossover_stats.csv")
out = []
L = ["% ===== Table S11.1: static protocol, mean loss (intact - lesioned) per subject pooled over scale, h in {5,10,15}, k in {1,2,3,5}; c=0.1, eps=1e-5. Percentile loss in percentile points (x100). Positive = degradation =====",
     "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
     " & \\multicolumn{2}{c}{ER loss} & \\multicolumn{2}{c}{PR loss} & \\multicolumn{2}{c}{ER-percentile loss} & \\multicolumn{2}{c}{PR-percentile loss} \\\\",
     "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
     "Lesion & Degree & Cycle & Degree & Cycle & Degree & Cycle & Degree & Cycle \\\\", "\\midrule"]
st = s32[s32.protocol == "static"].set_index(["lesion_type", "strategy"])
for lt in ["degree", "cycle", "random"]:
    dg, cy = st.loc[(lt, "degree_strength")], st.loc[(lt, "cycle_participation")]
    L.append(f"{lt.capitalize()} hubs ($h$ top) " if lt != "random" else "Random ($h$ nodes) ")
    L[-1] = (f"{lt.capitalize()} lesion" if lt != "random" else "Random lesion") + \
        f" & {f2(dg.dER)} & {f2(cy.dER)} & {f2(dg.dPR)} & {f2(cy.dPR)} & {100 * dg.dPct_ER:+.1f} & {100 * cy.dPct_ER:+.1f} & {100 * dg.dPct_PR:+.1f} & {100 * cy.dPct_PR:+.1f} \\\\"
    L[-1] = L[-1].replace(" -", " $-$")
L += ["\\bottomrule", "\\end{tabular}", ""]
out.append("\n".join(L))

# crossover tests
MN = {"dER": "ER", "dPR": "PR", "dPct_ER": "ER-pct", "dPct_PR": "PR-pct", "dER_excess": "ER excess"}
L = ["% ===== Table S11.2: crossover tests on pooled D = cycle loss - degree loss (static). D>0 = cycle sets degrade more. Bootstrap 95% CI of mean; Wilcoxon; Holm within metric x protocol (30 tests: 27 cells + 3 pooled); interactions Holm in separate family of 30. Pct metrics in percentile points (x100) =====",
     "\\begin{tabular}{llrcccr}", "\\toprule",
     "Metric & Lesion & Mean $D$ & 95\\% CI & $n(D>0)$ & $p_H$ & $r_{rb}$ \\\\", "\\midrule"]
for i, m in enumerate(["dER", "dPR", "dPct_ER", "dPct_PR"]):
    if i:
        L.append("\\addlinespace")
    mult = 100 if "Pct" in m else 1
    for lt in ["degree", "cycle", "random", "interaction cycle-degree"]:
        r = cs[(cs.metric == m) & (cs.protocol == "static") & (cs.lesion_type == lt) & (cs.scale == "pooled")].iloc[0]
        nm = "Interaction (cyc.\\ $D$ $-$ deg.\\ $D$)" if lt.startswith("inter") else lt.capitalize()
        if mult == 1:
            v, lo, hi = f2(r["mean"]), f2(r.ci_lo), f2(r.ci_hi)
        else:
            v, lo, hi = [f"{mult * x:+.1f}".replace("-", "$-$") for x in (r["mean"], r.ci_lo, r.ci_hi)]
        L.append(f"{MN[m] if lt == 'degree' else ''} & {nm} & {v} & [{lo}, {hi}] & {int(r.n_pos)}/70 & {pf(r.p_holm)} & {f2(r.r_rb)} \\\\")
L += ["\\bottomrule", "\\end{tabular}", ""]
out.append("\n".join(L))

# Holm-significant cells
L = ["% ===== Table S11.3: number of Holm-significant (p_H<0.05) scale x h cells out of 9, by sign of D (static); computed from crossover_stats.csv =====",
     "\\begin{tabular}{lccc}", "\\toprule",
     "Metric & Degree lesion & Cycle lesion & Random lesion \\\\", "\\midrule"]
for m, nm in [("dER", "ER"), ("dPR", "PR"), ("dER_excess", "ER excess"), ("dPct_ER", "ER percentile"),
              ("dPct_PR", "PR percentile"), ("dER_perN", "ER/$N$ (not recommended)")]:
    cells = []
    for lt in ["degree", "cycle", "random"]:
        c = cs[(cs.metric == m) & (cs.protocol == "static") & (cs.lesion_type == lt) & (cs.scale != "pooled")]
        assert len(c) == 9
        sig = c[c.p_holm < 0.05]
        neg, pos = int((sig["mean"] < 0).sum()), int((sig["mean"] > 0).sum())
        cells.append(f"{neg} ($D<0$) / {pos} ($D>0$)")
    L.append(nm + " & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", ""]
out.append("\n".join(L))

# protocol dependence
rc = s32[s32.protocol == "recompute"].set_index(["lesion_type", "strategy"])
L = ["% ===== Table S11.4: protocol dependence. Mean ER and PR loss (pooled) under static vs recompute protocols; D = cycle - degree with n(D>0)/70 and Holm p =====",
     "\\begin{tabular}{llrrrcc rrrcc}", "\\toprule",
     " & & \\multicolumn{5}{c}{ER loss} & \\multicolumn{5}{c}{PR loss} \\\\",
     "\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}",
     "Protocol & Lesion & Degree & Cycle & $D$ & $n(D>0)$ & $p_H$ & Degree & Cycle & $D$ & $n(D>0)$ & $p_H$ \\\\", "\\midrule"]
for prot, tab in [("static", st), ("recompute", rc)]:
    for lt in ["degree", "cycle", "random"]:
        cells = []
        for m in ["dER", "dPR"]:
            r = cs[(cs.metric == m) & (cs.protocol == prot) & (cs.lesion_type == lt) & (cs.scale == "pooled")].iloc[0]
            cells += [f2(r.mean_degree), f2(r.mean_cycle), f2(r["mean"]), f"{int(r.n_pos)}/70", pf(r.p_holm)]
        L.append(f"{prot.capitalize() if lt == 'degree' else ''} & {lt.capitalize()} & " + " & ".join(cells) + " \\\\")
    if prot == "static":
        L.append("\\addlinespace")
L += ["\\bottomrule", "\\end{tabular}", ""]
out.append("\n".join(L))

# absolute post-lesion ER (computed from per_subject_degradation.csv)
d = pd.read_csv(LE + "per_subject_degradation.csv")
L = ["% ===== Table S11.5: absolute post-lesion ER. Mean lesioned ER (degree / cycle sets) and per-subject D = ER_cyc - ER_deg after lesion, averaged over h (and over scales for 'pooled'); n(D>0)/70; uncorrected two-sided Wilcoxon. Computed read-only from per_subject_degradation.csv =====",
     "\\begin{tabular}{lllcccc}", "\\toprule",
     "Protocol & Lesion & Scale & Intact ER (deg / cyc) & Lesioned ER (deg / cyc) & $D$ (cyc $-$ deg) & $n(D>0)$; $p$ \\\\", "\\midrule"]
for prot in ["static", "recompute"]:
    s = d[d.protocol == prot]
    p = s.pivot_table(index=["lesion_type", "scale", "h", "subject"], columns="strategy",
                      values=["lesioned_effective_rank", "intact_effective_rank"])
    D = p[("lesioned_effective_rank", "cycle_participation")] - p[("lesioned_effective_rank", "degree_strength")]
    for lt in ["degree", "cycle", "random"]:
        for sc in ["pooled"] + SC:
            if sc == "pooled":
                g = D.loc[lt].groupby(level="subject").mean()
                pp = p.loc[lt]
            else:
                g = D.loc[(lt, sc)].groupby(level="subject").mean()
                pp = p.loc[(lt, sc)]
            I = pp["intact_effective_rank"].mean()
            Ls = pp["lesioned_effective_rank"].mean()
            L.append(f"{prot.capitalize() if (lt == 'degree' and sc == 'pooled') else ''} & {lt.capitalize() if sc == 'pooled' else ''} & {sc} & "
                     f"{I.degree_strength:.2f} / {I.cycle_participation:.2f} & {Ls.degree_strength:.2f} / {Ls.cycle_participation:.2f} & "
                     f"{f2(g.mean())} & {(g > 0).sum()}/70; {pf(wilcoxon(g).pvalue)} \\\\")
        L.append("\\addlinespace")
    if prot == "static":
        L.append("\\midrule")
L = L[:-1] if L[-1] == "\\addlinespace" else L
L += ["\\bottomrule", "\\end{tabular}", ""]
out.append("\n".join(L))

# energy-size diagnostics
es = pd.read_csv(LE + "energy_size_diagnostics.csv")
L = ["% ===== Table S11.6: lesioned - intact log10 E_avg (static protocol) vs size prediction log10((N-h)/N); floor fraction = fraction of W+eps*I eigenvalues at eps (mean of the two strategies' values in energy_size_diagnostics.csv) =====",
     "\\begin{tabular}{llrccccc}", "\\toprule",
     "Scale & Lesion & $h$ & Floor frac.\\ intact $\\to$ lesioned & $\\Delta\\log_{10}E$ deg. & $\\Delta\\log_{10}E$ cyc. & Random-set ref. & $\\log_{10}\\frac{N-h}{N}$ \\\\", "\\midrule"]
for sc in SC:
    for lt in ["degree", "cycle", "random"]:
        for h in [5, 10, 15]:
            g = es[(es.scale == sc) & (es.lesion_type == lt) & (es.h == h)].set_index("strategy")
            fi, fl = g.frac_floor_intact.mean(), g.frac_floor_lesioned.mean()
            dd, dc = g.loc["degree_strength"].dlog10E, g.loc["cycle_participation"].dlog10E
            ref = g.ref_dlog10E.iloc[0]
            L.append(f"{sc if (lt == 'degree' and h == 5) else ''} & {lt.capitalize() if h == 5 else ''} & {h} & {fi:.2f} $\\to$ {fl:.2f} & "
                     f"{dd:+.3f} & {dc:+.3f} & {ref:+.3f} & {g.size_pred.iloc[0]:+.3f} \\\\".replace("+-", "-").replace(" -0", " $-$0"))
    if sc != 219:
        L.append("\\addlinespace")
L += ["\\midrule",
      "\\multicolumn{8}{l}{corr($\\Delta\\log_{10}E$, $\\log_{10}((N-h)/N)$) $= 0.989$; corr($\\Delta\\log_{10}E$, $\\log_{10}(n_{\\mathrm{floor,les}}/n_{\\mathrm{floor,int}})$) $= 0.975$ (all rows).} \\\\"]
esd = cs[(cs.metric == "dlog10E") & (cs.protocol == "static") & (cs.scale == "pooled")].set_index("lesion_type")
txt = "; ".join(f"{lt}: {esd.loc[lt]['mean']:+.4f} ({int(esd.loc[lt].n_pos)}/70, $p_H$={pf(esd.loc[lt].p_holm)})" for lt in ["degree", "cycle", "random"]).replace(": -", ": $-$")
L += [f"\\multicolumn{{8}}{{l}}{{Pooled cyc.\\ $-$ deg.\\ $\\Delta\\log_{{10}}E$: {txt}}} \\\\".replace("+-", "-"),
      "\\bottomrule", "\\end{tabular}", ""]
out.append("\n".join(L))
open(OUT + "S11_lesion.tex", "w").write("\n".join(out))
print("ok")
