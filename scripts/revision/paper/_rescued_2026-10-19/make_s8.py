"""Build S8_nulls.tex from saved null-model results (read-only)."""
import numpy as np
import pandas as pd

R = "/Users/cartersale/Research/projects/higher_order_connectomes/results/revision/nulls/"
OUT = "/private/tmp/claude-501/-Users-cartersale-Research-projects-higher-order-connectomes/cea7c7b4-c39c-4a27-8881-388281d3a80d/scratchpad/tables/S8_nulls.tex"
NULLS = ["strength", "geometric", "weights"]
NL = {"strength": "Strength", "geometric": "Geometric", "weights": "Weight shuffle"}
SC = [68, 114, 219]


def f2(v):
    if pd.isna(v):
        return "--"
    s = f"{v:.2f}"
    return "0.00" if s == "-0.00" else s


def fz(v):
    return f2(v)


def fp(p):
    if pd.isna(p):
        return "--"
    if p >= 0.1:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.2g}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def n70(n):
    return f"{int(n)}/70"


L = []
add = L.append
add("% S8 -- Null networks. Auto-generated from results/revision/nulls/ (see S8_notes.txt).")
add("% Requires \\usepackage{booktabs}. Values rounded to 2 decimals; counts n/70.")
add("")

# ---------------- (a) fidelity
fid = pd.read_csv(R + "null_fidelity.csv")
add("% ---- Table S8.1: null fidelity (means over subjects and 50 nulls/subject)")
add("\\begin{tabular}{llrrrrrrr}")
add("\\toprule")
add("Network & Scale & Strength $r$ & Max rel.\\ strength err. & Top-5 hub Jaccard & Degree preserved & Edge overlap & Mean edge length & Weighted edge length \\\\")
add("\\midrule")
for nl in ["real"] + NULLS:
    for sc in SC:
        r = fid[(fid.null == nl) & (fid.scale == sc)].iloc[0]
        name = "Real" if nl == "real" else NL[nl]
        add(f"{name if sc == 68 else ''} & {sc} & {r.strength_r:.3f} & {f2(r.strength_max_rel_err)} & {f2(r.strength_top5_jaccard)} & {f2(r.degree_preserved)} & {f2(r.edge_overlap)} & {r.mean_edge_length:.2f} & {r.weighted_edge_length:.2f} \\\\")
    if nl != "weights":
        add("\\midrule")
add("\\bottomrule")
add("\\end{tabular}")
add("")

# ---------------- (b) spectral stats
gs = pd.read_csv(R + "gap_split.csv")
ni = pd.read_csv(R + "network_inference.csv")
add("% ---- Table S8.2: lambda1, lambda2, gap, IPR(v1): real vs each null (mean); median per-subject z; n above 95th / below 5th null pct; Wilcoxon p on z (uncorrected)")
add("\\begin{tabular}{llrrrrrrrrrrrr}")
add("\\toprule")
add(" & & \\multicolumn{4}{c}{68} & \\multicolumn{4}{c}{114} & \\multicolumn{4}{c}{219} \\\\")
add("\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14}")
add("Statistic & Null & Real & Null & $\\tilde z$ & $>$95th / $<$5th & Real & Null & $\\tilde z$ & $>$95th / $<$5th & Real & Null & $\\tilde z$ & $>$95th / $<$5th \\\\")
add("\\midrule")
for stat, lab, src in [("lambda1", "$\\lambda_1$", gs), ("lambda2", "$\\lambda_2$", gs), ("gap", "$\\lambda_1-\\lambda_2$", gs), ("ipr_v1", "IPR($v_1$)", ni)]:
    for nl in NULLS:
        cells = []
        for sc in SC:
            r = src[(src.null == nl) & (src.scale == sc) & (src.stat == stat)].iloc[0]
            cells += [f2(r.real_mean), f2(r.null_mean), fz(r.z_median), f"{n70(r.n_above95)} / {n70(r.n_below5)}"]
        add(f"{lab if nl == 'strength' else ''} & {NL[nl]} & " + " & ".join(cells) + " \\\\")
    add("\\midrule")
# lambda1 share of gap excess
add("\\multicolumn{14}{l}{Median per-subject share of the real$-$null gap excess carried by $\\lambda_1$ ($\\Delta\\lambda_1/\\Delta\\text{gap}$); count = subjects with share $>0.5$ / share $<0$} \\\\")
for nl in NULLS:
    cells = []
    for sc in SC:
        r = gs[(gs.null == nl) & (gs.scale == sc) & (gs.stat == "lambda1_share_of_gap_excess")].iloc[0]
        cells += [f"\\multicolumn{{3}}{{c}}{{{f2(r.delta_mean)}}}", f"{n70(r.n_above95)} / {n70(r.n_below5)}"]
    add(f"$\\Delta\\lambda_1$ share & {NL[nl]} & " + " & ".join(cells) + " \\\\")
add("\\bottomrule")
add("\\end{tabular}")
add("")

inf = pd.read_csv(R + "inference.csv")
inf["k"] = inf["k"].astype(str)


def inf_table(stat, title, cs=(0.01, 0.1, 1.0), modes=("fixed", "gap"), scales=SC, show_n=True):
    sub = inf[(inf.stat == stat) & (inf.k == "avg")]
    ncol = 4 if show_n else 3
    add(f"% ---- {title}")
    add("\\begin{tabular}{lll" + "r" * (ncol * len(scales)) + "}")
    add("\\toprule")
    add(" & & & " + " & ".join(f"\\multicolumn{{{ncol}}}{{c}}{{{sc}}}" for sc in scales) + " \\\\")
    cm = []
    for i, _ in enumerate(scales):
        a = 4 + i * ncol
        cm.append(f"\\cmidrule(lr){{{a}-{a + ncol - 1}}}")
    add("".join(cm))
    hdr = ["Mean", "$\\tilde z$"] + (["$>$95th"] if show_n else []) + ["$p_{\\mathrm{Holm}}$"]
    add("$c$ & $c$ mode & Network & " + " & ".join(" & ".join(hdr) for _ in scales) + " \\\\")
    add("\\midrule")
    for c in cs:
        # real row
        cells = []
        for sc in scales:
            r = sub[(sub.c_ref == c) & (sub.scale == sc)].iloc[0]
            cells += [f2(r.real_mean)] + [""] * (ncol - 1)
        add(f"{c:g} & -- & Real & " + " & ".join(cells) + " \\\\")
        for mode in modes:
            for nl in NULLS:
                cells = []
                for sc in scales:
                    r = sub[(sub.c_ref == c) & (sub.scale == sc) & (sub.null == nl) & (sub.c_mode == mode)].iloc[0]
                    cells += [f2(r.null_mean), fz(r.z_median)] + ([n70(r.n_above95)] if show_n else []) + [fp(r.p_holm)]
                ml = {"fixed": "fixed", "gap": "gap-matched"}[mode]
                add(f" & {ml if nl == 'strength' else ''} & {NL[nl]} & " + " & ".join(cells) + " \\\\")
        if c != cs[-1]:
            add("\\midrule")
    add("\\bottomrule")
    add("\\end{tabular}")
    add("")


# ---------------- (c) degree-set q1
inf_table("degree_q1", "Table S8.3: degree-set q1 (k-avg over k=1,2,3,5), real vs null mean, fixed and gap-matched c")
# ---------------- (d) W-perp
inf_table("degree_q1perp_pct", "Table S8.4a: W-perp residual -- percentile of degree-set q1(perp) among 100 random sets (k-avg)")
inf_table("degree_q1perp", "Table S8.4b: W-perp residual -- raw degree-set q1(perp) (k-avg)")

# ---------------- (e) Rayleigh accounting
ry = pd.read_csv(R + "rayleigh/summary.csv")
ps = pd.read_csv(R + "rayleigh/per_subject.csv")
for m in ["R1", "R2", "R3"]:
    ps[f"sl_{m}"] = ps[f"d_{m}"] / ps["d_lambda1"]
    ps[f"sg_{m}"] = ps[f"d_{m}"] / ps["d_gap"]
med = ps.groupby(["null", "scale"]).median(numeric_only=True)
net = pd.read_csv(R + "rayleigh/networks.csv.gz")
nm = net.groupby(["null", "scale"])[["cos_s_v1"]].mean()
add("% ---- Table S8.5: Rayleigh accounting of the real-null lambda1 excess (medians over 70 subjects of per-subject quantities)")
add("\\begin{tabular}{llrrr}")
add("\\toprule")
add("Quantity & Null & 68 & 114 & 219 \\\\")
add("\\midrule")
rows = [
    ("$\\Delta\\lambda_1$ (real $-$ null mean)", lambda nl, sc: med.loc[(nl, sc), "d_lambda1"]),
    ("$\\Delta$gap (real $-$ null mean)", lambda nl, sc: med.loc[(nl, sc), "d_gap"]),
    ("Share of $\\Delta\\lambda_1$: $R_0$ (mean strength)", lambda nl, sc: med.loc[(nl, sc), "share_l1_R0"]),
    ("Share of $\\Delta\\lambda_1$: $R_1=s^\\top Ws/s^\\top s$", lambda nl, sc: med.loc[(nl, sc), "sl_R1"]),
    ("Share of $\\Delta\\lambda_1$: $R_2$ (walks $\\le5$)", lambda nl, sc: med.loc[(nl, sc), "sl_R2"]),
    ("Share of $\\Delta\\lambda_1$: $R_3$ (walks $\\le7$)", lambda nl, sc: med.loc[(nl, sc), "sl_R3"]),
    ("Share of $\\Delta$gap: $R_1$", lambda nl, sc: med.loc[(nl, sc), "sg_R1"]),
    ("Share of $\\Delta$gap: $R_2$", lambda nl, sc: med.loc[(nl, sc), "sg_R2"]),
    ("Share of $\\Delta$gap: $R_3$", lambda nl, sc: med.loc[(nl, sc), "sg_R3"]),
    ("Slack $\\lambda_1-R_1$, null (real: see note)", lambda nl, sc: med.loc[(nl, sc), "slack_null"]),
    ("$\\rho(R_1,\\lambda_1)$ within nulls", lambda nl, sc: med.loc[(nl, sc), "rho_R1_l1"]),
    ("$\\rho(r_W,\\lambda_1)$ within nulls", lambda nl, sc: med.loc[(nl, sc), "rho_rW_l1"]),
    ("$\\rho(r_{\\mathrm{bct}},\\lambda_1)$ within nulls", lambda nl, sc: med.loc[(nl, sc), "rho_rbct_l1"]),
    ("$\\rho(r_W,\\mathrm{gap})$ within nulls", lambda nl, sc: med.loc[(nl, sc), "rho_rW_gap"]),
    ("Pearson $r(r_W, s^\\top Ws)$ within nulls", lambda nl, sc: med.loc[(nl, sc), "pearson_rW_sTWs"]),
    ("SD$(s^\\top s)$/mean within nulls", lambda nl, sc: med.loc[(nl, sc), "sd_sTs_rel"]),
    ("$\\cos(s,v_1)$, null mean", lambda nl, sc: nm.loc[(nl, sc), "cos_s_v1"]),
]
for lab, fn in rows:
    for nl in NULLS:
        vals = []
        for sc in SC:
            v = fn(nl, sc)
            vals.append(f"{v:.4f}" if "SD$(s" in lab else (f"{v:.3f}" if "Pearson" in lab else f2(v)))
        add(f"{lab if nl == 'strength' else ''} & {NL[nl]} & " + " & ".join(vals) + " \\\\")
    add("\\addlinespace")
real_cos = net[net.null == "real"].groupby("scale")["cos_s_v1"].mean()
real_slack = med.xs("strength", level=0)["slack_real"]
add(f"$\\cos(s,v_1)$, real mean & -- & " + " & ".join(f2(real_cos[sc]) for sc in SC) + " \\\\")
add(f"Slack $\\lambda_1-R_1$, real (median) & -- & " + " & ".join(f2(real_slack[sc]) for sc in SC) + " \\\\")
add("\\bottomrule")
add("\\end{tabular}")
add("")

# ---------------- (f) assortativity real vs null
real = net[net.null == "real"].set_index(["scale", "subject"])
add("% ---- Table S8.6: strength assortativity real vs null; r_W = edge-weighted (04_rayleigh_bound.py), r_bct = bct.assortativity_wei(flag=0). Computed here from rayleigh/networks.csv.gz")
add("\\begin{tabular}{ll" + "rrrr" * 3 + "}")
add("\\toprule")
add(" & & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{sc}}}" for sc in SC) + " \\\\")
add("\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14}")
add("Statistic & Null & " + " & ".join("Real & Null & $\\tilde z$ & $>$95th" for _ in SC) + " \\\\")
add("\\midrule")
for stat, lab in [("r_W", "$r_W$ (edge-weighted)"), ("r_bct", "$r_{\\mathrm{bct}}$ (\\texttt{assortativity\\_wei})")]:
    for nl in NULLS:
        cells = []
        for sc in SC:
            g = net[(net.null == nl) & (net.scale == sc)]
            zs, ab, nmn = [], 0, []
            for sid, gg in g.groupby("subject"):
                x = gg[stat].to_numpy(float)
                r = real.loc[(sc, sid), stat]
                zs.append((r - x.mean()) / x.std(ddof=1))
                ab += int(r > np.percentile(x, 95))
                nmn.append(x.mean())
            cells += [f2(real.loc[sc][stat].mean()), f2(np.mean(nmn)), fz(np.median(zs)), n70(ab)]
        add(f"{lab if nl == 'strength' else ''} & {NL[nl]} & " + " & ".join(cells) + " \\\\")
    if stat == "r_W":
        add("\\midrule")
add("\\bottomrule")
add("\\end{tabular}")
add("")

# ---------------- (g) mediation
mn = pd.read_csv(R + "mediation_noncircular.csv")
mn["k"] = mn["k"].astype(str)
MED = [("strength_assort", "Strength assortativity ($r_{\\mathrm{bct}}$)"),
       ("rc_bu", "Binary rich club (90th pct degree)"),
       ("rc_wu", "Weighted rich club (90th pct)"),
       ("rc_wu_p75", "Weighted rich club (75th pct)"),
       ("rc_wu_p85", "Weighted rich club (85th pct)"),
       ("rc_wu_p95", "Weighted rich club (95th pct)"),
       ("wdeg_top_share", "Weight share, top-20\\% degree nodes"),
       ("wdeg_rho", "Weight--degree correlation"),
       ("ipr_v1", "IPR($v_1$)"),
       ("hub_strength_frac", "Top-5 hub-to-hub strength share (circular)")]
add("% ---- Table S8.7: proportion mediated (1 - beta_direct/beta_total; OLS with subject fixed effects); c = 0.1 fixed")
add("\\begin{tabular}{ll" + "r" * 9 + "}")
add("\\toprule")
add(" & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{NL[nl]}}}" for nl in NULLS) + " \\\\")
add("\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}")
add("Outcome & Mediator & " + " & ".join("68 & 114 & 219" for _ in NULLS) + " \\\\")
add("\\midrule")
for oc, k, ol in [("degree_q1", "avg", "Degree-set $q_1$ (k-avg)"), ("gap", "-", "Spectral gap")]:
    for med_, lab in MED:
        cells = []
        for nl in NULLS:
            for sc in SC:
                rr = mn[(mn.outcome == oc) & (mn.k == k) & (mn.mediator == med_) & (mn.null == nl) & (mn.scale == sc)]
                cells.append(f2(rr.iloc[0].prop_mediated) if len(rr) else "--")
        add(f"{ol if med_ == 'strength_assort' else ''} & {lab} & " + " & ".join(cells) + " \\\\")
    add("\\midrule")
add("\\multicolumn{11}{l}{Strength assortativity ($r_{\\mathrm{bct}}$) by $k$, outcome degree-set $q_1$: proportion mediated [total effect $\\beta_{\\mathrm{real}}$]} \\\\")
for k in ["1", "2", "3", "5"]:
    cells = []
    for nl in NULLS:
        for sc in SC:
            r = mn[(mn.outcome == "degree_q1") & (mn.k == k) & (mn.mediator == "strength_assort") & (mn.null == nl) & (mn.scale == sc)].iloc[0]
            cells.append(f"{f2(r.prop_mediated)} [{r.beta_real_total:.3f}]")
    add(f"$k={k}$ & & " + " & ".join(cells) + " \\\\")
add("\\midrule")
add("\\multicolumn{11}{l}{Median within-subject $\\rho$(mediator, outcome) across the 50 nulls + real, strength assortativity} \\\\")
for oc, k, ol in [("degree_q1", "avg", "$q_1$ (k-avg)"), ("gap", "-", "gap")]:
    cells = []
    for nl in NULLS:
        for sc in SC:
            r = mn[(mn.outcome == oc) & (mn.k == k) & (mn.mediator == "strength_assort") & (mn.null == nl) & (mn.scale == sc)].iloc[0]
            cells.append(f2(r.median_within_subject_rho))
    add(f"{ol} & & " + " & ".join(cells) + " \\\\")
add("\\bottomrule")
add("\\end{tabular}")
add("")

# mediator contrasts (rejected mediators real vs null)
mc = pd.read_csv(R + "mediator_contrasts.csv")
add("% ---- Table S8.8: candidate mediators, real vs null (mean); median z; n above 95th null pct")
add("\\begin{tabular}{ll" + "rrrr" * 3 + "}")
add("\\toprule")
add(" & & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{sc}}}" for sc in SC) + " \\\\")
add("\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14}")
add("Mediator & Null & " + " & ".join("Real & Null & $\\tilde z$ & $>$95th" for _ in SC) + " \\\\")
add("\\midrule")
for med_, lab in MED:
    for nl in NULLS:
        cells = []
        for sc in SC:
            rr = mc[(mc.stat == med_) & (mc.null == nl) & (mc.scale == sc)]
            r = rr.iloc[0]
            cells += [f2(r.real_mean), f2(r.null_mean), fz(r.z_median), n70(r.n_above95)]
        add(f"{lab if nl == 'strength' else ''} & {NL[nl]} & " + " & ".join(cells) + " \\\\")
    if med_ != MED[-1][0]:
        add("\\addlinespace")
add("\\bottomrule")
add("\\end{tabular}")
add("")

# ---------------- (h) cycle - degree ER at 114
add("% ---- Table S8.9: cycle - degree ER and q1 at 114 regions (k-avg), real vs null, fixed vs gap-matched c")
add("\\begin{tabular}{lllrrrrrrrr}")
add("\\toprule")
add(" & & & \\multicolumn{4}{c}{Cycle $-$ degree ER} & \\multicolumn{4}{c}{Cycle $-$ degree $q_1$} \\\\")
add("\\cmidrule(lr){4-7}\\cmidrule(lr){8-11}")
add("$c$ & $c$ mode & Network & Value & $\\tilde z$ & $>$95th & $p_{\\mathrm{Holm}}$ & Value & $\\tilde z$ & $>$95th & $p_{\\mathrm{Holm}}$ \\\\")
add("\\midrule")
for c in (0.01, 0.1, 1.0):
    re = inf[(inf.stat == "cycle_minus_degree_er") & (inf.k == "avg") & (inf.c_ref == c)].iloc[0]
    rq = inf[(inf.stat == "cycle_minus_degree_q1") & (inf.k == "avg") & (inf.c_ref == c)].iloc[0]
    add(f"{c:g} & -- & Real & {f2(re.real_mean)} & & & & {f2(rq.real_mean)} & & & \\\\")
    for mode in ("fixed", "gap"):
        for nl in NULLS:
            cells = []
            for st in ("cycle_minus_degree_er", "cycle_minus_degree_q1"):
                r = inf[(inf.stat == st) & (inf.k == "avg") & (inf.c_ref == c) & (inf.null == nl) & (inf.c_mode == mode)].iloc[0]
                cells += [f2(r.null_mean), fz(r.z_median), n70(r.n_above95), fp(r.p_holm)]
            ml = {"fixed": "fixed", "gap": "gap-matched"}[mode]
            add(f" & {ml if nl == 'strength' else ''} & {NL[nl]} & " + " & ".join(cells) + " \\\\")
    if c != 1.0:
        add("\\midrule")
add("\\bottomrule")
add("\\end{tabular}")

open(OUT, "w").write("\n".join(L) + "\n")
print("wrote", OUT, len(L))
