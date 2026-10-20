"""Supplement S8: null networks (S8_1-S8_10), and the inline numbers of main.tex L190/L369.

Port of scripts/revision/paper/_rescued_2026-10-19/make_s8.py (reads <run>/revision/nulls/).
Table ids follow tables/index.txt of the rescued folder: the original's "Table S8.4a/b" are S8_4/S8_5,
so the original S8.5-S8.9 are S8_6-S8_10 here.
"""

from __future__ import annotations

from pathlib import Path

from paper_common import Table, np, pd

NULLS = ["strength", "geometric", "weights"]
NL = {"strength": "Strength", "geometric": "Geometric", "weights": "Weight shuffle"}
SC = [68, 114, 219]
CS = (0.01, 0.1, 1.0)
MODE_LABEL = {"fixed": "fixed", "gap": "gap-matched"}

COMMON_NOTE = """Root: results/revision/nulls/ (scripts/revision/nulls/00..04).
Common design: 70 subjects x 3 scales x 50 nulls/subject x 3 nulls (strength = Maslov-Sneppen + SA on weights;
geometric = Betzel-Bassett distance-binned swaps; weights = permute weights over the real binary edges).
Model A = A0 - (lambda_max + c)I, infinite-horizon Gramian, eps = 1e-5. Per-subject z = (real - null mean)/null SD
(ddof=1) against that subject's own 50 nulls; ">95th" = real > np.percentile(null, 95). Wilcoxon on the 70 z's.
Holm family (02_analyze.py): within each (null, c_mode, stat), across scale x c_ref x k (k in 1,2,3,5,avg,avg2+;
54 tests). k-avg = mean over k = 1,2,3,5. Gap-matched: c_null = c * gap_null / gap_real (per network), real uses c.
Rounding: 2 decimals; counts n/N (N = subjects, read from the data); p shown 2 dp if >= 0.1, 2 s.f. if >= 0.001,
else a x 10^b."""


# ---------------------------------------------------------------- formatting (as in make_s8.py)
def f2(v):
    if pd.isna(v):
        return "--"
    s = f"{v:.2f}"
    return "0.00" if s == "-0.00" else s


fz = f2


def fp(p):
    if pd.isna(p):
        return "--"
    if p >= 0.1:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.2g}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def nof(n, tot):
    return f"{int(n)}/{int(tot)}"


def one(df: pd.DataFrame, **eq) -> pd.Series:
    """The single row of ``df`` matching every column == value in ``eq``."""
    mask = np.ones(len(df), bool)
    for col, val in eq.items():
        mask &= (df[col] == val).to_numpy()
    sub = df[mask]
    assert len(sub) == 1, (eq, len(sub))
    return sub.iloc[0]


def note(header: str, body: str) -> str:
    return f"{header}\n\n{body.strip()}\n\n{COMMON_NOTE}"


# ---------------------------------------------------------------- shared computations
def _rayleigh(nulls: Path):
    """Per-(null, scale) medians over subjects of the per-subject Rayleigh quantities, with the R_m shares of
    delta-lambda1 and delta-gap (sl_R*, sg_R*) that summary.csv does not store."""
    ps = pd.read_csv(nulls / "rayleigh" / "per_subject.csv")
    for m in ["R1", "R2", "R3"]:
        ps[f"sl_{m}"] = ps[f"d_{m}"] / ps["d_lambda1"]
        ps[f"sg_{m}"] = ps[f"d_{m}"] / ps["d_gap"]
    return ps, ps.groupby(["null", "scale"]).median(numeric_only=True)


def _assortativity(net: pd.DataFrame) -> dict:
    """Real vs null strength assortativity per (stat, null, scale), computed from rayleigh/networks.csv.gz.

    Returns {(stat, null, scale): dict(real_mean, null_mean, z_median, n_above95, n)}.
    """
    real = net[net.null == "real"].set_index(["scale", "subject"])
    out = {}
    for stat in ("r_W", "r_bct"):
        for nl in NULLS:
            for sc in SC:
                g = net[(net.null == nl) & (net.scale == sc)]
                zs, ab, nmn = [], 0, []
                for sid, gg in g.groupby("subject"):
                    x = gg[stat].to_numpy(float)
                    r = real.loc[(sc, sid), stat]
                    zs.append((r - x.mean()) / x.std(ddof=1))
                    ab += int(r > np.percentile(x, 95))
                    nmn.append(x.mean())
                out[(stat, nl, sc)] = dict(real_mean=real.loc[sc][stat].mean(), null_mean=float(np.mean(nmn)),
                                           z_median=float(np.median(zs)), n_above95=ab, n=len(zs))
    return out


# ---------------------------------------------------------------- tables
def _s8_1(nulls: Path) -> Table:
    fid = pd.read_csv(nulls / "null_fidelity.csv")
    L = ["\\begin{tabular}{llrrrrrrr}", "\\toprule",
         "Network & Scale & Strength $r$ & Max rel.\\ strength err. & Top-5 hub Jaccard & Degree preserved & Edge overlap"
         " & Mean edge length & Weighted edge length \\\\", "\\midrule"]
    for nl in ["real"] + NULLS:
        for sc in SC:
            r = one(fid, null=nl, scale=sc)
            name = "Real" if nl == "real" else NL[nl]
            L.append(f"{name if sc == 68 else ''} & {sc} & {r.strength_r:.3f} & {f2(r.strength_max_rel_err)}"
                     f" & {f2(r.strength_top5_jaccard)} & {f2(r.degree_preserved)} & {f2(r.edge_overlap)}"
                     f" & {r.mean_edge_length:.2f} & {r.weighted_edge_length:.2f} \\\\")
        if nl != NULLS[-1]:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    return Table("S8_1", "\n".join(L), note(
        "S8.1 (a): null fidelity (means over subjects and 50 nulls/subject).", """
Source: results/revision/nulls/null_fidelity.csv (means over subjects and nulls). Strength r shown to 3 dp
(0.998/0.999/0.999; 2 dp would read 1.00). "Max rel. strength err." is the mean over networks of the per-network max
relative node-strength error (strength null 0.16-0.27: strengths are approximate, not exact). Edge lengths in centroid
units (Lausanne centroid Euclidean distance; presumably mm). CAVEAT: the strength null does NOT preserve geometry:
mean edge length 78-82 vs 58-61 real. Top-5 hub Jaccard 1.00 for strength null. Degree preserved = 1.00 for all nulls
(binary degree exact)."""))


def _s8_2(nulls: Path) -> Table:
    gs = pd.read_csv(nulls / "gap_split.csv")
    ni = pd.read_csv(nulls / "network_inference.csv").rename(columns={"n_subjects": "n"})
    L = ["\\begin{tabular}{llrrrrrrrrrrrr}", "\\toprule",
         " & & \\multicolumn{4}{c}{68} & \\multicolumn{4}{c}{114} & \\multicolumn{4}{c}{219} \\\\",
         "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14}",
         "Statistic & Null & Real & Null & $\\tilde z$ & $>$95th / $<$5th & Real & Null & $\\tilde z$ & $>$95th / $<$5th"
         " & Real & Null & $\\tilde z$ & $>$95th / $<$5th \\\\", "\\midrule"]
    for stat, lab, src in [("lambda1", "$\\lambda_1$", gs), ("lambda2", "$\\lambda_2$", gs),
                           ("gap", "$\\lambda_1-\\lambda_2$", gs), ("ipr_v1", "IPR($v_1$)", ni)]:
        for nl in NULLS:
            cells = []
            for sc in SC:
                r = one(src, null=nl, scale=sc, stat=stat)
                cells += [f2(r.real_mean), f2(r.null_mean), fz(r.z_median),
                          f"{nof(r.n_above95, r.n)} / {nof(r.n_below5, r.n)}"]
            L.append(f"{lab if nl == 'strength' else ''} & {NL[nl]} & " + " & ".join(cells) + " \\\\")
        L.append("\\midrule")
    L.append("\\multicolumn{14}{l}{Median per-subject share of the real$-$null gap excess carried by $\\lambda_1$"
             " ($\\Delta\\lambda_1/\\Delta\\text{gap}$); count = subjects with share $>0.5$ / share $<0$} \\\\")
    for nl in NULLS:
        cells = []
        for sc in SC:
            r = one(gs, null=nl, scale=sc, stat="lambda1_share_of_gap_excess")
            cells += [f"\\multicolumn{{3}}{{c}}{{{f2(r.delta_mean)}}}", f"{nof(r.n_above95, r.n)} / {nof(r.n_below5, r.n)}"]
        L.append(f"$\\Delta\\lambda_1$ share & {NL[nl]} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return Table("S8_2", "\n".join(L), note(
        "S8.2 (b): lambda1, lambda2, gap, IPR(v1): real vs each null (mean); median per-subject z; n above 95th / below"
        " 5th null pct; Wilcoxon p on z (uncorrected, not printed).", """
Source: lambda1/lambda2/gap -- results/revision/nulls/gap_split.csv; IPR(v1) -- network_inference.csv.
"Real"/"Null" = means; z~ = median z; counts real > 95th / real < 5th. Bottom block: lambda1_share_of_gap_excess =
MEDIAN over subjects of delta_lambda1/delta_gap (stored in the delta_mean column); its two counts are NOT percentiles --
per 03_gap_mediators.py they are (#subjects with share > 0.5) / (#share < 0) (stored in n_above95 / n_below5).
Wilcoxon p (uncorrected) not printed; all lambda1/gap p <= 1e-11; lambda2: 68 p = 0.056 (strength), 0.0066 (geo),
0.0096 (weights); 114 p <= 5e-4; 219 p ~ 0. IPR: n.s. at 219 (p 0.33-0.52) and at 114 for strength/weights (0.82, 0.35).
Also in network_inference.csv but not tabled: decay_ratio_c0.1 (real 0.20/0.17/0.20 vs null 0.31-0.39)."""))


def _inf_table(inf: pd.DataFrame, stat: str) -> str:
    sub = inf[(inf.stat == stat) & (inf.k == "avg")]
    ncol = 4
    L = ["\\begin{tabular}{lll" + "r" * (ncol * len(SC)) + "}", "\\toprule",
         " & & & " + " & ".join(f"\\multicolumn{{{ncol}}}{{c}}{{{sc}}}" for sc in SC) + " \\\\",
         "".join(f"\\cmidrule(lr){{{4 + i * ncol}-{3 + (i + 1) * ncol}}}" for i in range(len(SC)))]
    hdr = ["Mean", "$\\tilde z$", "$>$95th", "$p_{\\mathrm{Holm}}$"]
    L += ["$c$ & $c$ mode & Network & " + " & ".join(" & ".join(hdr) for _ in SC) + " \\\\", "\\midrule"]
    for c in CS:
        cells = []
        for sc in SC:
            real = sub[(sub.c_ref == c) & (sub.scale == sc)].real_mean
            assert real.nunique() == 1, (stat, c, sc)  # same real value in every null/mode row
            cells += [f2(real.iloc[0])] + [""] * (ncol - 1)
        L.append(f"{c:g} & -- & Real & " + " & ".join(cells) + " \\\\")
        for mode in ("fixed", "gap"):
            for nl in NULLS:
                cells = []
                for sc in SC:
                    r = one(sub, c_ref=c, scale=sc, null=nl, c_mode=mode)
                    cells += [f2(r.null_mean), fz(r.z_median), nof(r.n_above95, r.n_subjects), fp(r.p_holm)]
                L.append(f" & {MODE_LABEL[mode] if nl == 'strength' else ''} & {NL[nl]} & " + " & ".join(cells) + " \\\\")
        if c != CS[-1]:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def _s8_3_to_5(inf: pd.DataFrame) -> list[Table]:
    src = "Source: results/revision/nulls/inference.csv (filters: stat == '{}', k == 'avg'; c_ref in 0.01, 0.1, 1; " \
          "c_mode fixed and gap). Real row = real mean. Per null: null mean, z~, n > 95th, p_Holm."
    return [
        Table("S8_3", _inf_table(inf, "degree_q1"), note(
            "S8.3 (c): degree-set q1 (k-avg over k=1,2,3,5), real vs null mean, fixed and gap-matched c.",
            src.format("degree_q1"))),
        Table("S8_4", _inf_table(inf, "degree_q1perp_pct"), note(
            "S8.4a (d): W-perp residual -- percentile of degree-set q1(perp) among 100 random sets (k-avg).",
            src.format("degree_q1perp_pct") + """
This is the statistic behind REPORT.md's "median z 0.2-0.7, pct 0.74 vs 0.67-0.73". At c = 0.1: z~ 0.29-0.68 (fixed),
0.18-0.65 (gap); real 0.74/0.74/0.74 vs null 0.67-0.73; p_H <= 0.04.""")),
        Table("S8_5", _inf_table(inf, "degree_q1perp"), note(
            "S8.4b (d): W-perp residual -- raw degree-set q1(perp) (k-avg).",
            src.format("degree_q1perp") + """
Raw q1 after projecting out v1. Real vs null is essentially NULL here: c = 0.1 z~ -0.41..+0.29, most p_H = 1; the
68 geometric null even has real < null (fixed p_H 0.019, gap 6e-6). => "small residual real-brain excess" holds only
for the percentile-among-random-sets version (S8_4), not raw q1perp.""")),
    ]


def _s8_6(nulls: Path, net: pd.DataFrame) -> Table:
    _, med = _rayleigh(nulls)
    nm = net.groupby(["null", "scale"])[["cos_s_v1"]].mean()
    L = ["\\begin{tabular}{llrrr}", "\\toprule", "Quantity & Null & 68 & 114 & 219 \\\\", "\\midrule"]
    rows = [  # (label, source, column, decimals)
        ("$\\Delta\\lambda_1$ (real $-$ null mean)", med, "d_lambda1", 2),
        ("$\\Delta$gap (real $-$ null mean)", med, "d_gap", 2),
        ("Share of $\\Delta\\lambda_1$: $R_0$ (mean strength)", med, "share_l1_R0", 2),
        ("Share of $\\Delta\\lambda_1$: $R_1=s^\\top Ws/s^\\top s$", med, "sl_R1", 2),
        ("Share of $\\Delta\\lambda_1$: $R_2$ (walks $\\le5$)", med, "sl_R2", 2),
        ("Share of $\\Delta\\lambda_1$: $R_3$ (walks $\\le7$)", med, "sl_R3", 2),
        ("Share of $\\Delta$gap: $R_1$", med, "sg_R1", 2),
        ("Share of $\\Delta$gap: $R_2$", med, "sg_R2", 2),
        ("Share of $\\Delta$gap: $R_3$", med, "sg_R3", 2),
        ("Slack $\\lambda_1-R_1$, null (real: see note)", med, "slack_null", 2),
        ("$\\rho(R_1,\\lambda_1)$ within nulls", med, "rho_R1_l1", 2),
        ("$\\rho(r_W,\\lambda_1)$ within nulls", med, "rho_rW_l1", 2),
        ("$\\rho(r_{\\mathrm{bct}},\\lambda_1)$ within nulls", med, "rho_rbct_l1", 2),
        ("$\\rho(r_W,\\mathrm{gap})$ within nulls", med, "rho_rW_gap", 2),
        ("Pearson $r(r_W, s^\\top Ws)$ within nulls", med, "pearson_rW_sTWs", 3),
        ("SD$(s^\\top s)$/mean within nulls", med, "sd_sTs_rel", 4),
        ("$\\cos(s,v_1)$, null mean", nm, "cos_s_v1", 2),
    ]
    for lab, src, col, d in rows:
        for nl in NULLS:
            vals = [f2(v) if d == 2 else f"{v:.{d}f}" for v in (src.loc[(nl, sc), col] for sc in SC)]
            L.append(f"{lab if nl == 'strength' else ''} & {NL[nl]} & " + " & ".join(vals) + " \\\\")
        L.append("\\addlinespace")
    real_cos = net[net.null == "real"].groupby("scale")["cos_s_v1"].mean()
    real_slack = med.xs("strength", level=0)["slack_real"]
    # slack_real is a real-network quantity: it must not depend on which null's table it was taken from
    for nl in NULLS[1:]:
        assert np.allclose(med.xs(nl, level=0)["slack_real"], real_slack)
    L.append("$\\cos(s,v_1)$, real mean & -- & " + " & ".join(f2(real_cos[sc]) for sc in SC) + " \\\\")
    L.append("Slack $\\lambda_1-R_1$, real (median) & -- & " + " & ".join(f2(real_slack[sc]) for sc in SC) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return Table("S8_6", "\n".join(L), note(
        "S8.5 (e): Rayleigh accounting of the real-null lambda1 excess (medians over 70 subjects of per-subject"
        " quantities).", """
Source: results/revision/nulls/rayleigh/per_subject.csv (medians over subjects per null x scale; these equal
rayleigh/summary.csv) and rayleigh/networks.csv.gz (cos(s, v1)).
R_m = 1'W^(2m+1)1 / 1'W^(2m)1; R0 = mean strength (preserved by all nulls -> share exactly 0); R1 = s'Ws/s's.
Shares of delta-lambda1 by R2 and R3, and shares of delta-gap by R2/R3, are NOT in summary.csv: computed here as
median over the 70 subjects of d_R{m}/d_lambda1 (resp. /d_gap) from per_subject.csv (d_* = real - mean over that
subject's 50 nulls). R1 values reproduce summary.csv share_l1_R1/share_gap_R1 exactly.
Delta rows are medians -- they differ slightly from the means in S8_2 (e.g. geometric 219: 0.52 vs 0.49).
rho rows = median over subjects of within-subject Spearman across that subject's 50 nulls (nulls only).
Pearson r(r_W, s'Ws) shown to 3 dp. SD(s's)/mean shown to 4 dp (strength null 0.0005-0.0008 = <0.1%).
cos(s,v1): real and null means computed from rayleigh/networks.csv.gz (simple mean over networks).
Slack lambda1 - R1 real = median over subjects of slack_real (same for every null)."""), long=True)


def _s8_7(assort: dict) -> Table:
    L = ["\\begin{tabular}{ll" + "rrrr" * 3 + "}", "\\toprule",
         " & & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{sc}}}" for sc in SC) + " \\\\",
         "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14}",
         "Statistic & Null & " + " & ".join("Real & Null & $\\tilde z$ & $>$95th" for _ in SC) + " \\\\", "\\midrule"]
    for stat, lab in [("r_W", "$r_W$ (edge-weighted)"), ("r_bct", "$r_{\\mathrm{bct}}$ (\\texttt{assortativity\\_wei})")]:
        for nl in NULLS:
            cells = []
            for sc in SC:
                a = assort[(stat, nl, sc)]
                cells += [f2(a["real_mean"]), f2(a["null_mean"]), fz(a["z_median"]), nof(a["n_above95"], a["n"])]
            L.append(f"{lab if nl == 'strength' else ''} & {NL[nl]} & " + " & ".join(cells) + " \\\\")
        if stat == "r_W":
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    return Table("S8_7", "\n".join(L), note(
        "S8.6 (f): strength assortativity real vs null; r_W = edge-weighted (04_rayleigh_bound.py), r_bct ="
        " bct.assortativity_wei(flag=0). Computed here from rayleigh/networks.csv.gz.", """
Source: results/revision/nulls/rayleigh/networks.csv.gz (one row per network: real + 50 nulls x 70 subjects x 3 scales
x 3 nulls). For each subject: z vs own 50 nulls (ddof=1), real > np.percentile(null,95) -- identical rule to 02/03
scripts. Real/Null = mean over subjects. Validation: the r_bct rows reproduce gap_summary.md/mediator_contrasts.csv
strength_assort exactly (means, z~, n>95th), and lambda1 recomputed this way reproduces gap_split.csv exactly.
r_W (edge-weighted, 04_rayleigh_bound.py) real 0.40/0.46/0.37; strength null 0.06/0.03/0.00 (70/70 above 95th at every
scale); geometric null 0.26/0.14/0.11 (40/69/69); weight shuffle 0.14/0.15/0.14 (69/69/69).
r_bct (bct.assortativity_wei, binary-edge counting) real 0.14/0.21/0.19 vs null -0.03..0.05."""))


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


def _s8_8(nulls: Path) -> Table:
    mn = pd.read_csv(nulls / "mediation_noncircular.csv")
    mn["k"] = mn["k"].astype(str)
    L = ["\\begin{tabular}{ll" + "r" * 9 + "}", "\\toprule",
         " & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{NL[nl]}}}" for nl in NULLS) + " \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
         "Outcome & Mediator & " + " & ".join("68 & 114 & 219" for _ in NULLS) + " \\\\", "\\midrule"]
    for oc, k, ol in [("degree_q1", "avg", "Degree-set $q_1$ (k-avg)"), ("gap", "-", "Spectral gap")]:
        for med_, lab in MED:
            cells = []
            for nl in NULLS:
                for sc in SC:
                    rr = mn[(mn.outcome == oc) & (mn.k == k) & (mn.mediator == med_) & (mn.null == nl) & (mn.scale == sc)]
                    assert len(rr) <= 1
                    cells.append(f2(rr.iloc[0].prop_mediated) if len(rr) else "--")
            L.append(f"{ol if med_ == 'strength_assort' else ''} & {lab} & " + " & ".join(cells) + " \\\\")
        L.append("\\midrule")
    L.append("\\multicolumn{11}{l}{Strength assortativity ($r_{\\mathrm{bct}}$) by $k$, outcome degree-set $q_1$:"
             " proportion mediated [total effect $\\beta_{\\mathrm{real}}$]} \\\\")
    for k in ["1", "2", "3", "5"]:
        cells = []
        for nl in NULLS:
            for sc in SC:
                r = one(mn, outcome="degree_q1", k=k, mediator="strength_assort", null=nl, scale=sc)
                cells.append(f"{f2(r.prop_mediated)} [{r.beta_real_total:.3f}]")
        L.append(f"$k={k}$ & & " + " & ".join(cells) + " \\\\")
    L.append("\\midrule")
    L.append("\\multicolumn{11}{l}{Median within-subject $\\rho$(mediator, outcome) across the 50 nulls + real,"
             " strength assortativity} \\\\")
    for oc, k, ol in [("degree_q1", "avg", "$q_1$ (k-avg)"), ("gap", "-", "gap")]:
        cells = []
        for nl in NULLS:
            for sc in SC:
                r = one(mn, outcome=oc, k=k, mediator="strength_assort", null=nl, scale=sc)
                cells.append(f2(r.median_within_subject_rho))
        L.append(f"{ol} & & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return Table("S8_8", "\n".join(L), note(
        "S8.7 (g): proportion mediated (1 - beta_direct/beta_total; OLS with subject fixed effects); c = 0.1 fixed.", """
Source: results/revision/nulls/mediation_noncircular.csv (03_gap_mediators.py): outcome degree-set q1 at FIXED c = 0.1
(k-avg and each k) and the spectral gap; OLS stat ~ is_real (+ mediator) + C(subject); prop = 1 - beta_direct/beta_total.
The "strength assortativity" mediator is bct.assortativity_wei (r_bct), NOT r_W. Mediation has not been re-run with r_W.
rc_bu/rc_wu = binary/weighted rich club at the 90th-pct degree; rc_wu_p75/85/95 = weighted (Opsahl) rich club at those
real-degree percentiles; wdeg_top_share = weight share of top-20%-degree nodes; wdeg_rho = weight-degree correlation;
hub_strength_frac = top-5-strength hub-to-hub share (flagged circular=True in the CSV: same nodes as the k = 5 degree
set). Binary rich club under the weight-shuffle null is identical real vs null, so it is absent from the CSV ("--").
mediation.csv additionally has c = 0.01/1 and gap-matched rows; the gap-matched proportions are unstable (total effect
~0) -- not tabled. k = 1 row: total effect ~0, so proportions are meaningless; shown with beta_total.
Within-subject rho (Spearman of mediator vs outcome, median over subjects; column median_within_subject_rho)."""))


def _s8_9(nulls: Path) -> Table:
    mc = pd.read_csv(nulls / "mediator_contrasts.csv")
    L = ["\\begin{tabular}{ll" + "rrrr" * 3 + "}", "\\toprule",
         " & & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{sc}}}" for sc in SC) + " \\\\",
         "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14}",
         "Mediator & Null & " + " & ".join("Real & Null & $\\tilde z$ & $>$95th" for _ in SC) + " \\\\", "\\midrule"]
    for med_, lab in MED:
        for nl in NULLS:
            cells = []
            for sc in SC:
                r = one(mc, stat=med_, null=nl, scale=sc)
                cells += [f2(r.real_mean), f2(r.null_mean), fz(r.z_median), nof(r.n_above95, r.n)]
            L.append(f"{lab if nl == 'strength' else ''} & {NL[nl]} & " + " & ".join(cells) + " \\\\")
        if med_ != MED[-1][0]:
            L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}"]
    return Table("S8_9", "\n".join(L), note(
        "S8.8 (g): candidate mediators, real vs null (mean); median z; n above 95th null pct.",
        "Source: results/revision/nulls/mediator_contrasts.csv (same as gap_summary.md A2 block)."))


def _s8_10(inf: pd.DataFrame) -> Table:
    L = ["\\begin{tabular}{lllrrrrrrrr}", "\\toprule",
         " & & & \\multicolumn{4}{c}{Cycle $-$ degree ER} & \\multicolumn{4}{c}{Cycle $-$ degree $q_1$} \\\\",
         "\\cmidrule(lr){4-7}\\cmidrule(lr){8-11}",
         "$c$ & $c$ mode & Network & Value & $\\tilde z$ & $>$95th & $p_{\\mathrm{Holm}}$ & Value & $\\tilde z$ & $>$95th"
         " & $p_{\\mathrm{Holm}}$ \\\\", "\\midrule"]
    stats = ("cycle_minus_degree_er", "cycle_minus_degree_q1")
    sub = inf[inf.stat.isin(stats) & (inf.k == "avg")]
    assert set(sub.scale) == {114}, set(sub.scale)  # these stats exist only at 114 regions
    for c in CS:
        reals = []
        for st in stats:
            real = sub[(sub.stat == st) & (sub.c_ref == c)].real_mean
            assert real.nunique() == 1, (st, c)
            reals.append(f2(real.iloc[0]))
        L.append(f"{c:g} & -- & Real & {reals[0]} & & & & {reals[1]} & & & \\\\")
        for mode in ("fixed", "gap"):
            for nl in NULLS:
                cells = []
                for st in stats:
                    r = one(sub, stat=st, c_ref=c, null=nl, c_mode=mode)
                    cells += [f2(r.null_mean), fz(r.z_median), nof(r.n_above95, r.n_subjects), fp(r.p_holm)]
                L.append(f" & {MODE_LABEL[mode] if nl == 'strength' else ''} & {NL[nl]} & " + " & ".join(cells) + " \\\\")
        if c != CS[-1]:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    return Table("S8_10", "\n".join(L), note(
        "S8.9 (h): cycle - degree ER and q1 at 114 regions (k-avg), real vs null, fixed vs gap-matched c.", """
Source: results/revision/nulls/inference.csv, stats cycle_minus_degree_er / _q1 (k == 'avg'; only scale 114 is saved),
c in {0.01, 0.1, 1}, fixed and gap. Real 0.98 vs null 0.35-0.51 (fixed c = 0.1, p_H <= 1.7e-9) and 1.09-1.18
(gap-matched; geometric p_H 3.1e-4 with real BELOW null, others n.s.).
CAVEAT: at c = 0.1 fixed the real cycle - degree q1 (-0.21) is MORE negative than null (-0.13), i.e. the cycle set
loses more concentration relative to degree in the real brain; n > 95th is therefore small and z negative."""))


def build(res: Path) -> list[Table]:
    nulls = res / "nulls"
    inf = pd.read_csv(nulls / "inference.csv")
    inf["k"] = inf["k"].astype(str)
    net = pd.read_csv(nulls / "rayleigh" / "networks.csv.gz")
    return [_s8_1(nulls), _s8_2(nulls), *_s8_3_to_5(inf), _s8_6(nulls, net), _s8_7(_assortativity(net)),
            _s8_8(nulls), _s8_9(nulls), _s8_10(inf)]


def inline_values(res: Path) -> dict[str, float]:
    """Unrounded values quoted in main.tex that only make_s8.py computed (same code as S8_6 / S8_7).

    Keys (scale = 68, 114, 219; null = strength, geometric, weights):

    - ``rW_real_<scale>``: mean over subjects of the empirical edge-weighted strength assortativity r_W.
      main.tex L190: "The edge-weighted strength assortativity ... was 0.40, 0.46, and 0.37".
    - ``rW_null_<null>_<scale>``: mean over subjects of the subject's null-mean r_W; ``rW_null_min`` /
      ``rW_null_max``: their range. main.tex L190: "against 0.00 to 0.26 in the nulls" (min is -0.003,
      strength null at 219, printed 0.00).
    - ``rW_n_above95_<null>_<scale>``: subjects (of ``n_subjects``) whose r_W exceeds the 95th percentile of their
      own 50 nulls. main.tex L190: "exceeded the 95th null percentile in 69 or 70 of 70 subjects for every null and
      scale except the geometry-preserving nulls at 68 regions (40 subjects)".
    - ``share_l1_R<m>_strength_<scale>`` (m = 1, 2, 3): median over subjects of d_R<m>/d_lambda1 against the
      strength-preserving nulls. main.tex L190: "the bound alone guaranteed a median of 60%, 43%, and 36%" (R1) and
      "accounted for 103%, 87%, and 77%, and 112%, 102%, and 95%" (R2, then R3). Also Fig. 5D.
    - ``pearson_rW_sTWs_strength_median`` / ``_min``: median / minimum over all subjects and scales of the
      within-subject Pearson r(r_W, s'Ws) across the strength-preserving nulls. main.tex L369: "(median Pearson
      r = 0.998; minimum 0.95)" (L190: "almost perfectly correlated"). Not computed by make_s8.py, which printed
      only per-scale medians (S8_6); added here because the sentence quotes the pooled values.
    """
    nulls = res / "nulls"
    net = pd.read_csv(nulls / "rayleigh" / "networks.csv.gz")
    assort = _assortativity(net)
    ps, med = _rayleigh(nulls)
    out: dict[str, float] = {}
    for sc in SC:
        out[f"rW_real_{sc}"] = float(assort[("r_W", "strength", sc)]["real_mean"])
    for nl in NULLS:
        for sc in SC:
            a = assort[("r_W", nl, sc)]
            out[f"rW_null_{nl}_{sc}"] = a["null_mean"]
            out[f"rW_n_above95_{nl}_{sc}"] = float(a["n_above95"])
    null_means = [out[f"rW_null_{nl}_{sc}"] for nl in NULLS for sc in SC]
    out["rW_null_min"], out["rW_null_max"] = min(null_means), max(null_means)
    out["n_subjects"] = float(assort[("r_W", "strength", SC[0])]["n"])
    for m in (1, 2, 3):
        for sc in SC:
            out[f"share_l1_R{m}_strength_{sc}"] = float(med.loc[("strength", sc), f"sl_R{m}"])
    pr = ps[ps.null == "strength"].pearson_rW_sTWs
    out["pearson_rW_sTWs_strength_median"] = float(pr.median())
    out["pearson_rW_sTWs_strength_min"] = float(pr.min())
    return out
