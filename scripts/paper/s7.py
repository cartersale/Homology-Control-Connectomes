"""Supplement S7: set-level mechanism tables (S7_1-S7_11).

Port of scripts/revision/paper/_rescued_2026-10-19/scripts/make_s7.py. The
L_S-q1 candidate table (S7_11) used an intermediate CSV written by
scripts/ls_q1.py; that computation is now ``ls_q1_candidates`` below.
"""

from __future__ import annotations

from pathlib import Path

from paper_common import SCALES, Table, np, pd
from scipy.stats import spearmanr

S = list(SCALES)
C = [0.01, 0.1, 1.0]
K = [2, 3, 5]
CS = {0.01: "0.01", 0.1: "0.1", 1.0: "1"}

# Families that are neither counterfactual node-score sets nor matched sets (ls_q1.py `base`; cycle_resid not included).
BASE_FAMILIES = ["betweenness", "cycle_participation", "degree_strength", "eigenvector_centrality",
                 "participation_coefficient", "triangle_participation", "cycle_criticality",
                 "degree_normalized_cycle_participation", "tetrahedron_participation", "random"]

COMMON_NOTES = """Common settings: A = A0 - (lambda_max + c)I, infinite-horizon Gramian, eps = 1e-5 where the eps floor applies.
q1 = lambda1(W)/tr W. In the modal and q1 files, q1 is computed on the RAW W (no eps floor; q1/validation.json).
"ER" (effective_rank) = exp-entropy of the eps-floored spectrum. "er_raw" = the same without the floor.
None of the modal/* analyses include k = 1 (pairwise quantities are undefined). Exception: the L_S table, which uses set_q1.csv.gz."""


def f2(x) -> str:
    if not np.isfinite(x):
        return "--"
    s = f"{x:.2f}"
    if s == "-0.00":
        s = "0.00"
    return s.replace("-", "$-$")


def tab(colspec: str, header: list[str], rows: list[str]) -> str:
    """The rescued builder's tabular layout (\\toprule on the \\begin line)."""
    return "\n".join([f"\\begin{{tabular}}{{{colspec}}}\\toprule", *header, "\\midrule", *rows, "\\bottomrule",
                      "\\end{tabular}"])


def notes(label: str, caption: str, note: str | None, size: str, detail: str) -> str:
    out = [f"{label}.", f"Caption: {caption}"]
    if note:
        out.append(f"Note: {note}")
    out.append(f"Suggested font size: {size}")
    out.append(detail.strip("\n"))
    out.append(COMMON_NOTES)
    return "\n".join(out)


def _rho(x: pd.DataFrame) -> float:
    return spearmanr(x.loading, x.q1)[0]


def ls_q1_candidates(res: Path) -> pd.DataFrame:
    """Spearman rho(L_S, q1) from q1/set_q1.csv.gz under several poolings (was scripts/ls_q1.py)."""
    d = pd.read_csv(res / "q1" / "set_q1.csv.gz", usecols=["subject_id", "scale", "c", "k", "family", "loading", "q1"])
    out = []
    for c in C:
        for sc in S:
            g = d[(d.c == c) & (d.scale == sc)]
            for name, sub in [("all families, all k", g), ("all families, k>=2", g[g.k >= 2]),
                              ("rankings+random (no cf/matched), all k", g[g.family.isin(BASE_FAMILIES)]),
                              ("rankings+random, k>=2", g[g.family.isin(BASE_FAMILIES) & (g.k >= 2)])]:
                out.append((c, str(sc), name, "pooled", _rho(sub)))
            for name, sub in [("all families, all k", g), ("all families, k>=2", g[g.k >= 2])]:
                r = pd.Series({sid: _rho(x) for sid, x in sub.groupby("subject_id")})
                out.append((c, str(sc), name, "within-subj median", r.median()))
            per = pd.DataFrame([(k, _rho(x)) for (_, k), x in g.groupby(["subject_id", "k"])], columns=["k", "rho"])
            for k, v in per.groupby("k").rho.median().items():
                out.append((c, str(sc), f"all families, k={k}", "within-subj(k) median", v))
        g = d[d.c == c]
        out.append((c, "all", "all families, all k", "pooled", _rho(g)))
        out.append((c, "all", "all families, k>=2", "pooled", _rho(g[g.k >= 2])))
    return pd.DataFrame(out, columns=["c", "scale", "sets", "pooling", "rho"])


def build(res: Path) -> list[Table]:
    M, Q1, MECH = res / "modal", res / "q1", res / "mechanism"
    tables = []

    def add(label, caption, colspec, header, rows, note, detail, size="\\scriptsize"):
        tables.append(Table(f"S7_{len(tables) + 1}", tab(colspec, header, rows), notes(label, caption, note, size, detail)))

    # --- S7.1 node-level correspondence
    n_all = pd.read_csv(M / "node_level_correlations.csv")
    n = n_all[n_all.c == 0.1]
    nm = {"modal_gu": "Modal ctrl.\\ (Gu)", "modal_ct": "Modal ctrl.\\ (cont.-time)", "avg_ctrl": "Average ctrl.",
          "perron2": "$v_1(i)^2$"}
    rk = {"degree_strength": "Strength", "cycle_participation": "Cycle part.", "eigenvector_centrality": "Eigenvector",
          "betweenness": "Betweenness"}
    rows = []
    for a in nm:
        for b in rk:
            cells = []
            for s in S:
                r = n[(n.node_metric == a) & (n.ranking == b) & (n.scale == s)].iloc[0]
                cells.append(f"{f2(r['median'])} ({int(r.n_pos)}/70)")
            rows.append(f"{nm[a]} & {rk[b]} & " + " & ".join(cells) + " \\\\")
        rows.append("\\addlinespace")
    rows = rows[:-1]
    # Was typed in as "0.90 / 0.89 / 0.89"; now read from the same file at c = 0.01.
    ac001 = [n_all[(n_all.c == 0.01) & (n_all.node_metric == "avg_ctrl") & (n_all.ranking == "degree_strength")
                   & (n_all.scale == s)]["median"].iloc[0] for s in S]
    add("tab:S7_nodelevel",
        "Node-level Spearman correlation between single-node controllability scores and the ranking criteria (within subject, across regions; $c=0.1$). Median across subjects (number of subjects with $\\rho>0$).",
        "llccc", ["Node score & Ranking & 68 & 114 & 219 \\\\"], rows,
        "Modal controllability and $v_1(i)^2$ do not depend on $c$; average controllability does (at $c=0.01$ its $\\rho$ with strength is "
        + " / ".join(f"{v:.2f}" for v in ac001) + ").",
        """Source: modal/node_level_correlations.csv, c = 0.1.
  Within-subject Spearman rho across regions; median across subjects, n_pos = subjects with rho > 0.
  Modal (Gu and continuous-time) and v1^2 are c-independent (the file repeats them at every c). Average controllability varies with c.
  Key: modal_gu vs strength -0.94 / -0.92 / -0.93 (0/70). avg_ctrl vs strength 0.94 / 0.94 / 0.93 (70/70).""")

    # --- S7.2 within-subject over random sets
    w = pd.read_csv(M / "within_subject_summary.csv")
    preds = [("rho_q1_perron_loading", "$\\rho(L_S,q_1)$"), ("rho_q1_avg_ctrl", "$\\rho(\\overline{\\mathrm{AC}}_S,q_1)$"),
             ("rho_q1_modal_gu", "$\\rho(\\overline{\\mathrm{MC}}_S,q_1)$"), ("rho_q1_q1_G", "$\\rho(q_1(G_S),q_1)$"),
             ("rho_q1_red_G", "$\\rho(\\bar r_G,q_1)$"),
             ("prho_q1_redGperp|node", "partial $\\rho(\\bar r_{G\\perp},q_1\\mid\\text{node})$"),
             ("prho_q1_redF|node", "partial $\\rho(\\bar r_F,q_1\\mid\\text{node})$")]
    rows = []
    for c in C:
        rows.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{$c={CS[c]}$}}}} \\\\")
        for v, lab in preds:
            cells = []
            for s in S:
                for k in K:
                    r = w[(w.variable == v) & (w.scale == s) & (w.k == k) & (w.c == c)].iloc[0]
                    cells.append(f"{f2(r['median'])} ({int(r.n_pos)}/70)")
            rows.append(lab + " & " + " & ".join(cells) + " \\\\")
    kcols = "".join(f"& $k={k}$ " for _ in S for k in K)
    scale_hdr = ["& \\multicolumn{3}{c}{68} & \\multicolumn{3}{c}{114} & \\multicolumn{3}{c}{219} \\\\",
                 "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}"]
    add("tab:S7_withinsubj",
        "Within-subject Spearman correlations with $q_1$ across each subject's 100 random driver sets. Median across 70 subjects (subjects with $\\rho>0$, of 70). Node = $\\{L_S,\\ \\overline{\\mathrm{MC}}_S^{\\mathrm{Gu}},\\ \\overline{\\mathrm{AC}}_S\\}$.",
        "l" + "c" * 9, scale_hdr + ["Predictor " + kcols + "\\\\"], rows,
        "$q_1(G_S)=\\lambda_1(G_S)/\\operatorname{tr}G_S$ for the $k\\times k$ driver Gram matrix of slow-mode profiles $x_i=(u_a(i)/\\sqrt{-2\\mu_a})_a$; $\\bar r_G$ = mean pairwise cosine of these profiles; $\\bar r_{G\\perp}$ = same with the Perron mode removed; $\\bar r_F$ = mean pairwise Frobenius cosine of single-node Gramians. $k=1$ not computed (no pairs).",
        """Source: modal/within_subject_summary.csv.
  Spearman rho with q1 across each subject's 100 random sets (family == "random" only; script 02_analyze_modal.py §A).
  Rows: rho_q1_{perron_loading, avg_ctrl, modal_gu, q1_G, red_G}, prho_q1_redGperp|node, prho_q1_redF|node.
  c in {0.01, 0.1, 1}; k in {2, 3, 5}. Cells: median (n_pos / 70).
  Partial = rank-residualized on {L_S, mean modal_gu, mean avg_ctrl}.
  Plan check (R6): "within-subject rho 0.55–0.72 at c = 0.1"
    = rho(q1(G_S), q1) at k = 3 ONLY: 0.72 / 0.58 / 0.55. (k = 5: 0.92 / 0.86 / 0.80; k = 2: 0.24 / 0.09 / 0.14.)
    That predictor is the driver-Gram q1, not the redundancy statistic r_G. r_G itself at k = 3 is 0.46 / 0.34 / 0.22.
  Plan check: "partial rho 0.12–0.23 at c = 0.1, rising to 0.27–0.47 at c = 1" = prho_q1_redGperp|node at k = 3
    (0.23 / 0.19 / 0.12; 0.47 / 0.40 / 0.27). At k = 5: 0.31 / 0.29 / 0.25 and 0.56 / 0.54 / 0.45.""",
        size="\\tiny")

    # --- S7.3 driver-Gram proxy (pooled)
    g = pd.read_csv(M / "driver_gram_proxy.csv")
    rows = []
    for c in C:
        for k in K:
            cells = []
            for s in S:
                r = g[(g.scale == s) & (g.k == k) & (g.c == c)].iloc[0]
                cells += [f2(r.rho_q1G_q1raw), f2(r.median_q1G_minus_q1raw), f2(r.median_q1raw_minus_pshare)]
            rows.append(f"{CS[c]} & {k} & " + " & ".join(cells) + " \\\\")
        rows.append("\\addlinespace")
    rows = rows[:-1]
    add("tab:S7_drivergram",
        "Driver-Gram bounds on $q_1$, pooled over all driver sets (random, ranked and counterfactual) and subjects. $\\rho$: Spearman $\\rho(q_1(G_S),q_1)$; upper gap: median $q_1(G_S)-q_1$; lower gap: median $q_1-\\pi_S$, where $\\pi_S=v_1^\\top W v_1/\\operatorname{tr}W=(L_S/2c)/\\operatorname{tr}W$.",
        "cc" + "ccc" * 3,
        ["& & \\multicolumn{3}{c}{68} & \\multicolumn{3}{c}{114} & \\multicolumn{3}{c}{219} \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
         "$c$ & $k$ " + "& $\\rho$ & upper & lower " * 3 + "\\\\"], rows,
        "Both gaps are $\\ge 0$ in every cell, i.e.\\ $\\pi_S\\le q_1\\le q_1(G_S)$. Here $q_1$ is computed on the raw $W$ (no $\\varepsilon$ floor; \\texttt{q1\\_raw}).",
        """Source: modal/driver_gram_proxy.csv.
  Pooled over all sets (random + ranked + counterfactual) and all subjects, per scale × k × c.
  Shown: rho_q1G_q1raw, median_q1G_minus_q1raw, median_q1raw_minus_pshare.
  Both gaps are >= 0 in every cell, consistent with the bounds pi_S <= q1 <= q1(G_S).
  NOTE: these are medians, not proofs. The bound holds per set per the derivation, but this table doesn't check per-set violations.""")

    # --- S7.4 communicability vs geodesic
    rd = pd.read_csv(Q1 / "redundancy_summary.csv")
    rows = []
    for c in C:
        for s in S:
            r = rd[(rd.scale == s) & (rd.c == c)].iloc[0]
            rows.append(f"{CS[c]} & {s} & {f2(r.rho_L)} & {f2(r.rho_comm)} & {f2(r.rho_geo_inv)} & {f2(r['prho_comm|L'])} & "
                        f"{f2(r['prho_geo_inv|L'])} & {f2(r['prho_geo_log|L'])} & {int(r.n_comm_better_geo_inv_given_L)}/70 & "
                        f"{r.p_comm_vs_geo_inv_given_L:.1e} \\\\")
        rows.append("\\addlinespace")
    rows = rows[:-1]
    add("tab:S7_comm_geo",
        "Redundancy as shared slow-mode access vs adjacency. Within-subject Spearman $\\rho$ with $q_1$ across the 114 driver sets per subject (14 named criteria + 100 random), averaged over $k\\in\\{2,3,5\\}$, median across subjects. comm = mean pairwise driver-Gram cosine ($=\\bar r_G$); geo = mean pairwise shortest-path length with edge length $1/w$ or $-\\log w$.",
        "cc" + "c" * 8,
        ["& & \\multicolumn{3}{c}{raw $\\rho$} & \\multicolumn{3}{c}{partial $\\rho\\mid L_S$} & \\multicolumn{2}{c}{comm vs geo$_{1/w}$ $\\mid L_S$} \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}",
         "$c$ & $N$ & $L_S$ & comm & geo$_{1/w}$ & comm & geo$_{1/w}$ & geo$_{-\\log w}$ & $n$ better & $p$ \\\\"], rows,
        "$n$ better: subjects with $|\\rho_{\\text{comm}\\mid L}|>|\\rho_{\\text{geo}\\mid L}|$; $p$: paired Wilcoxon on $|\\rho_{\\text{comm}\\mid L}|-|\\rho_{\\text{geo}\\mid L}|$ across subjects (uncorrected).",
        """Source: q1/redundancy_summary.csv (per-subject rows in q1/redundancy_rho.csv).
  Sets: 114 per subject (14 named criteria + 100 random; matched sets excluded); k in {2, 3, 5}.
  Per subject, rho is averaged over k; the median across subjects is reported (per q1/REPORT.md).
  POOLING: the L_S column here (rho_L) is within subject, k >= 2 only, matched sets EXCLUDED. It is therefore not the same
  quantity as the pooled all-families rho of S7_11 rows (a)-(b), which include the matched sets and (row a) k = 1.
  p = paired Wilcoxon of |prho_comm|L| - |prho_geo_inv|L| across subjects, uncorrected.
  Matches the plan's R6 numbers:
    - comm | L: +0.24 / +0.28 / +0.27 (c = 0.1) and +0.50 / +0.46 / +0.38 (c = 1);
    - geo(1/w) | L: -0.10 / -0.13 / -0.14 and -0.20 / -0.18 / -0.16;
    - comm better in 68 / 68 / 70 (c = 0.1) and 70 / 70 / 70 (c = 1) of 70.
  The plan's "-0.16 to -0.30" range at c = 1 mixes the 1/w and -log w versions (1/w: -0.16 to -0.20; -log w: -0.24 to -0.30).
  The c = 0.01 n-better values are 56 / 38 / 40, as in the report.""")

    # --- S7.5 modal sets vs cycle sets
    o = pd.read_csv(M / "set_overlap.csv")
    o = o[o.c == 0.1]
    cf = pd.read_csv(M / "cycle_vs_counterfactual.csv")
    cf = cf[cf.c == 0.1]
    rows = []
    for a, al in [("top_modal_gu", "Top-$k$ modal (Gu)"), ("top_avg_ctrl", "Top-$k$ average ctrl."), ("low_perron", "Lowest $v_1(i)^2$")]:
        for b, bl in [("cycle_participation", "cycle"), ("degree_strength", "degree"), ("eigenvector_centrality", "eigenvector")]:
            cells = [f2(o[(o.a == a) & (o.b == b) & (o.scale == s) & (o.k == k)].jaccard.iloc[0]) for s in S for k in K]
            rows.append(f"Jaccard: {al} vs {bl} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    for con, cl in [("cycle - top_modal_gu", "cycle $-$ modal (Gu)"), ("cycle - top_avg_ctrl", "cycle $-$ avg.\\ ctrl."),
                    ("cycle - low_perron", "cycle $-$ lowest $v_1^2$")]:
        for t, tl in [("q1", "$\\Delta q_1$"), ("effective_rank", "$\\Delta$ER"), ("er_raw", "$\\Delta$ER (no floor)")]:
            cells = []
            for s in S:
                for k in K:
                    r = cf[(cf.contrast == con) & (cf.target == t) & (cf.scale == s) & (cf.k == k)].iloc[0]
                    cells.append(f"{f2(r['median'])} ({int(r.n_pos)}/70)")
            rows.append(f"{tl}: {cl} & " + " & ".join(cells) + " \\\\")
    add("tab:S7_modal_sets",
        "Counterfactual node-score sets vs cycle-participation sets ($c=0.1$). Top: mean Jaccard overlap across subjects. Bottom: paired difference cycle $-$ counterfactual set, median across subjects (subjects with difference $>0$, of 70).",
        "l" + "c" * 9, scale_hdr + [kcols + "\\\\"], rows,
        "ER = exp-entropy effective rank of the $\\varepsilon$-floored spectrum ($\\varepsilon=10^{-5}$, pipeline definition); ER (no floor) uses the raw spectrum. Overlap of modal/avg.-ctrl.\\ sets with ranked sets depends weakly on $c$ through average controllability; modal-set overlaps are 0.00 with cycle, degree and eigenvector sets at every $c$.",
        """Source: modal/set_overlap.csv (mean Jaccard across subjects) and modal/cycle_vs_counterfactual.csv.
  c = 0.1. Contrast = (cycle - random median) - (counterfactual - random median), which equals cycle - counterfactual; median (n_pos).
  Plan check: "Cycle sets beat them on ER (+0.44 / +0.48 / +0.74; 55 / 65 / 69)" = effective_rank (floored), k = 3, c = 0.1.
    Exact: 0.44 (55) / 0.48 (65) / 0.74 (69). At k = 5: 0.05 (35) / 0.55 (53) / 0.98 (67). The plan should state k = 3.
  q1: cycle - modal is small and positive (cycle is MORE concentrated than modal sets): k = 3, 0.08 (69) / 0.04 (56) / 0.02 (57).
  Modal (Gu / ct) sets share 0.00 nodes with cycle, degree and eigenvector sets at every k and scale. Confirms "zero overlap".
  top_avg_ctrl vs eigenvector Jaccard = 0.88–0.98 at c = 0.1. reframing_literature.md says 0.89–0.96; that is a k = 3 subset.
  LaTeX: this table is about 10 pt overfull at 1.5 cm margins in \\tiny. Use landscape or shorten the row labels.""",
        size="\\tiny")

    # --- S7.6 ranked-set prediction (q1)
    h = pd.read_csv(M / "ranked_set_prediction_summary.csv")
    h = h[(h.c == 0.1) & (h.target == "q1")]
    rows = []
    cells = [f2(h[(h.scale == s) & (h.k == k) & (h.model == "node3")].obs_median.iloc[0]) for s in S for k in K]
    rows.append("Observed cycle $-$ degree $\\Delta q_1$ & " + " & ".join(cells) + " \\\\")
    for m, ml in [("node3", "node scores"), ("node3+redGperp", "node scores + $\\bar r_{G\\perp}$"),
                  ("pshare", "Perron share $\\pi_S$"), ("q1G", "$q_1(G_S)$")]:
        cells = []
        for s in S:
            for k in K:
                r = h[(h.scale == s) & (h.k == k) & (h.model == m)].iloc[0]
                cells.append(f"{f2(r.pred_median)} [{f2(r.rho_obs_pred_across_subjects)}]")
        rows.append(f"Predicted: {ml} & " + " & ".join(cells) + " \\\\")
    add("tab:S7_rankedpred",
        "Out-of-sample prediction of the cycle $-$ degree $q_1$ difference from linear models fit on each subject's 100 random sets ($c=0.1$). Median predicted difference across subjects [Spearman $\\rho$ between observed and predicted differences across subjects].",
        "l" + "c" * 9, scale_hdr + [kcols + "\\\\"], rows, None,
        """Source: modal/ranked_set_prediction_summary.csv, target q1, c = 0.1.
  Models are fit per subject on the 100 random sets (z-scored predictors), then applied to the cycle and degree sets.
  Cells: median predicted cycle - degree Δq1 [Spearman rho between observed and predicted across subjects].
  The node-score model overshoots the observed difference (frac_explained 0.80–1.98).
  pi_S and q1(G_S) under-predict the magnitude (0.27–0.78) but track subjects best (rho up to 0.98).
  ER-target rows exist in the CSV but are not tabulated. Node models predict the wrong sign for ER at 114/219 for k = 2–3.""",
        size="\\tiny")

    # --- S7.7 coverage
    cb = pd.read_csv(MECH / "coverage_by_criterion.csv")
    cb = cb[cb.kgrp == "2+"]
    fams = [("betweenness", "Betweenness"), ("degree_strength", "Degree/strength"), ("cycle_participation", "Cycle participation"),
            ("cycle_resid", "Cycle (degree-resid.)"), ("eigenvector_centrality", "Eigenvector centrality"),
            ("top_avg_ctrl", "Top-$k$ average ctrl."), ("participation_coefficient", "Participation coef."),
            ("top_modal_gu", "Top-$k$ modal ctrl.\\ (Gu)"), ("low_perron", "Lowest $v_1(i)^2$")]
    rows = []
    for f, fl in fams:
        cells = []
        for s in S:
            r = cb[(cb.family == f) & (cb.scale == s)].iloc[0]
            cells.append(f"{f2(r.mean_d_inv)} & {f2(r.pct_inv)} & {int(r.n_closer_inv)}/70")
        rows.append(fl + " & " + " & ".join(cells) + " \\\\")
    add("tab:S7_coverage",
        "Coverage distance (mean over non-driver regions of the $1/w$ shortest-path distance to the nearest driver) of criterion sets vs each subject's random sets, $k\\ge2$. $\\Delta$: set $-$ random (subject mean over $k$, then mean across subjects); pct: percentile among random sets; $n$: subjects with set closer than random.",
        "l" + "ccc" * 3, scale_hdr + ["Criterion " + "& $\\Delta$ & pct & $n$ " * 3 + "\\\\"], rows,
        "Coverage is independent of $c$. Hop-count and $-\\log w$ versions are in \\texttt{coverage\\_by\\_criterion.csv}.",
        """Source: mechanism/coverage_by_criterion.csv, kgrp == "2+", 1/w distance columns (mean_d_inv, pct_inv, n_closer_inv).
  Matches mechanism/REPORT.md §3 (betweenness -3.9 / -4.0 / -5.3, 70/70; degree -2.2 / -2.6 / -3.9; cycle -1.9 / -2.2 / -2.6, 60–65/70; modal +21 to +26, 0/70).
  Small wording point: REPORT lists the modal percentile as "100th". The pct_inv values are 1.00 / 0.999 / 0.998.""")

    # --- S7.8 coverage rho (computed from per-subject rows)
    cr = pd.read_csv(MECH / "coverage_rho.csv")
    rk_ = cr[cr.k > 1].groupby(["scale", "subject_id", "c"]).mean(numeric_only=True).reset_index()
    rows = []
    for c in C:
        for s in S:
            g_ = rk_[(rk_.scale == s) & (rk_.c == c)]
            cells = []
            for col in ["rho_inv", "prho_inv|L", "rho_log", "prho_log|L", "rho_hop", "prho_hop|L"]:
                v = g_[col]
                cells.append(f"{f2(v.median())} ({int((v > 0).sum())}/70)")
            rows.append(f"{CS[c]} & {s} & " + " & ".join(cells) + " \\\\")
        rows.append("\\addlinespace")
    rows = rows[:-1]
    add("tab:S7_coverage_rho",
        "Within-subject Spearman $\\rho$(coverage distance, $q_1$) across driver sets, raw and partial given $L_S$; per-subject mean over $k\\in\\{2,3,5\\}$, median across subjects (subjects with $\\rho>0$).",
        "cc" + "c" * 6,
        ["& & \\multicolumn{2}{c}{$1/w$} & \\multicolumn{2}{c}{$-\\log w$} & \\multicolumn{2}{c}{hops} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
         "$c$ & $N$ & raw & $\\mid L_S$ & raw & $\\mid L_S$ & raw & $\\mid L_S$ \\\\"], rows,
        "Summary computed by us from per-subject rows in \\texttt{coverage\\_rho.csv} (read-only), reproducing the aggregation in \\texttt{03\\_coverage\\_distance.py}.",
        """Source: COMPUTED from mechanism/coverage_rho.csv (per subject × c × k rows).
  Took rows with k > 1, averaged per subject over k, then took the median across subjects and counted rho > 0.
  This mirrors 03_coverage_distance.py line 124.
  Raw 1/w: -0.08 to -0.45. Partial | L_S (1/w): +0.07 to +0.23. Matches the REPORT and plan ("+0.07–0.23").""")

    # --- S7.9 hub loading
    hl = pd.read_csv(MECH / "hub_loading_summary.csv")
    stats = [("cos_s_v1", "$\\cos(s,v_1)$"), ("cos_As_v1", "$\\cos(As,v_1)$"), ("rho_logs_logm", "$\\rho(s_i,m_i)$"),
             ("jac_top5", "Top-5 Jaccard ($s$ vs $v_1$)"), ("L_deg_5", "$L_S$, top-5 strength"),
             ("L_max_5", "$L_{\\max}$, top-5 by $v_1^2$"), ("frac", "$L_S/L_{\\max}$ ($k=5$)"),
             ("strength_assort", "Strength assort.\\ (bct)"), ("l1", "$\\ell_1$"), ("gap", "Gap $\\ell_1-\\ell_2$")]
    rows = []
    for st, sl in stats:
        for s in S:
            cells = []
            for nl in ["strength", "geometric", "weights"]:
                if st == "frac":  # ratio of across-subject means
                    a = hl[(hl.null == nl) & (hl.scale == s) & (hl.stat == "L_deg_5")].iloc[0]
                    b = hl[(hl.null == nl) & (hl.scale == s) & (hl.stat == "L_max_5")].iloc[0]
                    real = a.real_mean / b.real_mean
                    cells.append(f"{f2(a.null_mean / b.null_mean)} (--)")
                else:
                    r = hl[(hl.null == nl) & (hl.scale == s) & (hl.stat == st)].iloc[0]
                    real = r.real_mean
                    cells.append(f"{f2(r.null_mean)} ({int(r.n_real_higher)}/70)")
            rows.append((sl if s == 68 else "") + f" & {s} & {f2(real)} & " + " & ".join(cells) + " \\\\")
        rows.append("\\addlinespace")
    rows = rows[:-1]
    add("tab:S7_hubloading",
        "Hub-loading identity $v_1(i)=s_i m_i/\\ell_1$: real connectomes vs null networks. Means across subjects (null: mean of each subject's 50-null mean); in parentheses, subjects with real $>$ own null mean (of 70).",
        "llcccc", ["Statistic & $N$ & Real & Strength null & Geometric null & Weight null \\\\"], rows,
        "$m_i=\\sum_j A_{ij}v_1(j)/s_i$. $\\rho(s_i,m_i)$ is Spearman across regions (computed on $\\log s,\\log m$). $L_S/L_{\\max}$ is the ratio of the across-subject means (computed by us from the two rows above; no per-subject count). Wilcoxon $p$ for each row is in \\texttt{hub\\_loading\\_summary.csv}.",
        """Source: mechanism/hub_loading_summary.csv (real_mean, null_mean, n_real_higher).
  Null mean = mean over subjects of each subject's 50-null mean. n = subjects with real > own null mean. Wilcoxon p is in the CSV.
  The "Real" column takes real_mean from the last null block read (real_mean is the same in every null block).
  The row L_S/L_max (k = 5) is COMPUTED as mean(L_deg_5) / mean(L_max_5). It is a ratio of means, not a per-subject median.
    Real: 0.89 / 0.91 / 0.78.
    Strength null: 0.73 / 0.68 / 0.64. Geometric: 0.81 / 0.74 / 0.64. Weights: 0.81 / 0.74 / 0.65.
  DISCREPANCY: the plan (Open items (a)) gives "78–91% of attainable vs 64–73%". The 64–73% covers the strength null only. Across all three nulls it is 64–81%.
  cos(s, v1): real 0.82 / 0.76 / 0.69 vs strength null 0.77 / 0.79 / 0.78.
    The real value is LOWER than the null at 114 and 219 (22/70 and 15/70 real > null). This matches the plan's R5 note on why the R1 share falls with scale.
  rho(s_i, m_i): real 0.58 / 0.54 / 0.43 vs nulls 0.20–0.38. Matches.
  Top-5 Jaccard: real 0.59 / 0.63 / 0.43 vs nulls 0.30–0.43. Matches.
  L_max (largest attainable): real 0.74 / 0.67 / 0.58 vs nulls 0.53–0.79. Matches.
  "strength_assort" here is bct assortativity_wei (0.14 / 0.21 / 0.19), not the edge-weighted r_W of the 19/10 Rayleigh amendment.""")

    # --- S7.10 assortativity vs hub loading
    ha = pd.read_csv(MECH / "hub_loading_assort_corr.csv")
    rows = []
    for y, yl in [("rho_logs_logm", "$\\rho(s_i,m_i)$"), ("cos_s_v1", "$\\cos(s,v_1)$"), ("L_deg_5", "$L_S$ (top-5 strength)"),
                  ("l1", "$\\ell_1$"), ("gap", "Gap")]:
        cells = []
        for s in S:
            r = ha[(ha.scale == s) & (ha.y == y)].iloc[0]
            cells.append(f"{f2(r.rho_with_assort)} ({r.p:.2g})")
        rows.append(yl + " & " + " & ".join(cells) + " \\\\")
    add("tab:S7_assortcorr",
        "Across real subjects ($n=70$): Spearman $\\rho$ of strength assortativity (bct) with hub-loading quantities ($p$).",
        "lccc", ["& 68 & 114 & 219 \\\\"], rows, None,
        """Source: mechanism/hub_loading_assort_corr.csv (across 70 real subjects).
  bct assortativity predicts rho(s_i, m_i) (0.60 / 0.57 / 0.66) but not L_S, the gap or l1 (|rho| <= 0.29).
  REPORT caveat: "|rho| <= 0.29". Correct. The largest is l1 at 68, rho = -0.286, p = 0.016 nominal. That value is nominally significant, so "does not predict" is slightly strong for l1 at 68.""")

    # --- S7.11 L_S vs q1 candidates
    cand = ls_q1_candidates(res)
    pool = pd.read_csv(M / "pooled_spearman.csv")
    pool = pool[(pool.target == "q1") & (pool.predictor == "perron_loading")]
    xs = pd.read_csv(M / "cycle_minus_degree_node_vs_set.csv")
    rows = []

    def crow(lab, getter):
        rows.append(lab + " & " + " & ".join(f2(getter(c, s)) for c in C for s in S) + " \\\\")

    def cget(sets, pooling=None):
        def get(c, s):
            m = (cand.c == c) & (cand.scale == str(s)) & (cand.sets == sets)
            if pooling is not None:
                m &= cand.pooling == pooling
            x = cand[m]
            assert len(x) == 1, (sets, pooling, c, s)
            return x.rho.iloc[0]
        return get

    crow("(a) Pooled, all sets, all $k$", cget("all families, all k", "pooled"))
    crow("(b) Pooled, all sets, $k\\ge2$", cget("all families, k>=2", "pooled"))
    crow("(c) Within-subj., all sets, $k\\ge2$", cget("all families, k>=2", "within-subj median"))
    for k in [1, 2, 3, 5]:
        crow(f"(d) Within-subj., all sets, $k={k}$", cget(f"all families, k={k}"))
    for k in K:
        crow(f"(e) Pooled (modal file), $k={k}$", lambda c, s, k=k: pool[(pool.c == c) & (pool.scale == s) & (pool.k == k)].rho.iloc[0])
    for k in K:
        crow(f"(f) Across subj., $\\Delta L_S$ vs $\\Delta q_1$, $k={k}$",
             lambda c, s, k=k: xs[(xs.c == c) & (xs.scale == s) & (xs.k == k)]["q1~perron_loading"].iloc[0])
    f9 = xs[(xs.c == 0.1) & xs.k.isin(K) & xs.scale.isin(S)]["q1~perron_loading"]
    add("tab:S7_LS_q1",
        "Where does ``$L_S$ predicts $q_1$ ($\\rho=0.86$)'' come from? Spearman $\\rho(L_S,q_1)$ under different poolings. No saved result equals 0.86 under a single stated pooling (see notes).",
        "l" + "c" * 9,
        ["& \\multicolumn{3}{c}{$c=0.01$} & \\multicolumn{3}{c}{$c=0.1$} & \\multicolumn{3}{c}{$c=1$} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
         "Pooling " + "& 68 & 114 & 219 " * 3 + "\\\\"], rows,
        "(a)--(d) computed by us (read-only) from \\texttt{q1/set\\_q1.csv.gz} (all 18 set families incl.\\ random and matched sets; $q_1$ on raw $W$). (e) \\texttt{modal/pooled\\_spearman.csv} (random + ranked + counterfactual sets, pooled over subjects). (f) \\texttt{modal/cycle\\_minus\\_degree\\_node\\_vs\\_set.csv}: across 70 subjects, cycle $-$ degree differences.",
        f"""Source hunt for "L_S predicts q1 (rho = 0.86)".
  The ONLY occurrences of the claim are in planning text (reframing_literature.md line 34; revision_plan.md lines 47 and 166;
  the commented outline in network_neuro_resubmission/main.tex line 131). No saved CSV or report contains 0.86 for L_S vs q1.
  Searched: regime/task3_perron_spearman.csv (L_S vs ER, not q1), spectra/ (L_S vs ER), modal/pooled_spearman.csv, q1/, mechanism/.
  Candidates tabulated:
    (a)-(d) COMPUTED in s7.ls_q1_candidates from q1/set_q1.csv.gz (columns loading, q1; all 18 families incl. random AND
      matched sets; q1 on raw W). (Was scripts/ls_q1.py -> tables/ls_q1_candidates.csv; the computed frame equals that CSV.)
      The frame also has "rankings+random only" variants (not tabulated; that variant omits cycle_resid).
    (e) modal/pooled_spearman.csv, predictor perron_loading, target q1 (random + ranked + counterfactual; no matched sets).
    (f) modal/cycle_minus_degree_node_vs_set.csv "q1~perron_loading" (across subjects, cycle - degree differences).
  POOLING: rows (a)-(d) pool all 18 families INCLUDING the degree/spatial matched sets; (e) excludes matched sets; S7_4's
  rho_L (within subject, 14 named + 100 random, k >= 2) also excludes them. These are different set collections.
  Findings:
    - At c = 0.1, pooled across k, rho(L_S, q1) is ~0 or negative: 0.03 / -0.06 / -0.19 (all k); -0.05 for all scales pooled.
      This is because L_S grows with k while q1 does not, and at k = 1 the relation is strongly NEGATIVE (within-subject -0.89 / -0.87 / -0.87).
    - Within a fixed k >= 3 it is strong: within-subject median 0.80 / 0.52 / 0.35 (k = 3), 0.95 / 0.88 / 0.76 (k = 5).
      Pooled (modal file): 0.55 / 0.40 / 0.23 (k = 3), 0.85 / 0.74 / 0.59 (k = 5).
    - Strongest at c = 0.01: pooled k >= 2, 0.88 / 0.80 / 0.66 (0.80 all scales pooled); within-subject k >= 2, 0.87 / 0.83 / 0.66.
    - The closest match to the plan's R3 context (cycle vs degree) is (f): across subjects, Delta L_S vs Delta q1 at c = 0.1.
      The plain mean of those 9 cells is {f9.mean():.3f} (computed here). It is an across-subject correlation of differences, NOT "across all sets".""",
        size="\\tiny")
    return tables
