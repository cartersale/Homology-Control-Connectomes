"""Supplement S6: all criteria vs random sets, and matched controls (S6_1-S6_6).

Port of scripts/revision/paper/_rescued_2026-10-19/work/s6.py.
"""

from __future__ import annotations

from pathlib import Path

from paper_common import LABEL, SCALES, Table, npos, pd, ptex, tabular


def build(res: Path) -> list[Table]:
    Q1 = res / "q1"
    p1 = pd.read_csv(Q1 / "p1_vs_random.csv")
    p2 = pd.read_csv(Q1 / "p2_matched.csv")
    FAMS = ["degree_strength", "eigenvector_centrality", "top_avg_ctrl", "betweenness", "participation_coefficient",
            "triangle_participation", "tetrahedron_participation", "cycle_participation", "cycle_resid",
            "degree_normalized_cycle_participation", "cycle_criticality", "top_modal_gu", "top_modal_ct", "low_perron"]
    KS = ["1", "2", "3", "5", "avg2+", "avg"]
    KLAB = {"1": "1", "2": "2", "3": "3", "5": "5", "avg2+": "avg $k\\geq2$", "avg": "avg all"}
    MNAME = {"q1": "$q_1$", "er": "effective rank", "pr": "participation ratio"}
    parts = []

    for metric, d in (("q1", 3), ("er", 2), ("pr", 2)):
        rows = []
        for fam in FAMS:
            rows.append(("MULTI", LABEL[fam], 10))
            for k in KS:
                r = [KLAB[k]]
                for scale in SCALES:
                    x = p1[(p1.family == fam) & (p1.scale == scale) & (p1.c == 0.1) & (p1.k == k) & (p1.metric == metric)]
                    assert len(x) == 1, (fam, scale, k, metric)
                    x = x.iloc[0]
                    r += [f"{x['mean']:+.{d}f}", f"{x.pct_mean:.2f}", npos(x.n_positive)]
                rows.append(r)
        tab = tabular("l" + "ccc" * 3,
                      ["& \\multicolumn{3}{c}{68} & \\multicolumn{3}{c}{114} & \\multicolumn{3}{c}{219} \\\\",
                       "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
                       "$k$ & $\\Delta$ & Pct. & $n_>$ & $\\Delta$ & Pct. & $n_>$ & $\\Delta$ & Pct. & $n_>$ \\\\"], rows)
        note = f"""S6-{metric}. Every ranking criterion vs random driver sets of the same size, {MNAME[metric]} (c = 0.1, eps = 1e-5, T = inf).
    Source: results/revision/q1/p1_vs_random.csv (filters: c == 0.1, metric == '{metric}', all 14 families; k in 1,2,3,5 and the saved
    k-averages 'avg2+' = mean over k in {{2,3,5}} and 'avg' = mean over k in {{1,2,3,5}}). Same values as q1/tables.md (P1 tables, avg rows).
    Delta = criterion value - mean over the subject's 100 random sets, averaged over 70 subjects; Pct. = mean over subjects of the fraction
    of the subject's random sets below the criterion set (column pct_mean, 0-1); n_> = subjects with Delta > 0 (of 70).
    Wilcoxon p and Holm p (column p_holm) are in the CSV. q1 is on raw W; ER and PR on W_eps.
    For ER/PR, positive Delta = MORE evenly distributed than random; for q1, positive Delta = MORE concentrated than random.
    Top-k modal (Gu) and (CT) are near-identical sets; 'Lowest Perron loading' = the k nodes with smallest v1(i)^2."""
        parts.append(Table(f"S6_{len(parts) + 1}", tab, note, long=True))

    # matched controls
    COMPS = ["cycle - degree", "cycle - random", "cycle - degree_matched", "cycle - spatial_matched", "cycle - degree_spatial_matched",
             "cycle_resid - random", "cycle_resid - degree"]
    CLAB = {"cycle - degree": "Cycle $-$ degree", "cycle - random": "Cycle $-$ random",
            "cycle - degree_matched": "Cycle $-$ degree-matched", "cycle - spatial_matched": "Cycle $-$ spatially matched",
            "cycle - degree_spatial_matched": "Cycle $-$ degree+spatial matched",
            "cycle_resid - random": "Degree-residualized cycle $-$ random", "cycle_resid - degree": "Degree-residualized cycle $-$ degree"}
    for metric, d in (("q1", 3), ("er", 2), ("pr", 2)):
        rows = []
        for comp in COMPS:
            rows.append(("MULTI", CLAB[comp], 10))
            for k in KS:
                sub = p2[(p2.comparison == comp) & (p2.c == 0.1) & (p2.k == k) & (p2.metric == metric)]
                if sub.empty:
                    continue
                r = [KLAB[k] + (f" ($n_k$={int(sub.n_k.iloc[0])})" if k in ("avg", "avg2+") else "")]
                for scale in SCALES:
                    x = sub[sub.scale == scale].iloc[0]
                    r += [f"{x['mean']:+.{d}f}", npos(x.n_positive), ptex(x.p_holm)]
                rows.append(r)
        tab = tabular("l" + "ccc" * 3,
                      ["& \\multicolumn{3}{c}{68} & \\multicolumn{3}{c}{114} & \\multicolumn{3}{c}{219} \\\\",
                       "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
                       "$k$ & Mean & $n_+$ & $p_{\\mathrm{H}}$ & Mean & $n_+$ & $p_{\\mathrm{H}}$ & Mean & $n_+$ & $p_{\\mathrm{H}}$ \\\\"], rows)
        note = f"""S6-matched-{metric}. Matched and residualized controls, {MNAME[metric]} (c = 0.1, eps = 1e-5).
    Source: results/revision/q1/p2_matched.csv (filters: c == 0.1, metric == '{metric}'; all comparisons; k rows as saved).
    Mean = mean over subjects of (first set - second set); for matched/random comparisons the second set is the mean over that subject's
    matched/random sets. n_+ = subjects with a positive difference (of 70); p_H = Holm-corrected Wilcoxon p (column p_holm, Holm family as
    defined in scripts/revision/q1/02_tables.py). Spatially matched and degree+spatially matched sets exist only for k >= 2, so their
    'avg' rows average k in {{2,3,5}} (n_k = 3) and equal the 'avg k>=2' rows. 'Degree-residualized cycle' = cycle participation after
    regressing out degree strength (family cycle_resid)."""
        parts.append(Table(f"S6_{len(parts) + 1}", tab, note, long=True))
    return parts
