"""Supplement S9: representative-cycle conventions and perturbation stability (S9_1-S9_8).

Port of scripts/revision/paper/_rescued_2026-10-19/work/s9.py.
"""

from __future__ import annotations

from pathlib import Path

from paper_common import LABEL, SCALES, Table, np, pd, ptex, tabular

VAR = ["ripser", "gudhi", "geodesic", "birth_edge"]
LEVELS = (0.05, 0.1, 0.2)


def _perturb_label(fam: str, lvl: float) -> str:
    name = "Noise $\\sigma$" if fam == "noise" else "Edge removal $p$"
    return f"{name} $= {lvl:.2f}$"


def build(res: Path) -> list[Table]:
    st = res / "stability"
    tb = st / "tables"
    parts = []

    # ---------------- S9a: agreement with degree ----------------
    ag = pd.read_csv(tb / "rep_agreement.csv")
    rows = []
    for scale in SCALES:
        for v in VAR:
            x = ag[(ag.scale == scale) & (ag.variant == v)].iloc[0]
            rows.append([str(scale) if v == "ripser" else "", LABEL[v],
                         f"{x.rho_degree_median:.2f} ({x.rho_degree_q25:.2f}--{x.rho_degree_q75:.2f})",
                         "1" if v == "ripser" else f"{x.rho_ripser_median:.2f} ({x.rho_ripser_q25:.2f}--{x.rho_ripser_q75:.2f})"]
                        + [f"{x[f'shared_degree_k{k}']:.2f}" for k in (1, 2, 3, 5, 15)])
        if scale != 219:
            rows.append("MIDRULE")
    tab = tabular("llccccccc",
                  ["& & & & \\multicolumn{5}{c}{Mean nodes shared with degree top-$k$} \\\\", "\\cmidrule(lr){5-9}",
                   "Scale & Variant & $\\rho$ vs degree & $\\rho$ vs Ripser & $k=1$ & $k=2$ & $k=3$ & $k=5$ & $k=15$ \\\\"], rows)
    note = """S9a. Agreement of the four representative-cycle conventions with degree strength and with the Ripser-cocycle ranking.
Source: results/revision/stability/tables/rep_agreement.csv (all rows), summarised from
results/revision/stability/representatives/per_subject_scale-{68,114,219}.csv. rho = per-subject Spearman correlation of node scores,
median (IQR) across 70 subjects. Shared = mean number of top-k nodes in common with the degree top-k set.
Cross-check: the original S4 script re-run at all scales (results/revision/stability/reproduce_existing_114/,
existing_script_scale-68/, existing_script_scale-219/; *_selection_overlap_summary_*.csv) gives the same shared-node means
(e.g. 114: ripser 0.10/0.20/0.31/0.73/4.34; length-minimized 0.36/0.89/1.67/3.03/9.83) and Ripser-vs-variant score correlations
(mean rho ripser-gudhi 0.62/0.54/0.57, ripser-geodesic 0.48/0.40/0.45; score_correlation_summary_*.csv).
NOTE: per-subject rho(cycle participation, degree) = 0.48 / 0.38 / 0.40 here; the 0.75 / 0.71 / 0.61 quoted in the plan's Introduction is
the GROUP-level map correlation (results/revision/targets/REPORT.md, spin-tested), not a per-subject value."""
    parts.append(Table("S9_1", tab, note))

    # ---------------- S9b: result level ER / PR per variant, incl. k-average ----------------
    rl = pd.read_csv(tb / "rep_result_level.csv")
    ps = pd.concat([pd.read_csv(st / "representatives" / f"per_subject_scale-{s}.csv") for s in SCALES])
    kav = (ps[ps.k.isin([1, 2, 3, 5])].groupby(["scale", "variant", "subject_id"])[["er_diff_degree", "pr_diff_degree"]].mean()
           .reset_index())
    for metric, col in (("er", "er_diff_degree"), ("pr", "pr_diff_degree")):
        rows = []
        for scale in SCALES:
            for v in VAR:
                r = [str(scale) if v == "ripser" else "", LABEL[v]]
                for k in (1, 2, 3, 5, 15):
                    x = rl[(rl.scale == scale) & (rl.variant == v) & (rl.k == k) & (rl.metric == metric)].iloc[0]
                    r.append(f"{x['mean']:+.2f} ({int(x.n_positive)})")
                kk = kav[(kav.scale == scale) & (kav.variant == v)][col]
                assert len(kk) == 70, (scale, v, len(kk))
                r.append(f"{kk.mean():+.2f} ({int((kk > 0).sum())})")
                rows.append(r)
            if scale != 219:
                rows.append("MIDRULE")
        tab = tabular("llcccccc", ["Scale & Variant & $k=1$ & $k=2$ & $k=3$ & $k=5$ & $k=15$ & avg $k\\in\\{1,2,3,5\\}$ \\\\"], rows)
        mname = "effective rank" if metric == "er" else "participation ratio"
        note = f"""S9b-{metric}. Cycle - degree {mname} under each representative convention, at all 3 scales
including k = 1 (c = 0.1, eps = 1e-5, T = inf). Entries: mean paired difference (subjects positive, of 70).
Source: results/revision/stability/tables/rep_result_level.csv (metric == '{metric}'); medians, bootstrap CIs, Wilcoxon p and rank-biserial
are there and in representatives_tables.tex (tab:rep_er / tab:rep_pr). Last column computed here (read-only pandas) from
representatives/per_subject_scale-*.csv: per-subject mean of {col} over k in {{1,2,3,5}}, then mean over subjects and count > 0.
The Ripser rows reproduce the primary analysis exactly (e.g. avg 0.84 / 0.98 / 0.80; k = 1: +0.01 / -0.10 / -0.19).
Cross-check: identical to the original S4 script re-run (existing_script_scale-68/, reproduce_existing_114/, existing_script_scale-219/
result_level_summary_*.csv; birth-edge variant only in the new run).
MISSING: q1 for the representative variants was not computed in any saved stability file (per_subject_scale-*.csv has ER/PR only), so the
requested 'cycle - degree q1 per convention' cannot be filled without a new run."""
        parts.append(Table(f"S9_{len(parts) + 1}", tab, note))

    # ---------------- S9c: variants vs nulls ----------------
    nc = pd.read_csv(tb / "rep_null_comparisons.csv")
    rows = []
    for scale in SCALES:
        rows.append(("MULTI", f"{scale} regions", 5))
        for k in (1, 2, 3, 5):
            for v in ["degree"] + VAR:
                x = nc[(nc.scale == scale) & (nc.k == k) & (nc.variant == v)]
                if x.empty:
                    continue
                x = x.iloc[0]
                uni = f"{x.random_mean:+.2f} / {x.mean_pct_random:.0f} / {int(x.random_n_positive)}"
                if v == "degree":
                    ov = sm = "--"
                else:
                    ov = f"{x.overlap_mean:+.2f} / {x.mean_pct_overlap_null:.0f} / {int(x.overlap_n_positive)}"
                    sm = f"{x.strength_mean:+.2f} / {x.mean_pct_strength_null:.0f} / {int(x.strength_n_positive)}"
                rows.append([str(k) if v == "degree" else "", LABEL[v], uni, ov, sm])
    tab = tabular("llccc", ["$k$ & Set & Uniform random & Overlap-matched & Strength-matched \\\\"], rows)
    note = """S9c. ER of each variant's top-k set relative to three nulls (c = 0.1, eps = 1e-5). Cells: mean over subjects of (set ER - null-mean ER) /
mean percentile of the set within its null (0-100) / subjects above the null mean (of 70). Uniform = the primary 100 random sets (same seeds);
overlap-matched = 100 random sets sharing exactly as many nodes with the degree top-k set as the variant does; strength-matched = 100 random
sets whose members match the variant's nodes' strength ranks within +/-5% of N. Degree row: degree set vs uniform null.
Source: results/revision/stability/tables/rep_null_comparisons.csv (columns random_mean, mean_pct_random, random_n_positive, overlap_*,
strength_*). CIs and Wilcoxon p are in the CSV."""
    parts.append(Table("S9_4", tab, note, long=True))

    # ---------------- S9d: distance regressions ----------------
    dr = pd.read_csv(tb / "rep_distance_regressions.csv")
    rows = []
    # iterrows() upcasts the integer scale and k columns to float (the row Series is float64), so they print as
    # '68.0' / '1.0'. That is what the 2026-10-19 draft has; kept so the table reproduces byte for byte.
    for _, x in dr.iterrows():
        rows.append([str(x.scale), str(x.k), f"{x.jaccard_degree_beta_per_sd:+.2f}", f"{x.jaccard_degree_partial_r2:.2f}",
                     f"{x.jaccard_degree_variant_partial_r2_beyond:.2f}", ptex(x.jaccard_degree_variant_F_p),
                     f"{x.jaccard_degree_within_spearman:+.2f}"])
    tab = tabular("rrccccc", ["Scale & $k$ & $\\beta$ (per SD $J$) & $R^2_J$ & $R^2_{\\mathrm{var}|J}$ & $p_{\\mathrm{var}|J}$ & within-subj.\\ $\\rho$ \\\\"], rows)
    note = """S9d. Does distance from the degree set explain the cycle - degree ER differences across conventions? Observations = 4 variants x 70
subjects per scale and k, subject fixed effects. beta = change in cycle - degree ER per SD of the variant's top-k Jaccard overlap with the
degree set; R2_J = within-subject variance explained by overlap; R2_var|J = additional variance explained by variant identity given overlap
(nested F test p); last column = within-subject Spearman rho(ER difference, Jaccard). Source: results/revision/stability/tables/
rep_distance_regressions.csv (jaccard_degree_* columns; the CSV also has the same models with spearman_vs_degree and closeness_rank)."""
    parts.append(Table("S9_5", tab, note))

    # ---------------- S9e: perturbation rank stability ----------------
    rs = pd.read_csv(tb / "perturb_rank_stability.csv")
    ref = pd.read_csv(tb / "perturb_between_subject_reference.csv")
    rk_order = ["ripser", "gudhi", "geodesic", "birth_edge", "degree"]
    hdr = " & ".join(LABEL[r].replace("Degree strength", "Degree") for r in rk_order)
    rows, rows2 = [], []
    for scale in SCALES:
        for fam in ("noise", "dropout"):
            for lvl in LEVELS:
                r = [str(scale) if (fam == "noise" and lvl == 0.05) else "", _perturb_label(fam, lvl)]
                r2 = list(r)
                for rk in rk_order:
                    x = rs[(rs.scale == scale) & (rs.family == fam) & np.isclose(rs.level, lvl) & (rs.ranking == rk)].iloc[0]
                    r.append(f"{x.rho_median:.2f} [{x.rho_q25:.2f}, {x.rho_q75:.2f}]")
                    r2.append(f"{x.j1:.2f} / {x.j5:.2f} / {x.j15:.2f}")
                rows.append(r)
                rows2.append(r2)
        rows.append(["", "\\textit{Between-subject reference}"]
                    + [f"{ref[(ref.scale == scale) & (ref.ranking == rk)].iloc[0].between_subject_spearman_median:.2f}"
                       for rk in rk_order])
        if scale != 219:
            rows.append("MIDRULE")
            rows2.append("MIDRULE")
    tab = tabular("llccccc", [f"Scale & Perturbation & {hdr} \\\\"], rows)
    note = """S9e. Rank stability of node scores under edge-weight perturbation. Each subject's connectome was perturbed 50 times per level
(multiplicative Gaussian noise on existing edges with s.d. sigma, or removal of each edge with probability p; config [robustness]),
and all rankings were recomputed from scratch. Cells: median across 70 subjects [IQR] of the per-subject mean (over 50 draws) Spearman rho
between perturbed and unperturbed node scores. Reference row: median over all subject pairs of the between-subject Spearman rho of the
unperturbed scores (how different two people's rankings are).
Sources: results/revision/stability/tables/perturb_rank_stability.csv, perturb_between_subject_reference.csv (built from
perturbation/per_draw_scale-*.csv and unperturbed_scores_scale-*.npz by scripts/revision/stability/summarize_perturbation.py).
CAVEAT: after multiplicative noise, 51-65% of noise draws had to be re-normalized (max weight > 1 -> rescaled; perturb_rescaled_fraction.csv);
no dropout draws were rescaled."""
    parts.append(Table("S9_6", tab, note))
    tab = tabular("llccccc", [f"Scale & Perturbation & {hdr} \\\\"], rows2)
    note = """S9f. Top-k overlap under perturbation: mean over subjects (of the per-subject mean over 50 draws) of the Jaccard index between the
perturbed and unperturbed top-k sets, shown as J1 / J5 / J15. Source: results/revision/stability/tables/perturb_rank_stability.csv
(columns j1, j5, j15; medians j5_median / j15_median also saved). J2 and J3 exist only in perturbation/per_draw_scale-*.csv."""
    parts.append(Table("S9_7", tab, note))

    # ---------------- S9g: ER difference persistence (Ripser) ----------------
    pe = pd.read_csv(tb / "perturb_er_persistence.csv")
    rows = []
    for scale in SCALES:
        sub = pe[(pe.scale == scale) & (pe.variant == "ripser") & (pe.metric == "er")]
        r = [str(scale), "Unperturbed"]
        for k in (1, 2, 3, 5):
            x = sub[sub.k == k].iloc[0]  # unperturbed_* is repeated on every family/level row of this k
            r.append(f"{x.unperturbed_mean:+.2f} ({int(x.unperturbed_n_positive)})")
        rows.append(r)
        for fam in ("noise", "dropout"):
            for lvl in LEVELS:
                r = ["", _perturb_label(fam, lvl)]
                for k in (1, 2, 3, 5):
                    x = sub[(sub.family == fam) & np.isclose(sub.level, lvl) & (sub.k == k)].iloc[0]
                    r.append(f"{x.perturbed_mean:+.2f} ({int(x.n_subjects_positive)})")
                rows.append(r)
        if scale != 219:
            rows.append("MIDRULE")
    tab = tabular("llcccc", ["Scale & Condition & $k=1$ & $k=2$ & $k=3$ & $k=5$ \\\\"], rows)
    note = """S9g. Cycle (Ripser) - degree effective rank recomputed on perturbed connectomes (both rankings and Gramians recomputed; c = 0.1,
eps = 1e-5). Cells: mean over subjects of the per-subject mean over 50 draws (subjects with a positive per-subject mean, of 70).
Source: results/revision/stability/tables/perturb_er_persistence.csv (variant == ripser, metric == er). The CSV also holds PR, the other
three variants and k = 15, plus fraction of draws positive and Wilcoxon p. q1 not saved for perturbed networks."""
    parts.append(Table("S9_8", tab, note))
    assert [t.id for t in parts] == [f"S9_{i}" for i in range(1, 9)]
    return parts
