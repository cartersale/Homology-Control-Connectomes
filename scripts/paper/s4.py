"""Supplement S4: epsilon sweep, floor definitions, q1 vs eps, eigenvalue extremes (S4_1-S4_5).

Port of scripts/revision/paper/_rescued_2026-10-19/work/s4.py.
"""

from __future__ import annotations

from pathlib import Path

from paper_common import LABEL, SCALES, Table, np, npos, num, pd, ptex, tabular


def eps_tex(e):
    return f"$10^{{{int(round(np.log10(e)))}}}$"


def build(res: Path) -> list[Table]:
    REG = res / "regime"
    parts = []

    # ---------------- S4a: full eps sweep, c=0.1, T=inf, degree & cycle ----------------
    tidy = pd.read_csv(REG / "horizon_eps_largek_tidy.csv.gz")
    t = tidy[(tidy.c == 0.1) & np.isinf(tidy["T"]) & tidy.k.isin([1, 2, 3, 5])]
    # per subject k-average, then mean over subjects
    sub = t.groupby(["scale", "eps", "method", "subject_id"])[["log10_average_energy", "fraction_at_floor"]].mean().reset_index()
    agg = sub.groupby(["scale", "eps", "method"])[["log10_average_energy", "fraction_at_floor"]].mean().reset_index()
    s1 = pd.read_csv(REG / "task5_S1_epsilon_summary.csv")
    s1e = s1[(s1.c == 0.1) & (s1.metric == "log10_average_energy")]
    D = pd.read_csv(REG / "task6_delta_log10E_summary.csv")
    rows = []
    for scale in SCALES:
        rows.append(("MULTI", f"{scale} regions", 9))
        for eps in sorted(t.eps.unique()):
            g = agg[(agg.scale == scale) & (agg.eps == eps)].set_index("method")
            d = s1e[(s1e.scale == scale) & np.isclose(s1e.eps, eps)].iloc[0]
            rows.append([eps_tex(eps),
                         num(g.loc["degree_strength", "fraction_at_floor"]), num(g.loc["cycle_participation", "fraction_at_floor"]),
                         num(g.loc["degree_strength", "log10_average_energy"]), num(g.loc["cycle_participation", "log10_average_energy"]),
                         num(np.log10(scale / eps)),
                         f"{d.mean_diff * 1e3:+.2f}", npos(d.n_positive), ptex(d.p_holm)])
    tab = tabular("lcccccccc",
                  ["& \\multicolumn{2}{c}{Frac.\\ eig.\\ numerically 0} & \\multicolumn{2}{c}{$\\log_{10}E_{\\mathrm{avg}}$} & & \\multicolumn{3}{c}{Cycle $-$ degree $\\Delta\\log_{10}E_{\\mathrm{avg}}$} \\\\",
                   "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){7-9}",
                   "$\\epsilon$ & Degree & Cycle & Degree & Cycle & $\\log_{10}(N/\\epsilon)$ & Mean ($\\times10^{-3}$) & $n_+$ & $p_{\\mathrm{Holm}}$ \\\\"],
                  rows)
    # The two Holm families at 68 regions / eps = 1e-5 (quoted in the note; previously typed in).
    p_s1 = s1e[(s1e.scale == 68) & np.isclose(s1e.eps, 1e-5)].p_holm.iloc[0]
    p_t6 = D[(D.scale == 68) & np.isclose(D.eps, 1e-5)].p_holm.iloc[0]
    note = f"""S4a. Average energy and floor fraction across the full epsilon sweep (c = 0.1, infinite horizon, k in {{1,2,3,5}}).
Sources: results/revision/regime/horizon_eps_largek_tidy.csv.gz (per-subject rows; filter c == 0.1, T == inf, k in {{1,2,3,5}},
methods degree_strength / cycle_participation) and results/revision/regime/task5_S1_epsilon_summary.csv
(metric == log10_average_energy, c == 0.1) for the paired difference.
Computation (read-only pandas): per-subject mean over k, then mean over the 70 subjects.
'Frac. eig. numerically 0' = pipeline definition fraction_at_floor: eigenvalue of W (before flooring) <= max(machine eps, 1e-8*eps),
i.e. eig(W) is zero to numerical precision (scripts/revision/regime/_common.py::metrics_from_eigs).
CAVEAT: the tolerance scales with eps, so at eps = 1e-8 (tol ~1e-16) fewer eigenvalues pass the test; the count is a property of W,
not of eps, only for eps >= ~1e-6. The alternative definition eig(W) < eps is in S4b (available only at 3 eps values).
p_Holm: Holm over 7 eps x 3 scales within c = 0.1 (family used in task5_S1). Note: S4b reports a different Holm family
(3 eps x 3 scales, task6) — at 68 regions / eps = 1e-5 this gives p_Holm {p_t6:.3f} instead of {p_s1:.3f} here."""
    parts.append(Table("S4_1", tab, note))

    # ---------------- S4b: three eps, all methods, eig<eps, landscape ----------------
    a = pd.read_csv(REG / "task6_driver_sets.csv")
    asub = a.groupby(["scale", "eps", "method", "subject_id"])[["fraction_at_floor", "fraction_below_eps", "log10_average_energy",
                                                                  "log10_lambda_max"]].mean().reset_index()
    aagg = asub.groupby(["scale", "eps", "method"]).mean(numeric_only=True).reset_index()
    L = pd.read_csv(REG / "task6_landscape.csv")
    Lagg = L.groupby(["scale", "eps"])[["iqr_log10E", "iqr_log10E_above_eps", "range_log10E"]].mean().reset_index()
    rows = []
    for scale in SCALES:
        rows.append(("MULTI", f"{scale} regions", 9))
        for eps in (1e-8, 1e-5, 1e-2):
            g = aagg[(aagg.scale == scale) & np.isclose(aagg.eps, eps)]
            ff = g.fraction_at_floor
            fb = g.fraction_below_eps
            rnd = g[g.method == "random_mean"].iloc[0]
            d = D[(D.scale == scale) & np.isclose(D.eps, eps)].iloc[0]
            lnd = Lagg[(Lagg.scale == scale) & np.isclose(Lagg.eps, eps)].iloc[0]
            rows.append([eps_tex(eps),
                         f"{ff.min():.2f}--{ff.max():.2f}", f"{fb.min():.2f}--{fb.max():.2f}",
                         num(rnd.log10_average_energy), num(np.log10(scale / eps)),
                         f"{d.mean_diff * 1e3:+.2f} ({npos(d.n_positive)})", ptex(d.p_holm),
                         f"{lnd.iqr_log10E * 1e3:.2f}", f"{lnd.iqr_log10E_above_eps:.2f}"])
    tab = tabular("lcccccccc",
                  ["& \\multicolumn{2}{c}{Fraction of eigenvalues} & & & \\multicolumn{2}{c}{Cycle $-$ degree $\\Delta\\log_{10}E$} & \\multicolumn{2}{c}{Landscape IQR of $\\log_{10}E$} \\\\",
                   "\\cmidrule(lr){2-3}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
                   "$\\epsilon$ & numerically 0 & $\\lambda(W)<\\epsilon$ & $\\log_{10}E$ (random) & $\\log_{10}(N/\\epsilon)$ & Mean $\\times10^{-3}$ ($n_+$) & $p_{\\mathrm{Holm}}$ & All ($\\times10^{-3}$) & Above-$\\epsilon$ part \\\\"],
                  rows)
    note = """S4b. Floor definitions, energy, cycle-degree energy difference and landscape spread at eps in {1e-8, 1e-5, 1e-2} (c = 0.1, T = inf).
Sources: results/revision/regime/task6_driver_sets.csv (filters: all rows; k in {1,2,3,5}; methods degree_strength, cycle_participation,
eigenvector_centrality, betweenness, random_mean), task6_delta_log10E_summary.csv, task6_landscape.csv
(also printed in results/revision/regime/report_tables.md sections T6-methods / T6-delta / T6-landscape).
Fraction columns: per-subject k-average, mean over subjects, then the range (min--max) across the 5 set types.
'numerically 0' = eig(W) <= max(machine eps, 1e-8*eps) (pipeline floor test); 'lambda(W) < eps' = eig(W) below the floor value.
Delta log10 E: cycle - degree, k-averaged per subject; Wilcoxon signed-rank, Holm over 3 scales x 3 eps (task6 family).
Landscape: per subject x k, log10 E_avg of the primary run's saved candidate sets (results/primary/landscape_samples: 5000 random
k-sets, or exhaustive when C(N,k) <= 5000); IQR across sets, then mean over 70 subjects x 4 k values.
'Above-eps part' = IQR of log10 sum over eig(W) > eps of 1/(lambda + eps), i.e. energy carried by non-floored directions."""
    parts.append(Table("S4_2", tab, note))

    # ---------------- S4c: ER / PR cycle-degree across eps ----------------
    rows = []
    for c in (0.1, 0.01):
        rows.append(("MULTI", f"c = {c}", 7))
        for eps in sorted(s1.eps.unique()):
            r = [eps_tex(eps)]
            for metric in ("effective_rank", "participation_ratio"):
                for scale in SCALES:
                    x = s1[(s1.c == c) & np.isclose(s1.eps, eps) & (s1.metric == metric) & (s1.scale == scale)].iloc[0]
                    r.append(f"{x.mean_diff:+.2f} ({int(x.n_positive)})")
            rows.append(r)
    tab = tabular("lcccccc",
                  ["& \\multicolumn{3}{c}{Effective rank} & \\multicolumn{3}{c}{Participation ratio} \\\\",
                   "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
                   "$\\epsilon$ & 68 & 114 & 219 & 68 & 114 & 219 \\\\"], rows)
    shown = s1[s1.c.isin([0.1, 0.01]) & s1.metric.isin(["effective_rank", "participation_ratio"])]
    note = f"""S4c. Cycle - degree effective rank and participation ratio across eps (replaces published Table S1, now at the primary c = 0.1;
the c = 0.01 block reproduces the published c). Entries: mean paired difference (subjects positive, of 70).
Source: results/revision/regime/task5_S1_epsilon_summary.csv (metric in {{effective_rank, participation_ratio}}; T = inf; k-averaged over
{{1,2,3,5}}). All p_Holm <= {shown.p_holm.max():.1e} (Holm over eps x scale within c). CAVEAT: from eps = 1e-4 upward ER/PR are dominated by the floor
(the eps*I term), so the large values at eps >= 1e-3 reflect regularization, not network structure."""
    parts.append(Table("S4_3", tab, note))

    # ---------------- S4d: q1 across eps ----------------
    q = pd.read_csv(res / "q1" / "set_q1.csv.gz", usecols=["subject_id", "scale", "c", "k", "family", "q1"])
    q = q[(q.c == 0.1) & q.family.isin(["degree_strength", "cycle_participation", "random"])]
    qs = q.groupby(["scale", "family", "subject_id"]).q1.mean().reset_index().groupby(["scale", "family"]).q1.mean()
    spec = np.load(REG / "task6_spectra_k5_c0.1.npz")
    eps_list = (1e-8, 1e-6, 1e-5, 1e-4, 1e-2)
    rows = []
    for scale in SCALES:
        for fam, key in (("degree_strength", "degree_strength"), ("cycle_participation", "cycle_participation"), ("random", "random0")):
            w = spec[f"scale{scale}_{key}"]  # (70, N) eigenvalues of raw W, k = 5
            raw = (w.max(axis=1) / w.sum(axis=1)).mean()
            cells = [str(scale) if fam == "degree_strength" else "", LABEL[fam] if fam != "random" else "Random (set 0 only)",
                     num(qs.loc[(scale, fam)]), num(raw)]
            for eps in eps_list:
                ev = np.clip(w + eps, eps, None)
                cells.append(num((ev.max(axis=1) / ev.sum(axis=1)).mean()))
            rows.append(cells)
        if scale != SCALES[-1]:
            rows.append("MIDRULE")
    tab = tabular("llccccccc",
                  ["& & $q_1$ (raw $W$) & \\multicolumn{6}{c}{$k=5$: $\\lambda_{\\max}/\\mathrm{tr}$ computed on} \\\\",
                   "\\cmidrule(lr){4-9}",
                   "Scale & Set & $k$-avg & raw $W$ & $W_{10^{-8}}$ & $W_{10^{-6}}$ & $W_{10^{-5}}$ & $W_{10^{-4}}$ & $W_{10^{-2}}$ \\\\"], rows)
    note = """S4d. q1 = lambda_max(W)/tr W is computed on the raw Gramian and is therefore eps-invariant by construction; no eps sweep of q1 is
saved in the repo. Column 3: q1 (k-averaged over {1,2,3,5}, c = 0.1) from results/revision/q1/set_q1.csv.gz (family in
{degree_strength, cycle_participation, random}; random = mean over the 100 sets; per-subject k-average then subject mean).
Columns 4-9 (computed here, read-only, from results/revision/regime/task6_spectra_k5_c0.1.npz, which stores the raw eig(W) of the k = 5
degree, cycle and FIRST random set ('random0') per subject at c = 0.1): mean over subjects of max/sum of eig(W), and of the same ratio
after the pipeline floor W_eps = clip(W + eps I, eps). This shows how much a floored analogue of q1 would move with eps.
CAVEAT: k = 5 only, and the random row is a single random set per subject, not the 100-set mean, so it is not comparable to column 3."""
    parts.append(Table("S4_4", tab, note))

    # ---------------- S4e: lambda_min / lambda_max by criterion ----------------
    t2 = pd.read_csv(res / "spectra" / "tables" / "t2_eigenvalue_extremes.csv")
    rows = []
    for scale in SCALES:
        rows.append(("MULTI", f"{scale} regions", 7))
        for _, r in t2[t2.scale == scale].iterrows():
            rows.append([LABEL.get(r.family, r.family), num(r.log10_lambda_min_mean), f"{100 * r.frac_rows_lambda_min_eq_eps:.0f}",
                         f"{r.log10_lambda_max_mean:+.2f} ({r.log10_lambda_max_sd:.2f})", num(r.frac_at_floor_mean),
                         num(r.frac_W_below_eps_mean), num(r.top1_share_mean)])
    tab = tabular("lcccccc",
                  ["Set & $\\log_{10}\\lambda_{\\min}$ & \\% rows $\\lambda_{\\min}=\\epsilon$ & $\\log_{10}\\lambda_{\\max}$ (SD) & Frac.\\ numerically 0 & Frac.\\ $\\lambda(W)<\\epsilon$ & $\\lambda_{\\max}/\\mathrm{tr}\\,W_\\epsilon$ \\\\"],
                  rows)
    note = """S4e. Smallest and largest Gramian eigenvalues by criterion (c = 0.1, eps = 1e-5, T = inf, k in {1,2,3,5}).
Source: results/revision/spectra/tables/t2_eigenvalue_extremes.csv (= 'T2 eigenvalue extremes' in spectra/tables/all_tables.md),
built from results/revision/spectra/set_metrics.csv; means over all subject x k rows (280 per ranking criterion; random/matched rows pool
all sets). lambda_min, lambda_max are eigenvalues of the floored W_eps. lambda_min = eps in 100% of rows, so the condition number
equals lambda_max/eps. Last column is lambda_max/tr on the FLOORED Gramian (top1_share), which differs from the lead metric q1 (raw W)
in the 3rd decimal at most at eps = 1e-5. Random / matched rows: means over all 100 sets per subject (matched sets as generated
in the spectra run)."""
    parts.append(Table("S4_5", tab, note, long=True))
    return parts
