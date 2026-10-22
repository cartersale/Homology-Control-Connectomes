"""Supplement S5: damping c and horizon T (S5_1-S5_11).

Port of scripts/revision/paper/_rescued_2026-10-19/work/s5.py. Numbers quoted in the
notes (check values, flip points, D values, large-k sizes) are computed from the saved
CSVs rather than typed in.
"""

from __future__ import annotations

from pathlib import Path

from paper_common import SCALES, Table, np, npos, num, pd, ptex, tabular


def eps_tex(e):
    return f"$10^{{{int(round(np.log10(e)))}}}$"


def slash(values, d=2):
    return "/".join(f"{v:.{d}f}" for v in values)


def build(res: Path) -> list[Table]:
    REG = res / "regime"
    Q1 = res / "q1"
    parts = []

    # ---------------- S5a: finite horizon ----------------
    fh = pd.read_csv(REG / "task4_finite_horizon_summary.csv")
    rows = []
    for c in (0.01, 0.1, 1.0):
        rows.append(("MULTI", f"c = {c:g}", 8))
        for eps in (1e-8, 1e-5):
            for T in sorted(fh["T"].unique(), key=lambda x: (np.isinf(x), x)):
                r = [eps_tex(eps), "$\\infty$" if np.isinf(T) else f"{T:g}"]
                for metric in ("effective_rank", "participation_ratio"):
                    for scale in SCALES:
                        x = fh[(fh.c == c) & np.isclose(fh.eps, eps) & (fh["T"] == T) & (fh.metric == metric) & (fh.scale == scale)].iloc[0]
                        r.append(f"{x.mean_diff:+.2f} ({int(x.n_positive)})")
                rows.append(r)
    tab = tabular("llcccccc",
                  ["& & \\multicolumn{3}{c}{Effective rank} & \\multicolumn{3}{c}{Participation ratio} \\\\",
                   "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
                   "$\\epsilon$ & $T$ & 68 & 114 & 219 & 68 & 114 & 219 \\\\"], rows)

    def er_inf(eps):
        x = fh[(fh.c == 1.0) & np.isclose(fh.eps, eps) & np.isinf(fh["T"]) & (fh.metric == "effective_rank")]
        return x.set_index("scale").mean_diff.loc[list(SCALES)]

    # How closely finite T >= 2.5 reproduces T = inf at c = 1 (ER/PR; quoted in the note).
    x = fh[(fh.c == 1.0) & fh.metric.isin(["effective_rank", "participation_ratio"])]
    piv = x.pivot_table(index=["metric", "eps", "scale"], columns="T", values="mean_diff")
    t_dev = piv[[T for T in piv.columns if 2.5 <= T < np.inf]].sub(piv[np.inf], axis=0).abs().to_numpy().max()
    assert t_dev < 0.005, t_dev
    note = f"""S5a. Corrected finite-horizon table (replaces published Table S2). Cycle - degree, mean paired difference (subjects positive of 70),
k-averaged over {{1,2,3,5}}. Source: results/revision/regime/task4_finite_horizon_summary.csv (all rows; metric in
{{effective_rank, participation_ratio}}). Identical values in report_tables.md sections T4-effective_rank / T4-participation_ratio.
Full statistics (bootstrap CI, Wilcoxon p, Holm p, rank-biserial) are in the CSV; at eps = 1e-5 see report_tables.md T4-full-eps1e-5.
Check: at c = 1, T >= 2.5 is within {t_dev:.3f} of the infinite-horizon result for ER and PR (infinite-horizon ER {slash(er_inf(1e-5))} at eps = 1e-5;
{slash(er_inf(1e-8))} at eps = 1e-8). The published Table S2 values match none of these rows."""
    parts.append(Table("S5_1", tab, note))

    # ---------------- S5b1: c sweep ER / PR ----------------
    cd = pd.read_csv(REG / "task1_csweep_cycle_minus_degree.csv")
    vr = pd.read_csv(REG / "task1_csweep_vs_random.csv")
    vr = vr[vr.method == "degree_strength"]
    rows = []
    for c in sorted(cd.c.unique()):
        r = [f"{c:g}"]
        for metric in ("effective_rank", "participation_ratio"):
            for scale in SCALES:
                x = cd[(cd.c == c) & (cd.metric == metric) & (cd.scale == scale)].iloc[0]
                r.append(f"{x.mean_diff:+.2f} ({int(x.n_positive)})")
        rows.append(r)
    tab1 = tabular("lcccccc",
                   ["& \\multicolumn{3}{c}{Cycle $-$ degree ER} & \\multicolumn{3}{c}{Cycle $-$ degree PR} \\\\",
                    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
                    "$c$ & 68 & 114 & 219 & 68 & 114 & 219 \\\\"], rows)
    rows = []
    for c in sorted(vr.c.unique()):
        r = [f"{c:g}"]
        for metric in ("er_minus_random", "pr_minus_random"):
            for scale in SCALES:
                x = vr[(vr.c == c) & (vr.metric == metric) & (vr.scale == scale)].iloc[0]
                r.append(f"{x.mean_diff:+.2f} ({int(x.n_positive)})")
        rows.append(r)
    tab2 = tabular("lcccccc",
                   ["& \\multicolumn{3}{c}{Degree $-$ random ER} & \\multicolumn{3}{c}{Degree $-$ random PR} \\\\",
                    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
                    "$c$ & 68 & 114 & 219 & 68 & 114 & 219 \\\\"], rows)
    cs = sorted(cd.c.unique())
    note = f"""S5b(i). Full c sweep for effective rank and participation ratio (eps = 1e-5, T = inf, k-averaged over {{1,2,3,5}}).
Entries: mean paired difference (subjects positive, of 70).
Sources: results/revision/regime/task1_csweep_cycle_minus_degree.csv ({len(cs)} c values, {cs[0]:g}-{cs[-1]:g}); results/revision/regime/task1_csweep_vs_random.csv
(method == degree_strength; metric er_minus_random / pr_minus_random = degree set minus mean of the subject's 100 random sets).
Bootstrap CIs, Wilcoxon p and Holm p are in the CSVs and in report_tables.md (T1-full-*, T1-random-*).
q1 is NOT in these files; it is in S5b(ii) from a different c grid."""
    parts.append(Table("S5_2", tab1, note))
    parts.append(Table("S5_3", tab2, "Same source/filters as S5b(i); second half of the table."))

    # ---------------- S5b2: c sweep q1 (decay curves) ----------------
    dc = pd.read_csv(Q1 / "decay_curves.csv.gz")
    rows = []
    for c in sorted(dc.c.unique()):
        r = [f"{c:.3g}"]
        for curve in ("cd_q1", "dr_q1", "cd_er"):
            for scale in SCALES:
                v = dc[(dc.scale == scale) & np.isclose(dc.c, c)][curve]
                r.append(f"{v.mean():+.2f} ({int((v > 0).sum())})")
        rows.append(r)
    tab = tabular("lccccccccc",
                  ["& \\multicolumn{3}{c}{Cycle $-$ degree $q_1$} & \\multicolumn{3}{c}{Degree $-$ random $q_1$} & \\multicolumn{3}{c}{Cycle $-$ degree ER} \\\\",
                   "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
                   "$c$ & 68 & 114 & 219 & 68 & 114 & 219 & 68 & 114 & 219 \\\\"], rows)
    note = """S5b(ii). Full c sweep for q1 (lead metric) on the 25-point log grid c = 10^-2.5 ... 10^0.5 (eps = 1e-5 for ER; q1 uses raw W).
Entries: mean over 70 subjects (subjects with a positive difference). k-averaged over {1,2,3,5}.
Source: results/revision/q1/decay_curves.csv.gz (per-subject rows; columns cd_q1 = q1(cycle) - q1(degree), dr_q1 = q1(degree) - mean q1 of
the 100 primary random sets, cd_er = ER(cycle) - ER(degree)). Means and counts computed here with read-only pandas (no Wilcoxon tests are
saved for this grid). PR is not saved on this grid. The ER column reproduces S5b(i) at the shared c values (0.01, 0.1, 1).
For q1, 'positive' means cycle MORE concentrated than degree (cd_q1) or degree MORE concentrated than random (dr_q1)."""
    parts.append(Table("S5_4", tab, note))

    # ---------------- S5c: sign-flip points ----------------
    sf = pd.read_csv(REG / "task1_signflip.csv")
    sk = pd.read_csv(REG / "task1_signflip_per_k.csv")
    dfs = pd.read_csv(Q1 / "decay_flip_summary.csv")
    rows = []
    for metric in ("effective_rank", "participation_ratio"):
        for scale in SCALES:
            x = sf[(sf.metric == metric) & (sf.scale == scale)].iloc[0]
            rows.append([("ER" if metric == "effective_rank" else "PR") if scale == 68 else "", str(scale),
                         num(x.c_flip_mean), num(x.c_flip_median), num(x.c_npos_below_half),
                         f"{x.c_last_holm_sig_positive:g}", "--" if pd.isna(x.c_first_holm_sig_negative) else f"{x.c_first_holm_sig_negative:g}"])
    tab = tabular("llccccc",
                  ["Metric & Scale & Mean curve & Median curve & $n_+<35$ & Last Holm-sig.\\ $>0$ & First Holm-sig.\\ $<0$ \\\\"], rows)
    er_flip = sf[sf.metric == "effective_rank"].set_index("scale").c_flip_mean.loc[list(SCALES)]
    pr_none = sf[(sf.metric == "participation_ratio") & sf.c_flip_mean.isna()].scale.tolist()
    note = f"""S5c(i). c at which the cycle - degree difference changes sign (eps = 1e-5, k-averaged), from results/revision/regime/task1_signflip.csv:
'Mean curve' / 'Median curve' = linear interpolation of the zero crossing of the mean / median paired difference on the {len(cs)}-point c grid;
'n+ < 35' = c at which fewer than half the subjects are positive (interpolated); last / first grid c with a Holm-significant positive /
negative difference. '--' (PR, {' and '.join(map(str, pr_none))}): no sign change within c <= {cs[-1]:g}. Note the interpolated ER flip points are {slash(er_flip, 2).replace('/', ' / ')},
not 'c ~ 0.6-0.7 (68, 114)'."""
    parts.append(Table("S5_5", tab, note))

    rows = []
    for scale in SCALES:
        for k in (1, 2, 3, 5):
            x = sk[(sk.scale == scale) & (sk.k == k)].iloc[0]
            rows.append([str(scale) if k == 1 else "", str(k), num(x.c_flip_mean), num(x.c_flip_npos35)])
    tab = tabular("llcc", ["Scale & $k$ & Mean-curve flip & $n_+<35$ flip \\\\"], rows)
    note = """S5c(ii). Per-k ER sign-flip points (cycle - degree), from results/revision/regime/task1_signflip_per_k.csv (eps = 1e-5).
At k = 1 the ER difference is already near zero or negative at c = 0.1 (see S5f / T1-per-k), so its flip lies below the primary c.
'--' = no zero crossing within the saved grid (empty CSV value; at k = 5 the 68/114 difference stays positive to c = 2)."""
    parts.append(Table("S5_6", tab, note))

    rows = []
    for _, x in dfs.iterrows():
        rows.append([{"cd_er": "Cycle $-$ degree ER", "cd_q1": "Cycle $-$ degree $q_1$"}[x.curve],
                     {"flip_c": "$c$", "flip_decay": "decay ratio $d$"}[x.axis],
                     num(x.median_68), num(x.median_114), num(x.median_219),
                     f"{int(x.n_flip_68)} / {int(x.n_flip_114)} / {int(x.n_flip_219)}", num(x.max_over_min), ptex(x.kruskal_p)])
    tab = tabular("llcccccc", ["Curve & Axis & Median 68 & Median 114 & Median 219 & Subjects that flip & max/min & Kruskal $p$ \\\\"], rows)
    q1f = dfs[dfs.curve == "cd_q1"].iloc[0]
    note = f"""S5c(iii)/S5d(ii). Per-subject first zero crossing on the 25-point grid, on the c axis and on the decay-ratio axis d = c/(c + lambda1 - lambda2).
Source: results/revision/q1/decay_flip_summary.csv (medians over subjects that flip; 'Subjects that flip' of 70 per scale; max/min of the
three scale medians; Kruskal-Wallis across scales). Only {int(q1f.n_flip_68)} / {int(q1f.n_flip_114)} / {int(q1f.n_flip_219)} subjects ever show a sign change in cycle - degree q1."""
    parts.append(Table("S5_7", tab, note))

    # ---------------- S5d: decay collapse ----------------
    col = pd.read_csv(Q1 / "decay_collapse.csv")
    rows = []
    for _, x in col.iterrows():
        rows.append([{"cd_er": "Cycle $-$ degree ER", "cd_q1": "Cycle $-$ degree $q_1$", "dr_q1": "Degree $-$ random $q_1$"}[x.curve],
                     {"c": "$c$ (full)", "decay_ratio": "$d$ (full)", "c (matched window)": "$c$ (matched window)",
                      "decay_ratio (matched window)": "$d$ (matched window)"}[x.axis], f"[{x.grid_lo:.3f}, {x.grid_hi:.3f}]",
                     f"{x.between_scale_var:.4f}", f"{x.total_var:.4f}", f"{x.D:.3f}"])
    tab = tabular("llcccc", ["Curve & Axis & Grid range & Between-scale var. & Total var. & $D$ \\\\"], rows)
    Dv = col.set_index(["axis", "curve"]).D
    rng = col.set_index("axis")[["grid_lo", "grid_hi"]].groupby(level=0).first()
    mw = {cv: f"{Dv[('c (matched window)', cv)]:.3f} -> {Dv[('decay_ratio (matched window)', cv)]:.3f}" for cv in ("cd_q1", "cd_er", "dr_q1")}
    note = f"""S5d. Decay-ratio collapse statistics. Source: results/revision/q1/decay_collapse.csv (all {len(col)} rows).
D = mean over the common grid of the between-scale variance of the three scale-mean curves divided by the total variance of all subjects'
curve values (smaller = tighter collapse). Rows come in two kinds: the full shared range on each axis (c in [{rng.loc['c'].grid_lo:.4f}, {rng.loc['c'].grid_hi:.2f}] vs d in [{rng.loc['decay_ratio'].grid_lo:.3f}, {rng.loc['decay_ratio'].grid_hi:.3f}]) and
matched windows (c in [{rng.loc['c (matched window)'].grid_lo:.3f}, {rng.loc['c (matched window)'].grid_hi:.2f}], the c window that maps to the shared d range at the median gap; the d row is the same as the full d row).
The plan's 'D: cycle-degree q1 0.022 -> 0.015; ER 0.037 -> 0.026; degree-random q1 0.030 -> 0.021' = matched-window rows
(saved: cycle-degree q1 {mw['cd_q1']}; ER {mw['cd_er']}; degree-random q1 {mw['dr_q1']}).
On the full c range D is {Dv[('c', 'cd_q1')]:.3f} / {Dv[('c', 'cd_er')]:.3f} / {Dv[('c', 'dr_q1')]:.3f} (degree-random q1 on c vs d: {Dv[('c', 'dr_q1')]:.3f} vs {Dv[('decay_ratio', 'dr_q1')]:.3f})."""
    parts.append(Table("S5_8", tab, note))

    # ---------------- S5e: gap correlations ----------------
    gc = pd.read_csv(Q1 / "decay_gap_correlations.csv")
    pc = pd.read_csv(Q1 / "decay_gap_partial_correlations.csv")
    cvals = sorted(gc.c.unique())
    pick = [c for c in cvals if any(np.isclose(c, t, rtol=1e-6) for t in (0.01, 10 ** -1.5, 0.1, 10 ** -0.5, 1.0, 10 ** 0.5))]
    ctrls = sorted(pc.controls.unique())
    rows = []
    for curve in ("dr_q1", "cd_q1", "cd_er"):
        rows.append(("MULTI", {"dr_q1": "Degree $-$ random $q_1$", "cd_q1": "Cycle $-$ degree $q_1$", "cd_er": "Cycle $-$ degree ER"}[curve], 10))
        for c in pick:
            r = [f"{c:.3g}"]
            for scale in SCALES:
                x = gc[(gc.scale == scale) & np.isclose(gc.c, c) & (gc.curve == curve)].iloc[0]
                r.append(f"{x.rho_gap:+.2f} ({ptex(x.p)})")
            for ctrl in ctrls:
                for scale in SCALES:
                    x = pc[(pc.scale == scale) & np.isclose(pc.c, c) & (pc.curve == curve) & (pc.controls == ctrl)].iloc[0]
                    r.append(f"{x.partial_rho_gap:+.2f}")
            rows.append(r)
    tab = tabular("l" + "c" * 9,
                  ["& \\multicolumn{3}{c}{Spearman $\\rho$(gap, effect) ($p$)} & \\multicolumn{3}{c}{Partial $\\rho$ | " + ctrls[0].replace("_", "\\_") +
                   "} & \\multicolumn{3}{c}{Partial $\\rho$ | " + ctrls[1].replace("_", "\\_").replace("+", " + ") + "} \\\\",
                   "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
                   "$c$ & 68 & 114 & 219 & 68 & 114 & 219 & 68 & 114 & 219 \\\\"], rows)
    pmax = {}
    for curve in ("dr_q1", "cd_q1", "cd_er"):
        for c in pick:
            pmax[(curve, round(c, 4))] = pc[np.isclose(pc.c, c) & (pc.curve == curve)].p.max()
    note = f"""S5e. Across-subject Spearman correlation between the spectral gap (lambda1 - lambda2 of A0) and the effect size, at fixed c,
and partial Spearman rho controlling for {ctrls[0]} and for {ctrls[1]}. Sources: results/revision/q1/decay_gap_correlations.csv,
results/revision/q1/decay_gap_partial_correlations.csv ({len(cvals)} c values saved; {len(pick)} shown: c = 10^-2 ... 10^0.5 in half-decade steps).
n = 70 per scale. Partial-correlation p values are in the CSV (not shown); the largest p among the shown partial rows per curve/c:
""" + "; ".join(f"{k[0]} c={k[1]}: {v:.1e}" for k, v in pmax.items()) + """
Sign convention: dr_q1 positive = degree more concentrated than random; cd_q1 negative = cycle less concentrated than degree, so a
negative rho for cd_q1 means 'larger gap -> larger cycle-degree separation'."""
    parts.append(Table("S5_9", tab, note))

    # ---------------- S5f: large k at c = 0.1 ----------------
    lk = pd.read_csv(REG / "task5_S3_largek_summary.csv")
    rows = []
    large = {}
    for scale in SCALES:
        sub = lk[(lk.c == 0.1) & (lk.scale == scale)]
        ks = sorted(sub.k.unique())
        large[scale] = [k for k in ks if k > 5]
        for k in ks:
            r = [str(scale) if k == min(sub.k) else "", str(k)]
            for metric in ("effective_rank", "participation_ratio"):
                x = sub[(sub.k == k) & (sub.metric == metric)].iloc[0]
                r += [f"{x.mean_diff:+.2f}", npos(x.n_positive), ptex(x.p_holm)]
            rows.append(r)
        if scale != SCALES[-1]:
            rows.append("MIDRULE")
    tab = tabular("llcccccc",
                  ["& & \\multicolumn{3}{c}{Cycle $-$ degree ER} & \\multicolumn{3}{c}{Cycle $-$ degree PR} \\\\",
                   "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
                   "Scale & $k$ & Mean & $n_+$ & $p_{\\mathrm{Holm}}$ & Mean & $n_+$ & $p_{\\mathrm{Holm}}$ \\\\"], rows)
    sizes = "; ".join(f"{'/'.join(map(str, large[s]))} at {s}" for s in SCALES)
    note = f"""S5f. Larger driver sets at the primary c = 0.1 (replaces published Table S3, which was run at c = 0.01).
Source: results/revision/regime/task5_S3_largek_summary.csv (filter c == 0.1; eps = 1e-5; T = inf). Large-k values are ~5/10/20% of N
plus a common k = 10 ({sizes}) as saved. q1 is not computed for large k in any saved file."""
    parts.append(Table("S5_10", tab, note))

    # ---------------- S5g: nctpy equivalence ----------------
    nc = pd.read_csv(REG / "nctpy_cohort_summary.csv")
    det = pd.read_csv(REG / "nctpy_equivalence_detail.csv")
    rows = []
    for metric in ("d_er", "d_pr"):
        for scale in SCALES:
            r = ["ER" if metric == "d_er" and scale == 68 else ("PR" if scale == 68 else ""), str(scale)]
            for norm in ("ours_c1", "nctpy_c1"):
                x = nc[(nc.normalization == norm) & (nc.metric == metric) & (nc.scale == scale)].iloc[0]
                r += [f"{x.mean_diff:+.3f}", npos(x.n_positive), ptex(x.p_holm)]
            rows.append(r)
    tab = tabular("llcccccc",
                  ["& & \\multicolumn{3}{c}{$A_0-(\\lambda_{\\max}+1)I$ (ours, $c=1$)} & \\multicolumn{3}{c}{nctpy $A_0/(\\lambda_{\\max}+1)-I$} \\\\",
                   "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
                   "Metric & Scale & Mean & $n_+$ & $p_{\\mathrm{Holm}}$ & Mean & $n_+$ & $p_{\\mathrm{Holm}}$ \\\\"], rows)
    note = f"""S5g. nctpy equivalence. Cohort: cycle - degree ER / PR under our c = 1 model and under the nctpy continuous-time normalization
(eps = 1e-5 floor in both). Source: results/revision/regime/nctpy_cohort_summary.csv.
Spot-check detail: results/revision/regime/nctpy_equivalence_detail.csv ({len(det)} rows = subject x method x k checks):
max |W_nctpy - (lambda_max+1) W_ours| = {det.max_abs_W_diff.max():.1e} (max relative {det.rel_W_diff.max():.1e});
max |Delta ER| with the eps floor at 1e-5 in both = {det['d_er_eps1e-5'].abs().max():.3f}; with eps rescaled by (lambda_max+1) =
{det.d_er_eps_rescaled.abs().max():.1e}; without floor = {det.d_er_nofloor.abs().max():.1e}. So the two models are identical up to time
units; the small cohort-level differences come only from applying the same absolute eps to Gramians that differ by a scalar factor.
q1 is scale-invariant and therefore identical under both normalizations (by construction; not separately saved)."""
    parts.append(Table("S5_11", tab, note))
    return parts
