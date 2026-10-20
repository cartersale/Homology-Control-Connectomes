from common import *

REG = RES / "regime"
Q1 = RES / "q1"
parts = []


def eps_tex(e):
    return f"$10^{{{int(round(np.log10(e)))}}}$"


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
note = """S5a. Corrected finite-horizon table (replaces published Table S2). Cycle - degree, mean paired difference (subjects positive of 70),
k-averaged over {1,2,3,5}. Source: results/revision/regime/task4_finite_horizon_summary.csv (all rows; metric in
{effective_rank, participation_ratio}). Identical values in report_tables.md sections T4-effective_rank / T4-participation_ratio.
Full statistics (bootstrap CI, Wilcoxon p, Holm p, rank-biserial) are in the CSV; at eps = 1e-5 see report_tables.md T4-full-eps1e-5.
Check: at c = 1, T >= 2.5 reproduces the infinite-horizon result to 2 decimals (ER -0.02/-0.05/-0.24 at eps = 1e-5;
-0.02/-0.06/-0.25 at eps = 1e-8). The published Table S2 values match none of these rows."""
parts.append(block("S5a: finite horizon", note, tab))

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
note = """S5b(i). Full c sweep for effective rank and participation ratio (eps = 1e-5, T = inf, k-averaged over {1,2,3,5}).
Entries: mean paired difference (subjects positive, of 70).
Sources: results/revision/regime/task1_csweep_cycle_minus_degree.csv (13 c values, 0.01-2); results/revision/regime/task1_csweep_vs_random.csv
(method == degree_strength; metric er_minus_random / pr_minus_random = degree set minus mean of the subject's 100 random sets).
Bootstrap CIs, Wilcoxon p and Holm p are in the CSVs and in report_tables.md (T1-full-*, T1-random-*).
q1 is NOT in these files; it is in S5b(ii) from a different c grid."""
parts.append(block("S5b(i): c sweep, cycle - degree ER / PR", note, tab1))
parts.append(block("S5b(i'): c sweep, degree - random ER / PR", "Same source/filters as S5b(i); second half of the table.", tab2))

# ---------------- S5b2: c sweep q1 (decay curves) ----------------
dc = pd.read_csv(Q1 / "decay_curves.csv.gz")
rows = []
q1_checks = {}
for c in sorted(dc.c.unique()):
    r = [f"{c:.3g}"]
    for curve in ("cd_q1", "dr_q1", "cd_er"):
        for scale in SCALES:
            v = dc[(dc.scale == scale) & np.isclose(dc.c, c)][curve]
            r.append(f"{v.mean():+.2f} ({int((v > 0).sum())})")
            q1_checks[(curve, scale, round(c, 4))] = (v.mean(), int((v > 0).sum()), int((v < 0).sum()))
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
parts.append(block("S5b(ii): c sweep, q1", note, tab))

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
note = """S5c(i). c at which the cycle - degree difference changes sign (eps = 1e-5, k-averaged), from results/revision/regime/task1_signflip.csv:
'Mean curve' / 'Median curve' = linear interpolation of the zero crossing of the mean / median paired difference on the 13-point c grid;
'n+ < 35' = c at which fewer than half the subjects are positive (interpolated); last / first grid c with a Holm-significant positive /
negative difference. '--' (PR, 68 and 114): no sign change within c <= 2. Note the interpolated ER flip points are 0.88 / 0.69 / 0.29,
not 'c ~ 0.6-0.7 (68, 114)'."""
parts.append(block("S5c(i): sign-flip points", note, tab))
rows = []
for scale in SCALES:
    for k in (1, 2, 3, 5):
        x = sk[(sk.scale == scale) & (sk.k == k)].iloc[0]
        def f(v):
            try:
                return num(float(v))
            except ValueError:
                return "none in $[0.01,2]$"
        rows.append([str(scale) if k == 1 else "", str(k), f(x.c_flip_mean), f(x.c_flip_npos35)])
tab = tabular("llcc", ["Scale & $k$ & Mean-curve flip & $n_+<35$ flip \\\\"], rows)
note = """S5c(ii). Per-k ER sign-flip points (cycle - degree), from results/revision/regime/task1_signflip_per_k.csv (eps = 1e-5).
At k = 1 the ER difference is already near zero or negative at c = 0.1 (see S5f / T1-per-k), so its flip lies below the primary c.
'--' = no zero crossing within the saved grid (CSV value 'none (<=0.01 or >2)'; at k = 5 the 68/114 difference stays positive to c = 2)."""
parts.append(block("S5c(ii): per-k ER sign flips", note, tab))
rows = []
for _, x in dfs.iterrows():
    rows.append([{"cd_er": "Cycle $-$ degree ER", "cd_q1": "Cycle $-$ degree $q_1$"}[x.curve],
                 {"flip_c": "$c$", "flip_decay": "decay ratio $d$"}[x.axis],
                 num(x.median_68), num(x.median_114), num(x.median_219),
                 f"{int(x.n_flip_68)} / {int(x.n_flip_114)} / {int(x.n_flip_219)}", num(x.max_over_min), ptex(x.kruskal_p)])
tab = tabular("llcccccc", ["Curve & Axis & Median 68 & Median 114 & Median 219 & Subjects that flip & max/min & Kruskal $p$ \\\\"], rows)
note = """S5c(iii)/S5d(ii). Per-subject first zero crossing on the 25-point grid, on the c axis and on the decay-ratio axis d = c/(c + lambda1 - lambda2).
Source: results/revision/q1/decay_flip_summary.csv (medians over subjects that flip; 'Subjects that flip' of 70 per scale; max/min of the
three scale medians; Kruskal-Wallis across scales). Only 8 / 4 / 20 subjects ever show a sign change in cycle - degree q1."""
parts.append(block("S5c(iii): per-subject flip points (decay grid)", note, tab))

# ---------------- S5d: decay collapse ----------------
col = pd.read_csv(Q1 / "decay_collapse.csv")
rows = []
for _, x in col.iterrows():
    rows.append([{"cd_er": "Cycle $-$ degree ER", "cd_q1": "Cycle $-$ degree $q_1$", "dr_q1": "Degree $-$ random $q_1$"}[x.curve],
                 {"c": "$c$ (full)", "decay_ratio": "$d$ (full)", "c (matched window)": "$c$ (matched window)", "decay_ratio (matched window)": "$d$ (matched window)"}[x.axis], f"[{x.grid_lo:.3f}, {x.grid_hi:.3f}]",
                 f"{x.between_scale_var:.4f}", f"{x.total_var:.4f}", f"{x.D:.3f}"])
tab = tabular("llcccc", ["Curve & Axis & Grid range & Between-scale var. & Total var. & $D$ \\\\"], rows)
note = """S5d. Decay-ratio collapse statistics. Source: results/revision/q1/decay_collapse.csv (all 12 rows).
D = mean over the common grid of the between-scale variance of the three scale-mean curves divided by the total variance of all subjects'
curve values (smaller = tighter collapse). Rows come in two kinds: the full shared range on each axis (c in [0.0032, 3.16] vs d in [0.090, 0.786]) and
matched windows (c in [0.046, 1.72], the c window that maps to the shared d range at the median gap; the d row is the same as the full d row).
The plan's 'D: cycle-degree q1 0.022 -> 0.015; ER 0.037 -> 0.026; degree-random q1 0.030 -> 0.021' = matched-window rows (confirmed).
On the full c range D is 0.042 / 0.045 / 0.018 (degree-random q1 is then LOWER on c than on d: 0.018 vs 0.021)."""
parts.append(block("S5d: decay-ratio collapse", note, tab))

# ---------------- S5e: gap correlations ----------------
gc = pd.read_csv(Q1 / "decay_gap_correlations.csv")
pc = pd.read_csv(Q1 / "decay_gap_partial_correlations.csv")
cvals = sorted(gc.c.unique())
pick = [c for c in cvals if any(np.isclose(c, t, rtol=1e-6) for t in (0.01, 10 ** -1.5, 0.1, 10 ** -0.5, 1.0, 10 ** 0.5))]
rows = []
for curve in ("dr_q1", "cd_q1", "cd_er"):
    rows.append(("MULTI", {"dr_q1": "Degree $-$ random $q_1$", "cd_q1": "Cycle $-$ degree $q_1$", "cd_er": "Cycle $-$ degree ER"}[curve], 10))
    for c in pick:
        r = [f"{c:.3g}"]
        for scale in SCALES:
            x = gc[(gc.scale == scale) & np.isclose(gc.c, c) & (gc.curve == curve)].iloc[0]
            r.append(f"{x.rho_gap:+.2f} ({ptex(x.p)})")
        for ctrl in sorted(pc.controls.unique()):
            for scale in SCALES:
                x = pc[(pc.scale == scale) & np.isclose(pc.c, c) & (pc.curve == curve) & (pc.controls == ctrl)].iloc[0]
                r.append(f"{x.partial_rho_gap:+.2f}")
        rows.append(r)
ctrls = sorted(pc.controls.unique())
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
results/revision/q1/decay_gap_partial_correlations.csv (25 c values saved; 6 shown: c = 10^-2 ... 10^0.5 in half-decade steps).
n = 70 per scale. Partial-correlation p values are in the CSV (not shown); the largest p among the shown partial rows per curve/c:
""" + "; ".join(f"{k[0]} c={k[1]}: {v:.1e}" for k, v in pmax.items()) + """
Sign convention: dr_q1 positive = degree more concentrated than random; cd_q1 negative = cycle less concentrated than degree, so a
negative rho for cd_q1 means 'larger gap -> larger cycle-degree separation'."""
parts.append(block("S5e: gap-effect correlations", note, tab))

# ---------------- S5f: large k at c = 0.1 ----------------
lk = pd.read_csv(REG / "task5_S3_largek_summary.csv")
rows = []
for scale in SCALES:
    sub = lk[(lk.c == 0.1) & (lk.scale == scale)]
    for k in sorted(sub.k.unique()):
        r = [str(scale) if k == min(sub.k) else "", str(k)]
        for metric in ("effective_rank", "participation_ratio"):
            x = sub[(sub.k == k) & (sub.metric == metric)].iloc[0]
            r += [f"{x.mean_diff:+.2f}", npos(x.n_positive), ptex(x.p_holm)]
        rows.append(r)
    if scale != 219:
        rows.append("MIDRULE")
tab = tabular("llcccccc",
              ["& & \\multicolumn{3}{c}{Cycle $-$ degree ER} & \\multicolumn{3}{c}{Cycle $-$ degree PR} \\\\",
               "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
               "Scale & $k$ & Mean & $n_+$ & $p_{\\mathrm{Holm}}$ & Mean & $n_+$ & $p_{\\mathrm{Holm}}$ \\\\"], rows)
note = """S5f. Larger driver sets at the primary c = 0.1 (replaces published Table S3, which was run at c = 0.01).
Source: results/revision/regime/task5_S3_largek_summary.csv (filter c == 0.1; eps = 1e-5; T = inf). Large-k values are ~5/10/20% of N
(7/10/14 at 68; 6/11/23 at 114; 11/22/44 at 219) as saved. q1 is not computed for large k in any saved file."""
parts.append(block("S5f: large k", note, tab))

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
parts.append(block("S5g: nctpy equivalence", note, tab))

(OUT / "S5_c_horizon.tex").write_text("% Supplement S5 — damping c and horizon T. Generated read-only from saved CSVs.\n\n" + "\n".join(parts))

# ---- checks for SUMMARY
for key in [("cd_q1", s, c) for s in SCALES for c in (0.1, 1.0, 3.1623)] + [("dr_q1", s, c) for s in SCALES for c in (0.1, 1.0)] + [("cd_er", s, c) for s in SCALES for c in (0.01, 0.1, 1.0)]:
    print(key, q1_checks.get(key))
print(pick)
x = gc[np.isclose(gc.c, 0.1)]
print(x.round(3).to_string())
x = pc[np.isclose(pc.c, 0.1)]
print(x.round(4).to_string())
print(sorted(dc.c.unique()))
