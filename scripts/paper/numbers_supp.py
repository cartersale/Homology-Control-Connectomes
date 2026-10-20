"""Numbers quoted in the prose of the Supporting Information (supp_mat.tex).

``collect(res)`` returns one ``Number`` per result quoted in paragraph text or in a
caption written in supp_mat.tex (table bodies under ``supp_tables/`` are built by the
table modules). ``res`` is ``<run>/revision``; legacy stages are read from ``res.parent``.

Conventions for ``Number.text``:

- The number only: no ``$...$`` and no ``\\%``. Percent signs stay in the prose
  (``\\SRayleighShareROneSixtyEight\\%``).
- Every macro is safe in math mode. Negative values print a plain ``-``, so a macro
  holding a negative number (or an exponent such as ``2.4\\times10^{-3}``) must be
  placed inside ``$...$``: ``$\\rho = \\SRankDivGOneRhoSixtyEight$``. An explicit ``+``
  appears only where the draft prints one.
- Ranges ("86--87\\%", "0.64 to 0.71") get one macro per endpoint (``...Lo`` / ``...Hi``,
  or the endpoint the prose names first / second), computed as the min / max of the
  quantity described.
- Bounds ("within 1\\%", "$p < 10^{-3}$", "at most 25\\%") print the bound; ``value`` is
  the actual quantity and the code asserts it respects the bound where that is not
  automatic from rounding.
- When the prose quotes a supplement-table cell, the value is computed from the same
  file, filters and rounding as the table builder.
- ``where`` is ``supp_mat.tex:<line>`` of the 2026-10-19 revision draft. ``draft`` is
  set only where the saved result differs from the draft text.

Macro names: letters only, prefix ``S``; scales are spelled ``SixtyEight``,
``OneFourteen``, ``TwoNineteen``.
"""

from __future__ import annotations

import math
from pathlib import Path

from paper_common import Number, np, pd

SC = (68, 114, 219)
SCN = {68: "SixtyEight", 114: "OneFourteen", 219: "TwoNineteen"}
NULLS = ("strength", "geometric", "weights")


def _f(x, d=2):
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def _pct(x):
    """Fraction -> integer percent text."""
    return f"{100 * x:.0f}"


def _sig1(x: float) -> str:
    """One significant figure in LaTeX exponent form, e.g. 4.1e-15 -> '4\\times10^{-15}'."""
    e = math.floor(math.log10(abs(x)))
    m = round(x / 10**e)
    if m == 10:
        m, e = 1, e + 1
    return f"{m}\\times10^{{{e}}}"


def collect(res: Path) -> list[Number]:
    out = (_eps_and_damping(res) + _spin_and_targets(res) + _setlevel(res) + _nulls_and_stability(res)
           + _lesions(res))
    return sorted(out, key=lambda n: int(n.where[0].rsplit(":", 1)[1]))


# ---------------------------------------------------------------- S7 set-level (L191, L193)
def _setlevel(res: Path) -> list[Number]:
    M = res / "modal"
    out: list[Number] = []

    def add(name, value, text, source, line, draft=None):
        out.append(Number(name, float(value), text, source, [f"supp_mat.tex:{line}"], draft))

    # L191: upper driver-Gram bound gap, median q1(G_S) - q1 at c = 0.1 (S7_3 'upper' column, k = 2, 3, 5)
    g = pd.read_csv(M / "driver_gram_proxy.csv")
    gap = g[g.c == 0.1].median_q1G_minus_q1raw
    src = ("revision/modal/driver_gram_proxy.csv [c == 0.1; k 2,3,5; 3 scales]: {} of median_q1G_minus_q1raw"
           " (S7_3 'upper'), 2 dp")
    add("SDriverGramUpperGapLo", gap.min(), _f(gap.min()), src.format("min"), 191)
    add("SDriverGramUpperGapHi", gap.max(), _f(gap.max()), src.format("max"), 191)

    # L193: regions shared by top-k modal (Gu) sets and cycle-participation sets, c = 0.1, all scales and k
    s = pd.read_csv(M / "set_level_modal.csv.gz", usecols=["subject_id", "scale", "c", "k", "family", "sample", "nodes"])
    s = s[(s.c == 0.1) & (s["sample"] == -1) & s.family.isin(["top_modal_gu", "cycle_participation"])]
    w = s.pivot_table(index=["subject_id", "scale", "k"], columns="family", values="nodes", aggfunc="first").dropna()
    shared = w.apply(lambda r: len(set(r.top_modal_gu.split()) & set(r.cycle_participation.split())), axis=1)
    n_subj = shared[shared > 0].index.get_level_values("subject_id").nunique()
    n_reg = int(shared.max())
    assert (n_subj, n_reg) == (1, 1), (n_subj, n_reg)
    add("SModalCycleSharedSubjects", n_subj, "one",
        "revision/modal/set_level_modal.csv.gz [c == 0.1, sample == -1, top_modal_gu vs cycle_participation, all scales x"
        " k 2,3,5]: subjects with any shared region (the single case is 114 regions, k = 5); printed as a word", 193)

    # L193: cycle - top-k modal (Gu) effective rank at k = 3 (S7_5 bottom block)
    cf = pd.read_csv(M / "cycle_vs_counterfactual.csv")
    cf = cf[(cf.c == 0.1) & (cf.k == 3) & (cf.contrast == "cycle - top_modal_gu") & (cf.target == "effective_rank")]
    for sc in SC:
        r = cf[cf.scale == sc].iloc[0]
        src = (f"revision/modal/cycle_vs_counterfactual.csv [c == 0.1, k == 3, contrast == 'cycle - top_modal_gu',"
               f" target == effective_rank, scale == {sc}]: {{}} (S7_5)")
        add(f"SModalCycleERMedianDiff{SCN[sc]}", r["median"], _f(r["median"]), src.format("median, 2 dp"), 193)
        add(f"SModalCycleERNPos{SCN[sc]}", r.n_pos, str(int(r.n_pos)), src.format("n_pos"), 193)
    return out


# ---------------------------------------------------------------- S11 reciprocal hub lesions (L313, L315)
def _lesions(res: Path) -> list[Number]:
    LE = res / "lesion"
    out: list[Number] = []

    def add(name, value, text, source, line, draft=None):
        out.append(Number(name, float(value), text, source, [f"supp_mat.tex:{line}"], draft))

    rows = pd.read_csv(LE / "lesion_rows.csv.gz")

    # L313: k-averaged effective rank of unconstrained random sets on the intact network, every scale
    rr = rows[(rows.protocol == "recompute") & (rows.strategy == "degree_strength") & (rows.lesion_type == "degree")]
    er = rr.groupby(["scale", "subject", "k"]).ref_intact_er.mean().groupby(["scale", "k"]).mean().groupby("scale").mean()
    txt = {_f(v, 1) for v in er}
    assert len(txt) == 1, er
    add("SLesionRandomSetER", er.mean(), txt.pop(),
        "revision/lesion/lesion_rows.csv.gz [protocol == recompute (random sets drawn on the intact network)]:"
        f" ref_intact_er, mean over subjects and k = 1,2,3,5 per scale ({', '.join(f'{v:.2f}' for v in er)}), 1 dp", 313)

    # L315: crossover statistics, static protocol, pooled (S11_2)
    cs = pd.read_csv(LE / "crossover_stats.csv")
    pooled = cs[(cs.protocol == "static") & (cs.scale == "pooled")].set_index(["metric", "lesion_type"])
    r = pooled.loc[("dER", "degree")]
    add("SLesionDegreeDiffER", r["mean"], _f(r["mean"]),
        "revision/lesion/crossover_stats.csv [static, pooled, metric dER, degree lesion]: mean D (S11_2)", 315)
    n_neg = int(r.n) - int(r.n_pos)
    add("SLesionDegreeDiffERNNeg", n_neg, str(n_neg),
        "revision/lesion/crossover_stats.csv [static, pooled, metric dER, degree lesion]: n - n_pos"
        " (subjects with D < 0; S11_2 shows n(D>0) = 4/70)", 315)
    for m, nm in (("dPct_ER", "ER"), ("dPct_PR", "PR")):
        v = 100 * pooled.loc[(m, "cycle")]["mean"]
        add(f"SLesionCyclePctDiff{nm}", v, f"{v:.1f}",
            f"revision/lesion/crossover_stats.csv [static, pooled, metric {m}, cycle lesion]: 100 x mean D,"
            " percentile points (S11_2)", 315)

    # L315: intact effective rank of the strategies' sets (S11_5 'Intact ER', pooled over scales)
    d = pd.read_csv(LE / "per_subject_degradation.csv")

    def intact(protocol, strategy):
        p = d[(d.protocol == protocol) & (d.lesion_type == "degree")]
        p = p.pivot_table(index=["scale", "h", "subject"], columns="strategy", values="intact_effective_rank")
        return float(p[strategy].mean())

    v = intact("static", "degree_strength")
    add("SLesionNextRankedDegreeER", v, f"{v:.2f}",
        "revision/lesion/per_subject_degradation.csv [static, degree lesion]: mean intact_effective_rank of the"
        " degree-strength (next-ranked surviving) sets, pooled over scales and h (S11_5 'Intact ER' deg)", 315)
    v = intact("recompute", "degree_strength")
    add("SLesionTopDegreeER", v, f"{v:.2f}",
        "revision/lesion/per_subject_degradation.csv [recompute, degree lesion]: mean intact_effective_rank of the"
        " top-ranked degree-strength set on the intact network, pooled (S11_5 recompute 'Intact ER' deg)", 315)
    v = intact("recompute", "cycle_participation")
    add("SLesionTopCycleER", v, f"{v:.2f}",
        "revision/lesion/per_subject_degradation.csv [recompute, degree lesion]: mean intact_effective_rank of the"
        " top-ranked cycle-participation set on the intact network, pooled (S11_5 recompute 'Intact ER' cyc)."
        " The draft's 3.36 is the static degree-lesion row (cycle sets avoiding the removed degree hubs)", 315,
        draft="3.36")

    # L315: energy change vs network size, all static rows (as 02_analyze_reciprocal_lesion.py, 'energy_corr')
    st = rows[rows.protocol == "static"]
    rsz = float(np.corrcoef(st.lesioned_log10_average_energy - st.intact_log10_average_energy,
                            np.log10(st.n_lesioned / st.n_intact))[0, 1])
    add("SLesionEnergySizeCorr", rsz, f"{rsz:.3f}",
        "revision/lesion/lesion_rows.csv.gz [protocol == static, all rows]: Pearson r of lesioned - intact"
        " log10 E_avg with log10(n_lesioned / n_intact) (report_snippets 'energy_corr'; S11_6 footer)", 315)
    cells = cs[(cs.metric == "dlog10E") & (cs.protocol == "static") & (cs.scale != "pooled")
               & cs.lesion_type.isin(["degree", "cycle", "random"])]
    v = float(cells["mean"].abs().max())
    add("SLesionEnergyStrategyDiffMax", v, f"{v:.3f}",
        "revision/lesion/crossover_stats.csv [metric dlog10E, static, scale x h cells, 3 lesion types]: max |mean D|"
        " (cycle - degree; 68 regions, degree lesion, h = 10)", 315)
    return out


def _eps_and_damping(res: Path) -> list[Number]:
    out: list[Number] = []
    REG = res / "regime"
    Q1 = res / "q1"

    def add(name, value, text, source, where, draft=None):
        out.append(Number(name, float(value), text, source, where, draft))

    # ---------------- floored fractions (S4_2 columns, eps = 1e-5) ----------------
    a = pd.read_csv(REG / "task6_driver_sets.csv")
    asub = a.groupby(["scale", "eps", "method", "subject_id"])[["fraction_at_floor", "fraction_below_eps"]].mean().reset_index()
    aagg = asub.groupby(["scale", "eps", "method"]).mean(numeric_only=True).reset_index()
    src_frac = ("revision/regime/task6_driver_sets.csv [eps == 1e-5; methods degree_strength, eigenvector_centrality, "
                "cycle_participation, betweenness, random_mean]; per-subject k-mean, subject mean, min/max across the 5 set types; "
                "rounded to 2 decimals as in S4_2, x100")
    # how the draft prints each range: True = 'lo--hi', False = a single number
    printed = {("fraction_at_floor", 68): True, ("fraction_at_floor", 114): True, ("fraction_at_floor", 219): False,
               ("fraction_below_eps", 68): True, ("fraction_below_eps", 114): False, ("fraction_below_eps", 219): True}
    for col, tag, where in (("fraction_at_floor", "NumericallyZero", ["supp_mat.tex:93"]),
                            ("fraction_below_eps", "BelowEps", ["supp_mat.tex:60", "supp_mat.tex:93"])):
        for scale in SC:
            g = aagg[(aagg.scale == scale) & np.isclose(aagg.eps, 1e-5)][col]
            lo, hi = round(100 * float(f"{g.min():.2f}")), round(100 * float(f"{g.max():.2f}"))
            base = f"SEpsFrac{tag}{SCN[scale]}"
            src = f"{src_frac}; column {col}"
            if printed[(col, scale)]:
                add(base + "Lo", g.min() * 100, str(lo), src, where)
                add(base + "Hi", g.max() * 100, str(hi), src, where)
            else:
                assert lo == hi, (col, scale, lo, hi)
                add(base, g.mean() * 100, str(lo), src + " (min and max round to the same value)", where)

    # ---------------- L91: energy vs log10(N/eps); cycle - degree energy difference (S4_1) ----------------
    tidy = pd.read_csv(REG / "horizon_eps_largek_tidy.csv.gz")
    t = tidy[(tidy.c == 0.1) & np.isinf(tidy["T"]) & tidy.k.isin([1, 2, 3, 5])
             & tidy.method.isin(["degree_strength", "cycle_participation"])]
    sub = t.groupby(["scale", "eps", "method", "subject_id"]).log10_average_energy.mean().reset_index()
    agg = sub.groupby(["scale", "eps", "method"]).log10_average_energy.mean().reset_index()
    gapmax = float((np.log10(agg.scale / agg.eps) - agg.log10_average_energy).abs().max())
    add("SEpsEnergyWithinFloorBound", gapmax, f"{gapmax:.1f}",
        "revision/regime/horizon_eps_largek_tidy.csv.gz [c == 0.1, T == inf, k in 1,2,3,5, degree/cycle]; "
        "max over scale x eps x method of |log10(N/eps) - mean log10 E_avg| (S4_1 columns 4-6)", ["supp_mat.tex:91"])
    s1 = pd.read_csv(REG / "task5_S1_epsilon_summary.csv")
    e = s1[(s1.c == 0.1) & (s1.metric == "log10_average_energy")]
    src_e = "revision/regime/task5_S1_epsilon_summary.csv [c == 0.1, metric == log10_average_energy]; mean_diff (S4_1 column 7)"
    mx = float(e.mean_diff.abs().max())
    add("SEpsEnergyCycleDegreeMaxAbs", mx, f"{mx * 1e3:.1f}\\times10^{{-3}}", src_e + "; max |mean_diff| over scales and eps",
        ["supp_mat.tex:91"])
    for eps, nm, sign in ((1e-8, "EpsMinusEight", ""), (1e-2, "EpsMinusTwo", "+")):
        v = float(e[(e.scale == 68) & np.isclose(e.eps, eps)].mean_diff.iloc[0])
        add(f"SEpsEnergyCycleDegreeSixtyEight{nm}", v, f"{sign}{v * 1e3:.1f}\\times10^{{-3}}", src_e + f" [scale 68, eps {eps:g}]",
            ["supp_mat.tex:91"])

    # ---------------- L93: landscape IQR (S4_2 last two columns, eps = 1e-5) ----------------
    L = pd.read_csv(REG / "task6_landscape.csv")
    Lagg = L.groupby(["scale", "eps"])[["iqr_log10E", "iqr_log10E_above_eps"]].mean().reset_index()
    L5 = Lagg[np.isclose(Lagg.eps, 1e-5)].set_index("scale")
    src_l = "revision/regime/task6_landscape.csv [eps == 1e-5]; mean over subjects x k of the per-subject IQR across candidate sets"
    for scale in SC:
        v = float(L5.loc[scale, "iqr_log10E"])
        add(f"SEpsLandscapeIQR{SCN[scale]}", v * 1e3, f"{v * 1e3:.2f}",
            src_l + "; column iqr_log10E, in units of 1e-3 log10 units (S4_2 'All (x1e-3)')", ["supp_mat.tex:93"])
    ab = L5.iqr_log10E_above_eps
    add("SEpsLandscapeIQRAboveEpsLo", ab.min(), f"{ab.min():.2f}", src_l + "; column iqr_log10E_above_eps, min over scales",
        ["supp_mat.tex:93"])
    add("SEpsLandscapeIQRAboveEpsHi", ab.max(), f"{ab.max():.2f}", src_l + "; column iqr_log10E_above_eps, max over scales",
        ["supp_mat.tex:93"])

    # ---------------- L117: c sweep ----------------
    fl = pd.read_csv(Q1 / "decay_flip_summary.csv")
    r = fl[(fl.curve == "cd_q1") & (fl.axis == "flip_c")].iloc[0]
    for scale in SC:
        v = int(r[f"n_flip_{scale}"])
        add(f"SCsweepQOneSignChangeSubjects{SCN[scale]}", v, str(v),
            "revision/q1/decay_flip_summary.csv [curve == cd_q1, axis == flip_c]; n_flip_<scale> (S5_7)", ["supp_mat.tex:117"])
    sf = pd.read_csv(REG / "task1_signflip.csv")
    for scale in SC:
        v = float(sf[(sf.metric == "effective_rank") & (sf.scale == scale)].c_flip_mean.iloc[0])
        add(f"SCsweepERFlipC{SCN[scale]}", v, f"{v:.2f}",
            "revision/regime/task1_signflip.csv [metric == effective_rank]; c_flip_mean (S5_5 'Mean curve')", ["supp_mat.tex:117"])

    # ---------------- L127: decay-ratio collapse ----------------
    col = pd.read_csv(Q1 / "decay_collapse.csv")
    src_c = "revision/q1/decay_collapse.csv; column D (S5_8 matched-window rows)"
    red = []
    for curve, nm in (("cd_q1", "CycleDegreeQOne"), ("cd_er", "CycleDegreeER"), ("dr_q1", "DegreeRandomQOne")):
        dc = float(col[(col.curve == curve) & (col.axis == "c (matched window)")].D.iloc[0])
        dd = float(col[(col.curve == curve) & (col.axis == "decay_ratio (matched window)")].D.iloc[0])
        red.append(1 - dd / dc)
        add(f"SCollapseD{nm}C", dc, f"{dc:.3f}", src_c + f" [curve == {curve}, axis == c (matched window)]", ["supp_mat.tex:127"])
        add(f"SCollapseD{nm}Decay", dd, f"{dd:.3f}", src_c + f" [curve == {curve}, axis == decay_ratio (matched window)]",
            ["supp_mat.tex:127"])
    pct = 100 * float(np.mean(red))
    add("SCollapseDispersionDropPct", pct, f"{round(pct, -1):.0f}",
        src_c + "; mean over the 3 curves of 1 - D(d, matched)/D(c, matched), x100, rounded to 10 ('about 30%')",
        ["supp_mat.tex:127"])
    src_f = "revision/q1/decay_flip_summary.csv [curve == cd_er]; max_over_min (S5_7)"
    for axis, nm in (("flip_decay", "Decay"), ("flip_c", "C")):
        v = float(fl[(fl.curve == "cd_er") & (fl.axis == axis)].max_over_min.iloc[0])
        add(f"SCollapseERFlipSpread{nm}", v, f"{v:.1f}", src_f + f" [axis == {axis}]", ["supp_mat.tex:127"])
    kp = float(fl[(fl.curve == "cd_er") & (fl.axis == "flip_decay")].kruskal_p.iloc[0])
    assert kp < 1e-3, kp
    add("SCollapseERFlipKruskalBound", kp, "10^{-3}",
        "revision/q1/decay_flip_summary.csv [curve == cd_er, axis == flip_decay]; kruskal_p (2.8e-7), reported as the bound p < 1e-3",
        ["supp_mat.tex:127"])
    dcv = pd.read_csv(Q1 / "decay_curves.csv.gz", usecols=["subject_id", "scale", "gap"])
    gaps = dcv.drop_duplicates(["scale", "subject_id"]).groupby("scale").gap.mean()
    for scale in SC:
        v = float(gaps.loc[scale])
        add(f"SCollapseMeanGap{SCN[scale]}", v, f"{v:.2f}",
            "revision/q1/decay_curves.csv.gz; gap (l1 - l2 of A0) per subject, mean over the 70 subjects", ["supp_mat.tex:127"])

    # ---------------- L140: nctpy normalization ----------------
    det = pd.read_csv(REG / "nctpy_equivalence_detail.csv")
    src_n = "revision/regime/nctpy_equivalence_detail.csv"
    add("SNctpyChecks", len(det), str(len(det)), src_n + "; number of rows (subject x method x k checks)", ["supp_mat.tex:140"])
    v = float(det.max_abs_W_diff.max())
    add("SNctpyMaxGramianDiff", v, _sig1(v),
        src_n + "; max of max_abs_W_diff = max |W_nctpy - (l1 + 1) W_ours|, one significant figure", ["supp_mat.tex:140"])
    v = float(det.d_er_eps_rescaled.abs().max())
    add("SNctpyMaxRescaledDiff", v, _sig1(v),
        src_n + "; max |d_er_eps_rescaled| (ER only: no rescaled-eps PR column is saved), one significant figure",
        ["supp_mat.tex:140"])
    return out


def _spin_and_targets(res: Path) -> list[Number]:
    T = res / "targets"
    out: list[Number] = []

    def add(name, value, text, source, line, draft=None):
        out.append(Number(name, float(value), text, source, [f"supp_mat.tex:{line}"], draft))

    # ---- L79: Spearman rho degree strength vs cycle participation (S10_9 rows)
    spin = pd.read_csv(T / "spin_tests.csv")
    for s in SC:
        r = spin[(spin.scale == s) & (spin.x == "degree_strength") & (spin.y == "cycle_participation")].iloc[0]
        add(f"SCycleDegreeRho{SCN[s]}", r.spearman_rho, _f(r.spearman_rho, 2),
            "revision/targets/spin_tests.csv [x=degree_strength, y=cycle_participation]; spearman_rho (as S10_9)", 79)

    # ---- L79: top-15 regions shared by cycle participation and degree strength
    top = pd.read_csv(T / "top15_network_composition.csv")
    for s in SC:
        n = top[(top.scale == s) & (top.category == "both")].n.iloc[0]
        add(f"STopFifteenShared{SCN[s]}", n, str(int(n)),
            "revision/targets/top15_network_composition.csv [category=both]; n (subject-averaged connectome)", 79)

    # ---- L81: rank divergence vs registered G1: rho and Vasa spin p (S10_9 rows)
    for s in SC:
        r = spin[(spin.scale == s) & (spin.x == "rank_divergence") & (spin.y == "G1_registered")].iloc[0]
        add(f"SRankDivGOneRho{SCN[s]}", r.spearman_rho, _f(r.spearman_rho, 2),
            "revision/targets/spin_tests.csv [x=rank_divergence, y=G1_registered]; spearman_rho", 81)
        add(f"SRankDivGOnePspin{SCN[s]}", r.p_spin_vasa, _f(r.p_spin_vasa, 2),
            "revision/targets/spin_tests.csv [x=rank_divergence, y=G1_registered]; p_spin_vasa (2 dp)", 81)

    # ---- L287: Yeo-7 purity medians and precuneus parcels double-assigned in the legacy masks
    pa = pd.read_csv(T / "parcel_network_assignment.csv")
    pur = pa.groupby("scale").yeo7_purity.median()
    both = pa[pa.legacy_occipital & pa.legacy_association].groupby("scale").size()
    for s in SC:
        add(f"SYeoPurityMedian{SCN[s]}", pur[s], _f(pur[s], 2),
            "revision/targets/parcel_network_assignment.csv; median yeo7_purity per scale", 287)
    for s in SC:
        add(f"SLegacyPrecuneusParcels{SCN[s]}", both[s], str(int(both[s])),
            "revision/targets/parcel_network_assignment.csv; count parcels with legacy_occipital & legacy_association", 287)

    # ---- L289: named-target medians (S10_1 full, S10_2 drivers excluded), 1 dp as in prose
    nts = pd.read_csv(T / "named_target_summary.csv").set_index(["variant", "target_name", "scale"])
    for variant, tgt, stem, vlab in (("full", "kw_occipital_module", "STargetOccipitalFull", "full"),
                                     ("full", "net_Visual", "STargetVisualFull", "full"),
                                     ("drivers_excluded", "net_Visual", "STargetVisualExcl", "drivers_excluded")):
        for s in SC:
            v = nts.loc[(variant, tgt, s)].median_pct
            add(f"{stem}{SCN[s]}", v, _f(v, 1),
                f"revision/targets/named_target_summary.csv [variant={vlab}, target={tgt}]; median_pct (S10_1/S10_2)", 289)

    # overlap: fraction of subject x k cases with >=1 driver in the Yeo visual network (S10_3), as percent
    ov = pd.read_csv(T / "driver_target_overlap.csv").set_index(["target_name", "scale"])
    for col, stem in (("frac_any_driver_in_cyc", "SVisualOverlapCycle"), ("frac_any_driver_in_deg", "SVisualOverlapDegree")):
        for s in SC:
            v = ov.loc[("net_Visual", s)][col]
            # round the fraction to 2 dp first, as S10_3 prints it (degree at 68 is exactly 0.025 -> "0.03" -> 3)
            add(f"{stem}{SCN[s]}", 100 * v, str(round(100 * float(f"{v:.2f}"))),
                f"revision/targets/driver_target_overlap.csv [target=net_Visual]; {col}, 2 dp as S10_3, x100", 289)

    # rho(energy pct, coupling difference) for the visual network, full targets (S10_4)
    for s in SC:
        v = nts.loc[("full", "net_Visual", s)].rho_pct_vs_dcoupling
        add(f"SVisualCouplingRho{SCN[s]}", v, _f(v, 2),
            "revision/targets/named_target_summary.csv [variant=full, target=net_Visual]; rho_pct_vs_dcoupling (S10_4)", 289)

    # equal-distance intercept for the visual network (S10_4)
    pc = pd.read_csv(T / "proximity_controls.csv").set_index(["target_name", "scale"])
    for s in SC:
        v = pc.loc[("net_Visual", s)].eqdist_intercept
        add(f"SVisualEqualDist{SCN[s]}", v, _f(v, 1),
            "revision/targets/proximity_controls.csv [target=net_Visual]; eqdist_intercept (median regression, S10_4)", 289)

    # single-parcel targets: range over scales of rho across parcels with the coupling difference (S10_5)
    sp = pd.read_csv(T / "single_parcel_summary.csv")
    for variant, stem in (("full", "SSingleParcelCouplingRho"), ("drivers_excluded", "SSingleParcelCouplingRhoExcl")):
        rho = sp[sp.variant == variant].rho_parcels_pct_vs_dcoupling
        # prose orders the range "between -0.67 and -0.79": weakest (max) first, strongest (min) second
        add(f"{stem}Weak", rho.max(), _f(rho.max(), 2),
            f"revision/targets/single_parcel_summary.csv [variant={variant}]; max over scales of rho_parcels_pct_vs_dcoupling", 289)
        add(f"{stem}Strong", rho.min(), _f(rho.min(), 2),
            f"revision/targets/single_parcel_summary.csv [variant={variant}]; min over scales of rho_parcels_pct_vs_dcoupling", 289)

    # floor share of target energy: per-target median over subjects, k, methods (c=0.1, eps=1e-5), range over targets/scales (S10_8)
    sl = pd.read_csv(T / "sensitivity_long.csv.gz")
    sl = sl[(sl.c == 0.1) & (sl.eps == 1e-5)]
    med = sl.groupby(["scale", "target_name"]).floor_frac.median()
    add("STargetFloorShareLo", 100 * med.min(), f"{100 * med.min():.0f}",
        "revision/targets/sensitivity_long.csv.gz [c=0.1, eps=1e-5]; min over scale x target of median floor_frac, x100 (S10_8)", 289)
    add("STargetFloorShareHi", 100 * med.max(), f"{100 * med.max():.0f}",
        "revision/targets/sensitivity_long.csv.gz [c=0.1, eps=1e-5]; max over scale x target of median floor_frac, x100"
        " (S10_8; prose 'nearly 100')", 289)
    return out


def _nulls_and_stability(res: Path) -> list[Number]:
    out: list[Number] = []

    def add(name, value, text, source, line, draft=None):
        out.append(Number(name, float(value), text, source, [f"supp_mat.tex:{line}"], draft))

    nulls = res / "nulls"

    # ---------------- L217: null fidelity (Table S8_1) ----------------
    fid = pd.read_csv(nulls / "null_fidelity.csv")
    st = fid[fid.null == "strength"]
    src = "revision/nulls/null_fidelity.csv [null == strength], {col} over scales (Table S8_1, {d} dp)"
    add("SNullStrengthCorrLo", st.strength_r.min(), _f(st.strength_r.min(), 3),
        src.format(col="min strength_r", d=3), 217)
    add("SNullStrengthCorrHi", st.strength_r.max(), _f(st.strength_r.max(), 3),
        src.format(col="max strength_r", d=3), 217)
    add("SNullStrengthMaxRelErrLo", st.strength_max_rel_err.min(), _f(st.strength_max_rel_err.min()),
        src.format(col="min strength_max_rel_err", d=2), 217)
    add("SNullStrengthMaxRelErrHi", st.strength_max_rel_err.max(), _f(st.strength_max_rel_err.max()),
        src.format(col="max strength_max_rel_err", d=2), 217)
    real = fid[fid.null == "real"].set_index("scale").mean_edge_length
    geo = fid[fid.null == "geometric"].set_index("scale").mean_edge_length
    dev = (100 * (geo - real).abs() / real).max()
    assert dev < 1
    add("SNullGeomEdgeLengthWithinPct", dev, "1",
        "revision/nulls/null_fidelity.csv: max over scales of |geometric - real| / real mean_edge_length, in %"
        " (value 0.67%); text is the bound 'within 1\\%'", 217)

    # ---------------- L221: lambda2 at 114, median z (Table S8_2) ----------------
    gs = pd.read_csv(nulls / "gap_split.csv")
    z = gs[(gs.stat == "lambda2") & (gs.scale == 114) & gs.null.isin(NULLS)].z_median
    src = "revision/nulls/gap_split.csv [stat == lambda2, scale == 114, 3 nulls], {} of z_median (Table S8_2), 1 dp"
    add("SNullLambdaTwoZOneFourteenLo", z.min(), f"{z.min():.1f}", src.format("min"), 221)
    add("SNullLambdaTwoZOneFourteenHi", z.max(), f"{z.max():.1f}", src.format("max"), 221)

    # ---------------- L241, L243: Rayleigh accounting (Table S8_6) ----------------
    ps = pd.read_csv(nulls / "rayleigh" / "per_subject.csv")
    ps["sl_R1"] = ps["d_R1"] / ps["d_lambda1"]
    pss = ps[ps.null == "strength"]
    r = pss.pearson_rW_sTWs.median()
    add("SRayleighPearsonRWsAs", r, _f(r, 3),
        "revision/nulls/rayleigh/per_subject.csv [null == strength]: median over all subjects and scales of"
        " pearson_rW_sTWs (per scale 0.998/0.999/0.998, Table S8_6)", 241)
    med = pss.groupby("scale").median(numeric_only=True)
    for sc in SC:
        v = med.loc[sc, "sl_R1"]
        add(f"SRayleighShareROne{SCN[sc]}", v, _pct(v),
            f"revision/nulls/rayleigh/per_subject.csv [null == strength, scale == {sc}]: median of d_R1/d_lambda1"
            " (Table S8_6 'Share of Delta lambda1: R1'), in %", 243)
    for col, nm, line in (("rho_rW_l1", "RW", 243), ("rho_rbct_l1", "Bct", 243)):
        v = med[col]
        src = (f"revision/nulls/rayleigh/per_subject.csv [null == strength]: per-scale median of {col}"
               " (Table S8_6), {} over scales, 2 dp")
        add(f"SRayleighRho{nm}LambdaLo", v.min(), _f(v.min()), src.format("min"), line)
        add(f"SRayleighRho{nm}LambdaHi", v.max(), _f(v.max()), src.format("max"), line)

    # ---------------- L249: mediation (Table S8_8) ----------------
    mn = pd.read_csv(nulls / "mediation_noncircular.csv")
    mn["k"] = mn["k"].astype(str)
    hub = mn[(mn.outcome == "degree_q1") & (mn.k == "avg") & (mn.mediator == "hub_strength_frac")
             & mn.null.isin(NULLS)].prop_mediated
    src = ("revision/nulls/mediation_noncircular.csv [outcome == degree_q1, k == avg, mediator == hub_strength_frac,"
           " 3 nulls x 3 scales]: {} prop_mediated (Table S8_8), in %")
    add("SMediationHubFracLo", hub.min(), _pct(hub.min()), src.format("min"), 249)
    add("SMediationHubFracHi", hub.max(), _pct(hub.max()), src.format("max"), 249)
    rc = mn[mn.mediator.isin(["rc_bu", "rc_wu", "rc_wu_p75", "rc_wu_p85", "rc_wu_p95"])
            & (((mn.outcome == "degree_q1") & (mn.k == "avg")) | ((mn.outcome == "gap") & (mn.k == "-")))
            & mn.null.isin(NULLS)].prop_mediated
    add("SMediationRichClubMax", rc.max(), _pct(rc.max()),
        "revision/nulls/mediation_noncircular.csv [mediator in rc_bu/rc_wu/rc_wu_p75/p85/p95; outcome degree_q1 (k avg)"
        " or gap; 3 nulls x 3 scales]: max prop_mediated (Table S8_8; weighted 95th pct, strength null, 68), in %", 249)

    # ---------------- L259: representative agreement (Table S9_1) ----------------
    tb = res / "stability" / "tables"
    ag = pd.read_csv(tb / "rep_agreement.csv")
    for var, nm in (("geodesic", "Geodesic"), ("ripser", "Ripser")):
        v = ag[ag.variant == var].rho_degree_median
        src = (f"revision/stability/tables/rep_agreement.csv [variant == {var}]: {{}} over scales of"
               " rho_degree_median (Table S9_1), 2 dp")
        add(f"SRepRho{nm}DegreeLo", v.min(), _f(v.min()), src.format("min"), 259)
        add(f"SRepRho{nm}DegreeHi", v.max(), _f(v.max()), src.format("max"), 259)

    # ---------------- L261: distance regressions (S9_5) and strength-matched null (S9_4) ----------------
    dr = pd.read_csv(tb / "rep_distance_regressions.csv")
    r2 = dr[dr.k >= 2].jaccard_degree_partial_r2
    src = ("revision/stability/tables/rep_distance_regressions.csv [k >= 2, i.e. 2,3,5,15; all scales]: {}"
           " jaccard_degree_partial_r2 (Table S9_5 R^2_J), in %")
    add("SRepOverlapRsqLo", r2.min(), _pct(r2.min()), src.format("min"), 261)
    add("SRepOverlapRsqHi", r2.max(), _pct(r2.max()), src.format("max"), 261)
    nc = pd.read_csv(tb / "rep_null_comparisons.csv")
    x = nc[(nc.variant != "degree") & nc.k.isin([1, 2, 3, 5])].strength_mean
    add("SRepStrengthNullMaxER", x.max(), _f(x.max()),
        "revision/stability/tables/rep_null_comparisons.csv [4 conventions, k in 1,2,3,5, all scales]: max strength_mean"
        " (Table S9_4 strength-matched; GUDHI, 219, k = 1)", 261)

    # ---------------- L263: perturbation stability (Table S9_6) ----------------
    rs = pd.read_csv(tb / "perturb_rank_stability.csv")
    for fam, lvl, nm in (("noise", 0.05, "NoiseFive"), ("dropout", 0.2, "RemovalTwenty")):
        v = rs[(rs.ranking == "ripser") & (rs.family == fam) & np.isclose(rs.level, lvl)].rho_median
        src = (f"revision/stability/tables/perturb_rank_stability.csv [ranking == ripser, family == {fam},"
               f" level == {lvl}]: {{}} over scales of rho_median (Table S9_6), 2 dp")
        add(f"SPerturbRho{nm}Lo", v.min(), _f(v.min()), src.format("min"), 263)
        add(f"SPerturbRho{nm}Hi", v.max(), _f(v.max()), src.format("max"), 263)
    rf = pd.read_csv(tb / "perturb_rescaled_fraction.csv")
    v = rf[rf.family == "noise"].rescaled
    src = "revision/stability/tables/perturb_rescaled_fraction.csv [family == noise, all scales/levels]: {} rescaled, in %"
    add("SPerturbRescaledLo", v.min(), _pct(v.min()), src.format("min"), 263)
    add("SPerturbRescaledHi", v.max(), _pct(v.max()), src.format("max"), 263)

    return out
