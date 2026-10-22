"""Supplement S11: reciprocal hub lesions (S11_1-S11_6).

Port of scripts/revision/paper/_rescued_2026-10-19/scripts/make_s10_s11.py (S11 part).
All sources are in <run>/revision/lesion/.

Change from the rescued builder: the two correlations in the S11_6 footer (0.989, 0.975) were typed
in from report_snippets.md; they are now computed from lesion_rows.csv.gz exactly as
scripts/revision/lesion/02_analyze_reciprocal_lesion.py computes its "energy_corr" snippet.
"""

from __future__ import annotations

from pathlib import Path

from paper_common import SCALES, Table, np, pd
from s10_s11_fmt import f2, pf
from scipy.stats import wilcoxon

LESIONS = ["degree", "cycle", "random"]
COMMON = """Settings (lesion REPORT.md Methods): c = 0.1 re-applied to each lesioned (N-h)x(N-h) adjacency (A = A0 - (lmax + c)I),
infinite-horizon Gramian, eps = 1e-5 floor, k in {1,2,3,5}, h in {5,10,15}, 3 scales, 70 subjects. Lesions: top-h degree strength;
top-h cycle participation (both on the intact connectome); h uniformly random nodes (100 draws per subject x scale x h, averaged
within subject). Static protocol (primary): drivers = top-k SURVIVING nodes by the intact ranking, same identities evaluated
intact vs lesioned (k <= 5 <= h, so a lesion targeting a strategy's own ranking displaces all its primary drivers). Recompute
protocol: intact set = top-k intact; lesioned set = top-k by the ranking recomputed on the lesioned graph. Loss = intact - lesioned
(positive = degradation). Per subject: average over k, then draws; "pooled" = subject mean over the 9 scale x h cells.
D = cycle loss - degree loss (D > 0 = cycle sets degrade more). Stats: bootstrap 95% CI of mean (10,000), Wilcoxon signed-rank,
rank-biserial; Holm within metric x protocol family of 30 (27 cells + 3 pooled); interactions Holm in a separate family of 30."""


def _end(lines: list[str]) -> str:
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"])


def _pooled(cs: pd.DataFrame, metric: str, protocol: str, lesion: str) -> pd.Series:
    x = cs[(cs.metric == metric) & (cs.protocol == protocol) & (cs.lesion_type == lesion) & (cs.scale == "pooled")]
    assert len(x) == 1, (metric, protocol, lesion)
    return x.iloc[0]


def summary_table(s32: pd.DataFrame) -> str:
    st = s32[s32.protocol == "static"].set_index(["lesion_type", "strategy"])
    L = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
         " & \\multicolumn{2}{c}{ER loss} & \\multicolumn{2}{c}{PR loss} & \\multicolumn{2}{c}{ER-percentile loss} & "
         "\\multicolumn{2}{c}{PR-percentile loss} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
         "Lesion & Degree & Cycle & Degree & Cycle & Degree & Cycle & Degree & Cycle \\\\", "\\midrule"]
    for lt in LESIONS:
        dg, cy = st.loc[(lt, "degree_strength")], st.loc[(lt, "cycle_participation")]
        line = (f"{lt.capitalize()} lesion & {f2(dg.dER)} & {f2(cy.dER)} & {f2(dg.dPR)} & {f2(cy.dPR)} & "
                f"{100 * dg.dPct_ER:+.1f} & {100 * cy.dPct_ER:+.1f} & {100 * dg.dPct_PR:+.1f} & {100 * cy.dPct_PR:+.1f} \\\\")
        L.append(line.replace(" -", " $-$"))
    return _end(L)


def crossover_table(cs: pd.DataFrame) -> str:
    names = {"dER": "ER", "dPR": "PR", "dPct_ER": "ER-pct", "dPct_PR": "PR-pct"}
    L = ["\\begin{tabular}{llrcccr}", "\\toprule",
         "Metric & Lesion & Mean $D$ & 95\\% CI & $n(D>0)$ & $p_H$ & $r_{rb}$ \\\\", "\\midrule"]
    for i, m in enumerate(["dER", "dPR", "dPct_ER", "dPct_PR"]):
        if i:
            L.append("\\addlinespace")
        for lt in LESIONS + ["interaction cycle-degree"]:
            r = _pooled(cs, m, "static", lt)
            nm = "Interaction (cyc.\\ $D$ $-$ deg.\\ $D$)" if lt.startswith("inter") else lt.capitalize()
            if "Pct" in m:  # percentile points
                v, lo, hi = [f"{100 * x:+.1f}".replace("-", "$-$") for x in (r["mean"], r.ci_lo, r.ci_hi)]
            else:
                v, lo, hi = f2(r["mean"]), f2(r.ci_lo), f2(r.ci_hi)
            L.append(f"{names[m] if lt == 'degree' else ''} & {nm} & {v} & [{lo}, {hi}] & {int(r.n_pos)}/{int(r.n)} & "
                     f"{pf(r.p_holm)} & {f2(r.r_rb)} \\\\")
    return _end(L)


def holm_cells_table(cs: pd.DataFrame) -> str:
    L = ["\\begin{tabular}{lccc}", "\\toprule",
         "Metric & Degree lesion & Cycle lesion & Random lesion \\\\", "\\midrule"]
    for m, nm in [("dER", "ER"), ("dPR", "PR"), ("dER_excess", "ER excess"), ("dPct_ER", "ER percentile"),
                  ("dPct_PR", "PR percentile"), ("dER_perN", "ER/$N$ (not recommended)")]:
        cells = []
        for lt in LESIONS:
            c = cs[(cs.metric == m) & (cs.protocol == "static") & (cs.lesion_type == lt) & (cs.scale != "pooled")]
            assert len(c) == 9, (m, lt)
            sig = c[c.p_holm < 0.05]
            cells.append(f"{int((sig['mean'] < 0).sum())} ($D<0$) / {int((sig['mean'] > 0).sum())} ($D>0$)")
        L.append(nm + " & " + " & ".join(cells) + " \\\\")
    return _end(L)


def protocol_table(cs: pd.DataFrame) -> str:
    L = ["\\begin{tabular}{llrrrcc rrrcc}", "\\toprule",
         " & & \\multicolumn{5}{c}{ER loss} & \\multicolumn{5}{c}{PR loss} \\\\",
         "\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}",
         "Protocol & Lesion & Degree & Cycle & $D$ & $n(D>0)$ & $p_H$ & Degree & Cycle & $D$ & $n(D>0)$ & $p_H$ \\\\",
         "\\midrule"]
    for prot in ["static", "recompute"]:
        for lt in LESIONS:
            cells = []
            for m in ["dER", "dPR"]:
                r = _pooled(cs, m, prot, lt)
                cells += [f2(r.mean_degree), f2(r.mean_cycle), f2(r["mean"]), f"{int(r.n_pos)}/{int(r.n)}", pf(r.p_holm)]
            L.append(f"{prot.capitalize() if lt == 'degree' else ''} & {lt.capitalize()} & " + " & ".join(cells) + " \\\\")
        if prot == "static":
            L.append("\\addlinespace")
    return _end(L)


def absolute_er_table(d: pd.DataFrame) -> str:
    L = ["\\begin{tabular}{lllcccc}", "\\toprule",
         "Protocol & Lesion & Scale & Intact ER (deg / cyc) & Lesioned ER (deg / cyc) & $D$ (cyc $-$ deg) & "
         "$n(D>0)$; $p$ \\\\", "\\midrule"]
    for prot in ["static", "recompute"]:
        p = d[d.protocol == prot].pivot_table(index=["lesion_type", "scale", "h", "subject"], columns="strategy",
                                              values=["lesioned_effective_rank", "intact_effective_rank"])
        D = p[("lesioned_effective_rank", "cycle_participation")] - p[("lesioned_effective_rank", "degree_strength")]
        for lt in LESIONS:
            for sc in ["pooled", *SCALES]:
                key = lt if sc == "pooled" else (lt, sc)
                g = D.loc[key].groupby(level="subject").mean()  # per subject: mean over h (and scale if pooled)
                pp = p.loc[key]
                intact = pp["intact_effective_rank"].mean()
                lesioned = pp["lesioned_effective_rank"].mean()
                L.append(f"{prot.capitalize() if (lt == 'degree' and sc == 'pooled') else ''} & "
                         f"{lt.capitalize() if sc == 'pooled' else ''} & {sc} & "
                         f"{intact.degree_strength:.2f} / {intact.cycle_participation:.2f} & "
                         f"{lesioned.degree_strength:.2f} / {lesioned.cycle_participation:.2f} & "
                         f"{f2(g.mean())} & {(g > 0).sum()}/{len(g)}; {pf(wilcoxon(g).pvalue)} \\\\")
            L.append("\\addlinespace")
        if prot == "static":
            L.append("\\midrule")
    if L[-1] == "\\addlinespace":
        L = L[:-1]
    return _end(L)


def energy_correlations(rows_path: Path) -> tuple[float, float]:
    """corr(dlog10E, log10((N-h)/N)) and corr(dlog10E, log10(n_floor_les/n_floor_int)) over all static rows,
    as in 02_analyze_reciprocal_lesion.py (snippet "energy_corr")."""
    cols = ["protocol", "n_intact", "n_lesioned", "intact_log10_average_energy", "lesioned_log10_average_energy",
            "intact_fraction_at_floor", "lesioned_fraction_at_floor"]
    st = pd.read_csv(rows_path, usecols=cols)
    st = st[st.protocol == "static"]
    dlog10E = st.lesioned_log10_average_energy - st.intact_log10_average_energy
    size_pred = np.log10(st.n_lesioned / st.n_intact)
    pred_floor = np.log10((st.lesioned_fraction_at_floor * st.n_lesioned) / (st.intact_fraction_at_floor * st.n_intact))
    return float(np.corrcoef(dlog10E, size_pred)[0, 1]), float(np.corrcoef(dlog10E, pred_floor)[0, 1])


def energy_table(es: pd.DataFrame, cs: pd.DataFrame, r_size: float, r_floor: float) -> str:
    L = ["\\begin{tabular}{llrccccc}", "\\toprule",
         "Scale & Lesion & $h$ & Floor frac.\\ intact $\\to$ lesioned & $\\Delta\\log_{10}E$ deg. & "
         "$\\Delta\\log_{10}E$ cyc. & Random-set ref. & $\\log_{10}\\frac{N-h}{N}$ \\\\", "\\midrule"]
    for sc in SCALES:
        for lt in LESIONS:
            for h in [5, 10, 15]:
                g = es[(es.scale == sc) & (es.lesion_type == lt) & (es.h == h)].set_index("strategy")
                assert len(g) == 2, (sc, lt, h)
                fi, fl = g.frac_floor_intact.mean(), g.frac_floor_lesioned.mean()
                dd, dc = g.loc["degree_strength"].dlog10E, g.loc["cycle_participation"].dlog10E
                L.append(f"{sc if (lt == 'degree' and h == 5) else ''} & {lt.capitalize() if h == 5 else ''} & {h} & "
                         f"{fi:.2f} $\\to$ {fl:.2f} & {dd:+.3f} & {dc:+.3f} & {g.ref_dlog10E.iloc[0]:+.3f} & "
                         f"{g.size_pred.iloc[0]:+.3f} \\\\".replace("+-", "-").replace(" -0", " $-$0"))
        if sc != SCALES[-1]:
            L.append("\\addlinespace")
    L += ["\\midrule",
          f"\\multicolumn{{8}}{{l}}{{corr($\\Delta\\log_{{10}}E$, $\\log_{{10}}((N-h)/N)$) $= {r_size:.3f}$; "
          f"corr($\\Delta\\log_{{10}}E$, $\\log_{{10}}(n_{{\\mathrm{{floor,les}}}}/n_{{\\mathrm{{floor,int}}}})$) "
          f"$= {r_floor:.3f}$ (all rows).}} \\\\"]
    parts = []
    for lt in LESIONS:
        r = _pooled(cs, "dlog10E", "static", lt)
        parts.append(f"{lt}: {r['mean']:+.4f} ({int(r.n_pos)}/{int(r.n)}, $p_H$={pf(r.p_holm)})")
    txt = "; ".join(parts).replace(": -", ": $-$")
    L.append(f"\\multicolumn{{8}}{{l}}{{Pooled cyc.\\ $-$ deg.\\ $\\Delta\\log_{{10}}E$: {txt}}} \\\\".replace("+-", "-"))
    return _end(L)


def build(res: Path) -> list[Table]:
    LE = res / "lesion"
    s32 = pd.read_csv(LE / "summary_3x2.csv")
    cs = pd.read_csv(LE / "crossover_stats.csv")
    src = "Source: results/revision/lesion/"
    r_size, r_floor = energy_correlations(LE / "lesion_rows.csv.gz")
    return [
        Table("S11_1", summary_table(s32),
              f"""Static protocol, mean loss (intact - lesioned) per subject pooled over scale, h in {{5,10,15}}, k in {{1,2,3,5}};
c = 0.1, eps = 1e-5. Positive = degradation.
{COMMON}
{src}summary_3x2.csv (protocol = static): dER, dPR, dPct_ER, dPct_PR. Percentile losses are stored as fractions (0-1) and shown
x100 (percentile points, 1 dp). Percentile loss = percentile of the strategy set among 100 random k-sets from surviving nodes
(intact) minus the same in the lesioned network. Excess and ER/N variants are in the CSV (not tabulated)."""),
        Table("S11_2", crossover_table(cs),
              f"""Crossover tests on pooled D = cycle loss - degree loss (static). D > 0 = cycle sets degrade more. Bootstrap 95% CI
of mean; Wilcoxon; Holm within metric x protocol (30 tests: 27 cells + 3 pooled); interactions Holm in a separate family of 30.
Pct metrics in percentile points (x100).
{COMMON}
{src}crossover_stats.csv (protocol = static, scale = pooled; metrics dER, dPR, dPct_ER, dPct_PR)."""),
        Table("S11_3", holm_cells_table(cs),
              f"""Number of Holm-significant (p_H < 0.05) scale x h cells out of 9, by sign of D (static).
{COMMON}
Computed from {src}crossover_stats.csv: for each metric x lesion, the 9 non-pooled (scale, h) rows with p_holm < 0.05, split by
the sign of mean D."""),
        Table("S11_4", protocol_table(cs),
              f"""Protocol dependence. Mean ER and PR loss (pooled) under static vs recompute protocols; D = cycle - degree with
n(D>0)/70 and Holm p.
{COMMON}
{src}crossover_stats.csv pooled rows (mean_degree, mean_cycle, mean, n_pos, p_holm), static and recompute, dER and dPR."""),
        Table("S11_5", absolute_er_table(pd.read_csv(LE / "per_subject_degradation.csv")),
              f"""Absolute post-lesion ER. Mean intact / lesioned ER (degree / cycle sets) and per-subject D = ER_cyc - ER_deg after
the lesion, averaged over h (and over scales for 'pooled'); n(D>0)/70; uncorrected two-sided Wilcoxon.
{COMMON}
Computed from {src}per_subject_degradation.csv (intact_effective_rank, lesioned_effective_rank, already averaged over k and
draws). Intact/lesioned columns = means over subjects x (scale) x h.
Under the static protocol the "intact ER" differs between lesion types because the evaluated identities (surviving top-k)
differ; under recompute the intact set is the true top-k."""),
        Table("S11_6", energy_table(pd.read_csv(LE / "energy_size_diagnostics.csv"), cs, r_size, r_floor),
              f"""Lesioned - intact log10 E_avg (static protocol) vs size prediction log10((N-h)/N); floor fraction = fraction of
W+eps*I eigenvalues at eps (mean of the two strategies' values in energy_size_diagnostics.csv).
{COMMON}
{src}energy_size_diagnostics.csv (static): frac_floor_intact / frac_floor_lesioned (mean over strategies), dlog10E per strategy,
ref_dlog10E (random sets), size_pred = log10((N-h)/N). Footer correlations computed from lesion_rows.csv.gz over all static rows
(as in scripts/revision/lesion/02_analyze_reciprocal_lesion.py). Pooled cyc - deg dlog10E with Holm p from crossover_stats.csv
(metric dlog10E)."""),
    ]
