"""Numbers quoted in the prose of main.tex and response_to_reviewers.tex.

``collect(res)`` returns one :class:`paper_common.Number` per quoted result,
where ``res`` is ``<run>/revision`` (other stages are read from ``res.parent``).
Inputs that are settings rather than results (c, epsilon, k, numbers of
subjects/nulls/random sets, citation years, section numbers) are not macros.
Numbers inside ``\\input`` tables are built by ``make_tables.py``.

Conventions for ``text`` (what the macro expands to):

- The macro holds the number token only: sign, digits, decimals. It never
  contains ``$``/``\\(`` delimiters or ``\\%``; the manuscript keeps those, so
  ``86--87\\%`` becomes ``\\MFoo--\\MBar\\%`` and ``\\(-0.26\\)`` becomes
  ``\\(\\MBaz\\)``.
- Signs are printed exactly as in the draft: an explicit ``+`` only where the
  draft writes one, ``-`` for negatives (always inside math in the draft).
- Percentages are printed as the percentage number (``0.734`` -> ``73``).
- Ordinals ("99th percentile") hold the number only (``99``); counts of
  subjects hold the count only ("in 48 of 70 subjects" -> ``48``).
- Powers of ten are printed as math content without delimiters
  (``4\\times10^{-15}``, ``10^{-5}``).
- Where the draft gives a range ("0.19 to 0.27", "64th to 74th") each end
  is its own macro, computed as the min/max over the set the text describes.
- Where the draft states a bound ("within 0.07", "all p < 10^{-5}", "p > 0.84")
  the macro holds the bound, rounded so that the statement stays true (up for
  "within"/"<", down for ">"); the code asserts the bound and ``value`` holds
  the actual extreme.
- A few results are printed as words in the draft ("one subject", "about
  five times", "four to eight times", "one fifth", "with two exceptions"); their
  macros hold the word and ``value`` holds the unrounded number.
- A quantity printed both with and without a sign, or both as a percentage and
  as a decimal, gets one macro per printed form (e.g. L119 ``0.84`` vs L150/L172
  ``+0.84``; L115 ``73`` vs L135 ``0.73``).
- Macro names: ``M`` + readable CamelCase; per-scale macros end in
  ``SixtyEight`` / ``OneFourteen`` / ``TwoNineteen``.

``draft`` is set when the saved result does not reproduce the draft's text;
``value = nan`` with ``source = "NO SOURCE FOUND: ..."`` marks a number for
which no saved result was found.
"""

from __future__ import annotations

import json
import math
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from paper_common import SCALES, Number
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from hoc.preprocessing import stabilize_adjacency
from hoc.rankings import strength_scores
from hoc.topology import cycle_participation

NAN = float("nan")
SCALE_WORD = {68: "SixtyEight", 114: "OneFourteen", 219: "TwoNineteen"}


# --------------------------------------------------------------------------- helpers

def M(line: int) -> str:
    return f"main.tex:{line}"


def R(line: int) -> str:
    return f"response_to_reviewers.tex:{line}"


def fx(x: float, d: int = 2) -> str:
    """Fixed decimals; negative zero printed as 0."""
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def sg(x: float, d: int = 2) -> str:
    """Explicit sign (+/-)."""
    return f"{x:+.{d}f}"


def pc(x: float, d: int = 0) -> str:
    """A fraction printed as a percentage number (no % sign)."""
    return fx(100 * x, d)


def cnt(x: float) -> str:
    return str(int(round(x)))


def pow10(x: float, d: int = 0) -> str:
    """``a\\times10^{b}`` (or ``10^{b}`` when the mantissa is 1 and d == 0)."""
    m, e = f"{x:.{d}e}".split("e")
    if d == 0 and m == "1":
        return f"10^{{{int(e)}}}"
    return f"{m}\\times10^{{{int(e)}}}"


def num(name, value, text, source, where, draft=None) -> Number:
    if isinstance(where, str):
        where = [where]
    return Number(name=name, value=float(value), text=text, source=source, where=list(where), draft=draft)


def per_scale(prefix, values, fmt, source, where, drafts=None) -> list[Number]:
    """One Number per scale; ``values`` maps scale -> value; ``drafts`` maps scale -> draft text."""
    drafts = drafts or {}
    return [num(prefix + SCALE_WORD[s], values[s], fmt(values[s]), source, where, drafts.get(s)) for s in SCALES]


def nosource(name, text, searched, where) -> Number:
    return num(name, NAN, text, f"NO SOURCE FOUND: {searched}", where)


@lru_cache(maxsize=None)
def _csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def csv(res: Path, rel: str) -> pd.DataFrame:
    """Read ``<res>/<rel>`` (``rel`` may start with ``../`` for other stages); cached."""
    return _csv(str((res / rel).resolve()))


def one(df: pd.DataFrame, **eq) -> pd.Series:
    """The single row of ``df`` matching column == value for every keyword."""
    m = np.ones(len(df), bool)
    for k, v in eq.items():
        col = df[k]
        m &= np.isclose(col, v) if isinstance(v, float) else (col == v).to_numpy()
    x = df[m]
    assert len(x) == 1, (eq, len(x))
    return x.iloc[0]


# --------------------------------------------------------------------------- sections

# ---- section A

ENERGY_METHODS = ("degree_strength", "eigenvector_centrality", "cycle_participation", "random_mean")


def _fig1_q1(res: Path) -> dict[str, float]:
    """Figure 1B values, recomputed exactly as scripts/revision/figures/fig1_panels.py (panel_loading_q1) does."""
    a0 = np.asarray(np.load(res.parent / "processed" / "connectomes_scale-114.npz")["cohort_reference"], dtype=float)
    n, k = a0.shape[0], 3
    hub = np.argsort(-strength_scores(a0))[:k]
    spread = np.argsort(-cycle_participation(a0)[0])[:k]
    system, _ = stabilize_adjacency(a0, 0.1)
    lam, u = np.linalg.eigh(system)
    kern = -1.0 / (lam[:, None] + lam[None, :])

    def q1(drivers):
        ub = u[drivers, :].T  # = u.T @ B
        with np.errstate(all="ignore"):
            w = u @ ((ub @ ub.T) * kern) @ u.T
        e = np.linalg.eigvalsh(w)
        return float(e[-1] / e.sum())

    rng = np.random.default_rng(0)
    rand = [rng.choice(n, k, replace=False) for _ in range(1000)]
    return {"hub": q1(hub), "spread": q1(spread), "random": float(np.mean([q1(s) for s in rand]))}


def sec_A(res: Path) -> list[Number]:
    out: list[Number] = []

    # ---- L79: top-15 overlap of cycle participation and degree strength
    top = csv(res, "targets/top15_network_composition.csv")
    src = ("revision/targets/top15_network_composition.csv, category == 'both', column n (regions among the 15 highest-ranked "
           "by both cycle participation and degree strength; ranks of the subject-mean node scores degree_mean/cycle_mean "
           "from targets/parcel_maps_for_spin.csv, built by scripts/revision/targets/06_network_enrichment.py). "
           "Note: ranks the across-subject mean score, not the cohort-average connectome; supp_mat.tex L79 says 'subject-averaged connectome'.")
    out += per_scale("MTopFifteenSharedDegreeCycle", {s: one(top, scale=s, category="both").n for s in SCALES}, cnt, src, [M(79)])

    # ---- L88: Figure 1 caption
    f1 = _fig1_q1(res)
    src = ("recomputed as in scripts/revision/figures/fig1_panels.py panel_loading_q1: cohort_reference of "
           "processed/connectomes_scale-114.npz, c = 0.1, k = 3, unregularized infinite-horizon Gramian, q1 = lambda_max/trace; ")
    out.append(num("MFigOneHubQOnePct", f1["hub"], pc(f1["hub"]), src + "hub = top-3 degree strength; as a percentage", [M(88)]))
    out.append(num("MFigOneHubQOne", f1["hub"], fx(f1["hub"]), src + "hub = top-3 degree strength", [M(88)]))
    out.append(num("MFigOneSpreadQOnePct", f1["spread"], pc(f1["spread"]), src + "spread = top-3 cycle participation; as a percentage", [M(88)]))
    out.append(num("MFigOneRandomQOnePct", f1["random"], pc(f1["random"]),
                   src + "mean over 1000 random 3-sets (np.random.default_rng(0)); as a percentage", [M(88)]))
    out.append(nosource("MFigOneStrongestEdgesPct", "7",
                        "fig1_panels.py and fig1_layout.py (and their git history) draw no connectome edges and contain no 7%/0.93 "
                        "edge threshold; this is a display setting of a panel not produced by committed code", [M(88)]))

    # ---- L102 / L106 / L98 / R120: average energy vs the regularization floor
    st = csv(res, "../statistics/primary_statistics.csv")
    st = st[st.family == "static_energy_cycle_vs_degree"]
    rel = {s: 100 * (10 ** abs(one(st, scale=s).cycle_minus_degree_mean) - 1) for s in SCALES}
    out.append(num("MEnergyCycleDegreeMaxPct", max(rel.values()), fx(max(rel.values()), 2),
                   "statistics/primary_statistics.csv, family == 'static_energy_cycle_vs_degree' (eps = 1e-5, c = 0.1): "
                   "cycle_minus_degree_mean = mean over subjects of the k-averaged (k = 1,2,3,5) cycle - degree log10 E_avg "
                   "(= regime/task6_delta_log10E_summary.csv mean_diff at eps = 1e-5); printed as 100*(10^|mean| - 1), max over "
                   "scales (219 regions). 'At most' refers to the cohort-mean difference, not to every subject.", [M(102)]))

    a = csv(res, "regime/task6_driver_sets.csv")
    a = a[a.method.isin(ENERGY_METHODS)]
    g = a.groupby(["scale", "eps", "method"])[["fraction_at_floor", "fraction_below_eps", "log10_average_energy"]].mean()
    tsrc = ("regime/task6_driver_sets.csv (c = 0.1), methods degree_strength, eigenvector_centrality, cycle_participation, "
            "random_mean (mean of the 100 random sets); mean over subjects and k in {1,2,3,5} per (scale, eps, method) "
            "(= regime/report_tables.md T6-methods); ")
    e5 = g.xs(1e-5, level="eps")

    def rng_pct(col, scale, lo_name, hi_name, what, where):
        v = e5.xs(scale, level="scale")[col]
        lo, hi = v.min(), v.max()
        s = tsrc + f"eps = 1e-5, {what}, {scale} regions; "
        if pc(lo) == pc(hi):
            return [num(lo_name, v.mean(), pc(v.mean()), s + "all four methods round to the same percentage; value = mean", where)]
        return [num(lo_name, lo, pc(lo), s + "min over the four methods", where),
                num(hi_name, hi, pc(hi), s + "max over the four methods", where)]

    below = "fraction of eigenvalues of W below eps (fraction_below_eps)"
    out += rng_pct("fraction_below_eps", 68, "MFracBelowEpsMinSixtyEight", "MFracBelowEpsMaxSixtyEight", below, [M(106), R(120)])
    out += rng_pct("fraction_below_eps", 114, "MFracBelowEpsOneFourteen", None, below, [M(106), R(120)])
    out += rng_pct("fraction_below_eps", 219, "MFracBelowEpsMinTwoNineteen", "MFracBelowEpsMaxTwoNineteen", below, [M(106), R(120)])
    zero = "fraction of eigenvalues at the floor, i.e. numerically zero (fraction_at_floor)"
    out += rng_pct("fraction_at_floor", 68, "MFracZeroMinSixtyEight", "MFracZeroMaxSixtyEight", zero, [R(120)])
    out += rng_pct("fraction_at_floor", 114, "MFracZeroMinOneFourteen", "MFracZeroMaxOneFourteen", zero, [R(120)])
    out += rng_pct("fraction_at_floor", 219, "MFracZeroTwoNineteen", None, zero, [R(120)])

    le = e5["log10_average_energy"]
    vals = {}
    for s in SCALES:
        v = le.xs(s, level="scale")
        assert len({fx(x) for x in v}) == 1, (s, v)  # "for every criterion"
        vals[s] = v.mean()
    out += per_scale("MLogEavg", vals, fx, tsrc + "eps = 1e-5, log10 E_avg; every method rounds to this value; value = mean of the four",
                     [M(106)])
    out += per_scale("MLogNOverEps", {s: np.log10(s / 1e-5) for s in SCALES}, fx,
                     "log10(N/eps) with eps = 1e-5 (arithmetic, the reference value of Fig. 2A)", [M(106)])

    dev = np.log10(g.index.get_level_values("scale") / g.index.get_level_values("eps")) - g["log10_average_energy"]
    d5 = dev.xs(1e-5, level="eps").abs().max()
    def ceil2(x):  # "within x" is an upper bound: round up so the statement stays true
        return fx(math.ceil(x * 100 - 1e-9) / 100)

    out.append(num("MLogEavgFloorGapEpsFive", d5, ceil2(d5), tsrc + "max over scales and methods of |log10(N/eps) - mean log10 E_avg| "
                   "at eps = 1e-5 (0.065, 68 regions), rounded UP because the text states a bound ('within')",
                   [M(106)]))
    dall = dev.abs().max()
    src_all = tsrc + "max over scales, methods and eps in {1e-8, 1e-5, 1e-2} of |log10(N/eps) - mean log10 E_avg| (68 regions, eps = 1e-8, cycle participation)"
    out.append(num("MLogEavgFloorGapAllEps", dall, ceil2(dall),
                   src_all + "; rounded up (bound). Fig. 2A plots all three eps, so the caption's 0.07 (the eps = 1e-5 bound) "
                   "does not hold for the panel (0.101 at eps = 1e-8)",
                   [M(98)], draft="0.07"))
    out.append(num("MLogEavgFloorGapAllEpsOneDp", dall, fx(dall, 1),
                   src_all + "; nearest rounding to 1 dp (strictly 0.101 > 0.1)", [R(120)]))

    land = csv(res, "regime/task6_landscape.csv")
    iqr = land[np.isclose(land.eps, 1e-5)].groupby("scale").iqr_log10E_above_eps.mean()
    lsrc = ("regime/task6_landscape.csv, eps = 1e-5: iqr_log10E_above_eps (IQR across the saved landscape samples of all driver sets "
            "of log10 of the energy carried by eigenvalues above eps), mean over subjects and k (= report_tables.md T6-landscape "
            "iqr_above_eps); ")
    out.append(num("MEnergyAboveEpsIqrMin", iqr.min(), fx(iqr.min()), lsrc + "min over scales", [R(120)]))
    out.append(num("MEnergyAboveEpsIqrMax", iqr.max(), fx(iqr.max()), lsrc + "max over scales", [R(120)]))
    return out


# ---- section B

def sec_B(res: Path) -> list[Number]:
    """main.tex L115-L121 (hub concentration vs random, matched controls); response L87-L89."""
    out: list[Number] = []
    p1 = csv(res, "q1/p1_vs_random.csv")
    p2 = csv(res, "q1/p2_matched.csv")
    t0 = csv(res, "spectra/tables/t0_cycle_minus_degree.csv")
    ov = csv(res, "modal/set_overlap.csv")

    P1 = ("results/revision/q1/p1_vs_random.csv, c == 0.1, metric == '{m}', family == '{f}', k == '{k}'; column {col} "
          "({what}; per subject the k-values are averaged first, then the mean over 70 subjects)")

    def p1row(fam, scale, k, metric="q1"):
        return one(p1, family=fam, scale=scale, c=0.1, k=k, metric=metric)

    def p1vals(fam, k, col, metric="q1"):
        return {s: float(p1row(fam, s, k, metric)[col]) for s in SCALES}

    def pct_range(name, fams, k, where, metric="q1"):
        v = [float(p1row(f, s, k, metric).pct_mean) for f in fams for s in SCALES]
        src = (f"results/revision/q1/p1_vs_random.csv, c == 0.1, metric == '{metric}', k == '{k}', families {fams}, "
               f"all scales; column pct_mean x 100 (mean over subjects of the percentile among the subject's 100 random "
               f"sets); min / max over families x scales")
        return [num(name + "Min", min(v), pc(min(v)), src + " (min)", where),
                num(name + "Max", max(v), pc(max(v)), src + " (max)", where)]

    # ---------------------------------------------------------------- L115: k >= 2 shares
    w = M(115)
    hub = {}
    for fam, nm in [("random", "Random"), ("degree_strength", "Degree"), ("eigenvector_centrality", "Eig"),
                    ("top_avg_ctrl", "AvgCtrl"), ("betweenness", "Betweenness"),
                    ("participation_coefficient", "PartCoef"), ("cycle_participation", "Cycle")]:
        col = "random_mean" if fam == "random" else "value_mean"
        src_fam = "degree_strength" if fam == "random" else fam
        vals = p1vals(src_fam, "avg2+", col)
        nums = per_scale(f"MQOneTwoFive{nm}", vals, pc,
                         P1.format(m="q1", f=src_fam, k="avg2+", col=col,
                                   what="mean q1 of the 100 random sets" if fam == "random" else "mean q1 of the criterion set")
                         + "; x 100", w)
        hub[fam] = nums
        out += nums

    # response L87: "67% to 78%" (hub criteria) and "36% to 39%" (random): reuse the macros holding the min / max
    hubnums = hub["degree_strength"] + hub["eigenvector_centrality"] + hub["top_avg_ctrl"]
    for group in (hubnums, hub["random"]):
        lo = min(group, key=lambda n: n.value)
        hi = max(group, key=lambda n: n.value)
        lo.where.append(R(87))
        hi.where.append(R(87))
        lo.source += "; also the min over the group in response L87"
        hi.source += "; also the max over the group in response L87"

    out += pct_range("MHubPctTwoFive", ["degree_strength", "eigenvector_centrality", "top_avg_ctrl"], "avg2+",
                     [w, R(87)])
    for n in out[-2:]:
        n.source += " (draft rounds pct_mean x 100 to an integer percentile)"

    r = p1row("degree_strength", 114, "5")
    out.append(num("MQOneFiveDegreeOneFourteen", r.value_mean, pc(r.value_mean),
                   P1.format(m="q1", f="degree_strength", k="5", col="value_mean", what="mean q1 of degree sets") + "; scale 114; x 100", w))
    out.append(num("MQOneFiveRandomOneFourteen", r.random_mean, pc(r.random_mean),
                   P1.format(m="q1", f="degree_strength", k="5", col="random_mean", what="mean q1 of random sets") + "; scale 114; x 100", w))

    j = ov[(ov.a == "top_avg_ctrl") & (ov.b == "eigenvector_centrality") & (ov.c.round(6) == 0.1) & ov.k.isin([2, 3, 5])].jaccard
    assert len(j) == 9
    jsrc = ("results/revision/modal/set_overlap.csv, a == 'top_avg_ctrl', b == 'eigenvector_centrality', c == 0.1, "
            "k in {{2,3,5}}, all scales (9 rows); column jaccard (subject-mean Jaccard of the top-k sets); {} over the 9 rows")
    out.append(num("MJaccardAvgCtrlEigMin", j.min(), fx(j.min()), jsrc.format("min"), w))
    out.append(num("MJaccardAvgCtrlEigMax", j.max(), fx(j.max()), jsrc.format("max"), w))

    out += pct_range("MBetweennessPctTwoFive", ["betweenness"], "avg2+", w)
    out += pct_range("MCyclePctTwoFive", ["cycle_participation"], "avg2+", [w, R(88)])
    out += per_scale("MCycleAboveRandomTwoFiveN", p1vals("cycle_participation", "avg2+", "n_positive"), cnt,
                     P1.format(m="q1", f="cycle_participation", k="avg2+", col="n_positive",
                               what="subjects whose k-averaged q1 exceeds the mean of their random sets"), w)

    # modal controllability: the draft prints one "35% at every scale"; 219 rounds to 34
    mod = p1vals("top_modal_gu", "avg2+", "value_mean")
    msrc = (P1.format(m="q1", f="top_modal_gu", k="avg2+", col="value_mean", what="mean q1 of the modal sets")
            + "; x 100. Draft prints a single '35% at every scale'; 219 is 0.3448 -> 34 (top_modal_ct: 0.3449 -> 34), "
              "so the sentence needs three values")
    out += per_scale("MQOneTwoFiveModal", mod, pc, msrc, w, drafts={219: "35"})
    out += pct_range("MModalPctTwoFive", ["top_modal_gu"], "avg2+", w)
    for n in out[-2:]:
        n.source += " (Gu modal controllability, the definition in Methods; top_modal_ct also gives 28 to 33)"

    # ---------------------------------------------------------------- L117: single drivers, excess
    w = M(117)
    out += per_scale("MQOneSingleDegree", p1vals("degree_strength", "1", "value_mean"), pc,
                     P1.format(m="q1", f="degree_strength", k="1", col="value_mean", what="mean q1 of the single strongest region") + "; x 100", w)
    out += per_scale("MQOneSingleRandom", p1vals("degree_strength", "1", "random_mean"), pc,
                     P1.format(m="q1", f="degree_strength", k="1", col="random_mean",
                               what="mean q1 of the 100 random single regions") + "; x 100", w,
                     drafts={114: "96"})
    below = {s: float(p1row("degree_strength", s, "1").n - p1row("degree_strength", s, "1").n_positive) for s in SCALES}
    vb = min(below.values())
    out.append(num("MSingleDegreeBelowRandomN", vb, cnt(vb),
                   P1.format(m="q1", f="degree_strength", k="1", col="n - n_positive",
                             what="subjects whose single hub's q1 is not above the random mean") + "; min over scales (draft: 'all 70 ... at every scale')", w))
    out += per_scale("MExcessTwoDegreePoints", p1vals("degree_strength", "2", "mean"), pc,
                     P1.format(m="q1", f="degree_strength", k="2", col="mean",
                               what="mean over subjects of degree q1 minus the subject's random-set mean") + "; x 100 = percentage points",
                     w, drafts={114: "28"})
    out += per_scale("MExcessTwoDegreeN", p1vals("degree_strength", "2", "n_positive"), cnt,
                     P1.format(m="q1", f="degree_strength", k="2", col="n_positive", what="subjects with positive excess"), w)
    out += per_scale("MExcessFiveDegreePoints", p1vals("degree_strength", "5", "mean"), pc,
                     P1.format(m="q1", f="degree_strength", k="5", col="mean",
                               what="mean over subjects of degree q1 minus the subject's random-set mean") + "; x 100 = percentage points", w)
    n5 = min(p1vals("degree_strength", "5", "n_positive").values())
    out.append(num("MExcessFiveDegreeN", n5, cnt(n5),
                   P1.format(m="q1", f="degree_strength", k="5", col="n_positive", what="subjects with positive excess")
                   + "; min over scales (draft: 'all 70 subjects')", w))

    # ---------------------------------------------------------------- L119: effective rank, PR, q1 points
    w = M(119)
    for fam, nm in [("degree_strength", "Degree"), ("eigenvector_centrality", "Eig")]:
        vals = {s: -v for s, v in p1vals(fam, "avg2+", "mean", "er").items()}
        out += per_scale(f"MErDeficitTwoFive{nm}", vals, fx,
                         P1.format(m="er", f=fam, k="avg2+", col="mean",
                                   what="criterion effective rank minus random-set mean") + "; sign flipped (draft: 'lower')", w)
        out += pct_range(f"MErPctTwoFive{nm}", [fam], "avg2+", w, metric="er")

    def p2vals(comp, metric, col, k="avg"):
        return {s: float(one(p2, comparison=comp, scale=s, c=0.1, k=k, metric=metric)[col]) for s in SCALES}

    P2 = ("results/revision/q1/p2_matched.csv, comparison == '{cmp}', c == 0.1, k == 'avg' (mean over k in 1,2,3,5 "
          "within subject, n_k from file), metric == '{m}'; column {col} ({what})")
    out += per_scale("MErCycleMinusDegreeMean", p2vals("cycle - degree", "er", "mean"), fx,
                     P2.format(cmp="cycle - degree", m="er", col="mean", what="mean over subjects of cycle minus degree effective rank"), w)

    T0 = ("results/revision/spectra/tables/t0_cycle_minus_degree.csv, metric == 'effective_rank'; column {col} "
          "(cycle-minus-degree effective rank at c = 0.1, k-averaged over 1,2,3,5; {what})")
    t0v = lambda col: {s: float(one(t0, metric="effective_rank", scale=s)[col]) for s in SCALES}  # noqa: E731
    out += per_scale("MErCycleMinusDegreeMedian", t0v("median"), fx, T0.format(col="median", what="median over subjects"), w)
    out += per_scale("MErCycleMinusDegreeCILo", t0v("ci_lo"), fx,
                     T0.format(col="ci_lo", what="bootstrap 95% CI of the median, lower end")
                     + "; regime/task1_csweep_cycle_minus_degree.csv (c == 0.1, median_ci_low) has a separate bootstrap "
                       "giving 0.71, 0.84, 0.674 -- also 0.67 at 219", w, drafts={219: "0.68"})
    out += per_scale("MErCycleMinusDegreeCIHi", t0v("ci_hi"), fx, T0.format(col="ci_hi", what="bootstrap 95% CI of the median, upper end"), w)
    out += per_scale("MPrCycleMinusDegreeMean", p2vals("cycle - degree", "pr", "mean"), fx,
                     P2.format(cmp="cycle - degree", m="pr", col="mean", what="mean over subjects of cycle minus degree participation ratio"), w)
    out += per_scale("MQOneCycleBelowDegreePoints", {s: -v for s, v in p2vals("cycle - degree", "q1", "mean").items()}, pc,
                     P2.format(cmp="cycle - degree", m="q1", col="mean", what="mean over subjects of cycle minus degree q1")
                     + "; sign flipped, x 100 = percentage points", w)
    lower = {s: p2vals("cycle - degree", "q1", "n")[s] - p2vals("cycle - degree", "q1", "n_positive")[s] for s in SCALES}
    out += per_scale("MQOneCycleBelowDegreeN", lower, cnt,
                     P2.format(cmp="cycle - degree", m="q1", col="n - n_positive",
                               what="subjects whose cycle q1 is not above degree q1 (draft: 'lower'; no ties expected)"), w)

    # ---------------------------------------------------------------- L121: matched controls
    w = M(121)
    out += per_scale("MQOneCycleMinusMatched", p2vals("cycle - degree_matched", "q1", "mean"), sg,
                     P2.format(cmp="cycle - degree_matched", m="q1", col="mean",
                               what="mean over subjects of cycle q1 minus the mean of its 100 strength-matched sets"), w)
    out += per_scale("MErCycleMinusMatched", p2vals("cycle - degree_matched", "er", "mean"), sg,
                     P2.format(cmp="cycle - degree_matched", m="er", col="mean",
                               what="mean over subjects of cycle ER minus the mean of its strength-matched sets"), w)
    out += per_scale("MErCycleAboveMatchedN", p2vals("cycle - degree_matched", "er", "n_positive"), cnt,
                     P2.format(cmp="cycle - degree_matched", m="er", col="n_positive", what="subjects with cycle ER higher"), w)
    out += per_scale("MQOneResidMinusRandom", p2vals("cycle_resid - random", "q1", "mean"), lambda x: sg(x, 3),
                     P2.format(cmp="cycle_resid - random", m="q1", col="mean",
                               what="mean over subjects of degree-residualized cycle q1 minus random-set mean"), w)
    out += per_scale("MErResidMinusRandom", p2vals("cycle_resid - random", "er", "mean"), sg,
                     P2.format(cmp="cycle_resid - random", m="er", col="mean",
                               what="mean over subjects of degree-residualized cycle ER minus random-set mean"), w)
    return out


# ---- section C

# ----------------------------------------------------------------------------- worker C
# main.tex L133-L139 (leading mode), the hand-typed Table tab:perp (L149-L151), L197
# (driver overlap / coverage); repeats at L168, L172, response L93, L128.

_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

_HUB_FAMS = ["degree_strength", "eigenvector_centrality", "cycle_participation", "random"]


@lru_cache(maxsize=None)
def _set_q1(path: str) -> pd.DataFrame:
    cols = ["subject_id", "scale", "c", "k", "family", "q1", "q1_perp", "loading", "perron_share"]
    return pd.read_csv(path, usecols=cols)


@lru_cache(maxsize=None)
def _set_metrics(path: str) -> pd.DataFrame:
    cols = ["subject_id", "scale", "k", "family", "log10_lambda_max", "lambda_min_equals_eps"]
    return pd.read_csv(path, usecols=cols)


def _kavg(d: pd.DataFrame, col: str, ks) -> pd.DataFrame:
    """Mean within (family, scale, subject, k) [over random/matched draws], then over k, then over subjects.
    Returns family x scale."""
    d = d[d.k.isin(ks)]
    return (d.groupby(["family", "scale", "subject_id", "k"])[col].mean()
            .groupby(["family", "scale", "subject_id"]).mean()
            .groupby(["family", "scale"]).mean().unstack())


def sec_C(res: Path) -> list[Number]:
    out: list[Number] = []
    SQ1 = "revision/q1/set_q1.csv.gz"

    # --- L133: strength vs v1 (mechanism/hub_loading.csv.gz, real connectomes) ---------------------
    hl = pd.read_csv(res / "mechanism" / "hub_loading.csv.gz")
    real = hl[hl.null == "real"].groupby("scale")
    src = ("revision/mechanism/hub_loading.csv.gz, null == 'real' (70 subjects per scale), column {col}, "
           "mean over subjects (= real_mean in mechanism/hub_loading_summary.csv)")
    cos = real.cos_s_v1.mean()
    out += per_scale("MCosStrengthVOne", cos, fx, src.format(col="cos_s_v1 (cosine of s and v1)"), M(133))
    rsm = real.rho_logs_logm.mean()
    out += per_scale("MRhoStrengthNeighbourLoading", rsm, fx,
                     src.format(col="rho_logs_logm (within-subject Spearman rho across regions of s_i and m_i; "
                                    "computed on log s, log m, which Spearman ignores)"), M(133))
    cap = real.L_deg_5.mean() / real.L_max_5.mean()
    out += per_scale("MTopFiveLoadingCapture", cap, pc,
                     "revision/mechanism/hub_loading.csv.gz, null == 'real': mean(L_deg_5) / mean(L_max_5) over subjects "
                     "(ratio of means, as in supp table S7 hub loading; mean of per-subject ratios: 88.7/90.9/78.2 %); "
                     "L_deg_5 = L_S of the 5 strongest regions, L_max_5 = L_S of the 5 regions with largest v1(i)^2. "
                     "MISMATCH at 68: 0.8852 -> 89 % (draft 88 %, probably from the ratio of the rounded means "
                     "0.653/0.738); either definition gives 89", M(133), drafts={68: "88"})

    # --- L135: L_S, leading-mode share, q1 (rescued a1.py) ----------------------------------------------
    sq = _set_q1(str((res / "q1" / "set_q1.csv.gz").resolve()))
    c1 = sq[(sq.c == 0.1) & sq.family.isin(_HUB_FAMS)]
    agg = ("mean over draws within (family, scale, subject, k), then over k in {ks}, then over 70 subjects "
           "(scripts/revision/paper/_rescued_2026-10-19/a1.py)")
    load = _kavg(c1, "loading", [1, 2, 3, 5])
    src = f"{SQ1}, c == 0.1, family == '{{fam}}', column loading (L_S; c-independent); " + agg.format(ks="{{1,2,3,5}}")
    out += per_scale("MLoadingDegree", load.loc["degree_strength"], fx, src.format(fam="degree_strength"), M(135))
    out += per_scale("MLoadingEigenvector", load.loc["eigenvector_centrality"], fx,
                     src.format(fam="eigenvector_centrality"), M(135))
    out += per_scale("MLoadingCycle", load.loc["cycle_participation"], fx,
                     src.format(fam="cycle_participation")
                     + ". MISMATCH at 68: saved 0.1247 rounds to 0.12 (draft 0.13; likely double rounding via 0.125). "
                       "spectra/set_metrics.csv perron_loading gives the same 0.1247",
                     M(135), drafts={68: "0.13"})
    out += per_scale("MLoadingRandom", load.loc["random"], fx, src.format(fam="random"), M(135))

    share = _kavg(c1, "perron_share", [2, 3, 5])
    q1 = _kavg(c1, "q1", [2, 3, 5])
    src = (f"{SQ1}, c == 0.1, family == '{{fam}}', column {{col}}; " + agg.format(ks="{{2,3,5}}"))
    out += per_scale("MLeadShareDegree", share.loc["degree_strength"], fx,
                     src.format(fam="degree_strength", col="perron_share (v1' W v1 / tr W)"), M(135))
    out += per_scale("MLeadShareRandom", share.loc["random"], fx,
                     src.format(fam="random", col="perron_share (v1' W v1 / tr W)"), M(135))
    out += per_scale("MLeadModeQOneDegree", q1.loc["degree_strength"], fx,
                     src.format(fam="degree_strength", col="q1 (raw W)")
                     + "; same quantity as the 73/74/67 % at main.tex:115 (q1/p1_vs_random.csv value_mean, k == 'avg2+')",
                     M(135))
    out += per_scale("MLeadModeQOneRandom", q1.loc["random"], fx,
                     src.format(fam="random", col="q1 (raw W)")
                     + "; same quantity as the 39/37/36 % at main.tex:115", M(135))

    # L_S - q1 within-subject Spearman (rescued a2.py, 'all' families, per-k then averaged)
    x = sq[sq.k.isin([2, 3, 5])]
    for c, word, ln in ((0.01, "WeakDamping", 135), (0.1, "PrimaryDamping", 135)):
        y = x[x.c == c]
        rho = (y.groupby(["scale", "subject_id", "k"])[["loading", "q1"]]
               .apply(lambda g: g.loading.corr(g.q1, method="spearman"))
               .groupby(["scale", "subject_id"]).mean().groupby("scale").median())
        out += per_scale(
            f"MRhoLoadingQOne{word}", rho, fx,
            f"{SQ1}, c == {c}, k in {{2,3,5}}, ALL 18 families (14 criteria + random + degree-, spatially and "
            "degree+spatially matched sets, i.e. ~414 sets per subject and k): within-subject Spearman rho(loading, q1) "
            "per k, mean over k, median over 70 subjects (rescued a2.py, 'all' / 'perk-avg'). Matches the text "
            "('all criterion, random, and matched sets'). NOTE: supp table S7_4 (tab:S7_LS_q1) does not contain this "
            "pooling (its rows (d) are per-k medians, not the k-average), and supp table S7_comm_geo / "
            "q1/redundancy_summary.csv rho_L use 114 sets per subject (matched sets excluded; e.g. 0.75/0.59/0.35 at "
            "c = 0.01), so the main-text rho is not reproducible from any supplement table", M(ln))

    # --- L137 / R128: condition numbers ---------------------------------------------------------------
    sm = _set_metrics(str((res / "spectra" / "set_metrics.csv").resolve()))
    assert bool(sm.lambda_min_equals_eps.all())
    out.append(num("MNumSetsLambdaMinFloor", len(sm), f"{len(sm):,}",
                   "revision/spectra/set_metrics.csv: number of rows (every criterion, random and matched set, "
                   "k in {1,2,3,5}, c = 0.1, 3 scales x 70 subjects); lambda_min_equals_eps is True in all rows",
                   [M(137), R(128)]))
    lm = _kavg(sm[sm.family.isin(_HUB_FAMS)], "log10_lambda_max", [1, 2, 3, 5])
    src = ("revision/spectra/set_metrics.csv, family == '{fam}', column log10_lambda_max (log10 lambda_max(W_eps), "
           "c = 0.1); mean over draws within (subject, k), then over k in {{1,2,3,5}}, then over subjects")
    out += per_scale("MLogLambdaMaxDegree", lm.loc["degree_strength"], fx, src.format(fam="degree_strength"), M(137))
    out += per_scale("MLogLambdaMaxCycle", lm.loc["cycle_participation"], fx, src.format(fam="cycle_participation"),
                     M(137))
    out += per_scale("MLogLambdaMaxRandom", lm.loc["random"], fx, src.format(fam="random"), M(137))
    r_dr = 10 ** (lm.loc["degree_strength"] - lm.loc["random"])
    r_dc = 10 ** (lm.loc["degree_strength"] - lm.loc["cycle_participation"])
    ratio_src = ("10**(difference of the k- and subject-averaged log10 lambda_max(W_eps) above, i.e. a ratio of "
                 "geometric means), revision/spectra/set_metrics.csv; per scale {vals}; printed as a spelled-out "
                 "integer ('about N times'); value = {what}")
    v = float(r_dr.mean())
    out.append(num("MLambdaMaxRatioDegreeRandom", v, _WORDS[round(v)],
                   ratio_src.format(vals=", ".join(f"{s}: {r_dr[s]:.2f}" for s in SCALES),
                                    what="mean over scales; degree / random"), [M(137), R(128)]))
    v = float(r_dc.min())
    out.append(num("MLambdaMaxRatioDegreeCycleLow", v, _WORDS[round(v)],
                   ratio_src.format(vals=", ".join(f"{s}: {r_dc[s]:.2f}" for s in SCALES),
                                    what="min over scales; degree / cycle participation"), [M(137), R(128)]))
    v = float(r_dc.max())
    out.append(num("MLambdaMaxRatioDegreeCycleHigh", v, _WORDS[round(v)],
                   ratio_src.format(vals=", ".join(f"{s}: {r_dc[s]:.2f}" for s in SCALES),
                                    what="max over scales; degree / cycle participation (219: 4.3, so 'three to four' "
                                         "is at the edge)"), [M(137), R(128)]))

    # --- L139 + Table tab:perp (L149-L151) + L168/L172 + R93: projection of v1 ------------------------
    p3 = csv(res, "q1/p3_perp.csv")
    p3 = p3[np.isclose(p3.c, 0.1) & (p3.k == "avg")]
    src = ("revision/q1/p3_perp.csv, c == 0.1, k == 'avg' (cycle - degree, mean over subjects of the per-subject "
           "mean over k in {{1,2,3,5}}), metric == '{m}', column {col}")
    rows = {m: {s: one(p3, scale=s, metric=m) for s in SCALES} for m in ("q1", "er", "pr")}
    MN = {"q1": "QOne", "er": "EffRank", "pr": "PartRatio"}
    for m, d in (("q1", 2), ("er", 2), ("pr", 2)):
        full = {s: rows[m][s].full_mean for s in SCALES}
        perp = {s: rows[m][s].perp_mean for s in SCALES}
        frac = {s: rows[m][s].fraction_removed for s in SCALES}
        w_full = [M(150 if m == "er" else 149 if m == "q1" else 151)]
        if m == "q1":
            w_full.append(M(168))
        if m == "er":
            w_full.append(M(172))
        out += per_scale(f"MCycleMinusDegree{MN[m]}Full", full, lambda v, d=d: sg(v, d),
                         src.format(m=m, col="full_mean") + "; same as q1/p2_matched.csv comparison 'cycle - degree'",
                         w_full)
        out += per_scale(f"MCycleMinusDegree{MN[m]}Perp", perp, lambda v, d=d: sg(v, d),
                         src.format(m=m, col="perp_mean (W_perp = P W P, P = I - v1 v1')"),
                         [M(150 if m == "er" else 149 if m == "q1" else 151)])
        w_frac = {"q1": [M(139), M(149)], "er": [M(150)], "pr": [M(139), M(151)]}[m]
        nums_frac = per_scale(f"MPerpFractionRemoved{MN[m]}", frac, pc,
                              src.format(m=m, col="fraction_removed (1 - perp_mean / full_mean)"), w_frac)
        if m == "q1":  # R93 quotes the range '59% to 75%' = 68 and 219 values (= min and max)
            assert frac[68] == min(frac.values()) and frac[219] == max(frac.values())
            nums_frac[0].where.append(R(93))
            nums_frac[2].where.append(R(93))
            nums_frac[0].source += "; R93 '59% to 75%' = min (68) to max (219) over scales"
            nums_frac[2].source += "; R93 '59% to 75%' = min (68) to max (219) over scales"
        if m == "er":  # L139 quotes only 68 ('reduced by 86% at 68 regions')
            nums_frac[0].where.append(M(139))
        out += nums_frac
    nless = {s: 70 - rows["q1"][s].perp_n_positive for s in SCALES}
    out += per_scale("MPerpCycleLessConcentratedCount", nless, cnt,
                     src.format(m="q1", col="70 - perp_n_positive") + " (subjects with cycle - degree q1(W_perp) < 0; "
                     "no ties)", M(139))

    # degree - random in q1(W_perp), k >= 2 (rescued a1.py)
    y = sq[(sq.c == 0.1) & sq.family.isin(["degree_strength", "random"]) & sq.k.isin([2, 3, 5])]
    s = y.groupby(["family", "scale", "subject_id", "k"]).q1_perp.mean()
    diff = (s.xs("degree_strength", level="family") - s.xs("random", level="family")).groupby(["scale", "subject_id"]).mean()
    out += per_scale("MPerpDegreeExcessQOne", diff.groupby("scale").mean(), fx,
                     f"{SQ1}, c == 0.1, k in {{2,3,5}}, column q1_perp: degree_strength minus the mean of the subject's 100 "
                     "random sets, per k, mean over k, mean over subjects (rescued a1.py)", M(139))
    npos = (diff > 0).groupby("scale").sum()
    out.append(num("MPerpDegreeExcessNPositive", npos.min(), cnt(npos.min()),
                   f"{SQ1}, as MPerpDegreeExcessQOne: subjects with a positive difference; min over scales "
                   f"({', '.join(f'{k}: {int(v)}' for k, v in npos.items())}); text 'in all 70 subjects'", M(139)))

    # --- L197: driver-Gram overlap vs shortest-path distance (q1/redundancy_summary.csv) -------------
    rs = csv(res, "q1/redundancy_summary.csv")
    src = ("revision/q1/redundancy_summary.csv, c == {c}, column '{col}': within-subject partial Spearman rho with q1 "
           "given L_S across each subject's 114 sets (14 criteria + 100 random; matched sets excluded), "
           "k in {{2,3,5}} averaged within subject, median over 70 subjects (scripts/revision/q1/04_redundancy_geodesic.py)")
    for c, tag in ((0.1, "PrimaryDamping"), (1.0, "StrongDamping")):
        vals = {s: one(rs, scale=s, c=c)["prho_comm|L"] for s in SCALES}
        out += per_scale(f"MPartialRhoOverlap{tag}", vals, fx, src.format(c=c, col="prho_comm|L"), M(197))
    vals = {s: one(rs, scale=s, c=0.1)["prho_geo_inv|L"] for s in SCALES}
    out += per_scale("MPartialRhoPathDistance", vals, fx, src.format(c=0.1, col="prho_geo_inv|L (edge length 1/w)"),
                     M(197))
    nb = rs[rs.c >= 0.1 - 1e-12]["n_comm_better_geo_inv_given_L"]
    nb_src = ("revision/q1/redundancy_summary.csv, c in {{0.1, 1}}, column n_comm_better_geo_inv_given_L (subjects with "
              "|prho_comm|L| > |prho_geo_inv|L|); {what} over scales and c")
    out.append(num("MOverlapBetterCountLow", nb.min(), cnt(nb.min()), nb_src.format(what="min"), M(197)))
    out.append(num("MOverlapBetterCountHigh", nb.max(), cnt(nb.max()), nb_src.format(what="max"), M(197)))

    cr = csv(res, "mechanism/coverage_rho.csv")
    rk = cr[cr.k > 1].groupby(["scale", "subject_id", "c"]).mean(numeric_only=True).groupby(["scale", "c"])["prho_inv|L"].median()
    cov_src = ("revision/mechanism/coverage_rho.csv, k in {{2,3,5}}, column 'prho_inv|L' (within-subject partial Spearman "
               "rho of coverage distance (mean distance to nearest driver, edge length 1/w) with q1 given L_S); mean over k "
               "per subject, median over subjects (as in scripts/revision/mechanism/03_coverage_distance.py); {what} over "
               "scales and c in {{0.01, 0.1, 1}}")
    out.append(num("MPartialRhoCoverageLow", rk.min(), fx(rk.min()), cov_src.format(what="min"), M(197)))
    out.append(num("MPartialRhoCoverageHigh", rk.max(), fx(rk.max()), cov_src.format(what="max"), M(197)))
    return out


# ---- section D

WORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def sec_D(res: Path) -> list:
    """Damping and spectral gap: main.tex L168, L170, L172, L259; response L101, L105, L106, L114."""
    out = []
    p1 = csv(res, "q1/p1_vs_random.csv")
    p2 = csv(res, "q1/p2_matched.csv")

    # ---- L168: excess q1 of degree sets over random, k-averaged (k in 1,2,3,5), per c
    def dr(scale, c):
        return one(p1, family="degree_strength", scale=scale, c=c, k="avg", metric="q1")

    src_dr = ("q1/p1_vs_random.csv: family == 'degree_strength', metric == 'q1', k == 'avg' (mean over k in 1,2,3,5 of "
              "criterion q1 minus mean of the subject's 100 random sets), c == {c}; column 'mean' (mean over 70 subjects)")
    for c, word in ((0.01, "Hundredth"), (0.1, "Tenth"), (1.0, "One")):
        out += per_scale(f"MDegExcessQOneC{word}", {s: dr(s, c)["mean"] for s in SCALES}, lambda v: fx(v, 2),
                         src_dr.format(c=c), M(168))
    out += per_scale("MDegExcessQOneCOneNPos", {s: dr(s, 1.0)["n_positive"] for s in SCALES}, cnt,
                     "q1/p1_vs_random.csv: family == 'degree_strength', metric == 'q1', k == 'avg', c == 1; column "
                     "'n_positive' (subjects with degree q1 above their random mean)", M(168))

    # cycle - degree q1 at c = 1 (the c = 0.1 values are worker C's)
    def cd(scale, c, metric):
        return one(p2, comparison="cycle - degree", scale=scale, c=c, k="avg", metric=metric)

    src_cd = ("q1/p2_matched.csv: comparison == 'cycle - degree', metric == '{m}', k == 'avg' (mean over k in 1,2,3,5), "
              "c == {c}; column '{col}'")
    out += per_scale("MCycMinusDegQOneCOne", {s: cd(s, 1.0, "q1")["mean"] for s in SCALES}, lambda v: fx(v, 2),
                     src_cd.format(m="q1", c=1.0, col="mean"), M(168))
    out += per_scale("MCycLessConcCOneN", {s: cd(s, 1.0, "q1")["n"] - cd(s, 1.0, "q1")["n_positive"] for s in SCALES}, cnt,
                     src_cd.format(m="q1", c=1.0, col="n - n_positive") + " (subjects with cycle q1 below degree q1; "
                     "no exact ties in continuous q1)", M(168))

    # sign changes of the cycle - degree q1 curve on the 25-value c grid (up to c = 10^0.5)
    fl = csv(res, "q1/decay_flip_summary.csv")
    row = one(fl, curve="cd_q1", axis="flip_c")
    out += per_scale("MCycDegQOneSignFlipN", {s: row[f"n_flip_{s}"] for s in SCALES}, cnt,
                     "q1/decay_flip_summary.csv: curve == 'cd_q1', axis == 'flip_c'; columns n_flip_<scale> (subjects whose "
                     "per-subject k-averaged cycle - degree q1 curve over the 25 log-spaced c in [10^-2.5, 10^0.5] "
                     "changes sign; per-subject flips in q1/decay_flips.csv)", [M(168), R(105)])

    # "about four to eight times smaller" between c = 0.1 and c = 1
    ratios = []
    for s in SCALES:
        ratios.append(dr(s, 0.1)["mean"] / dr(s, 1.0)["mean"])
        ratios.append(cd(s, 0.1, "q1")["mean"] / cd(s, 1.0, "q1")["mean"])
    lo, hi = min(ratios), max(ratios)
    src_ratio = ("ratio of the c = 0.1 to the c = 1 effect (degree - random q1 from q1/p1_vs_random.csv and cycle - degree q1 "
                 "from q1/p2_matched.csv, k == 'avg', column 'mean') over both effects and 3 scales; {} = {:.2f}, printed "
                 "as the spelled-out rounded integer. Ratios: " + ", ".join(f"{r:.2f}" for r in ratios))
    out.append(num("MDampingShrinkMin", lo, WORD[round(lo)], src_ratio.format("min", lo), [M(168), R(105)]))
    out.append(num("MDampingShrinkMax", hi, WORD[round(hi)], src_ratio.format("max", hi), [M(168), R(105)]))

    # ---- L170: gap range, gap correlations
    flips = csv(res, "q1/decay_flips.csv")
    gap = flips[flips.curve == "cd_q1"].groupby("scale").gap
    src_gap = ("q1/decay_flips.csv: curve == 'cd_q1' (one row per subject); column 'gap' = l1 - l2 of A0; {} over the 70 "
               "subjects")
    out += per_scale("MGapMin", gap.min().to_dict(), lambda v: fx(v, 2), src_gap.format("min"), M(170))
    out += per_scale("MGapMax", gap.max().to_dict(), lambda v: fx(v, 2), src_gap.format("max"), M(170))

    gc = csv(res, "q1/decay_gap_correlations.csv")
    gc1 = gc[np.isclose(gc.c, 0.1)]
    src_gc = ("q1/decay_gap_correlations.csv: c == 0.1, curve == '{}'; column 'rho_gap' (Spearman across 70 subjects between "
              "the gap and the k-averaged {} )")
    rho = {s: one(gc1, scale=s, curve="dr_q1")["rho_gap"] for s in SCALES}
    out += per_scale("MGapRhoDegExcess", rho, lambda v: fx(v, 2),
                     src_gc.format("dr_q1", "degree - random q1"), [M(170), R(106)])
    pmax = gc1[gc1.curve == "dr_q1"].p.max()
    out.append(num("MGapRhoDegExcessPBound", pmax, "10^{-5}",
                   f"q1/decay_gap_correlations.csv: c == 0.1, curve == 'dr_q1'; max p over scales = {pmax:.2g}; the text "
                   "states the bound 'all p < 10^{-5}' (macro is the bound, verified)", M(170)))
    assert pmax < 1e-5
    out += per_scale("MGapRhoCycDeg", {s: one(gc1, scale=s, curve="cd_q1")["rho_gap"] for s in SCALES},
                     lambda v: fx(v, 2), src_gc.format("cd_q1", "cycle - degree q1"), M(170))

    pc_ = csv(res, "q1/decay_gap_partial_correlations.csv")
    full = pc_[pc_.controls == "strength+density+cv"]
    full1 = full[np.isclose(full.c, 0.1)]
    src_pc = ("q1/decay_gap_partial_correlations.csv: c == 0.1, controls == 'strength+density+cv', curve == '{}'; column "
              "'partial_rho_gap' (rank-residualized partial Spearman across subjects)")
    out += per_scale("MGapPartialRhoDegExcess", {s: one(full1, scale=s, curve="dr_q1")["partial_rho_gap"] for s in SCALES},
                     lambda v: fx(v, 2), src_pc.format("dr_q1"), M(170))
    out += per_scale("MGapPartialRhoCycDeg", {s: one(full1, scale=s, curve="cd_q1")["partial_rho_gap"] for s in SCALES},
                     lambda v: fx(v, 2), src_pc.format("cd_q1"), M(170))
    pmax = full1[full1.curve.isin(["dr_q1", "cd_q1"])].p.max()
    assert pmax < 3e-4
    out.append(num("MGapPartialPBound", pmax, "3 \\times 10^{-4}",
                   f"q1/decay_gap_partial_correlations.csv: c == 0.1, controls == 'strength+density+cv', curves dr_q1 and "
                   f"cd_q1; max p = {pmax:.2g}; the text states the bound 'all p < 3 x 10^-4' (macro is the bound, verified)",
                   M(170)))
    rng = full[(full.curve == "dr_q1") & (full.c >= 10 ** -1.5 - 1e-9) & (full.c <= 1 + 1e-9)]
    src_rng = ("q1/decay_gap_partial_correlations.csv: controls == 'strength+density+cv', curve == 'dr_q1', the 13 grid values "
               "10^-1.5 <= c <= 1, all scales; {}")
    out.append(num("MGapPartialRhoRangeMin", rng.partial_rho_gap.min(), fx(rng.partial_rho_gap.min(), 2),
                   src_rng.format("min partial_rho_gap"), M(170)))
    out.append(num("MGapPartialRhoRangeMax", rng.partial_rho_gap.max(), fx(rng.partial_rho_gap.max(), 2),
                   src_rng.format("max partial_rho_gap"), M(170)))
    out.append(num("MGapPartialRhoRangePMax", rng.p.max(), fx(math.ceil(rng.p.max() * 1000) / 1000, 3),
                   src_rng.format(f"max p = {rng.p.max():.4f}; the text states the bound 'all p <= 0.002', so the macro "
                                  "is the max p rounded up to 3 decimals"), M(170)))

    # ---- L172: cycle - degree effective rank across c; flip points; ER vs average controllability
    er01 = {s: cd(s, 0.01, "er")["mean"] for s in SCALES}
    out += per_scale("MCycMinusDegERCHundredth", er01, lambda v: sg(v, 2),
                     src_cd.format(m="er", c=0.01, col="mean"), M(172))
    er1 = {s: cd(s, 1.0, "er")["mean"] for s in SCALES}
    out += per_scale("MCycMinusDegERCOne", er1, lambda v: fx(v, 2),
                     src_cd.format(m="er", c=1.0, col="mean") + ". Response L114 quotes these infinite-horizon values for "
                     "'T >= 2.5 give the same result'; regime/task4_finite_horizon_summary.csv (eps 1e-5, c 1, T 2.5) gives "
                     "-0.02, -0.05, -0.23 and T 5 gives -0.02, -0.05, -0.24", [M(172), R(114)])
    out += per_scale("MCycMinusDegERCOneNPos", {s: cd(s, 1.0, "er")["n_positive"] for s in SCALES}, cnt,
                     src_cd.format(m="er", c=1.0, col="n_positive"), M(172))
    sf = csv(res, "regime/task1_signflip.csv")
    out += per_scale("MCycMinusDegERFlipC",
                     {s: one(sf, metric="effective_rank", scale=s)["c_flip_mean"] for s in SCALES}, lambda v: fx(v, 2),
                     "regime/task1_signflip.csv: metric == 'effective_rank'; column c_flip_mean (zero crossing, interpolated "
                     "in c, of the cohort-mean cycle - degree effective-rank curve over the 13-value c sweep)",
                     [M(172), R(105)])
    ws = csv(res, "modal/within_subject_summary.csv")
    x = ws[(ws.variable == "rho_effective_rank_avg_ctrl") & np.isclose(ws.c, 1.0) & ws.k.isin([2, 3, 5])]
    assert len(x) == 9
    src_ws = ("modal/within_subject_summary.csv: variable == 'rho_effective_rank_avg_ctrl', c == 1, k in 2,3,5, 3 scales; "
              "column 'median' (median over subjects of the within-subject Spearman rho across sets between effective rank "
              "and the drivers' mean average controllability); {} over the 9 scale x k cells")
    out.append(num("MERAvgCtrlRhoMin", x["median"].min(), fx(x["median"].min(), 2), src_ws.format("min"), M(172)))
    out.append(num("MERAvgCtrlRhoMax", x["median"].max(), fx(x["median"].max(), 2), src_ws.format("max"), M(172)))

    # ---- L259: closed-form Gramian vs direct Lyapunov solver
    v = json.loads((res / "regime/validation.json").read_text())["library_vs_eigbasis"]
    worst = max(v["max_abs_W"], v["rel_W"])
    bound = 10.0 ** math.ceil(math.log10(worst))
    out.append(num("MLyapunovAgreement", worst, pow10(bound),
                   f"regime/validation.json: library_vs_eigbasis max_abs_W = {v['max_abs_W']:.2g}, rel_W = {v['rel_W']:.2g}; "
                   "text gives the next power of ten as the bound", M(259)))

    # ---- R101: nctpy equivalence
    eq = csv(res, "regime/nctpy_equivalence_detail.csv")
    out.append(num("MNctpyEquivMaxDiff", eq.max_abs_W_diff.max(), pow10(eq.max_abs_W_diff.max(), 0),
                   f"regime/nctpy_equivalence_detail.csv: max of max_abs_W_diff over all rows = {eq.max_abs_W_diff.max():.3g} "
                   f"(rounded to one significant figure; 'within' is loose: max rel_W_diff = {eq.rel_W_diff.max():.2g})",
                   R(101)))
    out.append(num("MNctpyEquivNCombos", len(eq), str(len(eq)),
                   "regime/nctpy_equivalence_detail.csv: number of rows (subject x criterion x k x scale combinations)",
                   R(101)))
    return out


# ---- section E

NULLS = ("strength", "geometric", "weights")


def sec_E(res: Path) -> list[Number]:
    """Null networks and model networks (main.tex L184-L190) and the null/Rayleigh methods (L245, L353, L355, L369)."""
    out: list[Number] = []

    # ---------------------------------------------------------------- L184: canonical model networks
    cs = csv(res, "mechanism/canonical_summary.csv")
    w = cs[(cs.weighting == "weighted") & np.isclose(cs.c, 0.1) & (cs.kgrp == "2+")]
    mod, conn = w[w.model != "CONN"], w[w.model == "CONN"]
    src = ("revision/mechanism/canonical_summary.csv (weighting == 'weighted', c == 0.1, kgrp == '2+' i.e. mean over k in {{2,3,5}} "
           "and the graphs of each model); {what}")
    out += [
        num("MModelExcessQOneMin", mod.deg_minus_rand_q1.min(), fx(mod.deg_minus_rand_q1.min()),
            src.format(what="min of deg_minus_rand_q1 over the 5 models (ER, BA, WS, SBM, RGG) x 3 scales"), M(184)),
        num("MModelExcessQOneMax", mod.deg_minus_rand_q1.max(), fx(mod.deg_minus_rand_q1.max()),
            src.format(what="max of deg_minus_rand_q1 over the 5 models x 3 scales"), M(184)),
        num("MModelExcessPctMin", mod.deg_q1_pct.min(), pc(mod.deg_q1_pct.min()),
            src.format(what="min of deg_q1_pct (percentile of the degree set among random sets, x100) over models x scales"), M(184)),
        num("MModelExcessPctMax", mod.deg_q1_pct.max(), pc(mod.deg_q1_pct.max()),
            src.format(what="max of deg_q1_pct over models x scales"), M(184)),
        num("MConnExcessQOneMin", conn.deg_minus_rand_q1.min(), fx(conn.deg_minus_rand_q1.min()),
            src.format(what="model == 'CONN' (the 69 connected connectomes), min of deg_minus_rand_q1 over scales"), M(184)),
        num("MConnExcessQOneMax", conn.deg_minus_rand_q1.max(), fx(conn.deg_minus_rand_q1.max()),
            src.format(what="model == 'CONN', max of deg_minus_rand_q1 over scales"), M(184)),
        num("MConnExcessPct", conn.deg_q1_pct.min(), pc(conn.deg_q1_pct.min()),
            src.format(what="model == 'CONN', deg_q1_pct, min over scales (range 99.0-99.8; '99th percentile')"), M(184)),
    ]
    g = csv(res, "mechanism/canonical_graphs.csv.gz")
    g = g[(g.weighting == "weighted") & np.isclose(g.c, 0.1) & (g.k >= 2)].assign(ex=lambda x: x.deg_q1 - x.rand_q1)
    per_graph = g.groupby(["model", "scale", "graph"])[["ex", "gap", "deg_L", "strength_cv"]].mean().reset_index()
    for var, name, what in (("gap", "MModelRhoGap", "spectral gap"), ("deg_L", "MModelRhoLoading", "degree-set loading L_S"),
                            ("strength_cv", "MModelRhoStrengthCV", "strength CV (heterogeneity)")):
        vals = {s: spearmanr(x.ex, x[var]).statistic for s, x in per_graph.groupby("scale")}
        out += per_scale(name, vals, fx,
                         "revision/mechanism/canonical_graphs.csv.gz (weighting == 'weighted', c == 0.1, k in {2,3,5}); per graph "
                         f"(all 6 models incl. CONN) mean over k of deg_q1 - rand_q1 and of {var}; Spearman rho across graphs "
                         f"between excess q1 and {what}, per scale (recomputed; quoted in mechanism/REPORT.md, not saved)", M(184))

    # ---------------------------------------------------------------- L186: spectral gap, l1, l2
    ni = csv(res, "nulls/network_inference.csv")
    gap = ni[ni.stat == "gap"]
    out += per_scale("MGapReal", {s: one(gap, null="strength", scale=s).real_mean for s in SCALES}, fx,
                     "revision/nulls/network_inference.csv (stat == 'gap'), real_mean (mean over 70 subjects of l1 - l2 of A0)", M(186))
    out += [
        num("MGapNullMin", gap.null_mean.min(), fx(gap.null_mean.min()),
            "revision/nulls/network_inference.csv (stat == 'gap'), min of null_mean over 3 null models x 3 scales", M(186)),
        num("MGapNullMax", gap.null_mean.max(), fx(gap.null_mean.max()),
            "revision/nulls/network_inference.csv (stat == 'gap'), max of null_mean over 3 null models x 3 scales", M(186)),
    ]
    sg_ = gap[gap.null.isin(["strength", "geometric"])].n_above95
    out += [
        num("MGapAboveNinetyFiveSGMin", sg_.min(), cnt(sg_.min()),
            "revision/nulls/network_inference.csv (stat == 'gap', null in {strength, geometric}), min of n_above95 over nulls x scales", M(186)),
        num("MGapAboveNinetyFiveSGMax", sg_.max(), cnt(sg_.max()),
            "revision/nulls/network_inference.csv (stat == 'gap', null in {strength, geometric}), max of n_above95 over nulls x scales", M(186)),
    ]
    out += per_scale("MGapAboveNinetyFiveWeights", {s: one(gap, null="weights", scale=s).n_above95 for s in SCALES}, cnt,
                     "revision/nulls/network_inference.csv (stat == 'gap', null == 'weights'), n_above95 (subjects above their 95th null pct)",
                     M(186))
    gs = csv(res, "nulls/gap_split.csv")
    l1, l2 = gs[gs.stat == "lambda1"], gs[gs.stat == "lambda2"]
    out += per_scale("MLeadEigReal", {s: one(l1, null="strength", scale=s).real_mean for s in SCALES}, fx,
                     "revision/nulls/gap_split.csv (stat == 'lambda1'), real_mean", M(186))
    out += [
        num("MLeadEigNullMin", l1.null_mean.min(), fx(l1.null_mean.min()),
            "revision/nulls/gap_split.csv (stat == 'lambda1'), min of null_mean over nulls x scales", M(186)),
        num("MLeadEigNullMax", l1.null_mean.max(), fx(l1.null_mean.max()),
            "revision/nulls/gap_split.csv (stat == 'lambda1'), max of null_mean over nulls x scales", M(186)),
    ]
    lo = l2[l2.scale.isin([68, 114])].delta_mean
    hi = l2[l2.scale == 219].delta_mean
    s2 = "revision/nulls/gap_split.csv (stat == 'lambda2'), delta_mean (mean over subjects of real - null mean l2), {}"
    out += [
        num("MSecondEigDiffLowMin", lo.min(), fx(lo.min()), s2.format("min over 3 nulls at 68 and 114"), M(186)),
        num("MSecondEigDiffLowMax", lo.max(), sg(lo.max()), s2.format("max over 3 nulls at 68 and 114"), M(186)),
        num("MSecondEigDiffHighMin", hi.min(), sg(hi.min()), s2.format("min over 3 nulls at 219"), M(186)),
        num("MSecondEigDiffHighMax", hi.max(), sg(hi.max()), s2.format("max over 3 nulls at 219"), M(186)),
    ]

    # ---------------------------------------------------------------- L188: degree-set q1 and cycle-minus-degree ER in nulls
    inf = csv(res, "nulls/inference.csv")
    q = inf[(inf.stat == "degree_q1") & np.isclose(inf.c_ref, 0.1) & (inf.k == "avg")]
    fixed, gapm = q[q.c_mode == "fixed"], q[q.c_mode == "gap"]
    sq = ("revision/nulls/inference.csv (stat == 'degree_q1', c_ref == 0.1, k == 'avg' = mean over k in {{1,2,3,5}}); "
          "NOTE k = avg includes k = 1, unlike the k >= 2 averages of main.tex L115/L135; {}")
    out += per_scale("MNullRealQOne", {s: one(fixed, null="strength", scale=s).real_mean for s in SCALES}, fx,
                     sq.format("real_mean"), M(188))
    out += [
        num("MNullFixedQOneMin", fixed.null_mean.min(), fx(fixed.null_mean.min()), sq.format("c_mode == 'fixed', min null_mean over nulls x scales"), M(188)),
        num("MNullFixedQOneMax", fixed.null_mean.max(), fx(fixed.null_mean.max()), sq.format("c_mode == 'fixed', max null_mean over nulls x scales"), M(188)),
    ]
    pmax = fixed.p_holm.max()
    assert pmax < 1e-3, pmax
    out.append(num("MNullFixedQOneHolmBound", pmax, "10^{-3}",
                   sq.format(f"c_mode == 'fixed', bound on max p_holm over nulls x scales (max = {pmax:.1e}); text is the bound"), M(188)))
    out += [
        num("MNullGapQOneMin", gapm.null_mean.min(), fx(gapm.null_mean.min()), sq.format("c_mode == 'gap', min null_mean over nulls x scales"), M(188)),
        num("MNullGapQOneMax", gapm.null_mean.max(), fx(gapm.null_mean.max()), sq.format("c_mode == 'gap', max null_mean over nulls x scales"), M(188)),
    ]
    r = one(gapm, null="strength", scale=114)
    out += [
        num("MNullGapStrengthQOneOneFourteen", r.null_mean, fx(r.null_mean),
            sq.format("c_mode == 'gap', null == 'strength', scale == 114, null_mean (0.7447; the draft's 0.75 matches neither k = avg nor "
                      "avg2+ (0.708); it is the rounding in nulls/REPORT.md)"), M(188), draft="0.75"),
        num("MNullGapStrengthHolmOneFourteen", r.p_holm, pow10(r.p_holm, 1),
            sq.format("c_mode == 'gap', null == 'strength', scale == 114, p_holm (6.1e-4; the draft's 0.001 is the k == 'avg2+' row, 9.2e-4)"),
            M(188), draft="0.001"),
    ]
    for sc, name in ((68, "SixtyEight"), (219, "TwoNineteen")):
        r = one(gapm, null="geometric", scale=sc)
        out += [
            num(f"MNullGapGeomQOne{name}", r.null_mean, fx(r.null_mean),
                sq.format(f"c_mode == 'gap', null == 'geometric', scale == {sc}, null_mean"), M(188)),
            num(f"MNullGapGeomHolm{name}", r.p_holm, fx(r.p_holm, 3),
                sq.format(f"c_mode == 'gap', null == 'geometric', scale == {sc}, p_holm"), M(188)),
        ]
    er = inf[(inf.stat == "cycle_minus_degree_er") & np.isclose(inf.c_ref, 0.1) & (inf.k == "avg") & (inf.scale == 114)]
    se = "revision/nulls/inference.csv (stat == 'cycle_minus_degree_er', c_ref == 0.1, k == 'avg', scale == 114); {}"
    ef, eg = er[er.c_mode == "fixed"], er[er.c_mode == "gap"]
    out += [
        num("MNullRealCycleMinusDegreeER", ef.real_mean.iloc[0], fx(ef.real_mean.iloc[0]),
            se.format("real_mean (same value as the empirical cycle-minus-degree ER at 114 in main.tex L119)"), M(188)),
        num("MNullFixedCycleMinusDegreeERMin", ef.null_mean.min(), fx(ef.null_mean.min()), se.format("c_mode == 'fixed', min null_mean over nulls"), M(188)),
        num("MNullFixedCycleMinusDegreeERMax", ef.null_mean.max(), fx(ef.null_mean.max()), se.format("c_mode == 'fixed', max null_mean over nulls"), M(188)),
        num("MNullGapCycleMinusDegreeERMin", eg.null_mean.min(), fx(eg.null_mean.min()), se.format("c_mode == 'gap', min null_mean over nulls"), M(188)),
        num("MNullGapCycleMinusDegreeERMax", eg.null_mean.max(), fx(eg.null_mean.max()), se.format("c_mode == 'gap', max null_mean over nulls"), M(188)),
    ]

    # ---------------------------------------------------------------- L190: assortativity, Rayleigh bounds, rich club
    from s8 import inline_values  # scripts/paper/s8.py (same code as tables S8_6 / S8_7)

    iv = inline_values(res)
    s8src = "scripts/paper/s8.py inline_values(res)['{}'] (revision/nulls/rayleigh/networks.csv.gz, per_subject.csv; see its docstring)"
    out += per_scale("MStrengthAssortReal", {s: iv[f"rW_real_{s}"] for s in SCALES}, fx, s8src.format("rW_real_<scale>"), M(190))
    out += [
        num("MStrengthAssortNullMin", iv["rW_null_min"], fx(iv["rW_null_min"]),
            s8src.format("rW_null_min") + "; -0.003 (strength null, 219) printed 0.00", M(190)),
        num("MStrengthAssortNullMax", iv["rW_null_max"], fx(iv["rW_null_max"]), s8src.format("rW_null_max"), M(190)),
    ]
    above = [iv[f"rW_n_above95_{n}_{s}"] for n in NULLS for s in SCALES if (n, s) != ("geometric", 68)]
    out += [
        num("MStrengthAssortAboveMin", min(above), cnt(min(above)),
            s8src.format("rW_n_above95_<null>_<scale>") + ", min over the 8 null x scale cells other than geometric/68", M(190)),
        num("MStrengthAssortAboveMax", max(above), cnt(max(above)),
            s8src.format("rW_n_above95_<null>_<scale>") + ", max over the 8 null x scale cells other than geometric/68", M(190)),
        num("MStrengthAssortAboveGeomSixtyEight", iv["rW_n_above95_geometric_68"], cnt(iv["rW_n_above95_geometric_68"]),
            s8src.format("rW_n_above95_geometric_68"), M(190)),
    ]
    for m, name in ((1, "MRayleighShareROne"), (2, "MRayleighShareRTwo"), (3, "MRayleighShareRThree")):
        out += per_scale(name, {s: iv[f"share_l1_R{m}_strength_{s}"] for s in SCALES}, pc,
                         s8src.format(f"share_l1_R{m}_strength_<scale>") + f": median over subjects of d_R{m}/d_lambda1, strength nulls",
                         M(190))
    nc = csv(res, "nulls/mediation_noncircular.csv")
    rc = nc[(nc.outcome == "degree_q1") & (nc.k == "avg") & nc.mediator.str.startswith("rc")]
    out.append(num("MRichClubMediatedMax", rc.prop_mediated.max(), pc(rc.prop_mediated.max()),
                   "revision/nulls/mediation_noncircular.csv (outcome == 'degree_q1' at fixed c = 0.1, k == 'avg', mediator in "
                   "{rc_bu, rc_wu, rc_wu_p75, rc_wu_p85, rc_wu_p95}), max of prop_mediated over mediators x nulls x scales "
                   "(rc_wu_p95, strength null, 68)", M(190)))

    # ---------------------------------------------------------------- L245: disconnected connectomes
    n_disc, n_iso = {}, {}
    for s in SCALES:
        A = np.load(res.parent / "processed" / f"connectomes_scale-{s}.npz")["adjacency"]
        n_disc[s] = sum(connected_components(a > 0, directed=False)[0] > 1 for a in A)
        n_iso[s] = int((A.sum(2) == 0).sum())
    assert len(set(n_disc.values())) == 1 and all(n_iso[s] == n_disc[s] for s in SCALES), (n_disc, n_iso)
    out.append(num("MDisconnectedSubjects", n_disc[68], {1: "one"}.get(n_disc[68], str(n_disc[68])),
                   "processed/connectomes_scale-<N>.npz (run root): subjects whose graph (A > 0) has more than one connected component, "
                   f"per scale {n_disc}; total zero-strength regions per scale {n_iso} (subject index 8, sub-008); recomputed here. "
                   "Also consistent with 69 CONN graphs in mechanism/canonical_graphs.csv.gz. Text is the word 'one'", M(245)))

    # ---------------------------------------------------------------- L353, L355: null fidelity
    nf = csv(res, "nulls/null_fidelity.csv")
    sr = nf[nf.null == "strength"].strength_r
    out += [
        num("MStrengthNullFidelityMin", sr.min(), fx(sr.min(), 3),
            "revision/nulls/null_fidelity.csv (null == 'strength'), strength_r (mean over subjects and nulls of corr(null, real strengths)), "
            "min over scales", M(353)),
        num("MStrengthNullFidelityMax", sr.max(), fx(sr.max(), 3),
            "revision/nulls/null_fidelity.csv (null == 'strength'), strength_r, max over scales", M(353)),
    ]
    real = nf[nf.null == "real"].set_index("scale").mean_edge_length
    geo = nf[nf.null == "geometric"].set_index("scale").mean_edge_length
    rel = float((geo / real - 1).abs().max())
    assert rel < 0.01, rel
    out.append(num("MGeomNullEdgeLengthBound", rel, "1",
                   "revision/nulls/null_fidelity.csv (null in {geometric, real}), mean_edge_length (mean over subjects and nulls); "
                   f"max over scales of |geometric/real - 1| = {rel:.4f}; text is the bound 1%. Per subject, the null-mean length "
                   "differs from the subject's own by up to 1.04% (68 regions; nulls/networks.csv.gz)", M(355)))

    # ---------------------------------------------------------------- L369: r_W vs s'A0 s in strength nulls
    out += [
        num("MAssortRayleighPearsonMedian", iv["pearson_rW_sTWs_strength_median"], fx(iv["pearson_rW_sTWs_strength_median"], 3),
            s8src.format("pearson_rW_sTWs_strength_median") + " (pooled over subjects and scales)", M(369)),
        num("MAssortRayleighPearsonMin", iv["pearson_rW_sTWs_strength_min"], fx(iv["pearson_rW_sTWs_strength_min"]),
            s8src.format("pearson_rW_sTWs_strength_min") + " (pooled over subjects and scales)", M(369)),
    ]
    return out


# ---- section F

_FRACTION_WORDS = {2: "one half", 3: "one third", 4: "one quarter", 5: "one fifth", 6: "one sixth"}


def _pc_half_up(x: float, d: int = 0) -> str:
    """Percentage with half-up rounding (2.5% -> 3%), for exact fractions such as 7/280."""
    q = Decimal(1).scaleb(-d)
    return str(Decimal(repr(round(100 * x, 10))).quantize(q, rounding=ROUND_HALF_UP))


def _floor(x: float, d: int = 2) -> str:
    """Round down to d decimals (for 'p > x' lower bounds)."""
    return f"{math.floor(x * 10**d) / 10**d:.{d}f}"


def sec_F(res: Path) -> list[Number]:
    """Response letter: gradients (R1 Minor 6), representatives (R2 Major 1), targets (R2 Major 6),
    lesions (R2 Major 7), edge perturbations (R2 minor comments)."""
    out: list[Number] = []
    T = "targets/"

    # ---- L75: rank divergence vs registered gradients, spin tests (Vasa) -------------------------------
    sp = csv(res, T + "spin_tests.csv")
    src = ("revision/targets/spin_tests.csv, x == 'rank_divergence', y == 'G1_registered'; {col} "
           "(Spearman across parcels of subject-averaged degree rank - cycle rank vs registered G1; 10,000 Vasa spins)")
    g1 = {s: one(sp, scale=s, x="rank_divergence", y="G1_registered") for s in SCALES}
    out += per_scale("MGradientRhoG", {s: g1[s].spearman_rho for s in SCALES}, lambda v: fx(v, 2),
                     src.format(col="spearman_rho"), R(75))
    out += per_scale("MGradientPspinG", {s: g1[s].p_spin_vasa for s in SCALES}, lambda v: fx(v, 2),
                     src.format(col="p_spin_vasa"), R(75))
    g2min = min(one(sp, scale=s, x="rank_divergence", y="G2_registered").p_spin_vasa for s in SCALES)
    out.append(num("MGradientTwoPspinMin", g2min, _floor(g2min, 2),
                   "revision/targets/spin_tests.csv, x == 'rank_divergence', y == 'G2_registered'; min over scales of "
                   "p_spin_vasa (0.849), rounded DOWN to 2 dp because the text states a lower bound ('p_spin > 0.84')",
                   R(75)))

    # ---- L95: variance in cycle - degree ER explained by distance from the degree set (k >= 2) -----------
    dr = csv(res, "stability/tables/rep_distance_regressions.csv")
    x = dr[dr.k >= 2].jaccard_degree_partial_r2
    src = ("revision/stability/tables/rep_distance_regressions.csv, k in {{2,3,5,15}}, all scales; {mm} of "
           "jaccard_degree_partial_r2 (within-subject variance of the cycle - degree ER difference across the 4 "
           "representative conventions explained by Jaccard overlap with the degree set; table S9d)")
    out.append(num("MRepDistanceRsqMin", x.min(), pc(x.min()), src.format(mm="min"), R(95)))
    out.append(num("MRepDistanceRsqMax", x.max(), pc(x.max()), src.format(mm="max"), R(95)))

    # ---- L136: precuneus parcels in the legacy occipital mask -------------------------------------------
    tm = csv(res, T + "target_mask_parcels.csv")
    prec = {s: one(tm, scale=s, target_name="legacy_occipital_module").n_positive
            - one(tm, scale=s, target_name="kw_occipital_module").n_positive for s in SCALES}
    out += per_scale("MPrecuneusParcels", prec, cnt,
                     "revision/targets/target_mask_parcels.csv: n_positive of legacy_occipital_module minus "
                     "kw_occipital_module (corrected = legacy minus precuneus; REPORT.md sec. 2)", R(136))

    # ---- L139/L141/L142: named-target medians (pct = 100 (E_cyc - E_deg)/E_deg, k-averaged, median) -----
    ns = csv(res, T + "named_target_summary.csv")
    src = ("revision/targets/named_target_summary.csv, variant == '{v}', target_name == '{t}'; median_pct "
           "(median over 70 subjects of 100 (E_cyc - E_deg)/E_deg averaged over k; c = 0.1, eps = 1e-5)")
    for name, t, v, line in [("MTargetOccipitalPct", "kw_occipital_module", "full", 139),
                             ("MTargetVisualPct", "net_Visual", "full", 139),
                             ("MTargetVisualExclPct", "net_Visual", "drivers_excluded", 141)]:
        out += per_scale(name, {s: one(ns, scale=s, target_name=t, variant=v).median_pct for s in SCALES},
                         lambda val: fx(val, 1), src.format(v=v, t=t), R(line))
    out += per_scale("MTargetVisualCouplingRho",
                     {s: one(ns, scale=s, target_name="net_Visual", variant="full").rho_pct_vs_dcoupling for s in SCALES},
                     lambda v: fx(v, 2),
                     "revision/targets/named_target_summary.csv, variant == 'full', target_name == 'net_Visual'; "
                     "rho_pct_vs_dcoupling (Spearman across subjects of pct vs cycle - degree structural coupling)", R(142))

    # ---- L140: drivers inside the Yeo visual network; sign agreement ------------------------------------
    ov = csv(res, T + "driver_target_overlap.csv")
    vis = ov[ov.target_name == "net_Visual"]
    src = ("revision/targets/driver_target_overlap.csv, target_name == 'net_Visual'; {mm} over scales of {col} "
           "(fraction of subject x k cases with any driver inside the target)")
    for name, col, fn in [("MVisualCycleDriversInMin", "frac_any_driver_in_cyc", min),
                          ("MVisualCycleDriversInMax", "frac_any_driver_in_cyc", max),
                          ("MVisualDegreeDriversInMin", "frac_any_driver_in_deg", min),
                          ("MVisualDegreeDriversInMax", "frac_any_driver_in_deg", max)]:
        v = fn(vis[col])
        out.append(num(name, v, _pc_half_up(v), src.format(mm=fn.__name__, col=col)
                       + "; rounded half-up (degree max is exactly 7/280 = 2.5%)", R(140)))
    f = ns[(ns.variant == "full") & (ns.family == "yeo7_network")].merge(ov, on=["scale", "family", "target_name"])
    single = f[~f.target_name.isin(["net_unimodal_VisSomMot", "net_transmodal_DefFP", "net_transmodal_vs_unimodal"])]
    sig = single[single.p_holm < 0.05]
    n_exc = int((np.sign(sig.median_pct) != np.sign(sig.frac_any_driver_in_deg - sig.frac_any_driver_in_cyc)).sum())
    out.append(num("MTargetSignExceptions", n_exc, _WORDS[n_exc],
                   "revision/targets/named_target_summary.csv (variant 'full') merged with driver_target_overlap.csv; the "
                   "7 single Yeo-7 networks x 3 scales with Holm p < 0.05 (18 cells); count of cells where sign(median_pct) "
                   "!= sign(frac_any_driver_in_deg - frac_any_driver_in_cyc) (68 SomMot, 219 Default). Definition "
                   "reconstructed: counting all 21 single-network cells regardless of significance gives 6. Printed as a word",
                   R(140)))

    # ---- L143: share of target energy carried by floored directions ------------------------------------
    sl = csv(res, T + "sensitivity_long.csv.gz")
    sl = sl[np.isclose(sl.c, 0.1) & np.isclose(sl.eps, 1e-5)]
    tmed = sl.groupby(["scale", "target_name"]).floor_frac.median()
    src = ("revision/targets/sensitivity_long.csv.gz, c == 0.1, eps == 1e-5; median of floor_frac over subjects x k x "
           "method per (scale, target), then {mm} over the 22 targets x 3 scales (as table S10.7 'range of per-target medians')")
    out.append(num("MTargetFloorShareMin", tmed.min(), pc(tmed.min()), src.format(mm="min"), R(143)))
    out.append(num("MTargetFloorShareMax", tmed.max(), pc(tmed.max()),
                   src.format(mm="max") + "; draft says 'nearly 100' (0.999)", R(143)))

    # ---- L155/L156: reciprocal lesions (static protocol, pooled over scale x h) ---------------------------
    cs = csv(res, "lesion/crossover_stats.csv")
    pooled = cs[(cs.protocol == "static") & (cs.scale == "pooled")]
    deg = one(pooled, metric="dER", lesion_type="degree")
    src = ("revision/lesion/crossover_stats.csv, metric == 'dER', protocol == 'static', lesion_type == 'degree', "
           "scale == 'pooled'; {col} (D = cycle-set ER loss - degree-set ER loss, subject mean over 9 scale x h cells)")
    out.append(num("MLesionDegreeERDiff", deg["mean"], fx(deg["mean"], 2), src.format(col="mean"), R(155)))
    out.append(num("MLesionDegreeERNeg", deg.n - deg.n_pos, cnt(deg.n - deg.n_pos),
                   src.format(col="n - n_pos (subjects with D < 0)"), R(155)))
    for name, metric, label in [("MLesionPctRatioER", "dPct_ER", "ER"), ("MLesionPctRatioPR", "dPct_PR", "PR")]:
        cyc = one(pooled, metric=metric, lesion_type="cycle")["mean"]
        dg = one(pooled, metric=metric, lesion_type="degree")["mean"]
        ratio = abs(cyc / dg)
        out.append(num(name, ratio, _FRACTION_WORDS[int(round(1 / ratio))],
                       f"revision/lesion/crossover_stats.csv, metric == '{metric}', protocol == 'static', scale == 'pooled'; "
                       f"|mean D (cycle lesion) / mean D (degree lesion)| = {abs(cyc):.4f}/{abs(dg):.4f}; printed as the "
                       f"nearest unit fraction 1/round(1/ratio) ({label})", R(156)))

    # ---- L158: intact ER of the replacement degree set vs the true top-k degree set ----------------------
    ps = csv(res, "lesion/per_subject_degradation.csv")
    rep = ps[(ps.protocol == "static") & (ps.lesion_type == "degree") & (ps.strategy == "degree_strength")].intact_effective_rank.mean()
    top = ps[(ps.protocol == "recompute") & (ps.strategy == "degree_strength")].intact_effective_rank.mean()
    out.append(num("MLesionReplacementIntactER", rep, fx(rep, 2),
                   "revision/lesion/per_subject_degradation.csv, protocol == 'static', lesion_type == 'degree', strategy == "
                   "'degree_strength'; mean intact_effective_rank over subjects x scale x h (degree ranks h+1..h+k)", R(158)))
    out.append(num("MLesionTopDegreeIntactER", top, fx(top, 2),
                   "revision/lesion/per_subject_degradation.csv, protocol == 'recompute', strategy == 'degree_strength'; mean "
                   "intact_effective_rank (true top-k degree set; identical for every lesion type)", R(158)))

    # ---- L159: energy change is a size effect ---------------------------------------------------------
    rows = pd.read_csv(res / "lesion" / "lesion_rows.csv.gz",
                       usecols=["protocol", "lesioned_log10_average_energy", "intact_log10_average_energy", "n_lesioned", "n_intact"])
    st = rows[rows.protocol == "static"]
    r = np.corrcoef(st.lesioned_log10_average_energy - st.intact_log10_average_energy,
                    np.log10(st.n_lesioned / st.n_intact))[0, 1]
    out.append(num("MLesionEnergySizeR", r, fx(r, 3),
                   "revision/lesion/lesion_rows.csv.gz, protocol == 'static'; Pearson r over all rows of (lesioned - intact "
                   "log10_average_energy) vs log10(n_lesioned/n_intact) (as report_snippets.md 'energy_corr')", R(159)))
    cells = cs[(cs.metric == "dlog10E") & (cs.protocol == "static") & (cs.scale != "pooled")
               & cs.lesion_type.isin(["degree", "cycle", "random"])]
    mx = cells["mean"].abs().max()
    out.append(num("MLesionEnergyStrategyDiffMax", mx, fx(mx, 3),
                   "revision/lesion/crossover_stats.csv, metric == 'dlog10E', protocol == 'static', the 27 scale x h x lesion "
                   "cells; max |mean (cycle - degree) change in log10 E_avg| (0.0052, degree lesion, 68, h = 10). "
                   "REPORT.md says <= 0.003 (pooled/rounded); the pooled degree-lesion value is 0.0024", R(159)))

    # ---- L169: cycle-participation rank stability under perturbation (Ripser ranking) --------------------
    rs = csv(res, "stability/tables/perturb_rank_stability.csv")
    rip = rs[rs.ranking == "ripser"]
    for name, fam, lev in [("MPerturbNoiseRho", "noise", 0.05), ("MPerturbDropoutRho", "dropout", 0.2)]:
        v = rip[(rip.family == fam) & np.isclose(rip.level, lev)].rho_median
        assert len(v) == 3
        src = (f"revision/stability/tables/perturb_rank_stability.csv, ranking == 'ripser', family == '{fam}', level == {lev}; "
               "{mm} over scales of rho_median (median over subjects x 50 draws of Spearman rho, perturbed vs unperturbed)")
        out.append(num(name + "Min", v.min(), fx(v.min(), 2), src.format(mm="min"), R(169)))
        out.append(num(name + "Max", v.max(), fx(v.max(), 2), src.format(mm="max"), R(169)))
    return out



SECTIONS = [sec_A, sec_B, sec_C, sec_D, sec_E, sec_F]


def collect(res: Path) -> list[Number]:
    res = Path(res)
    out: list[Number] = []
    for fn in SECTIONS:
        out.extend(fn(res))
    names = [n.name for n in out]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, dupes
    for n in out:
        assert not (isinstance(n.value, float) and math.isnan(n.value)) or n.source.startswith("NO SOURCE"), n.name
    return out
