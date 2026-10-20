#!/usr/bin/env python
"""Summaries and supplement tables for Task 1 (representatives x scales x k)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, wilcoxon

from stability_common import OUTPUT_DIR, VARIANTS, make_rng, paired_summary

SCALES = (68, 114, 219)
K_MAIN = (1, 2, 3, 5)
SHORT = {"ripser": "Ripser cocycle", "gudhi": "GUDHI pairing", "geodesic": "Length-minimized", "birth_edge": "Birth edge"}
IN = OUTPUT_DIR / "representatives"
OUT = OUTPUT_DIR / "tables"


def pfmt(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    if p >= 0.001:
        return f"{p:.3f}"
    mant, exp = f"{p:.1e}".split("e")
    return f"{mant}e{int(exp)}"


def ptex(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    if p >= 0.001:
        return f"{p:.3f}"
    mant, exp = f"{p:.1e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def load():
    per = pd.concat([pd.read_csv(IN / f"per_subject_scale-{s}.csv") for s in SCALES], ignore_index=True)
    agr = pd.concat([pd.read_csv(IN / f"score_agreement_scale-{s}.csv") for s in SCALES], ignore_index=True)
    return per, agr


def result_level(per: pd.DataFrame) -> pd.DataFrame:
    rng = make_rng(20260925)
    rows = []
    for (scale, k, variant), sub in per.groupby(["scale", "k", "variant"]):
        for metric in ("er", "pr"):
            rows.append(
                {"scale": scale, "k": k, "variant": variant, "metric": metric,
                 **paired_summary(sub[f"{metric}_diff_degree"].to_numpy(), rng)}
            )
    return pd.DataFrame(rows)


def random_level(per: pd.DataFrame) -> pd.DataFrame:
    rng = make_rng(20260926)
    rows = []
    for (scale, k), sub_k in per.groupby(["scale", "k"]):
        deg = sub_k[sub_k.variant == "ripser"]
        rows.append({"scale": scale, "k": k, "variant": "degree",
                     "mean_pct_random": deg.degree_er_pct_random.mean(),
                     **{f"random_{key}": v for key, v in paired_summary(deg.degree_er_diff_random.to_numpy(), rng).items()}})
        for variant, sub in sub_k.groupby("variant"):
            row = {"scale": scale, "k": k, "variant": variant,
                   "mean_pct_random": sub.er_pct_random.mean(),
                   "mean_pct_overlap_null": sub.er_pct_overlap_null.mean(),
                   "mean_pct_strength_null": sub.er_pct_strength_null.mean(),
                   "mean_er_diff_degree": sub.er_diff_degree.mean(),
                   # decomposition: variant - degree = (null - degree) + (variant - null)
                   "overlap_null_minus_degree": (sub.er_overlap_null_mean - sub.er_degree).mean(),
                   "strength_null_minus_degree": (sub.er_strength_null_mean - sub.er_degree).mean()}
            for tag, col in (("random", "er_diff_random"), ("overlap", "er_diff_overlap_null"),
                             ("strength", "er_diff_strength_null")):
                for key, v in paired_summary(sub[col].to_numpy(), rng).items():
                    row[f"{tag}_{key}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def distance_regressions(per: pd.DataFrame) -> pd.DataFrame:
    """Pool variants x subjects; relate ER advantage to closeness to the degree ranking."""
    rows = []
    for (scale, k), sub in per.groupby(["scale", "k"]):
        d = sub.copy()
        d["closeness_rank"] = 100 - d.mean_strength_rank_pct  # higher = more hub-like
        row = {"scale": scale, "k": k, "n_obs": len(d)}
        for pred in ("jaccard_degree", "spearman_vs_degree", "closeness_rank"):
            if d[pred].std() == 0:
                continue
            z = d.assign(x=(d[pred] - d[pred].mean()) / d[pred].std())
            m0 = smf.ols("er_diff_degree ~ C(subject_id)", z).fit()
            m1 = smf.ols("er_diff_degree ~ x + C(subject_id)", z).fit()
            m2 = smf.ols("er_diff_degree ~ x + C(variant) + C(subject_id)", z).fit()
            mv = smf.ols("er_diff_degree ~ C(variant) + C(subject_id)", z).fit()
            # within-subject partial R^2 of the predictor, and of variant identity beyond it
            row[f"{pred}_beta_per_sd"] = m1.params["x"]
            row[f"{pred}_beta_p"] = m1.pvalues["x"]
            row[f"{pred}_partial_r2"] = (m0.ssr - m1.ssr) / m0.ssr
            row[f"{pred}_variant_partial_r2_beyond"] = (m1.ssr - m2.ssr) / m1.ssr
            row[f"{pred}_variant_F_p"] = m2.compare_f_test(m1)[1]
            row["variant_only_partial_r2"] = (m0.ssr - mv.ssr) / m0.ssr
            # within-subject Spearman (subject-demeaned)
            dm = d.groupby("subject_id")[[pred, "er_diff_degree"]].transform(lambda s: s - s.mean())
            row[f"{pred}_within_spearman"] = spearmanr(dm[pred], dm.er_diff_degree).statistic
        rows.append(row)
    return pd.DataFrame(rows)


def within_variant_correlations(per: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scale, k, variant), sub in per.groupby(["scale", "k", "variant"]):
        rows.append({"scale": scale, "k": k, "variant": variant,
                     "spearman_erdiff_vs_jaccard": spearmanr(sub.er_diff_degree, sub.jaccard_degree).statistic,
                     "spearman_erdiff_vs_rho_degree": spearmanr(sub.er_diff_degree, sub.spearman_vs_degree).statistic})
    return pd.DataFrame(rows)


def agreement_table(per, agr) -> pd.DataFrame:
    a = agr.groupby(["scale", "variant"]).agg(
        rho_degree_median=("spearman_vs_degree", "median"),
        rho_degree_q25=("spearman_vs_degree", lambda s: s.quantile(0.25)),
        rho_degree_q75=("spearman_vs_degree", lambda s: s.quantile(0.75)),
        rho_ripser_median=("spearman_vs_ripser", "median"),
        rho_ripser_q25=("spearman_vs_ripser", lambda s: s.quantile(0.25)),
        rho_ripser_q75=("spearman_vs_ripser", lambda s: s.quantile(0.75)),
    ).reset_index()
    ov = per.pivot_table(index=["scale", "variant"], columns="k", values="n_shared_degree", aggfunc="mean")
    ov.columns = [f"shared_degree_k{k}" for k in ov.columns]
    jr = per.pivot_table(index=["scale", "variant"], columns="k", values="jaccard_ripser", aggfunc="mean")
    jr.columns = [f"jaccard_ripser_k{k}" for k in jr.columns]
    return a.merge(ov.reset_index()).merge(jr.reset_index())


# ----------------------------------------------------------------------------- tables
def tex_agreement(t: pd.DataFrame) -> str:
    lines = [r"\begin{table}[h]", r"\centering",
             r"\caption{\textbf{Agreement of cycle-participation variants with degree strength and with the Ripser ranking.} "
             r"Median (interquartile range) per-subject Spearman $\rho$ across 70 subjects, and mean number of top-$k$ nodes "
             r"shared with the degree top-$k$ set.}",
             r"\label{tab:rep_agreement}", r"\begin{tabular}{llcccccc}", r"\toprule",
             r"& & & & \multicolumn{4}{c}{Shared with degree set} \\ \cmidrule(lr){5-8}",
             r"Scale & Variant & $\rho$ vs degree & $\rho$ vs Ripser & $k=1$ & $k=2$ & $k=3$ & $k=5$ \\", r"\midrule"]
    for scale in SCALES:
        for i, v in enumerate(VARIANTS):
            r = t[(t.scale == scale) & (t.variant == v)].iloc[0]
            rip = "1" if v == "ripser" else f"{r.rho_ripser_median:.2f} ({r.rho_ripser_q25:.2f}--{r.rho_ripser_q75:.2f})"
            lines.append(f"{scale if i == 0 else ''} & {SHORT[v]} & {r.rho_degree_median:.2f} ({r.rho_degree_q25:.2f}--{r.rho_degree_q75:.2f}) & {rip} & "
                         + " & ".join(f"{r[f'shared_degree_k{k}']:.2f}" for k in K_MAIN) + r" \\")
        if scale != SCALES[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def tex_result(res: pd.DataFrame, metric: str, ks=K_MAIN) -> str:
    name = "effective rank" if metric == "er" else "participation ratio"
    lines = [r"\begin{table}[h]", r"\centering", r"\small",
             rf"\caption{{\textbf{{Cycle-minus-degree {name} under each representative variant.}} Mean and median of the "
             r"per-subject difference, bootstrap 95\% CI of the median (2{,}000 resamples), number of subjects with a positive difference "
             r"(of 70), two-sided Wilcoxon signed-rank $p$ (zero differences dropped), and matched-pairs rank-biserial $r$. $c=0.1$, $\epsilon=10^{-5}$.}",
             rf"\label{{tab:rep_{metric}}}", r"\begin{tabular}{rlrrcrrr}", r"\toprule",
             r"$k$ & Variant & Mean & Median & 95\% CI & $n_+$ & $p$ & $r$ \\"]
    for scale in SCALES:
        lines += [r"\midrule", rf"\multicolumn{{8}}{{l}}{{\textit{{{scale} regions}}}} \\"]
        for k in ks:
            for i, v in enumerate(VARIANTS):
                r = res[(res.scale == scale) & (res.k == k) & (res.variant == v) & (res.metric == metric)].iloc[0]
                lines.append(f"{k if i == 0 else ''} & {SHORT[v]} & {r['mean']:+.2f} & {r['median']:+.2f} & "
                             f"[{r.median_ci_low:+.2f}, {r.median_ci_high:+.2f}] & {r.n_positive} & {ptex(r.wilcoxon_p)} & {r.rank_biserial:+.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def tex_nulls(rl: pd.DataFrame) -> str:
    lines = [r"\begin{table}[h]", r"\centering", r"\small",
             r"\caption{\textbf{Effective rank of each variant relative to random and degree-matched random driver sets.} "
             r"For each subject, 100 random sets of the same size were drawn uniformly (the main analysis' random baseline), "
             r"matched on the number of nodes shared with the degree top-$k$ set (overlap-matched), or matched node-by-node on "
             r"strength rank within $\pm5\%$ of $N$ (strength-matched). Entries: mean over subjects of (variant ER $-$ null mean ER) "
             r"/ mean percentile of the variant within its null / subjects above the null mean (of 70). Degree row: degree set vs uniform null.}",
             r"\label{tab:rep_nulls}", r"\begin{tabular}{rlccc}", r"\toprule",
             r"$k$ & Variant & Uniform random & Overlap-matched & Strength-matched \\"]
    for scale in SCALES:
        lines += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{\textit{{{scale} regions}}}} \\"]
        for k in K_MAIN:
            sub = rl[(rl.scale == scale) & (rl.k == k)]
            d = sub[sub.variant == "degree"].iloc[0]
            lines.append(f"{k} & Degree & {d.random_mean:+.2f} / {d.mean_pct_random:.0f} / {d.random_n_positive} & -- & -- " + r"\\")
            for v in VARIANTS:
                r = sub[sub.variant == v].iloc[0]
                cells = [f"{r[f'{t}_mean']:+.2f} / {r[f'mean_pct_{c}']:.0f} / {int(r[f'{t}_n_positive'])}"
                         for t, c in (("random", "random"), ("overlap", "overlap_null"), ("strength", "strength_null"))]
                lines.append(f" & {SHORT[v]} & " + " & ".join(cells) + r" \\")
        if scale != SCALES[-1]:
            pass
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def tex_decomposition(rl: pd.DataFrame, ks=(2, 5)) -> str:
    lines = [r"\begin{table}[h]", r"\centering", r"\small",
             r"\caption{\textbf{Decomposition of the cycle-minus-degree effective-rank advantage.} For each variant, the mean "
             r"advantage over the degree set (Total) is split into the part obtained by random nodes with the same strength ranks "
             r"(Strength-matched $-$ degree) and the remainder attributable to which nodes of that strength the variant picks "
             r"(Variant $-$ strength-matched; subjects positive of 70 and Wilcoxon $p$ in parentheses).}",
             r"\label{tab:rep_decomposition}", r"\begin{tabular}{ll" + "ccc" * len(ks) + "}", r"\toprule",
             "& & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{$k={k}$}}" for k in ks) + r" \\ "
             + "".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(ks))),
             "Scale & Variant & " + " & ".join(["Total & Strength & Residual"] * len(ks)) + r" \\", r"\midrule"]
    for scale in SCALES:
        for i, v in enumerate(VARIANTS):
            cells = []
            for k in ks:
                r = rl[(rl.scale == scale) & (rl.k == k) & (rl.variant == v)].iloc[0]
                cells += [f"{r.mean_er_diff_degree:+.2f}", f"{r.strength_null_minus_degree:+.2f}",
                          f"{r.strength_mean:+.2f} ({int(r.strength_n_positive)}; {ptex(r.strength_wilcoxon_p)})"]
            lines.append(f"{scale if i == 0 else ''} & {SHORT[v]} & " + " & ".join(cells) + r" \\")
        if scale != SCALES[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def tex_regression(reg: pd.DataFrame) -> str:
    lines = [r"\begin{table}[h]", r"\centering",
             r"\caption{\textbf{Does distance from the degree set explain the effective-rank advantage?} Observations are "
             r"variants $\times$ subjects (4 $\times$ 70) within each scale and $k$; all models include subject fixed effects. "
             r"$\beta$: change in the cycle-minus-degree ER per SD of the variant's top-$k$ Jaccard overlap with the degree set; "
             r"$R^2_J$: within-subject variance explained by overlap; $R^2_{\text{var}|J}$: additional within-subject variance "
             r"explained by variant identity once overlap is controlled ($p$ from the nested $F$ test).}",
             r"\label{tab:rep_distance}", r"\begin{tabular}{rrcccc}", r"\toprule",
             r"Scale & $k$ & $\beta$ (per SD) & $R^2_J$ & $R^2_{\text{var}|J}$ & $p_{\text{var}|J}$ \\", r"\midrule"]
    for scale in SCALES:
        for i, k in enumerate(K_MAIN):
            r = reg[(reg.scale == scale) & (reg.k == k)].iloc[0]
            lines.append(f"{scale if i == 0 else ''} & {k} & {r.jaccard_degree_beta_per_sd:+.2f} & {r.jaccard_degree_partial_r2:.2f} & "
                         f"{r.jaccard_degree_variant_partial_r2_beyond:.2f} & {ptex(r.jaccard_degree_variant_F_p)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per, agr = load()
    res = result_level(per)
    rl = random_level(per)
    reg = distance_regressions(per)
    wv = within_variant_correlations(per)
    ag = agreement_table(per, agr)
    res.to_csv(OUT / "rep_result_level.csv", index=False)
    rl.to_csv(OUT / "rep_null_comparisons.csv", index=False)
    reg.to_csv(OUT / "rep_distance_regressions.csv", index=False)
    wv.to_csv(OUT / "rep_within_variant_correlations.csv", index=False)
    ag.to_csv(OUT / "rep_agreement.csv", index=False)
    tex = "\n\n".join([tex_agreement(ag), tex_result(res, "er"), tex_result(res, "pr"), tex_nulls(rl), tex_decomposition(rl), tex_regression(reg)])
    (OUT / "representatives_tables.tex").write_text(tex + "\n")

    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
    print(ag.round(2).to_string(index=False))
    print(res[res.metric == "er"][["scale", "k", "variant", "mean", "median", "median_ci_low", "median_ci_high", "n_positive", "n_zero", "wilcoxon_p", "rank_biserial"]].round(3).to_string(index=False))
    print(res[res.metric == "pr"][["scale", "k", "variant", "mean", "median", "n_positive", "wilcoxon_p", "rank_biserial"]].round(3).to_string(index=False))
    cols = ["scale", "k", "variant", "mean_er_diff_degree", "random_mean", "mean_pct_random", "random_n_positive", "random_wilcoxon_p",
            "overlap_null_minus_degree", "overlap_mean", "mean_pct_overlap_null", "overlap_n_positive", "overlap_wilcoxon_p",
            "strength_null_minus_degree", "strength_mean", "mean_pct_strength_null", "strength_n_positive", "strength_wilcoxon_p"]
    print(rl[cols].round(3).to_string(index=False))
    print(reg.round(3).to_string(index=False))
    print(wv.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
