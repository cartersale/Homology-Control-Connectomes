#!/usr/bin/env python
"""Statistics and tables for the spectra / random-baseline / matched-control revision analyses.

Reads results/revision/spectra/set_metrics.csv (from 01_compute_spectra.py) and writes
CSV + Markdown tables into results/revision/spectra/tables/.

Statistics convention (every subject-level comparison, n = 70 subjects):
  subject value = mean over k of the per-(subject, k) paired difference;
  mean, median, bootstrap 95 % CI of the median (2000 resamples, seeded),
  n positive / 70, two-sided Wilcoxon signed-rank p, matched-pairs rank-biserial r,
  Holm correction within the family stated in each table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from hoc.config import load_config  # noqa: E402

OUT = REPO_ROOT / "results" / "revision" / "spectra"
TAB = OUT / "tables"
CFG = load_config(REPO_ROOT / "config/default.toml")
SEED = int(CFG["analysis"]["random_seed"])
EPS = float(CFG["control"]["primary_epsilon"])
SCALES = [68, 114, 219]
RANKINGS = [
    "degree_strength",
    "eigenvector_centrality",
    "betweenness",
    "cycle_participation",
    "participation_coefficient",
    "triangle_participation",
    "tetrahedron_participation",
    "degree_normalized_cycle_participation",
    "cycle_criticality",
    "cycle_resid",
]
SHORT = {
    "degree_strength": "Degree (strength)",
    "eigenvector_centrality": "Eigenvector centrality",
    "betweenness": "Betweenness",
    "cycle_participation": "Cycle participation",
    "participation_coefficient": "Participation coefficient",
    "triangle_participation": "Triangle participation",
    "tetrahedron_participation": "Tetrahedron participation",
    "degree_normalized_cycle_participation": "Degree-normalized CP",
    "cycle_criticality": "Cycle criticality",
    "cycle_resid": "CP residualized on strength",
    "random": "Random (100/subject)",
    "degree_matched": "Degree-matched random",
    "spatial_matched": "Spatially matched random",
    "degree_spatial_matched": "Degree+spatially matched random",
}


def paired_stats(values: np.ndarray, rng_seed: int = SEED) -> dict:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(rng_seed)
    boots = np.median(rng.choice(x, size=(2000, x.size), replace=True), axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    nz = x[x != 0]
    if nz.size == 0:
        p, rrb = 1.0, 0.0
    else:
        p = float(wilcoxon(nz, zero_method="wilcox", alternative="two-sided").pvalue)
        r = rankdata(np.abs(nz))
        wp, wm = r[nz > 0].sum(), r[nz < 0].sum()
        rrb = float((wp - wm) / (wp + wm))
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_pos": int((x > 0).sum()),
        "n": int(x.size),
        "p": p,
        "r_rb": rrb,
    }


def holm(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["p_holm"] = np.nan
    for _, idx in df.groupby(by).groups.items():
        df.loc[idx, "p_holm"] = multipletests(df.loc[idx, "p"].to_numpy(), method="holm")[1]
    return df


def fmt_p(p: float) -> str:
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.to_numpy()) + " |")
    return "\n".join(lines)


def stat_cells(s: dict) -> dict:
    return {
        "mean": f"{s['mean']:+.2f}",
        "median [95% CI]": f"{s['median']:+.2f} [{s['ci_lo']:+.2f}, {s['ci_hi']:+.2f}]",
        "n>0/70": f"{s['n_pos']}",
        "p (Holm)": fmt_p(s["p_holm"]),
        "r_rb": f"{s['r_rb']:+.2f}",
    }


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT / "set_metrics.csv")
    md: list[str] = []

    det = df[df["sample"] == -1]
    rnd = df[df.family == "random"]
    key = ["subject_id", "scale", "k"]

    # ---------------- Task 0: reproduction ----------------
    piv = det.pivot_table(index=key, columns="family", values="effective_rank")
    cd = (piv["cycle_participation"] - piv["degree_strength"]).groupby(["subject_id", "scale"]).mean()
    rows = []
    for metric in ("effective_rank", "participation_ratio"):
        p2 = det.pivot_table(index=key, columns="family", values=metric)
        d = (p2["cycle_participation"] - p2["degree_strength"]).groupby(["subject_id", "scale"]).mean()
        for sc in SCALES:
            rows.append({"metric": metric, "scale": sc, **paired_stats(d.xs(sc, level="scale").to_numpy())})
    t0 = holm(pd.DataFrame(rows), ["metric"])
    t0.to_csv(TAB / "t0_cycle_minus_degree.csv", index=False)
    md.append("## T0 cycle - degree (reproduction)\n")
    md.append(md_table(pd.DataFrame([{"metric": r.metric, "scale": r.scale, **stat_cells(r._asdict())} for r in t0.itertuples()])))

    # ---------------- Task 1: rankings vs random ----------------
    rstats = rnd.groupby(key).agg(
        er_rand_mean=("effective_rank", "mean"), pr_rand_mean=("participation_ratio", "mean")
    )
    rows = []
    per_subject_rows = []
    for metric, rcol in (("effective_rank", "er_rand_mean"), ("participation_ratio", "pr_rand_mean")):
        rvals = rnd.set_index(key)[metric]
        for m in RANKINGS:
            dm = det[det.family == m].set_index(key)[metric]
            diff = dm - rstats[rcol].reindex(dm.index)
            # within-subject percentile (ties count half)
            joined = rvals.to_frame("r").join(dm.rename("m"), how="inner")
            pct = joined.assign(lt=lambda f: (f.r < f.m - 1e-12).astype(float) + 0.5 * (np.abs(f.r - f.m) <= 1e-12)).groupby(key)["lt"].mean()
            subj_diff = diff.groupby(["subject_id", "scale"]).mean()
            subj_pct = pct.groupby(["subject_id", "scale"]).mean()
            subj_val = dm.groupby(["subject_id", "scale"]).mean()
            subj_rand = rstats[rcol].groupby(["subject_id", "scale"]).mean()
            for sc in SCALES:
                s = paired_stats(subj_diff.xs(sc, level="scale").to_numpy())
                rows.append(
                    {
                        "metric": metric,
                        "scale": sc,
                        "ranking": m,
                        "value_mean": float(subj_val.xs(sc, level="scale").mean()),
                        "random_mean": float(subj_rand.xs(sc, level="scale").mean()),
                        "pct_mean": float(subj_pct.xs(sc, level="scale").mean()),
                        "pct_median": float(subj_pct.xs(sc, level="scale").median()),
                        **s,
                    }
                )
            per_subject_rows.append(
                pd.DataFrame({"metric": metric, "ranking": m, "diff_vs_random": subj_diff, "percentile": subj_pct}).reset_index()
            )
    t1 = holm(pd.DataFrame(rows), ["metric", "scale"])
    t1.to_csv(TAB / "t1_rankings_vs_random.csv", index=False)
    pd.concat(per_subject_rows).to_csv(TAB / "t1_rankings_vs_random_per_subject.csv", index=False)
    # per-k view
    rows = []
    for metric, rcol in (("effective_rank", "er_rand_mean"), ("participation_ratio", "pr_rand_mean")):
        for m in RANKINGS:
            dm = det[det.family == m].set_index(key)[metric]
            diff = (dm - rstats[rcol].reindex(dm.index)).rename("diff").reset_index()
            for (sc, k), g in diff.groupby(["scale", "k"]):
                rows.append({"metric": metric, "ranking": m, "scale": sc, "k": k, "mean_diff": g["diff"].mean(), "n_pos": int((g["diff"] > 0).sum())})
    pd.DataFrame(rows).to_csv(TAB / "t1_rankings_vs_random_by_k.csv", index=False)

    for metric in ("effective_rank", "participation_ratio"):
        md.append(f"\n## T1 {metric}: ranking - random-set mean (k-averaged), Holm over 10 rankings within scale\n")
        sub = t1[t1.metric == metric]
        out = []
        for r in sub.itertuples():
            out.append(
                {
                    "scale": r.scale,
                    "ranking": SHORT[r.ranking],
                    "value": f"{r.value_mean:.2f}",
                    "random": f"{r.random_mean:.2f}",
                    **stat_cells(r._asdict()),
                    "pctile (mean)": f"{100 * r.pct_mean:.0f}",
                }
            )
        md.append(md_table(pd.DataFrame(out)))

    # ---------------- Task 2 & 6: eigenvalue extremes ----------------
    fam_order = RANKINGS + ["random", "degree_matched", "spatial_matched", "degree_spatial_matched"]
    rows = []
    for (sc, fam), g in df.groupby(["scale", "family"]):
        subj = g.groupby(["subject_id", "k"])[["log10_lambda_min", "log10_lambda_max", "frac_at_floor", "frac_W_below_eps", "top1_share"]].mean()
        subj = subj.groupby("subject_id").mean()  # k-averaged per subject
        rows.append(
            {
                "scale": sc,
                "family": fam,
                "n_sets": len(g),
                "log10_lambda_min_mean": subj.log10_lambda_min.mean(),
                "frac_rows_lambda_min_eq_eps": g.lambda_min_equals_eps.mean(),
                "log10_lambda_max_mean": subj.log10_lambda_max.mean(),
                "log10_lambda_max_sd": subj.log10_lambda_max.std(),
                "frac_at_floor_mean": subj.frac_at_floor.mean(),
                "frac_at_floor_min": g.frac_at_floor.min(),
                "frac_at_floor_max": g.frac_at_floor.max(),
                "frac_W_below_eps_mean": subj.frac_W_below_eps.mean(),
                "top1_share_mean": subj.top1_share.mean(),
            }
        )
    t2 = pd.DataFrame(rows)
    t2["family"] = pd.Categorical(t2.family, fam_order)
    t2 = t2.sort_values(["scale", "family"])
    t2.to_csv(TAB / "t2_eigenvalue_extremes.csv", index=False)
    # floor fraction by k for degree / cycle / random
    fk = (
        df[df.family.isin(["degree_strength", "cycle_participation", "random"])]
        .groupby(["scale", "k", "family"])[["frac_at_floor", "frac_W_below_eps"]]
        .mean()
        .reset_index()
    )
    fk.to_csv(TAB / "t2_floor_fraction_by_k.csv", index=False)
    md.append("\n## T2 eigenvalue extremes of W_eps (k-averaged per subject, then mean over 70)\n")
    md.append(
        md_table(
            pd.DataFrame(
                [
                    {
                        "scale": r.scale,
                        "set": SHORT[r.family],
                        "log10 λmin": f"{r.log10_lambda_min_mean:.2f}",
                        "% rows λmin=ε": f"{100 * r.frac_rows_lambda_min_eq_eps:.1f}",
                        "log10 λmax (SD)": f"{r.log10_lambda_max_mean:.2f} ({r.log10_lambda_max_sd:.2f})",
                        "frac. at floor": f"{r.frac_at_floor_mean:.3f}",
                        "frac. λ(W)<ε": f"{r.frac_W_below_eps_mean:.3f}",
                        "λmax/trace": f"{r.top1_share_mean:.2f}",
                    }
                    for r in t2.itertuples()
                ]
            )
        )
    )
    md.append("\n### Floor fraction by k (mean over subjects / sets)\n")
    md.append(md_table(fk.assign(frac_at_floor=fk.frac_at_floor.map("{:.3f}".format), frac_W_below_eps=fk.frac_W_below_eps.map("{:.3f}".format))))
    # λmax paired differences cycle - degree
    rows = []
    p2 = det.pivot_table(index=key, columns="family", values="log10_lambda_max")
    d = (p2["cycle_participation"] - p2["degree_strength"]).groupby(["subject_id", "scale"]).mean()
    for sc in SCALES:
        rows.append({"scale": sc, **paired_stats(d.xs(sc, level="scale").to_numpy())})
    t2b = pd.DataFrame(rows)
    t2b["p_holm"] = multipletests(t2b.p, method="holm")[1]
    t2b.to_csv(TAB / "t2_log10_lambda_max_cycle_minus_degree.csv", index=False)
    md.append("\n### log10 λmax, cycle - degree (k-averaged), Holm over 3 scales\n")
    md.append(md_table(pd.DataFrame([{"scale": r.scale, **stat_cells(r._asdict())} for r in t2b.itertuples()])))

    # Task 6: condition number identity
    kappa_resid = np.abs(df.log10_condition_number - (df.log10_lambda_max - np.log10(EPS)))
    t6 = {
        "n_sets": int(len(df)),
        "frac_lambda_min_eq_eps": float(df.lambda_min_equals_eps.mean()),
        "max_abs_log10kappa_minus_log10(lmax/eps)": float(kappa_resid.max()),
    }
    pd.Series(t6).to_csv(TAB / "t6_condition_number_check.csv")
    md.append("\n## T6 condition number\n")
    md.append("\n".join(f"- {k}: {v}" for k, v in t6.items()))

    # ---------------- Task 4: Perron loading ----------------
    rows = []
    pool = df[df.family.isin(RANKINGS[:-1] + ["random"])]
    for (sc, fam), g in df[df.family.isin(fam_order)].groupby(["scale", "family"]):
        subj = g.groupby(["subject_id", "k"]).perron_loading.mean()
        rows.append(
            {
                "scale": sc,
                "family": fam,
                **{f"k{k}": (subj.xs(k, level="k").mean() if k in subj.index.get_level_values("k") else np.nan) for k in (1, 2, 3, 5)},
                "k_avg": subj.groupby("subject_id").mean().mean(),
            }
        )
    t4a = pd.DataFrame(rows)
    t4a["family"] = pd.Categorical(t4a.family, fam_order)
    t4a = t4a.sort_values(["scale", "family"])
    t4a.to_csv(TAB / "t4_perron_loading_by_method.csv", index=False)
    rows = []
    for sc in SCALES:
        g = pool[pool.scale == sc]
        rho_all = spearmanr(g.perron_loading, g.effective_rank).statistic
        rows.append({"scale": sc, "k": "pooled (all k)", "rho_pooled": rho_all, "n_sets": len(g)})
        for k in (1, 2, 3, 5):
            gk = g[g.k == k]
            within = []
            for _, gs in gk.groupby("subject_id"):
                within.append(spearmanr(gs.perron_loading, gs.effective_rank).statistic)
            within = np.array(within)
            rows.append(
                {
                    "scale": sc,
                    "k": k,
                    "rho_pooled": spearmanr(gk.perron_loading, gk.effective_rank).statistic,
                    "rho_pooled_PR": spearmanr(gk.perron_loading, gk.participation_ratio).statistic,
                    "rho_within_mean": np.nanmean(within),
                    "rho_within_median": np.nanmedian(within),
                    "rho_within_n_neg": int((within < 0).sum()),
                    "n_sets": len(gk),
                }
            )
        # within-subject, all k pooled
        within = np.array([spearmanr(gs.perron_loading, gs.effective_rank).statistic for _, gs in g.groupby("subject_id")])
        rows.append({"scale": sc, "k": "within-subject (all k)", "rho_within_mean": np.nanmean(within), "rho_within_median": np.nanmedian(within), "rho_within_n_neg": int((within < 0).sum())})
    t4b = pd.DataFrame(rows)
    t4b.to_csv(TAB / "t4_perron_er_spearman.csv", index=False)
    md.append("\n## T4 Perron loading (sum of v_i^2 over drivers, v = leading eigvec of A0)\n")
    show = t4a.copy()
    show["family"] = show.family.map(SHORT)
    md.append(md_table(show.round(3)))
    md.append("\n### Spearman(Perron loading, ER) across sets (9 rankings + 100 random per subject)\n")
    md.append(md_table(t4b.round(3).fillna("")))
    # difference in Perron loading degree vs cycle, stats
    p2 = det.pivot_table(index=key, columns="family", values="perron_loading")
    rows = []
    for sc in SCALES:
        d = (p2["degree_strength"] - p2["cycle_participation"]).groupby(["subject_id", "scale"]).mean().xs(sc, level="scale")
        rows.append({"scale": sc, **paired_stats(d.to_numpy())})
    t4c = pd.DataFrame(rows)
    t4c["p_holm"] = multipletests(t4c.p, method="holm")[1]
    t4c.to_csv(TAB / "t4_perron_degree_minus_cycle.csv", index=False)
    md.append("\n### Perron loading, degree - cycle (k-averaged), Holm over 3 scales\n")
    md.append(md_table(pd.DataFrame([{"scale": r.scale, **stat_cells(r._asdict())} for r in t4c.itertuples()])))

    # ---------------- Task 5: matched controls ----------------
    rows = []
    quality = []
    for metric in ("effective_rank", "participation_ratio"):
        cyc = det[det.family == "cycle_participation"].set_index(key)[metric]
        res = det[det.family == "cycle_resid"].set_index(key)[metric]
        deg = det[det.family == "degree_strength"].set_index(key)[metric]
        rmean = rnd.groupby(key)[metric].mean()
        comps = {}
        for fam in ("degree_matched", "spatial_matched", "degree_spatial_matched"):
            fm = df[df.family == fam].groupby(key)[metric].mean()
            comps[f"cycle - {fam}"] = (cyc.reindex(fm.index) - fm)
        comps["cycle - degree_matched (k>=2)"] = comps["cycle - degree_matched"][comps["cycle - degree_matched"].index.get_level_values("k") >= 2]
        comps["cycle - random (k>=2)"] = (cyc - rmean)[cyc.index.get_level_values("k") >= 2]
        comps["cycle_resid - degree"] = res - deg
        comps["cycle_resid - random"] = res - rmean.reindex(res.index)
        comps["cycle_resid - cycle"] = res - cyc
        for name, d in comps.items():
            subj = d.groupby(["subject_id", "scale"]).mean()
            for sc in SCALES:
                rows.append({"metric": metric, "comparison": name, "scale": sc, **paired_stats(subj.xs(sc, level="scale").to_numpy())})
        # percentile of cycle within its matched distribution
        for fam in ("degree_matched", "spatial_matched", "degree_spatial_matched"):
            j = df[df.family == fam].set_index(key)[[metric]].join(cyc.rename("c"), how="inner")
            pct = ((j[metric] < j.c).astype(float) + 0.5 * (j[metric] == j.c)).groupby(key).mean().groupby(["subject_id", "scale"]).mean()
            for sc in SCALES:
                quality.append({"metric": metric, "family": fam, "scale": sc, "cycle_pctile_in_matched": pct.xs(sc, level="scale").mean()})
    t5 = holm(pd.DataFrame(rows), ["metric", "scale"])
    t5.to_csv(TAB / "t5_matched_controls.csv", index=False)
    # match quality
    mq = []
    for (sc, fam), g in df[df.family.isin(["cycle_participation", "degree_strength", "cycle_resid", "random", "degree_matched", "spatial_matched", "degree_spatial_matched"])].groupby(["scale", "family"]):
        s = g.groupby(["subject_id", "k"])[["mean_strength_pct", "mean_pairwise_dist", "perron_loading"]].mean()
        s2 = s[s.index.get_level_values("k") >= 2]
        mq.append(
            {
                "scale": sc,
                "family": fam,
                "strength_pctile (k-avg)": s.mean_strength_pct.groupby("subject_id").mean().mean(),
                "mean pairwise dist mm (k>=2)": s2.mean_pairwise_dist.groupby("subject_id").mean().mean(),
                "perron loading (k-avg)": s.perron_loading.groupby("subject_id").mean().mean(),
            }
        )
    mq = pd.DataFrame(mq)
    mq.to_csv(TAB / "t5_match_quality.csv", index=False)
    qdf = pd.DataFrame(quality)
    qdf.to_csv(TAB / "t5_cycle_percentile_in_matched.csv", index=False)
    for metric in ("effective_rank", "participation_ratio"):
        md.append(f"\n## T5 matched controls, {metric} (k-averaged; spatial & k>=2 rows average k in {{2,3,5}}), Holm over 8 comparisons within scale\n")
        md.append(md_table(pd.DataFrame([{"scale": r.scale, "comparison": r.comparison, **stat_cells(r._asdict())} for r in t5[t5.metric == metric].sort_values(["scale"]).itertuples()])))
    md.append("\n### Match quality\n")
    md.append(md_table(mq.round(3)))
    md.append("\n### Cycle-set percentile within its matched distribution (mean over subjects)\n")
    md.append(md_table(qdf.round(3)))

    ex = pd.read_csv(OUT / "subject_extras.csv")
    md.append("\n### Spearman correlations of scores (mean over subjects)\n")
    md.append(md_table(ex.groupby("scale")[["cp_strength_spearman", "resid_strength_spearman", "resid_cp_spearman"]].mean().round(3).reset_index()))
    # overlap of residualized top-k with cycle/degree top-k
    ov = []
    for (sc, k), g in det[det.family.isin(["cycle_resid", "cycle_participation", "degree_strength"])].groupby(["scale", "k"]):
        p = g.pivot_table(index="subject_id", columns="family", values="nodes", aggfunc="first")
        def jac(a, b):
            A, B = set(a.split(";")), set(b.split(";"))
            return len(A & B) / k
        ov.append({"scale": sc, "k": k, "resid∩cycle": np.mean([jac(a, b) for a, b in zip(p.cycle_resid, p.cycle_participation)]), "resid∩degree": np.mean([jac(a, b) for a, b in zip(p.cycle_resid, p.degree_strength)]), "cycle∩degree": np.mean([jac(a, b) for a, b in zip(p.cycle_participation, p.degree_strength)])})
    ov = pd.DataFrame(ov)
    ov.to_csv(TAB / "t5_set_overlap.csv", index=False)
    md.append("\n### Fraction of top-k nodes shared (mean over subjects)\n")
    md.append(md_table(ov.round(3)))

    (TAB / "all_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
