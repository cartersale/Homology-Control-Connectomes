#!/usr/bin/env python
"""Tables for P1-P3 from results/revision/q1/set_q1.csv.gz.

P1  every criterion vs its subject's 100 random sets: mean difference from the random mean,
    mean percentile among random sets, subjects above the random mean, Wilcoxon on the
    differences (Holm within metric x c x k across criteria and scales).
P2  cycle participation vs degree-/spatially-matched sets (difference from the matched-set
    mean), and degree-residualized cycle participation (cycle_resid) vs random and vs degree.
P3  cycle - degree before and after projecting out v1 (q1, ER, PR vs q1_perp, er_perp, pr_perp).

k = "avg" averages the per-k metric (or per-k percentile) over k in {1,2,3,5} within subject;
"avg2+" over k >= 2. Outputs: p1_vs_random.csv, p2_matched.csv, p3_perp.csv, tables.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "results" / "revision" / "q1"
METRICS = ["q1", "er", "pr", "q1_perp", "er_perp", "pr_perp", "loading", "perron_share"]
CRITERIA = [
    "degree_strength", "eigenvector_centrality", "top_avg_ctrl", "betweenness", "participation_coefficient",
    "triangle_participation", "tetrahedron_participation", "cycle_participation", "cycle_resid",
    "degree_normalized_cycle_participation", "cycle_criticality", "top_modal_gu", "top_modal_ct", "low_perron",
]
K_GROUPS = {"1": (1,), "2": (2,), "3": (3,), "5": (5,), "avg": (1, 2, 3, 5), "avg2+": (2, 3, 5)}


def paired(d: np.ndarray) -> dict:
    d = d[np.isfinite(d)]
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return {"n": d.size, "mean": d.mean(), "median": np.median(d), "n_positive": int((d > 0).sum()), "p": p}


def holm(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    df["p_holm"] = np.nan
    for _, idx in df.groupby(by).groups.items():
        df.loc[idx, "p_holm"] = multipletests(df.loc[idx, "p"].fillna(1.0), method="holm")[1]
    return df


def per_k_frames(df: pd.DataFrame):
    """Per subject x scale x c x k: random mean, random percentiles of each named set, matched means."""
    keys = ["subject_id", "scale", "c", "k"]
    rnd = df[df["family"] == "random"]
    named = df[df["family"].isin(CRITERIA)].set_index(keys + ["family"])[METRICS]
    rmean = rnd.groupby(keys)[METRICS].mean()
    pct_rows = []
    for key, g in rnd.groupby(keys):
        ref = {m: g[m].to_numpy() for m in ("q1", "er", "q1_perp")}
        for fam in CRITERIA:
            try:
                row = named.loc[key + (fam,)]
            except KeyError:
                continue
            pct_rows.append({**dict(zip(keys, key)), "family": fam,
                             **{f"{m}_pct": (np.sum(ref[m] < row[m]) + 0.5 * np.sum(ref[m] == row[m])) / ref[m].size for m in ref}})
    pct = pd.DataFrame(pct_rows).set_index(keys + ["family"])
    matched = df[df["family"].str.endswith("matched")].groupby(keys + ["family"])[METRICS].mean()
    return named, rmean, pct, matched


def kgroup(frame: pd.DataFrame, ks) -> pd.DataFrame:
    """Average over the k values in ks within subject x scale x c (x family)."""
    f = frame.reset_index()
    f = f[f["k"].isin(ks)]
    by = [c for c in ("subject_id", "scale", "c", "family") if c in f.columns]
    return f.drop(columns="k").groupby(by).mean(numeric_only=True)


def p1(named, rmean, pct) -> pd.DataFrame:
    rows = []
    for kl, ks in K_GROUPS.items():
        n, r, p = kgroup(named, ks), kgroup(rmean, ks), kgroup(pct, ks)
        diff = n.sub(r, axis=0, level=None) if False else n.reset_index().merge(r.reset_index(), on=["subject_id", "scale", "c"], suffixes=("", "_rand"))
        diff = diff.merge(p.reset_index(), on=["subject_id", "scale", "c", "family"])
        for (fam, scale, c), g in diff.groupby(["family", "scale", "c"]):
            for m in ("q1", "er", "pr", "q1_perp"):
                st = paired((g[m] - g[f"{m}_rand"]).to_numpy())
                rows.append({"family": fam, "scale": scale, "c": c, "k": kl, "metric": m, **st,
                             "value_mean": g[m].mean(), "random_mean": g[f"{m}_rand"].mean(),
                             "pct_mean": g[f"{m}_pct"].mean() if f"{m}_pct" in g else np.nan})
    return holm(pd.DataFrame(rows), ["metric", "c", "k"])


def p2(named, rmean, matched) -> pd.DataFrame:
    """Differences are taken per k, then averaged over the k values present for both sets
    (spatial matching is undefined at k = 1, so its "avg" is effectively k in {2,3,5}; see n_k)."""
    n, r, mt = named.reset_index(), rmean.reset_index(), matched.reset_index()
    cyc = n[n["family"] == "cycle_participation"].drop(columns="family")
    deg = n[n["family"] == "degree_strength"].drop(columns="family")
    res = n[n["family"] == "cycle_resid"].drop(columns="family")
    on = ["subject_id", "scale", "c", "k"]
    comparisons = [("cycle - degree", cyc, deg), ("cycle - random", cyc, r), ("cycle_resid - random", res, r), ("cycle_resid - degree", res, deg)]
    for fam in ("degree_matched", "spatial_matched", "degree_spatial_matched"):
        comparisons.append((f"cycle - {fam}", cyc, mt[mt["family"] == fam].drop(columns="family")))
    rows = []
    for label, a, b in comparisons:
        m = a.merge(b, on=on, suffixes=("_a", "_b"))
        for met in ("q1", "er", "pr"):
            m[f"d_{met}"] = m[f"{met}_a"] - m[f"{met}_b"]
        for kl, ks in K_GROUPS.items():
            sub = m[m["k"].isin(ks)]
            n_k = sub["k"].nunique()
            if n_k == 0:
                continue
            per_subject = sub.groupby(["subject_id", "scale", "c"])[[f"d_{x}" for x in ("q1", "er", "pr")]].mean().reset_index()
            for (scale, c), g in per_subject.groupby(["scale", "c"]):
                for met in ("q1", "er", "pr"):
                    rows.append({"comparison": label, "scale": scale, "c": c, "k": kl, "n_k": n_k, "metric": met,
                                 **paired(g[f"d_{met}"].to_numpy())})
    return holm(pd.DataFrame(rows), ["comparison", "metric", "c", "k"])


def p3(named) -> pd.DataFrame:
    rows = []
    for kl, ks in K_GROUPS.items():
        n = kgroup(named, ks).reset_index()
        cyc = n[n["family"] == "cycle_participation"].drop(columns="family")
        deg = n[n["family"] == "degree_strength"].drop(columns="family")
        m = cyc.merge(deg, on=["subject_id", "scale", "c"], suffixes=("_cyc", "_deg"))
        for (scale, c), g in m.groupby(["scale", "c"]):
            for met in ("q1", "er", "pr"):
                full = (g[f"{met}_cyc"] - g[f"{met}_deg"]).to_numpy()
                perp = (g[f"{met}_perp_cyc"] - g[f"{met}_perp_deg"]).to_numpy()
                a, b = paired(full), paired(perp)
                rows.append({"scale": scale, "c": c, "k": kl, "metric": met,
                             "full_mean": a["mean"], "full_n_positive": a["n_positive"], "full_p": a["p"],
                             "perp_mean": b["mean"], "perp_n_positive": b["n_positive"], "p": b["p"],
                             "fraction_removed": 1 - b["mean"] / a["mean"] if a["mean"] != 0 else np.nan})
    return holm(pd.DataFrame(rows), ["metric", "c", "k"])


def fmt3(df, value, label_cols, extra=None):
    out = []
    for keys, g in df.groupby(label_cols, sort=False):
        cells = []
        for s in (68, 114, 219):
            r = g[g["scale"] == s]
            if r.empty:
                cells.append("")
                continue
            r = r.iloc[0]
            cell = f"{r[value]:+.3f}" if value != "pct_mean" else f"{r[value]:.2f}"
            if extra:
                cell += extra(r)
            cells.append(cell)
        keys = keys if isinstance(keys, tuple) else (keys,)
        out.append("| " + " | ".join(map(str, keys)) + " | " + " | ".join(cells) + " |")
    return out


def main():
    df = pd.read_csv(OUT / "set_q1.csv.gz")
    assert np.isfinite(df[METRICS].to_numpy()).all(), "non-finite metrics"
    named, rmean, pct, matched = per_k_frames(df)
    t1, t2, t3 = p1(named, rmean, pct), p2(named, rmean, matched), p3(named)
    t1.to_csv(OUT / "p1_vs_random.csv", index=False)
    t2.to_csv(OUT / "p2_matched.csv", index=False)
    t3.to_csv(OUT / "p3_perp.csv", index=False)

    L = ["# q1 tables (P1-P3; auto-generated by 02_tables.py)", "",
         "Cells are 68 / 114 / 219. c = 0.1 and k-averaged unless stated. Δ = criterion − mean of the subject's 100 random sets.", ""]
    order = {f: i for i, f in enumerate(CRITERIA)}
    for metric, title in (("q1", "q1 = λ1/tr W"), ("er", "Effective rank"), ("q1_perp", "q1 of W⊥ (v1 projected out)")):
        s = t1[(t1["metric"] == metric) & (t1["c"] == 0.1) & (t1["k"] == "avg")].copy()
        s["o"] = s["family"].map(order)
        s = s.sort_values(["o", "scale"])
        L += [f"## P1: {title} vs random", "", "| criterion | 68 | 114 | 219 |", "|---|---|---|---|"]
        L += fmt3(s, "mean", ["family"], lambda r: f" (pct {r['pct_mean']:.2f}, {r['n_positive']}/70)" if np.isfinite(r["pct_mean"]) else f" ({r['n_positive']}/70)")
        L.append("")
    s = t1[(t1["metric"] == "q1") & (t1["c"] == 0.1) & t1["k"].isin(["1", "2", "3", "5"]) & t1["family"].isin(["degree_strength", "eigenvector_centrality", "top_avg_ctrl", "cycle_participation"])]
    L += ["## P1: q1 vs random by k (degree, eigenvector, average controllability, cycle)", "", "| criterion | k | 68 | 114 | 219 |", "|---|---|---|---|---|"]
    L += fmt3(s.sort_values(["family", "k", "scale"]), "mean", ["family", "k"], lambda r: f" ({r['n_positive']}/70)")
    L.append("")
    s = t2[(t2["c"] == 0.1) & (t2["k"] == "avg")]
    for metric in ("q1", "er"):
        L += [f"## P2: matched and residualized controls, {metric} (positive = first set higher)", "", "| comparison | 68 | 114 | 219 |", "|---|---|---|---|"]
        L += fmt3(s[s["metric"] == metric], "mean", ["comparison"], lambda r: f" ({r['n_positive']}/70, p_H {r['p_holm']:.1g})")
        L.append("")
    for c in (0.01, 0.1, 1.0):
        s = t3[(t3["c"] == c) & (t3["k"] == "avg")]
        L += [f"## P3: cycle − degree before → after W⊥, c = {c}", "", "| metric | 68 | 114 | 219 |", "|---|---|---|---|"]
        L += fmt3(s, "full_mean", ["metric"], lambda r: f" ({r['full_n_positive']}/70) → {r['perp_mean']:+.3f} ({r['perp_n_positive']}/70, p_H {r['p_holm']:.1g})")
        L.append("")
    (OUT / "tables.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
