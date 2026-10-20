#!/usr/bin/env python
"""Inference for the network-null analysis (paper/new_idea.md).

For every (null, scale, c_mode, c_ref, k) and statistic: each subject's real value is
z-scored against that subject's own nulls; z-scores are tested across subjects with a
two-sided Wilcoxon signed-rank test (Holm within null x c_mode x statistic across
scales x c_ref x k); we also report the share of subjects whose real value lies above
the 95th (below the 5th) percentile of their null distribution. k = "avg" averages the
per-network statistic over k in {1,2,3,5} before comparison (the paper's k-averaging);
"avg2+" averages over k >= 2.

Mediation (rich club / hub connectivity): within each null type, pooled over subjects,
stat ~ is_real + C(subject) with and without the network statistic; proportion mediated
= 1 - beta_real(with) / beta_real(without). Also the median within-subject Spearman rho
between the network statistic and the set statistic across that subject's nulls.

Outputs (results/revision/nulls/): networks.csv.gz, sets.csv.gz, null_fidelity.csv,
inference.csv, network_inference.csv, mediation.csv, summary_tables.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

from _common import OUT_DIR

SET_STATS = [
    "degree_q1_pct",  # headline
    "degree_q1",
    "degree_loading",
    "degree_perron_share",
    "rho_loading_q1_all",
    "degree_q1perp",
    "degree_q1perp_pct",
    "degree_er_pct",
    "degree_er",
    "cycle_q1_pct",
    "cycle_er_pct",
    "cycle_minus_degree_er",
    "cycle_minus_degree_q1",
]
NET_STATS = ["ipr_v1", "gap", "rc_bu", "rc_wu", "hub_strength_frac", "decay_ratio_c0.1"]
MEDIATORS = ["rc_bu", "rc_wu", "hub_strength_frac", "ipr_v1"]
MEDIATED = ["degree_q1_pct", "degree_q1"]


def load_chunks() -> tuple[pd.DataFrame, pd.DataFrame]:
    nets, sets = [], []
    for path in sorted((OUT_DIR / "chunks").glob("*/scale-*/sub-*_network.csv.gz")):
        nets.append(pd.read_csv(path))
        sets.append(pd.read_csv(path.with_name(path.name.replace("_network", "_sets"))))
    net = pd.concat(nets, ignore_index=True)
    st = pd.concat(sets, ignore_index=True)
    net["decay_ratio_c0.1"] = 0.1 / (0.1 + net["gap"])
    if "cycle_er" in st:
        st["cycle_minus_degree_er"] = st["cycle_er"] - st["degree_er"]
        st["cycle_minus_degree_q1"] = st["cycle_q1"] - st["degree_q1"]
    return net, st


def add_k_averages(st: pd.DataFrame) -> pd.DataFrame:
    keys = ["null", "null_index", "scale", "subject_index", "subject_id", "c_mode", "c_ref"]
    cols = [c for c in st.columns if c not in keys + ["k", "c", "decay_ratio"]]
    out = [st.assign(k=st["k"].astype(str))]
    for label, ks in (("avg", (1, 2, 3, 5)), ("avg2+", (2, 3, 5))):
        sub = st[st["k"].isin(ks)]
        agg = sub.groupby(keys, as_index=False)[cols + ["c", "decay_ratio"]].mean()
        out.append(agg.assign(k=label))
    return pd.concat(out, ignore_index=True)


def compare(real: pd.Series, nulls: pd.DataFrame, stat: str) -> dict | None:
    """real: one row per subject indexed by subject_id; nulls: rows with subject_id + stat."""
    rows = []
    for sid, grp in nulls.groupby("subject_id"):
        if sid not in real.index:
            continue
        x = grp[stat].to_numpy(dtype=float)
        r = float(real.loc[sid])
        x = x[np.isfinite(x)]
        if x.size < 2 or not np.isfinite(r):
            continue
        sd = x.std(ddof=1)
        sd = sd if sd > 1e-12 * max(1.0, abs(x.mean())) else 0.0  # identical values: avoid float-noise z
        z = (r - x.mean()) / sd if sd > 0 else (0.0 if np.isclose(r, x.mean()) else np.nan)
        rows.append(
            {
                "z": z,
                "delta": r - x.mean(),
                "real": r,
                "null_mean": x.mean(),
                "above95": r > np.percentile(x, 95),
                "below5": r < np.percentile(x, 5),
                "p_upper": (np.sum(x >= r) + 1) / (x.size + 1),
            }
        )
    if not rows:
        return None
    d = pd.DataFrame(rows)
    z = d["z"].dropna().to_numpy()
    p = wilcoxon(z).pvalue if np.any(z != 0) and z.size >= 5 else np.nan
    return {
        "n_subjects": len(d),
        "real_mean": d["real"].mean(),
        "null_mean": d["null_mean"].mean(),
        "delta_mean": d["delta"].mean(),
        "delta_median": d["delta"].median(),
        "z_median": float(np.median(z)) if z.size else np.nan,
        "z_mean": float(np.mean(z)) if z.size else np.nan,
        "n_z_undefined": int(d["z"].isna().sum()),
        "wilcoxon_p": p,
        "share_above95": d["above95"].mean(),
        "share_below5": d["below5"].mean(),
        "n_above95": int(d["above95"].sum()),
        "n_below5": int(d["below5"].sum()),
    }


def set_inference(st: pd.DataFrame) -> pd.DataFrame:
    rows = []
    real = st[st["null"] == "real"]
    for (null, scale, mode, c_ref, k), grp in st[st["null"] != "real"].groupby(["null", "scale", "c_mode", "c_ref", "k"]):
        r = real[(real["scale"] == scale) & (real["c_mode"] == mode) & (real["c_ref"] == c_ref) & (real["k"] == k)].set_index("subject_id")
        for stat in SET_STATS:
            if stat not in grp or grp[stat].isna().all():
                continue
            res = compare(r[stat], grp, stat)
            if res:
                rows.append({"null": null, "scale": scale, "c_mode": mode, "c_ref": c_ref, "k": k, "stat": stat, **res})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for _, idx in out.groupby(["null", "c_mode", "stat"]).groups.items():
        p = out.loc[idx, "wilcoxon_p"]
        ok = p.notna()
        if ok.any():
            out.loc[p[ok].index, "p_holm"] = multipletests(p[ok], method="holm")[1]
    return out


def network_inference(net: pd.DataFrame) -> pd.DataFrame:
    rows = []
    real = net[net["null"] == "real"]
    for (null, scale), grp in net[net["null"] != "real"].groupby(["null", "scale"]):
        r = real[real["scale"] == scale].set_index("subject_id")
        for stat in NET_STATS:
            res = compare(r[stat], grp, stat)
            if res:
                rows.append({"null": null, "scale": scale, "stat": stat, **res})
    return pd.DataFrame(rows)


def null_fidelity(net: pd.DataFrame) -> pd.DataFrame:
    cols = ["strength_r", "strength_max_rel_err", "strength_top5_jaccard", "degree_preserved", "edge_overlap", "mean_edge_length", "weighted_edge_length", "gap", "ipr_v1", "rc_bu"]
    return net.groupby(["null", "scale"])[cols].mean().reset_index()


def mediation(net: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = st[st["k"].isin(["avg", "3"])]
    merged = sub.merge(net[["null", "null_index", "scale", "subject_id"] + MEDIATORS], on=["null", "null_index", "scale", "subject_id"])
    for null in sorted(set(merged["null"]) - {"real"}):
        for (scale, mode, c_ref, k), grp in merged[merged["null"].isin(["real", null])].groupby(["scale", "c_mode", "c_ref", "k"]):
            if not (grp["null"] == null).any() or not (grp["null"] == "real").any():
                continue
            grp = grp.assign(is_real=(grp["null"] == "real").astype(float))
            for y in MEDIATED:
                base = smf.ols(f"{y} ~ is_real + C(subject_id)", data=grp).fit()
                for m in MEDIATORS:
                    if grp[m].std() == 0:
                        continue
                    full = smf.ols(f"{y} ~ is_real + {m} + C(subject_id)", data=grp).fit()
                    nulls_only = grp[grp["null"] == null]
                    rhos = [
                        spearmanr(g[m], g[y]).statistic
                        for _, g in nulls_only.groupby("subject_id")
                        if g[m].nunique() > 1 and g[y].nunique() > 1
                    ]
                    b0 = base.params["is_real"]
                    b1 = full.params["is_real"]
                    rows.append(
                        {
                            "null": null, "scale": scale, "c_mode": mode, "c_ref": c_ref, "k": k,
                            "stat": y, "mediator": m,
                            "beta_real_total": b0, "beta_real_direct": b1,
                            "beta_mediator": full.params[m], "p_mediator": full.pvalues[m],
                            "prop_mediated": 1 - b1 / b0 if b0 != 0 else np.nan,
                            "median_within_subject_rho": float(np.nanmedian(rhos)) if rhos else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def fmt_cell(r: pd.Series) -> str:
    return f"{r['delta_mean']:+.3f} | z̃ {r['z_median']:+.1f} | {r['n_above95']}/{r['n_subjects']} > 95th | p_H {r['p_holm']:.1g}"


def summary_tables(inf: pd.DataFrame, ninf: pd.DataFrame, fid: pd.DataFrame, med: pd.DataFrame) -> str:
    lines = ["# Null-model summary tables (auto-generated by 02_analyze.py)", ""]
    lines += [
        "Cells: mean(real - null mean) | median per-subject z | subjects with real above their 95th null percentile | Holm p (Wilcoxon on z).",
        "k = avg (k-averaged over 1,2,3,5) unless stated.",
        "",
    ]
    for stat, title in [
        ("degree_q1_pct", "Headline: percentile of the degree set's q1 among random sets"),
        ("degree_q1", "Degree-set q1 = λ1/tr W"),
        ("degree_loading", "Degree-set Perron loading L_S"),
        ("rho_loading_q1_all", "ρ(L_S, q1) across all sets"),
        ("degree_q1perp_pct", "W⊥: percentile of degree-set q1 after projecting out v1"),
        ("degree_er_pct", "Percentile of degree-set ER among random sets"),
        ("cycle_minus_degree_er", "Cycle − degree ER (114 only)"),
    ]:
        sub = inf[(inf["stat"] == stat) & (inf["k"] == "avg")]
        if sub.empty:
            continue
        lines += [f"## {title}", "", "| null | c mode | c | " + " | ".join(str(s) for s in sorted(sub["scale"].unique())) + " |"]
        lines.append("|" + "---|" * (3 + sub["scale"].nunique()))
        for (null, mode, c_ref), g in sub.groupby(["null", "c_mode", "c_ref"]):
            cells = [fmt_cell(g[g["scale"] == s].iloc[0]) if (g["scale"] == s).any() else "" for s in sorted(sub["scale"].unique())]
            lines.append(f"| {null} | {mode} | {c_ref} | " + " | ".join(cells) + " |")
        lines.append("")
    scales = sorted(ninf["scale"].unique())
    lines += ["## Network-level statistics (real vs null)", "", "| null | stat | " + " | ".join(str(s) for s in scales) + " |", "|" + "---|" * (2 + len(scales))]
    for (null, stat), g in ninf.groupby(["null", "stat"]):
        cells = [f"{r['real_mean']:.3g} vs {r['null_mean']:.3g} (z̃ {r['z_median']:+.1f}, {r['n_above95']}↑/{r['n_below5']}↓)" for _, r in g.sort_values("scale").iterrows()]
        lines.append(f"| {null} | {stat} | " + " | ".join(cells) + " |")
    lines += ["", "## Null fidelity (means over subjects and nulls)", "", fid.round(3).to_markdown(index=False), ""]
    if not med.empty:
        m = med[(med["stat"] == "degree_q1_pct") & (med["k"] == "avg") & (med["c_ref"] == 0.1)]
        lines += ["## Mediation of the headline statistic (k = avg, c = 0.1)", "", m.round(3).to_markdown(index=False), ""]
    return "\n".join(lines)


def main():
    net, st = load_chunks()
    print("networks", net.groupby(["null", "scale"]).size().to_dict(), flush=True)
    st = add_k_averages(st)
    net.to_csv(OUT_DIR / "networks.csv.gz", index=False)
    st.to_csv(OUT_DIR / "sets.csv.gz", index=False)
    fid = null_fidelity(net)
    fid.to_csv(OUT_DIR / "null_fidelity.csv", index=False)
    inf = set_inference(st)
    inf.to_csv(OUT_DIR / "inference.csv", index=False)
    ninf = network_inference(net)
    ninf.to_csv(OUT_DIR / "network_inference.csv", index=False)
    med = mediation(net, st)
    med.to_csv(OUT_DIR / "mediation.csv", index=False)
    (OUT_DIR / "summary_tables.md").write_text(summary_tables(inf, ninf, fid, med))
    print((OUT_DIR / "summary_tables.md").read_text())


if __name__ == "__main__":
    main()
