"""Generate markdown tables for REPORT.md from the regime CSV outputs -> results/revision/regime/report_tables.md
Also writes task1_signflip_per_k.csv and task1_csweep_per_k.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import OUT_DIR, holm, paired_stats

S = (68, 114, 219)
out = []


def fmtp(p):
    return "<1e-12" if p < 1e-12 else f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def cell(r, nd=2):
    return f"{r['mean_diff']:+.{nd}f} ({int(r['n_positive'])})"


def full_rows(t, keys, nd=3):
    t = t.copy()
    t["mean [95% CI]"] = t.apply(lambda r: f"{r['mean_diff']:+.{nd}f} [{r['mean_ci_low']:+.{nd}f}, {r['mean_ci_high']:+.{nd}f}]", axis=1)
    t["median [95% CI]"] = t.apply(lambda r: f"{r['median_diff']:+.{nd}f} [{r['median_ci_low']:+.{nd}f}, {r['median_ci_high']:+.{nd}f}]", axis=1)
    t["n+/70"] = t["n_positive"].astype(int)
    t["p"] = t["wilcoxon_p"].map(fmtp)
    t["p_Holm"] = t["p_holm"].map(fmtp)
    t["r_rb"] = t["rank_biserial"].map(lambda x: f"{x:+.2f}")
    return t[keys + ["mean [95% CI]", "median [95% CI]", "n+/70", "p", "p_Holm", "r_rb"]]


# ---- Task 1 ----
t1 = pd.read_csv(OUT_DIR / "task1_csweep_cycle_minus_degree.csv")
for metric in ("effective_rank", "participation_ratio"):
    x = t1[t1.metric == metric].copy()
    x["cell"] = x.apply(cell, axis=1)
    out.append(f"### T1-compact-{metric}\n\n" + md(x.pivot(index="c", columns="scale", values="cell").reset_index()))
    out.append(f"### T1-full-{metric}\n\n" + md(full_rows(x.sort_values(["scale", "c"]), ["scale", "c"])))
flip = pd.read_csv(OUT_DIR / "task1_signflip.csv")
out.append("### T1-signflip\n\n" + md(flip.round(3)))

# per-k
m = pd.read_csv(OUT_DIR / "csweep_sets_methods.csv")
w = m[m.method.isin(["cycle_participation", "degree_strength"])].pivot_table(index=["scale", "c", "k", "subject_id"], columns="method", values="effective_rank")
dk = (w["cycle_participation"] - w["degree_strength"]).rename("d").reset_index()
rows = [{"scale": s, "c": c, "k": k, **paired_stats(g["d"].to_numpy())} for (s, c, k), g in dk.groupby(["scale", "c", "k"])]
pk = holm(pd.DataFrame(rows), ["scale", "k"])
pk.to_csv(OUT_DIR / "task1_csweep_per_k.csv", index=False)
pk["cell"] = pk.apply(cell, axis=1)
tab = pk.pivot_table(index="c", columns=["scale", "k"], values="cell", aggfunc="first")
tab.columns = [f"{s}/k={k}" for s, k in tab.columns]
out.append("### T1-per-k\n\n" + md(tab.reset_index()))


def crossing(cs, ys, level=0.0):
    x = np.log10(np.asarray(cs, float)); y = np.asarray(ys, float) - level
    for i in range(len(y) - 1):
        if y[i] > 0 and y[i + 1] <= 0:
            return 10 ** (x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1]))
    return np.nan if y[0] > 0 else 0.0


fk = []
for (s, k), g in pk.groupby(["scale", "k"]):
    g = g.sort_values("c")
    fk.append({"scale": s, "k": k, "c_flip_mean": crossing(g.c, g.mean_diff), "c_flip_npos35": crossing(g.c, g.n_positive, 35)})
fk = pd.DataFrame(fk)
fk.to_csv(OUT_DIR / "task1_signflip_per_k.csv", index=False)
fk["c_flip_mean"] = fk["c_flip_mean"].map(lambda v: "none (≤0.01 or >2)" if not np.isfinite(v) else ("<0.01" if v == 0 else f"{v:.3f}"))
fk["c_flip_npos35"] = fk["c_flip_npos35"].map(lambda v: "none" if not np.isfinite(v) else ("<0.01" if v == 0 else f"{v:.3f}"))
out.append("### T1-signflip-per-k\n\n" + md(fk))

# vs random
r = pd.read_csv(OUT_DIR / "task1_csweep_vs_random.csv")
for metric in ("er_minus_random", "pr_minus_random"):
    x = r[(r.metric == metric) & r.method.isin(["cycle_participation", "degree_strength", "eigenvector_centrality", "betweenness"])].copy()
    x["cell"] = x.apply(lambda q: f"{q['mean_diff']:+.2f} / {q['mean_percentile']:.0f}", axis=1)
    tab = x.pivot_table(index="c", columns=["method", "scale"], values="cell", aggfunc="first")
    tab = tab[[(mm, s) for mm in ("cycle_participation", "degree_strength", "eigenvector_centrality", "betweenness") for s in S]]
    tab.columns = [f"{mm.split('_')[0]} {s}" for mm, s in tab.columns]
    out.append(f"### T1-random-{metric}\n\n" + md(tab.reset_index()))
    x["cell2"] = x.apply(lambda q: f"{int(q['n_positive'])} ({fmtp(q['p_holm'])})", axis=1)
    tab = x[x.method.isin(["cycle_participation", "degree_strength", "eigenvector_centrality"])].pivot_table(index="c", columns=["method", "scale"], values="cell2", aggfunc="first")
    tab.columns = [f"{mm.split('_')[0]} {s}" for mm, s in tab.columns]
    out.append(f"### T1-random-npos-{metric}\n\n" + md(tab.reset_index()))

# ---- Task 3 ----
t3m = pd.read_csv(OUT_DIR / "task3_perron_by_method.csv")
pl = t3m[t3m.c == 0.1].pivot_table(index="method", columns="scale", values="perron_loading").round(3)
out.append("### T3-loading\n\n" + md(pl.reset_index()))
x = t3m[t3m.method.isin(["degree_strength", "cycle_participation", "eigenvector_centrality", "random_mean"]) & t3m.c.isin([0.01, 0.03, 0.1, 0.3, 1.0, 2.0])]
tab = x.pivot_table(index="c", columns=["method", "scale"], values="perron_trace_fraction").round(3)
tab.columns = [f"{mm.split('_')[0]} {s}" for mm, s in tab.columns]
out.append("### T3-tracefrac\n\n" + md(tab.reset_index()))
sp = pd.read_csv(OUT_DIR / "csweep_spearman.csv")
tab = sp.groupby(["c", "scale", "k"]).rho_perronload_er.median().round(2).unstack(["scale", "k"])
tab.columns = [f"{s}/k={k}" for s, k in tab.columns]
out.append("### T3-rho-per-k\n\n" + md(tab.reset_index()))
t3s = pd.read_csv(OUT_DIR / "task3_perron_spearman.csv")
x = t3s[["scale", "c", "median_rho_rankings_only", "rho_across_subjects_dPerron_dER", "p_across_subjects"]].copy()
x["p_across_subjects"] = x["p_across_subjects"].map(fmtp)
x = x.round(2)
tab = x.pivot_table(index="c", columns="scale", values=["median_rho_rankings_only", "rho_across_subjects_dPerron_dER"])
tab.columns = [f"{a.replace('median_rho_rankings_only', 'within-subj, 9 rankings').replace('rho_across_subjects_dPerron_dER', 'across-subj ΔPerron vs ΔER')} {s}" for a, s in tab.columns]
out.append("### T3-rho-rankings-subjects\n\n" + md(tab.reset_index()))
dg = pd.read_csv(OUT_DIR / "task3_perron_diagnostic.csv")
tab = dg.groupby(["scale", "c", "k"])[["rho_er_strength", "rho_er_perron", "partial_rho_er_perron_given_strength"]].median().round(2).reset_index()
out.append("### T3-diagnostic\n\n" + md(tab))

# ---- Task 2 ----
n = pd.read_csv(OUT_DIR / "nctpy_cohort_summary.csv")
out.append("### T2-cohort\n\n" + md(full_rows(n.sort_values(["metric", "normalization", "scale"]), ["normalization", "metric", "scale"])))

# ---- Task 4 ----
t4 = pd.read_csv(OUT_DIR / "task4_finite_horizon_summary.csv")
t4["T"] = t4["T"].map(lambda v: "∞" if not np.isfinite(v) else f"{v:g}")
for metric in ("effective_rank", "participation_ratio"):
    x = t4[t4.metric == metric].copy()
    x["cell"] = x.apply(cell, axis=1)
    tab = x.pivot_table(index=["eps", "c", "T"], columns="scale", values="cell", aggfunc="first").reset_index()
    tab["eps"] = tab["eps"].map(lambda v: f"{v:.0e}")
    out.append(f"### T4-{metric}\n\n" + md(tab))
x = t4[(t4.metric == "effective_rank") & (t4.eps == 1e-5)]
out.append("### T4-full-eps1e-5\n\n" + md(full_rows(x.sort_values(["c", "T", "scale"]), ["c", "T", "scale"])))

# ---- Task 5 ----
s1 = pd.read_csv(OUT_DIR / "task5_S1_epsilon_summary.csv")
for c in (0.1, 0.01):
    for metric in ("effective_rank", "participation_ratio"):
        x = s1[(s1.c == c) & (s1.metric == metric)].copy()
        x["cell"] = x.apply(lambda r: f"{r['mean_diff']:.3f} ({int(r['n_positive'])}; p_Holm {fmtp(r['p_holm'])})", axis=1)
        tab = x.pivot_table(index="eps", columns="scale", values="cell", aggfunc="first").reset_index()
        tab["eps"] = tab["eps"].map(lambda v: f"{v:.0e}")
        out.append(f"### S1-c{c}-{metric}\n\n" + md(tab))
s3 = pd.read_csv(OUT_DIR / "task5_S3_largek_summary.csv")
for c in (0.1, 0.01):
    for metric in ("effective_rank", "participation_ratio"):
        x = s3[(s3.c == c) & (s3.metric == metric)].copy()
        blocks = []
        for s in S:
            y = x[x.scale == s].sort_values("k")
            blocks.append([(int(r.k), f"{r.mean_diff:.2f}", int(r.n_positive), fmtp(r.p_holm)) for r in y.itertuples()])
        L = max(map(len, blocks))
        lines = ["| k (68) | Mean | n+ | p_Holm | k (114) | Mean | n+ | p_Holm | k (219) | Mean | n+ | p_Holm |", "|" + "---|" * 12]
        for i in range(L):
            cells = []
            for b in blocks:
                cells += list(map(str, b[i])) if i < len(b) else ["", "", "", ""]
            lines.append("| " + " | ".join(cells) + " |")
        out.append(f"### S3-c{c}-{metric}\n\n" + "\n".join(lines))

# ---- Task 6 ----
a = pd.read_csv(OUT_DIR / "task6_driver_sets.csv")
g = a.groupby(["scale", "eps", "method"])[["fraction_at_floor", "fraction_below_eps", "log10_average_energy", "log10_energy_above_eps", "log10_lambda_max"]].mean().reset_index()
g["log10(N/eps)"] = np.log10(g.scale / g.eps).round(3)
g["eps"] = g["eps"].map(lambda v: f"{v:.0e}")
g = g[g.method.isin(["degree_strength", "cycle_participation", "eigenvector_centrality", "random_mean"])].round(3)
out.append("### T6-methods\n\n" + md(g))
d6 = pd.read_csv(OUT_DIR / "task6_delta_log10E_summary.csv")
d6["eps"] = d6["eps"].map(lambda v: f"{v:.0e}")
out.append("### T6-delta\n\n" + md(full_rows(d6, ["scale", "eps"], nd=4)))
b = pd.read_csv(OUT_DIR / "task6_landscape.csv")
gb = b.groupby(["scale", "eps"]).agg(var_log10E=("var_log10E", "mean"), iqr_log10E=("iqr_log10E", "mean"), range_log10E=("range_log10E", "mean"),
                                     median_log10E=("median_log10E", "mean"), log10_N_over_eps=("log10_N_over_eps", "first"),
                                     near_opt_1pct=("near_opt_01pct", "mean"), var_above_eps=("var_log10E_above_eps", "mean"),
                                     iqr_above_eps=("iqr_log10E_above_eps", "mean"), var_nonfloor=("var_log10E_nonfloor", "mean"),
                                     frac_below_eps=("mean_fraction_below_eps", "mean")).reset_index()
gb["eps"] = gb["eps"].map(lambda v: f"{v:.0e}")
for col in ("var_log10E", "var_nonfloor"):
    gb[col] = gb[col].map(lambda v: f"{v:.2e}")
out.append("### T6-landscape\n\n" + md(gb.round(4)))
gk = b.groupby(["scale", "k", "eps"]).agg(var_log10E=("var_log10E", "mean"), iqr_log10E=("iqr_log10E", "mean"), var_above_eps=("var_log10E_above_eps", "mean")).reset_index()
gk["eps"] = gk["eps"].map(lambda v: f"{v:.0e}")
gk["var_log10E"] = gk["var_log10E"].map(lambda v: f"{v:.2e}")
out.append("### T6-landscape-per-k\n\n" + md(gk.round(4)))

(OUT_DIR / "report_tables.md").write_text("\n\n".join(out) + "\n")
print("written")
