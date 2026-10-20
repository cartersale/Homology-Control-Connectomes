#!/usr/bin/env python
"""Statistics, tables and draft figure for the reciprocal hub-lesion analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, spearmanr, wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lesion_core import OUT_DIR  # noqa: E402

LESIONS = ["degree", "cycle", "random"]
STRATS = ["degree_strength", "cycle_participation"]
SLABEL = {"degree_strength": "degree", "cycle_participation": "cycle"}
METRICS = {
    "dER": "ER loss (intact - lesioned)",
    "dPR": "PR loss",
    "dER_perN": "ER/N loss",
    "dER_excess": "ER loss in excess of random driver sets",
    "dPR_excess": "PR loss in excess of random driver sets",
    "dPct_ER": "ER percentile loss vs 100 random sets",
    "dPct_PR": "PR percentile loss vs 100 random sets",
    "dlog10E": "log10 E_avg change (lesioned - intact)",
}
BOOT = 10000
RNG_SEED = 20260925


def derive(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["dER"] = d.intact_effective_rank - d.lesioned_effective_rank
    d["dPR"] = d.intact_participation_ratio - d.lesioned_participation_ratio
    d["dER_perN"] = d.intact_effective_rank / d.n_intact - d.lesioned_effective_rank / d.n_lesioned
    d["dER_excess"] = d.dER - (d.ref_intact_er - d.ref_lesioned_er)
    d["dPR_excess"] = d.dPR - (d.ref_intact_pr - d.ref_lesioned_pr)
    d["dPct_ER"] = d.pct_intact_er - d.pct_lesioned_er
    d["dPct_PR"] = d.pct_intact_pr - d.pct_lesioned_pr
    d["dlog10E"] = d.lesioned_log10_average_energy - d.intact_log10_average_energy
    d["dlog10E_nonfloor"] = d.lesioned_log10_nonfloor_energy - d.intact_log10_nonfloor_energy
    d["size_pred_log10E"] = np.log10(d.n_lesioned / d.n_intact)
    return d


def stats(x: np.ndarray, rng: np.random.Generator) -> dict:
    x = np.asarray(x, dtype=float)
    boots = rng.choice(x, size=(BOOT, x.size), replace=True).mean(axis=1)
    nz = x[x != 0]
    if nz.size:
        ranks = rankdata(np.abs(nz))
        rbc = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
        p = float(wilcoxon(nz).pvalue)
    else:
        rbc, p = 0.0, 1.0
    return {"mean": x.mean(), "median": np.median(x), "ci_lo": np.quantile(boots, 0.025),
            "ci_hi": np.quantile(boots, 0.975), "n_pos": int((x > 0).sum()), "n": x.size,
            "p": p, "r_rb": rbc}


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (p.size - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def per_subject(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Average random-lesion draws, then k -> one value per subject/scale/h/lesion/strategy/protocol."""
    keys = ["protocol", "lesion_type", "scale", "h", "subject", "strategy"]
    return d.groupby(keys + ["k"])[cols].mean().groupby(keys)[cols].mean().reset_index()


def crossover_tables(ps: pd.DataFrame, metric: str, protocol: str, rng) -> pd.DataFrame:
    sub = ps[ps.protocol == protocol]
    wide = sub.pivot_table(index=["lesion_type", "scale", "h", "subject"], columns="strategy", values=metric)
    wide["D"] = wide["cycle_participation"] - wide["degree_strength"]
    wide = wide.reset_index()
    rows = []
    for (lt, sc, h), g in wide.groupby(["lesion_type", "scale", "h"]):
        rows.append({"lesion_type": lt, "scale": str(sc), "h": str(h),
                     "mean_degree": g.degree_strength.mean(), "mean_cycle": g.cycle_participation.mean(),
                     **stats(g.D.values, rng)})
    for lt, g in wide.groupby("lesion_type"):
        pooled = g.groupby("subject")[["degree_strength", "cycle_participation", "D"]].mean()
        rows.append({"lesion_type": lt, "scale": "pooled", "h": "pooled",
                     "mean_degree": pooled.degree_strength.mean(), "mean_cycle": pooled.cycle_participation.mean(),
                     **stats(pooled.D.values, rng)})
    # crossover interaction: D(cycle lesion) - D(degree lesion), and vs random
    piv = wide.pivot_table(index=["scale", "h", "subject"], columns="lesion_type", values="D").reset_index()
    for a, b in [("cycle", "degree"), ("cycle", "random"), ("degree", "random")]:
        piv["I"] = piv[a] - piv[b]
        for (sc, h), g in piv.groupby(["scale", "h"]):
            rows.append({"lesion_type": f"interaction {a}-{b}", "scale": str(sc), "h": str(h),
                         "mean_degree": np.nan, "mean_cycle": np.nan, **stats(g.I.values, rng)})
        pooled = piv.groupby("subject").I.mean()
        rows.append({"lesion_type": f"interaction {a}-{b}", "scale": "pooled", "h": "pooled",
                     "mean_degree": np.nan, "mean_cycle": np.nan, **stats(pooled.values, rng)})
    out = pd.DataFrame(rows)
    out.insert(0, "metric", metric)
    out.insert(1, "protocol", protocol)
    main_family = ~out.lesion_type.str.startswith("interaction")
    out["p_holm"] = np.nan
    out.loc[main_family, "p_holm"] = holm(out.loc[main_family, "p"].values)
    out.loc[~main_family, "p_holm"] = holm(out.loc[~main_family, "p"].values)
    return out


def fmt_p(p: float) -> str:
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def md_table(df: pd.DataFrame, cols: list[str], floatfmt: str = "{:.3f}") -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if c in ("p", "p_holm"):
                vals.append(fmt_p(v))
            elif isinstance(v, (float, np.floating)):
                vals.append(floatfmt.format(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default=str(OUT_DIR / "lesion_rows.csv.gz"))
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    d = derive(pd.read_csv(args.rows))
    extra = ["dlog10E_nonfloor", "size_pred_log10E", "intact_fraction_at_floor", "lesioned_fraction_at_floor",
             "lesion_hits_topk_degree", "lesion_hits_topk_cycle", "drivers_displaced",
             "intact_effective_rank", "lesioned_effective_rank"]
    ps = per_subject(d, list(METRICS) + extra)
    ps.to_csv(out / "per_subject_degradation.csv", index=False)

    # --- reproduction check (static protocol, degree lesion) ---
    rep = d[(d.protocol == "static") & (d.lesion_type == "degree")].groupby(["strategy", "h"]).dER.mean().unstack()

    # --- crossover statistics ---
    tabs = []
    for protocol in ["static", "recompute"]:
        for metric in METRICS:
            if metric == "dlog10E" and protocol == "recompute":
                continue
            tabs.append(crossover_tables(ps, metric, protocol, rng))
    cross = pd.concat(tabs, ignore_index=True)
    cross.to_csv(out / "crossover_stats.csv", index=False)

    # --- 3x2 summary (pooled over scale, h, k; random draws averaged) ---
    summ = (ps.groupby(["protocol", "lesion_type", "strategy", "subject"])[list(METRICS)].mean()
            .groupby(["protocol", "lesion_type", "strategy"]).mean().reset_index())
    summ.to_csv(out / "summary_3x2.csv", index=False)

    # --- energy / size diagnostics ---
    st = d[d.protocol == "static"]
    energy = (st.groupby(["scale", "lesion_type", "h", "strategy"])
              .agg(frac_floor_intact=("intact_fraction_at_floor", "mean"),
                   frac_floor_lesioned=("lesioned_fraction_at_floor", "mean"),
                   dlog10E=("dlog10E", "mean"), size_pred=("size_pred_log10E", "mean"),
                   ref_dlog10E=("ref_lesioned_log10E", "mean"), ref_i=("ref_intact_log10E", "mean"),
                   dlog10E_nonfloor=("dlog10E_nonfloor", "mean"))
              .reset_index())
    energy["ref_dlog10E"] = energy.ref_dlog10E - energy.ref_i
    energy = energy.drop(columns="ref_i")
    energy.to_csv(out / "energy_size_diagnostics.csv", index=False)
    # floor-count prediction: E_avg ~ n_floor/eps => dlog10E ~ log10(n_floor_les / n_floor_int)
    pred_floor = np.log10((st.lesioned_fraction_at_floor * st.n_lesioned) / (st.intact_fraction_at_floor * st.n_intact))
    r_size = np.corrcoef(st.dlog10E, st.size_pred_log10E)[0, 1]
    r_floor = np.corrcoef(st.dlog10E, pred_floor)[0, 1]
    resid_floor = (st.dlog10E - pred_floor)
    lesion_energy_diff = cross[(cross.metric == "dlog10E") & (cross.protocol == "static")]

    # --- random-lesion draw variability & mechanism diagnostics ---
    hits = (ps[ps.protocol == "static"].groupby(["lesion_type", "strategy"])
            [["lesion_hits_topk_degree", "lesion_hits_topk_cycle", "drivers_displaced"]].mean().reset_index())
    lesion_overlap = (d[(d.protocol == "static") & (d.lesion_type != "random") & (d.k == 1)
                        & (d.strategy == "degree_strength")]
                      .groupby(["scale", "h", "lesion_type"])[["lesion_hits_topk_degree", "lesion_hits_topk_cycle"]]
                      .mean().reset_index())

    # --- draft figure ---
    make_figure(ps, cross, energy, out / "fig_lesion_draft.png")

    # --- write markdown snippets consumed by the report ---
    snippets = {}
    snippets["reproduction"] = rep.round(3).to_markdown()
    for protocol in ["static", "recompute"]:
        s = summ[summ.protocol == protocol].copy()
        s["strategy"] = s.strategy.map(SLABEL)
        snippets[f"summary_{protocol}"] = md_table(
            s, ["lesion_type", "strategy"] + [m for m in METRICS if not (protocol == "recompute" and m == "dlog10E")])
    cols = ["lesion_type", "scale", "h", "mean_degree", "mean_cycle", "mean", "median", "ci_lo", "ci_hi",
            "n_pos", "p", "p_holm", "r_rb"]
    for protocol in ["static", "recompute"]:
        for metric in METRICS:
            c = cross[(cross.metric == metric) & (cross.protocol == protocol)]
            if c.empty:
                continue
            snippets[f"cross_{protocol}_{metric}_pooled"] = md_table(c[c.scale == "pooled"], cols)
            snippets[f"cross_{protocol}_{metric}_cells"] = md_table(
                c[(c.scale != "pooled") & ~c.lesion_type.str.startswith("interaction")], cols)
            snippets[f"cross_{protocol}_{metric}_inter"] = md_table(
                c[c.lesion_type.str.startswith("interaction")], cols)
    e = energy[energy.strategy == "cycle_participation"].drop(columns="strategy")
    snippets["energy"] = md_table(e, list(e.columns))
    snippets["energy_corr"] = (f"corr(dlog10E, log10((N-h)/N)) = {r_size:.3f}; corr(dlog10E, log10(n_floor_les/n_floor_int)) "
                               f"= {r_floor:.3f}; residual vs floor-count prediction: mean {resid_floor.mean():.4f}, "
                               f"SD {resid_floor.std():.4f}, max |.| {resid_floor.abs().max():.4f}")
    snippets["energy_strategy_diff"] = md_table(lesion_energy_diff[lesion_energy_diff.scale == "pooled"], cols, "{:.4f}")
    snippets["hits"] = md_table(hits, list(hits.columns), "{:.2f}")
    snippets["lesion_overlap"] = md_table(lesion_overlap, list(lesion_overlap.columns), "{:.2f}")
    # N-dependence of ER for random driver sets (does ER itself scale with N?)
    rr = (d[(d.protocol == "static") & (d.strategy == "degree_strength")]
          .groupby(["scale", "lesion_type", "h", "k"])[["ref_intact_er", "ref_lesioned_er"]].mean().reset_index())
    rr["ref_dER"] = rr.ref_intact_er - rr.ref_lesioned_er
    snippets["ref_er"] = md_table(rr.groupby(["scale", "lesion_type", "h"])[["ref_intact_er", "ref_lesioned_er", "ref_dER"]]
                                  .mean().reset_index(), ["scale", "lesion_type", "h", "ref_intact_er", "ref_lesioned_er", "ref_dER"])
    # correlation between raw and normalized degradation
    cor = {m: spearmanr(ps[ps.protocol == "static"].dER, ps[ps.protocol == "static"][m]).statistic
           for m in ["dER_perN", "dER_excess", "dPct_ER"]}
    snippets["norm_corr"] = ", ".join(f"rho(dER, {m}) = {v:.3f}" for m, v in cor.items())
    with open(out / "report_snippets.md", "w") as fh:
        for key, val in snippets.items():
            fh.write(f"### {key}\n\n{val}\n\n")
    print(open(out / "report_snippets.md").read()[:3000])


def make_figure(ps: pd.DataFrame, cross: pd.DataFrame, energy: pd.DataFrame, path: Path) -> None:
    colors = {"degree_strength": "#3b6ea8", "cycle_participation": "#d0752b"}
    st = ps[ps.protocol == "static"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    panels = [("dER", "Effective-rank loss\n(intact − lesioned)"),
              ("dPR", "Participation-ratio loss"),
              ("dPct_ER", "ER percentile loss\n(vs 100 random driver sets)")]
    for ax, (metric, title) in zip(axes[0], panels):
        for j, strat in enumerate(STRATS):
            for i, lt in enumerate(LESIONS):
                vals = st[(st.lesion_type == lt) & (st.strategy == strat)].groupby("subject")[metric].mean()
                xpos = i + (j - 0.5) * 0.32
                ax.scatter(np.full(vals.size, xpos) + np.random.default_rng(i * 2 + j).uniform(-0.07, 0.07, vals.size),
                           vals, s=6, alpha=0.35, color=colors[strat], lw=0)
                ax.errorbar(xpos, vals.mean(), yerr=1.96 * vals.std() / np.sqrt(vals.size), fmt="o",
                            color=colors[strat], mec="k", ms=7, capsize=3,
                            label=SLABEL[strat] + " drivers" if i == 0 else None)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks(range(3), [f"{lt}\nlesion" for lt in LESIONS])
        ax.set_title(title, fontsize=10)
    axes[0, 0].legend(fontsize=8, frameon=False)
    # row 2: cycle-degree difference per cell
    for ax, metric in zip(axes[1, :2], ["dER", "dPct_ER"]):
        c = cross[(cross.metric == metric) & (cross.protocol == "static") & (cross.scale != "pooled")
                  & cross.lesion_type.isin(LESIONS)]
        for i, lt in enumerate(LESIONS):
            cc = c[c.lesion_type == lt].reset_index(drop=True)
            xs = i * 10 + np.arange(len(cc))
            ax.errorbar(xs, cc["mean"], yerr=[cc["mean"] - cc.ci_lo, cc.ci_hi - cc["mean"]], fmt="o", ms=4,
                        color=["#3b6ea8", "#d0752b", "0.4"][i], capsize=2)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks([4, 14, 24], [f"{lt} lesion" for lt in LESIONS])
        ax.set_title(f"cycle − degree {'ER loss' if metric == 'dER' else 'ER-percentile loss'}\n"
                     "(9 cells each: scale 68/114/219 × h 5/10/15; 95% boot CI)", fontsize=10)
        ax.set_ylabel(">0: cycle sets degrade more")
    ax = axes[1, 2]
    e = energy[energy.strategy == "cycle_participation"]
    mk = {"degree": "o", "cycle": "s", "random": "^"}
    for lt in LESIONS:
        ee = e[e.lesion_type == lt]
        ax.scatter(ee.size_pred, ee.dlog10E, marker=mk[lt], s=30, label=f"{lt} lesion", color="0.3")
    lim = [e.size_pred.min() - 0.01, 0.005]
    ax.plot(lim, lim, "k--", lw=0.8, label="y = log10((N−h)/N)")
    ax.set_xlabel("log10((N − h)/N)")
    ax.set_ylabel("Δ log10 E_avg (lesioned − intact)")
    ax.set_title("Energy change tracks network size", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
