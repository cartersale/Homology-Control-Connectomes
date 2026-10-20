#!/usr/bin/env python
"""Draft spectra figure + group-mean normalized spectrum summary.

Panels
  A  group-mean normalized spectrum (lambda_i / trace, top 15) for degree, cycle, random
  B  effective rank and participation ratio minus the subject's random-set mean, all rankings
  C  within-subject percentile of cycle and degree sets among the 100 random sets
  D  per-set Perron loading vs effective rank (rankings + random sets)

Writes runs/<name>/revision/spectra/{spectrum_summary.csv, fig_spectra_draft.png}
(choose the run with ``--run <name>``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from hoc.runs import stage_from_argv  # noqa: E402

STAGE = "revision/spectra/03_make_figure"
OUT: Path  # set in main(): <run>/revision/spectra
TAB: Path  # set in main(): OUT / "tables"
SCALES = [68, 114, 219]
K_VALUES = [1, 2, 3, 5]
N_TOP = 15
REP_SCALE, REP_K = 114, 5

C_DEG, C_CYC, C_RND = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RANK_ORDER = [
    "eigenvector_centrality",
    "degree_strength",
    "betweenness",
    "triangle_participation",
    "tetrahedron_participation",
    "cycle_criticality",
    "participation_coefficient",
    "degree_normalized_cycle_participation",
    "cycle_participation",
    "cycle_resid",
]
LABEL = {
    "degree_strength": "Degree",
    "eigenvector_centrality": "Eigenvector",
    "betweenness": "Betweenness",
    "cycle_participation": "Cycle participation",
    "participation_coefficient": "Participation coef.",
    "triangle_participation": "Triangle part.",
    "tetrahedron_participation": "Tetrahedron part.",
    "degree_normalized_cycle_participation": "Degree-norm. CP",
    "cycle_criticality": "Cycle criticality",
    "cycle_resid": "CP | strength (resid.)",
}


def spectrum_summary() -> pd.DataFrame:
    rows = []
    for sc in SCALES:
        z = np.load(OUT / f"spectra_scale-{sc}.npz")
        fams = {
            "degree_strength": z["method__degree_strength"],  # (subj, k, N)
            "cycle_participation": z["method__cycle_participation"],
            "random": z["random"],  # (subj, k, 100, N)
        }
        for fam, arr in fams.items():
            norm = arr / arr.sum(axis=-1, keepdims=True)
            if fam == "random":
                norm = norm.mean(axis=2)  # subject's mean random spectrum
            for ki, k in enumerate(K_VALUES):
                x = norm[:, ki, :N_TOP]
                mean, sem = x.mean(0), x.std(0, ddof=1) / np.sqrt(x.shape[0])
                for i in range(N_TOP):
                    rows.append({"scale": sc, "k": k, "family": fam, "index": i + 1, "mean_norm_eig": mean[i], "sem_norm_eig": sem[i]})
    return pd.DataFrame(rows)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    global OUT, TAB
    ctx = stage_from_argv(STAGE)
    ctx.require_stage("revision/spectra/01_compute_spectra")
    ctx.require_stage("revision/spectra/02_analyze")
    ctx.begin()
    OUT = ctx.root / "revision" / "spectra"
    TAB = OUT / "tables"
    summ = spectrum_summary()
    summ.to_csv(OUT / "spectrum_summary.csv", index=False)
    t1 = pd.read_csv(TAB / "t1_rankings_vs_random.csv")
    pers = pd.read_csv(TAB / "t1_rankings_vs_random_per_subject.csv")
    sets = pd.read_csv(OUT / "set_metrics.csv", usecols=["subject_id", "scale", "k", "family", "sample", "effective_rank", "perron_loading"])

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(14, 9), facecolor="white")
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.62)

    # A: spectra
    ax = fig.add_subplot(gs[0, 0])
    style(ax)
    for fam, col, lab in (("degree_strength", C_DEG, "Degree"), ("cycle_participation", C_CYC, "Cycle"), ("random", C_RND, "Random")):
        s = summ[(summ.scale == REP_SCALE) & (summ.k == REP_K) & (summ.family == fam)]
        ax.plot(s["index"], s.mean_norm_eig, "-o", color=col, lw=2, ms=4, label=lab, zorder=3)
        ax.fill_between(s["index"], s.mean_norm_eig - s.sem_norm_eig, s.mean_norm_eig + s.sem_norm_eig, color=col, alpha=0.2, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue index (descending)", color=INK)
    ax.set_ylabel(r"$\lambda_i(W_\epsilon)\,/\,\mathrm{tr}\,W_\epsilon$", color=INK)
    ax.set_title(f"A  Normalized spectra\n({REP_SCALE} regions, k = {REP_K}, mean ± SEM)", loc="left")
    ax.legend(frameon=False, fontsize=8)

    # B: ER and PR vs random, all rankings
    markers = {68: "o", 114: "s", 219: "D"}
    shades = {68: "#b4b3ad", 114: "#6f6e69", 219: "#0b0b0b"}
    for col_i, metric, title in ((1, "effective_rank", "B  Effective rank\n− subject random-set mean"), (2, "participation_ratio", "B′ Participation ratio\n− subject random-set mean")):
        ax = fig.add_subplot(gs[0, col_i])
        style(ax)
        ys = np.arange(len(RANK_ORDER))
        for j, sc in enumerate(SCALES):
            sub = t1[(t1.metric == metric) & (t1.scale == sc)].set_index("ranking").loc[RANK_ORDER]
            off = (j - 1) * 0.22
            ax.errorbar(sub["median"], ys + off, xerr=[sub["median"] - sub.ci_lo, sub.ci_hi - sub["median"]], fmt=markers[sc], color=shades[sc], ms=4.5, lw=1, capsize=0, label=f"{sc}", zorder=3)
        # random spread: central 90 % of random sets around their own subject mean (k-averaged sample index), 114
        r = sets[(sets.family == "random") & (sets.scale == REP_SCALE)]
        if metric == "effective_rank":
            dev = r.effective_rank - r.groupby(["subject_id", "k"]).effective_rank.transform("mean")
            lo, hi = np.percentile(dev.groupby([r.subject_id, r["sample"]]).mean(), [5, 95])
            ax.axvspan(lo, hi, color=C_RND, alpha=0.15, lw=0, zorder=1)
            ax.text(hi, len(RANK_ORDER) - 0.4, " 90% of random\n sets (114)", fontsize=6.5, color=INK2, va="top")
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([LABEL[m] for m in RANK_ORDER], fontsize=8)
        for t, m in zip(ax.get_yticklabels(), RANK_ORDER):
            if m in ("degree_strength", "cycle_participation"):
                t.set_fontweight("bold")
        ax.set_xlabel("Median paired difference (95% CI), k-averaged", color=INK)
        ax.set_title(title, loc="left")
        if col_i == 1:
            ax.legend(title="Scale", frameon=False, fontsize=7, title_fontsize=7, loc="lower right")

    # C: percentile distributions
    ax = fig.add_subplot(gs[1, 0:2])
    style(ax)
    pos = []
    for j, sc in enumerate(SCALES):
        for i, (m, col) in enumerate((("degree_strength", C_DEG), ("cycle_participation", C_CYC))):
            v = 100 * pers[(pers.metric == "effective_rank") & (pers.ranking == m) & (pers.scale == sc)].percentile.to_numpy()
            x = j * 3 + i
            pos.append(x)
            parts = ax.violinplot(v, positions=[x], widths=0.8, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(col)
                b.set_alpha(0.35)
                b.set_edgecolor("none")
            jitter = np.random.default_rng(0).uniform(-0.12, 0.12, v.size)
            ax.scatter(x + jitter, v, s=6, color=col, zorder=3, lw=0)
            ax.plot([x - 0.3, x + 0.3], [np.median(v)] * 2, color=INK, lw=2)
    ax.axhline(50, color=INK2, lw=1, ls="--")
    ax.set_xticks([0.5, 3.5, 6.5])
    ax.set_xticklabels([f"{s} regions" for s in SCALES])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentile among 100 random sets\n(effective rank, k-averaged)", color=INK)
    ax.set_title("C  Degree (blue) and cycle (orange) sets relative to the random distribution", loc="left")

    # D: Perron loading vs ER
    ax = fig.add_subplot(gs[1, 2])
    style(ax)
    d = sets[(sets.scale == REP_SCALE) & (sets.k == 3)]
    rr = d[d.family == "random"]
    ax.scatter(rr.perron_loading, rr.effective_rank, s=3, color=C_RND, alpha=0.25, lw=0, label="Random", zorder=2)
    others = d[(d["sample"] == -1) & ~d.family.isin(["degree_strength", "cycle_participation", "cycle_resid"])]
    ax.scatter(others.perron_loading, others.effective_rank, s=8, facecolor="none", edgecolor=INK2, lw=0.6, label="Other rankings", zorder=3)
    for fam, col, lab in (("degree_strength", C_DEG, "Degree"), ("cycle_participation", C_CYC, "Cycle")):
        g = d[d.family == fam]
        ax.scatter(g.perron_loading, g.effective_rank, s=14, color=col, edgecolor="white", lw=0.4, label=lab, zorder=4)
    rho = pd.read_csv(TAB / "t4_perron_er_spearman.csv")
    rr_ = rho[(rho.scale == REP_SCALE) & (rho.k.astype(str) == "3")].iloc[0]
    pw = pd.read_csv(TAB / "t4c_perron_piecewise.csv")
    pw = pw[(pw.c == 0.1) & (pw.scale == REP_SCALE) & (pw.k == 3)].iloc[0]
    ax.text(0.03, 0.97, f"ρ (loading ≤ 0.05) = {pw.rho_low:+.2f}\nρ (loading > 0.05) = {pw.rho_high:+.2f}\nρ all sets = {rr_.rho_pooled:+.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1.2)
    ax.axvline(0.05, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("Perron loading  Σ$_{i∈S}$ v$_i^2$ (log)", color=INK)
    ax.set_ylabel("Effective rank", color=INK)
    ax.set_title(f"D  Perron loading vs ER\n({REP_SCALE} regions, k = 3)", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower left", markerscale=1.5)

    fig.suptitle("DRAFT — c = 0.1, ε = 1e-5, infinite horizon, n = 70 subjects", x=0.01, ha="left", fontsize=9, color=INK2)
    fig.savefig(OUT / "fig_spectra_draft.png", dpi=200, bbox_inches="tight")
    print("saved", OUT / "fig_spectra_draft.png")
    ctx.finish()


if __name__ == "__main__":
    main()
