#!/usr/bin/env python
"""Main-text figure: driver-selection criteria produce different controllability spectra.

Panels
  A  group-mean normalized Gramian spectra (lambda_i / trace) for degree, cycle and random sets
  B  effective rank and participation ratio across ranking methods and random sets (k-averaged)
  C  within-subject percentile of cycle and degree sets among the subject's 100 random sets
  D  Perron loading of the driver set vs spectral concentration (top-eigenvalue share)

Reads runs/<name>/revision/spectra/{spectrum_summary.csv, set_metrics.csv, tables/}.
Writes runs/<name>/figures/spectra_methods.{svg,png} (choose the run with ``--run <name>``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from hoc.runs import figures_dir, stage_from_argv  # noqa: E402

STAGE = "revision/spectra/06_make_spectra_figure"
SPECTRA: Path  # set in main(): <run>/revision/spectra
FIGURES_DIR: Path  # set in main(): <run>/figures

SCALES = [68, 114, 219]
REP_SCALE, REP_K = 114, 5
N_TOP = 10

# Degree/cycle colors match Fig. 1 and the original geometry figure.
DEG_COLOR, CYC_COLOR, RND_COLOR, OTHER_COLOR = "#ef3b2c", "#2b8cbe", "#8c8c8c", "#3a3a3a"
INK, INK2, GRID = "#1a1a1a", "#5a5a5a", "#e6e6e6"
SCALE_MARKERS = {68: "o", 114: "s", 219: "D"}

# Rows of panel B, top to bottom.
METHODS = [
    ("cycle_participation", "Cycle participation", CYC_COLOR),
    ("random", "Random sets", RND_COLOR),
    ("participation_coefficient", "Participation coefficient", OTHER_COLOR),
    ("triangle_participation", "Triangle participation", OTHER_COLOR),
    ("betweenness", "Betweenness", OTHER_COLOR),
    ("degree_strength", "Degree strength", DEG_COLOR),
    ("eigenvector_centrality", "Eigenvector centrality", OTHER_COLOR),
]


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, letter: str) -> None:
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", ha="right", va="bottom", color=INK)


def load_sets() -> pd.DataFrame:
    cols = ["subject_id", "scale", "k", "family", "sample", "effective_rank", "participation_ratio", "top1_share", "perron_loading"]
    return pd.read_csv(SPECTRA / "set_metrics.csv", usecols=cols)


def per_subject_method_means(sets: pd.DataFrame) -> pd.DataFrame:
    """k-averaged value per subject for each ranking; random = mean over the subject's 100 random sets."""
    keep = sets[sets["family"].isin([m for m, _, _ in METHODS])]
    metrics = ["effective_rank", "participation_ratio", "top1_share", "perron_loading"]
    # Random sets: average over samples within (subject, k) first, then over k, like the rankings.
    by_k = keep.groupby(["subject_id", "scale", "family", "k"])[metrics].mean()
    return by_k.groupby(["subject_id", "scale", "family"]).mean().reset_index()


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    half = stats.t.ppf(0.975, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size)
    return values.mean(), half


def draw_spectra(ax) -> None:
    summ = pd.read_csv(SPECTRA / "spectrum_summary.csv")
    for fam, color, label in (("degree_strength", DEG_COLOR, "Degree strength"), ("cycle_participation", CYC_COLOR, "Cycle participation"), ("random", RND_COLOR, "Random sets")):
        s = summ[(summ.scale == REP_SCALE) & (summ.k == REP_K) & (summ.family == fam) & (summ["index"] <= N_TOP)]
        ax.fill_between(s["index"], s.mean_norm_eig - s.sem_norm_eig, s.mean_norm_eig + s.sem_norm_eig, color=color, alpha=0.2, lw=0)
        ax.plot(s["index"], s.mean_norm_eig, "-o", color=color, lw=2, ms=5, mec="white", mew=0.8, label=label, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(1, N_TOP + 1))
    ax.axvline(REP_K + 0.5, color=INK2, lw=0.8, ls=":")
    ax.text(REP_K + 0.6, 0.97, f"k = {REP_K} drivers", fontsize=8, color=INK2, va="top", transform=ax.get_xaxis_transform())
    ax.set_xlabel("Eigenvalue index (descending)", fontsize=10, color=INK)
    ax.set_ylabel(r"$\lambda_i(W_\epsilon)\,/\,\mathrm{tr}\,W_\epsilon$", fontsize=10, color=INK)
    ax.set_title(f"Normalized Gramian spectra ({REP_SCALE} regions, k = {REP_K})", fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")


def draw_methods(ax, means: pd.DataFrame, metric: str, title: str, show_labels: bool) -> None:
    ys = np.arange(len(METHODS))[::-1]
    offsets = {68: 0.22, 114: 0.0, 219: -0.22}
    rnd_mean = means[means.family == "random"].groupby("scale")[metric].mean()
    for y, (method, label, color) in zip(ys, METHODS):
        for sc in SCALES:
            m, h = mean_ci(means[(means.family == method) & (means.scale == sc)][metric])
            ax.errorbar(m, y + offsets[sc], xerr=h, fmt=SCALE_MARKERS[sc], color=color, ms=5.5, mec="white", mew=0.6, elinewidth=1.3, capsize=0, zorder=3)
    for sc in SCALES:
        ax.axvline(rnd_mean[sc], color=RND_COLOR, lw=0.8, ls="--", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, label, _ in METHODS] if show_labels else [], fontsize=9.5)
    for tick, (method, _, color) in zip(ax.get_yticklabels(), METHODS):
        if method != "random" and color != OTHER_COLOR:
            tick.set_color(color)
            tick.set_fontweight("bold")
    ax.set_ylim(-0.6, len(METHODS) - 0.4)
    ax.set_xlabel(f"{title} (mean ± 95% CI)", fontsize=10, color=INK)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.grid(False, axis="y")


def draw_percentiles(ax) -> None:
    pers = pd.read_csv(SPECTRA / "tables" / "t1_rankings_vs_random_per_subject.csv")
    pers = pers[pers.metric == "effective_rank"]
    rng = np.random.default_rng(0)
    for j, sc in enumerate(SCALES):
        for i, (method, color) in enumerate((("degree_strength", DEG_COLOR), ("cycle_participation", CYC_COLOR))):
            v = 100 * pers[(pers.ranking == method) & (pers.scale == sc)].percentile.to_numpy()
            x = j * 2.6 + i
            parts = ax.violinplot(v, positions=[x], widths=0.85, showextrema=False, bw_method=0.35)
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.25)
                body.set_edgecolor("none")
            ax.scatter(x + rng.uniform(-0.13, 0.13, v.size), v, s=9, color=color, lw=0, alpha=0.85, zorder=3)
            ax.plot([x - 0.28, x + 0.28], [np.median(v)] * 2, color=INK, lw=2, zorder=4)
    ax.axhline(50, color=RND_COLOR, lw=1, ls="--", zorder=1)
    ax.text(-0.6, 51.5, "random median", fontsize=8, color=INK2, ha="left", va="bottom")
    ax.set_xticks([j * 2.6 + 0.5 for j in range(len(SCALES))])
    ax.set_xticklabels([f"{sc} regions" for sc in SCALES], fontsize=10)
    ax.set_xlim(-0.7, 2 * 2.6 + 1.7)
    ax.set_ylim(-2, 102)
    ax.set_ylabel("Percentile among the subject's\n100 random sets (effective rank)", fontsize=10, color=INK)
    ax.set_title("Position relative to random driver placement", fontsize=10.5, color=INK, loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=6) for c in (DEG_COLOR, CYC_COLOR)]
    ax.legend(handles, ["Degree strength", "Cycle participation"], frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    ax.grid(False, axis="x")


def draw_perron(ax, sets: pd.DataFrame) -> None:
    d = sets[(sets.scale == REP_SCALE) & (sets.k == REP_K)]
    rnd = d[d.family == "random"]
    ax.scatter(rnd.perron_loading, rnd.top1_share, s=4, color=RND_COLOR, alpha=0.18, lw=0, zorder=2, label="Random sets")
    for method, label, color in METHODS:
        if method == "random":
            continue
        g = d[d.family == method]
        if color == OTHER_COLOR:
            ax.scatter(g.perron_loading, g.top1_share, s=12, facecolor="none", edgecolor=OTHER_COLOR, lw=0.6, zorder=3)
        else:
            ax.scatter(g.perron_loading, g.top1_share, s=18, color=color, edgecolor="white", lw=0.4, zorder=4, label=label)
    ax.scatter([], [], s=12, facecolor="none", edgecolor=OTHER_COLOR, lw=0.6, label="Other rankings")
    rho = stats.spearmanr(d.perron_loading, d.top1_share).statistic
    ax.text(0.03, 0.97, f"Spearman ρ = {rho:+.2f}\n(all sets, {REP_SCALE} regions, k = {REP_K})", transform=ax.transAxes, fontsize=8.5, color=INK, va="top")
    ax.set_xscale("log")
    ax.set_xlabel(r"Perron loading of driver set, $\sum_{i\in S} v_i^2$", fontsize=10, color=INK)
    ax.set_ylabel(r"Top-eigenvalue share, $\lambda_1 / \mathrm{tr}\,W_\epsilon$", fontsize=10, color=INK)
    ax.set_title("Hub loading and spectral concentration", fontsize=10.5, color=INK, loc="left")
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.84), markerscale=1.3)
    leg.legend_handles[0].set_alpha(0.8)
    leg.legend_handles[0].set_sizes([14])


def main() -> None:
    global SPECTRA, FIGURES_DIR
    ctx = stage_from_argv(STAGE)
    ctx.require_stage("revision/spectra/01_compute_spectra")
    ctx.require_stage("revision/spectra/02_analyze")
    ctx.require_stage("revision/spectra/03_make_figure")
    ctx.begin()
    SPECTRA = ctx.root / "revision" / "spectra"
    FIGURES_DIR = figures_dir(ctx.root)
    sets = load_sets()
    means = per_subject_method_means(sets)

    fig = plt.figure(figsize=(14, 9.2), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], hspace=0.42, wspace=0.3)
    # Top row gets its own spacing: a gutter for the method labels of panel B.
    top = gs[0, :].subgridspec(1, 4, width_ratios=[1, 0.45, 0.85, 0.85], wspace=0.1)
    ax_a = fig.add_subplot(top[0])
    ax_b1 = fig.add_subplot(top[2])
    ax_b2 = fig.add_subplot(top[3])
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_d = fig.add_subplot(gs[1, 2])
    for ax in (ax_a, ax_b1, ax_b2, ax_c, ax_d):
        style(ax)

    draw_spectra(ax_a)
    draw_methods(ax_b1, means, "effective_rank", "Effective rank", show_labels=True)
    draw_methods(ax_b2, means, "participation_ratio", "Participation ratio", show_labels=False)
    scale_handles = [plt.Line2D([], [], marker=SCALE_MARKERS[sc], ls="", color=INK2, ms=6) for sc in SCALES]
    ax_b2.legend(scale_handles + [plt.Line2D([], [], color=RND_COLOR, lw=0.8, ls="--")], [f"{sc} regions" for sc in SCALES] + ["random mean"], frameon=False, fontsize=8.5, loc="lower right")
    draw_percentiles(ax_c)
    draw_perron(ax_d, sets)

    for ax, letter in ((ax_a, "A"), (ax_b1, "B"), (ax_c, "C"), (ax_d, "D")):
        panel_label(ax, letter)

    for suffix in ("svg", "png"):
        fig.savefig(FIGURES_DIR / f"spectra_methods.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", FIGURES_DIR / "spectra_methods.png")
    ctx.finish()


if __name__ == "__main__":
    main()
