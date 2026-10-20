#!/usr/bin/env python
"""Figure: the cycle-degree spectral difference depends on the stabilization margin c.

Main figure (<run>/figures/csweep_mechanism.{svg,png})
  A  Delta r_eff = r_eff(cycle) - r_eff(degree) vs c (half the figure width)
  B  Delta q1 = q1(degree) - q1(cycle), q1 = lambda_1 / tr W_eps
  C  R(c) = v1'W_degree v1 / v1'W_cycle v1 (= L_degree / L_cycle, constant in c)

Alternative version (<run>/figures/csweep_perron_removed.{svg,png}): same A and B, with
  C  Delta r_eff for the full Gramian vs after projecting out the Perron vector (W_perp1 = P W P).

All values are k-averaged within subject (k in {1,2,3,5}); lines are means over 70 subjects with
95% t intervals. R(c) is a geometric mean (per-subject mean of log10 ratios over k).
Reads <run>/revision/regime/perron_mode_sweep.csv, where <run> = runs/<name>/ (choose it with
``--run <name>``).
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

STAGE = "revision/regime/09_make_csweep_figure"
DATA: Path  # set in main(): <run>/revision/regime/perron_mode_sweep.csv
FIGURES_DIR: Path  # set in main(): <run>/figures

SCALES = [68, 114, 219]
# Ordinal scale encoding: one hue light -> dark, plus the marker shapes used in the spectra figure.
SCALE_COLORS = {68: "#b3a6d6", 114: "#7a64b5", 219: "#3f1f7a"}
SCALE_MARKERS = {68: "o", 114: "s", 219: "D"}
PRIMARY_C = 0.1
NCTPY_C = 1.0
INK, INK2, GRID = "#1a1a1a", "#5a5a5a", "#e6e6e6"


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xscale("log")
    ax.set_xlabel("Stabilization margin $c$", fontsize=10, color=INK)


def panel_label(ax, letter: str) -> None:
    ax.text(-0.02, 1.05, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", ha="right", va="bottom", color=INK)


def reference_lines(ax) -> None:
    ax.axhline(0, color=INK, lw=1, zorder=1)
    for c, text in ((PRIMARY_C, "primary"), (NCTPY_C, "nctpy")):
        ax.axvline(c, color=INK2, lw=0.8, ls=":", zorder=1)
        ax.text(c * 1.06, 0.98, text, transform=ax.get_xaxis_transform(), fontsize=8, color=INK2, va="top")


def per_subject(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(index=["subject_id", "scale", "c", "k"], columns="method",
                          values=["effective_rank", "effective_rank_perp1", "q1", "perron_energy"])
    out = pd.DataFrame({
        "d_er": wide[("effective_rank", "cycle_participation")] - wide[("effective_rank", "degree_strength")],
        "d_er_perp1": wide[("effective_rank_perp1", "cycle_participation")] - wide[("effective_rank_perp1", "degree_strength")],
        "d_q1": wide[("q1", "degree_strength")] - wide[("q1", "cycle_participation")],
        "log10_R": np.log10(wide[("perron_energy", "degree_strength")] / wide[("perron_energy", "cycle_participation")]),
    })
    return out.groupby(["subject_id", "scale", "c"]).mean().reset_index()


def summarize(subj: pd.DataFrame, column: str) -> pd.DataFrame:
    def agg(values: pd.Series) -> pd.Series:
        v = values.to_numpy()
        half = stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
        return pd.Series({"mean": v.mean(), "lo": v.mean() - half, "hi": v.mean() + half})
    return subj.groupby(["scale", "c"])[column].apply(agg).unstack().reset_index()


def draw_lines(ax, summary: pd.DataFrame, transform=None, ls="-", label_suffix="", alpha_band=0.18, markers=True) -> None:
    for sc in SCALES:
        s = summary[summary.scale == sc].sort_values("c")
        mean, lo, hi = s["mean"], s["lo"], s["hi"]
        if transform is not None:
            mean, lo, hi = transform(mean), transform(lo), transform(hi)
        ax.fill_between(s.c, lo, hi, color=SCALE_COLORS[sc], alpha=alpha_band, lw=0)
        ax.plot(s.c, mean, ls=ls, marker=SCALE_MARKERS[sc] if markers else None, color=SCALE_COLORS[sc], lw=2,
                ms=5, mec="white", mew=0.6, label=f"{sc} regions{label_suffix}", zorder=3)


def draw_ratio(ax, subj: pd.DataFrame) -> None:
    draw_lines(ax, summarize(subj, "log10_R"), transform=lambda x: 10 ** x)
    ax.set_yscale("log")
    for c, text in ((PRIMARY_C, "primary"), (NCTPY_C, "nctpy")):
        ax.axvline(c, color=INK2, lw=0.8, ls=":", zorder=1)
        ax.text(c * 1.06, 0.98, text, transform=ax.get_xaxis_transform(), fontsize=8, color=INK2, va="top")
    ax.axhline(1, color=INK, lw=1, zorder=1)
    ax.set_ylim(0.8, 2000)
    ax.set_ylabel(r"$R = v_1^\top W_{\mathrm{deg}} v_1 \,/\, v_1^\top W_{\mathrm{cyc}} v_1$", fontsize=10, color=INK)
    ax.set_title("Perron-mode input ratio", fontsize=10.5, color=INK, loc="left")
    ax.text(0.03, 0.9, r"$= L_{\mathrm{deg}}/L_{\mathrm{cyc}}$, independent of $c$" "\n(geometric mean over subjects)",
            transform=ax.transAxes, fontsize=8, color=INK2, va="top")


def draw_perron_removed(ax, subj: pd.DataFrame) -> None:
    draw_lines(ax, summarize(subj, "d_er"))
    draw_lines(ax, summarize(subj, "d_er_perp1"), ls="--", alpha_band=0.12, markers=False)
    reference_lines(ax)
    ax.set_ylabel(r"$\Delta r_{\mathrm{eff}}$ (cycle − degree)", fontsize=10, color=INK)
    ax.set_title("Removing the Perron mode", fontsize=10.5, color=INK, loc="left")
    handles = [plt.Line2D([], [], color=INK2, lw=2, marker="o", ms=4), plt.Line2D([], [], color=INK2, lw=2, ls="--")]
    labels = [r"full $W$", r"$W_{\perp 1}$, $v_1$ removed"]
    ax.legend(handles, labels, frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.9))


def csweep_figure(subj: pd.DataFrame, draw_c, filename: str) -> None:
    fig = plt.figure(figsize=(14, 4.6), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1], wspace=0.38)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))
    for ax in (ax_a, ax_b, ax_c):
        style(ax)

    draw_lines(ax_a, summarize(subj, "d_er"))
    reference_lines(ax_a)
    ax_a.set_ylabel(r"$\Delta r_{\mathrm{eff}} = r_{\mathrm{eff}}^{\mathrm{cycle}} - r_{\mathrm{eff}}^{\mathrm{degree}}$", fontsize=10, color=INK)
    ax_a.set_title("Cycle − degree effective rank", fontsize=10.5, color=INK, loc="left")
    ax_a.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.9))

    draw_lines(ax_b, summarize(subj, "d_q1"))
    reference_lines(ax_b)
    ax_b.set_ylabel(r"$\Delta q_1 = q_1^{\mathrm{degree}} - q_1^{\mathrm{cycle}},\; q_1 = \lambda_1/\mathrm{tr}\,W_\epsilon$", fontsize=10, color=INK)
    ax_b.set_title("Leading-eigenvalue share", fontsize=10.5, color=INK, loc="left")

    draw_c(ax_c, subj)

    for ax, letter in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C")):
        panel_label(ax, letter)
    for suffix in ("svg", "png"):
        fig.savefig(FIGURES_DIR / f"{filename}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    global DATA, FIGURES_DIR
    ctx = stage_from_argv(STAGE)
    ctx.require_stage("revision/regime/08_perron_mode_sweep")
    ctx.begin()
    DATA = ctx.root / "revision" / "regime" / "perron_mode_sweep.csv"
    FIGURES_DIR = figures_dir(ctx.root)
    subj = per_subject(pd.read_csv(DATA))
    csweep_figure(subj, draw_ratio, "csweep_mechanism")
    csweep_figure(subj, draw_perron_removed, "csweep_perron_removed")
    table = pd.concat({col: summarize(subj, col).set_index(["scale", "c"])["mean"] for col in ("d_er", "d_er_perp1", "d_q1", "log10_R")}, axis=1)
    table["R_geomean"] = 10 ** table.pop("log10_R")
    table.round(4).to_csv(DATA.with_name("perron_mode_sweep_summary.csv"))
    print(f"saved {FIGURES_DIR}/csweep_mechanism.* and {FIGURES_DIR}/csweep_perron_removed.*")
    ctx.finish()


if __name__ == "__main__":
    main()
