#!/usr/bin/env python
"""Single-parcel target-state distribution panel (across-scale violin).

Reproduces the concentrated-baseline distribution: for each parcellation scale, the
across-parcel distribution of the subject-mean cycle-minus-degree percent energy
difference for single cortical-parcel targets, with the four named landmarks annotated
and significance-starred. The y-axis is clipped to a readable range (the single-parcel
outliers otherwise span several orders of magnitude).

Usage: python scripts/12_make_target_distribution.py [--run <name>]
(default $HOC_RUN or "main"; ``--run results`` reads the legacy tree, with the
figure written to runs/results-derived/figures/).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hoc.runs import figures_dir, reader_run_dir

SCALES = [68, 114, 219]
SCALE_COLORS = {68: "#4C72B0", 114: "#55A868", 219: "#C44E52"}
Y_LOW, Y_HIGH = -5.0, 19.0
METRIC = "pct_cycle_minus_degree"
LANDMARKS = {
    "landmark_v1": ("V1", -0.28),
    "landmark_m1": ("M1", -0.10),
    "landmark_pcc_precuneus": ("PCC/PCu", 0.10),
    "landmark_lpfc": ("lPFC", 0.28),
}


def stars(p_holm: float, reject: bool) -> str:
    if not reject:
        return ""
    return "***" if p_holm < 1e-3 else ("**" if p_holm < 1e-2 else "*")


def main() -> None:
    run_dir = reader_run_dir()
    figures = figures_dir(run_dir)
    stats = run_dir / "statistics"
    subj = pd.read_csv(stats / "target_subject_differences.csv")
    subj = subj[subj["selection_strategy"] == "static"]
    named = pd.read_csv(stats / "target_named_summary.csv")
    named = named[named["metric"] == METRIC]

    cortical = subj[subj["target_family"] == "single_cortical_parcel"]
    parcel = cortical.groupby(["scale", "target_name"], as_index=False).agg(value=(METRIC, "mean"))

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for xi, scale in enumerate(SCALES):
        vals = parcel[parcel["scale"] == scale]["value"].to_numpy()
        vals = vals[(vals >= Y_LOW) & (vals <= Y_HIGH)]  # clip to the readable window for the KDE
        body = ax.violinplot([vals], positions=[xi], widths=0.72, showextrema=False)["bodies"][0]
        body.set_facecolor(SCALE_COLORS[scale])
        body.set_edgecolor("#404040")
        body.set_alpha(0.30)
        body.set_linewidth(0.8)
        ax.scatter(xi, float(np.median(vals)), color="#202020", s=22, zorder=4)

        land = named[(named["scale"] == scale) & (named["target_name"].isin(LANDMARKS))]
        for _, r in land.iterrows():
            name, dx = LANDMARKS[r["target_name"]]
            x, y = xi + dx, float(r["median"])
            ax.scatter(x, y, s=42, color="white", edgecolor="#202020", linewidth=0.9, zorder=5)
            ax.text(x, y + 0.35, f"{name}{stars(r['p_value_holm'], bool(r['reject_holm']))}",
                    ha="center", va="bottom", fontsize=7.4, color="#202020", zorder=6)

    ax.axhline(0.0, color="#333333", linewidth=1.0, zorder=0)
    ax.set_xticks(range(len(SCALES)))
    ax.set_xticklabels([str(s) for s in SCALES], fontsize=11)
    ax.set_xlabel("Parcellation scale", fontsize=12)
    ax.set_ylabel("Cycle $-$ degree energy (%)", fontsize=12)
    ax.set_ylim(Y_LOW, Y_HIGH)
    ax.tick_params(labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7)

    handles = [Line2D([0], [0], marker="o", linestyle="", color=SCALE_COLORS[s], markersize=7,
                      markeredgecolor="white", label=str(s)) for s in SCALES]
    ax.legend(handles=handles, title="Parcellation scale", fontsize=9, title_fontsize=9,
              frameon=False, loc="upper right", ncol=3, columnspacing=1.0, handletextpad=0.3)
    fig.tight_layout()
    for suffix in ("svg", "png"):
        fig.savefig(figures / f"target_distribution_violin.{suffix}"
, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
