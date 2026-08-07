#!/usr/bin/env python
"""Combined forest plot of target-state energy differences (module + distributed targets).

Targets on the x-axis, cycle-minus-degree percent energy difference on the y-axis, with
one point per parcellation scale (68/114/219). Uses the corrected (Margulies-gradient)
target-energy statistics.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SCALE_COLORS = {68: "#4C72B0", 114: "#55A868", 219: "#C44E52"}
SCALES = [68, 114, 219]
# Left-to-right order on the x-axis.
TARGET_ORDER = [
    ("association_module", "Association"),
    ("association_vs_occipital", "Association vs.\nOccipital"),
    ("occipital_module", "Occipital"),
    ("uniform_cortical", "Uniform"),
    ("bipolar_gradient", "Bipolar"),
    ("principal_gradient", "Principal"),
]
DODGE = 0.24  # horizontal offset between scales within a target


def stars(p_holm: float, reject: bool) -> str:
    if not reject:
        return ""
    return "***" if p_holm < 1e-3 else ("**" if p_holm < 1e-2 else "*")


def main() -> None:
    summary = pd.read_csv(REPO_ROOT / "results" / "statistics" / "target_named_summary.csv")
    summary = summary[summary["metric"] == "pct_cycle_minus_degree"]

    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    for xi, (target_name, _) in enumerate(TARGET_ORDER):
        for si, scale in enumerate(SCALES):
            r = summary[(summary["target_name"] == target_name) & (summary["scale"] == scale)]
            if r.empty:
                continue
            r = r.iloc[0]
            x = xi + (si - 1) * DODGE  # 68 left, 114 center, 219 right
            color = SCALE_COLORS[scale]
            ax.errorbar(
                x, r["median"],
                yerr=[[r["median"] - r["ci_low"]], [r["ci_high"] - r["median"]]],
                fmt="o", color=color, markersize=7, capsize=3, linewidth=1.6, markeredgecolor="white", markeredgewidth=0.7,
            )
            mark = stars(r["p_value_holm"], bool(r["reject_holm"]))
            if mark:
                ax.text(x, r["ci_high"] + 0.5, mark, color=color, va="bottom", ha="center", fontsize=9)

    ax.axhline(0.0, color="#333333", linewidth=1.0, zorder=0)
    ax.set_xticks(range(len(TARGET_ORDER)))
    ax.set_xticklabels([label for _, label in TARGET_ORDER], fontsize=10)
    ax.set_xlim(-0.5, len(TARGET_ORDER) - 0.5)
    ax.set_ylabel("Cycle $-$ degree energy (%)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [Line2D([0], [0], marker="o", color=SCALE_COLORS[s], linestyle="", markersize=7,
                      markeredgecolor="white", label=str(s)) for s in SCALES]
    ax.legend(handles=handles, title="Parcellation scale", fontsize=9, title_fontsize=9,
              frameon=False, loc="upper right", ncol=3, columnspacing=1.0, handletextpad=0.3)
    fig.tight_layout()
    for suffix in ("svg", "png"):
        fig.savefig(FIGURES_DIR / f"target_forest_combined.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
