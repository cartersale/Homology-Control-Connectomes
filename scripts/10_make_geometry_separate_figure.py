#!/usr/bin/env python
"""Control-geometry metrics for degree and cycle driver sets shown as two separate lines.

Companion to the cycle-minus-degree difference figure: instead of the difference, this
plots the absolute effective rank, participation ratio, and log10 condition number for
degree-strength and cycle-participation static driver sets separately, across parcellation
scales (averaged over k in {1,2,3,5}).

Usage: python scripts/10_make_geometry_separate_figure.py [--run <name>]
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hoc.runs import figures_dir, reader_run_dir

# Degree/cycle colors match the rank-divergence figure (degree-only red, cycle-only blue).
DEG_COLOR, CYC_COLOR = "#ef3b2c", "#2b8cbe"
SCALES = [68, 114, 219]
K_VALUES = [1, 2, 3, 5]
METHODS = [("degree_strength", DEG_COLOR, "Degree strength"), ("cycle_participation", CYC_COLOR, "Cycle participation")]
PANELS = [
    ("effective_rank", "Effective rank"),
    ("participation_ratio", "Participation ratio"),
    ("log10_condition_number", r"$\log_{10}$ condition number"),
]


def load_metrics(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "primary" / "geometry_metrics.csv")
    df = df[(df["selection_strategy"] == "static") & df["k"].isin(K_VALUES) & df["method"].isin([m for m, _, _ in METHODS])]
    metrics = [metric for metric, _ in PANELS]
    # Average across driver-set sizes within subject, matching the difference figure.
    return df.groupby(["subject_id", "scale", "method"])[metrics].mean().reset_index()


def main() -> None:
    run_dir = reader_run_dir()
    figures = figures_dir(run_dir)
    per_subject = load_metrics(run_dir)
    x = np.arange(len(SCALES))

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    for ax, (metric, title) in zip(axes, PANELS):
        for method, color, label in METHODS:
            group = per_subject[per_subject["method"] == method].groupby("scale")[metric]
            means = group.mean().reindex(SCALES).to_numpy()
            sems = group.sem().reindex(SCALES).to_numpy()
            ax.errorbar(x, means, yerr=sems, color=color, marker="o", markersize=8, capsize=3,
                        linewidth=2, label=label, markeredgecolor="white", markeredgewidth=0.8)
        ax.set_title(title, fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(SCALES, fontsize=12)
        ax.tick_params(labelsize=11)
        ax.set_xlabel("Parcellation scale", fontsize=13)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.15)
    axes[0].set_ylabel("Metric value", fontsize=13)
    axes[0].legend(fontsize=11, frameon=False, loc="center right")
    fig.tight_layout()
    for suffix in ("svg", "png"):
        fig.savefig(figures / f"geometry_effects_separate.{suffix}"
, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
