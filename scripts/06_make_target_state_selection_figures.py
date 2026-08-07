#!/usr/bin/env python
"""Make target-state figures for the concentrated-to-distributed redesign."""

from __future__ import annotations

import sys
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


RESULTS_DIR = REPO_ROOT / "results"
STATISTICS_DIR = RESULTS_DIR / "statistics"
FIGURES_DIR = REPO_ROOT / "figures" / "section_target_state_selection"

SCALES = [68, 114, 219]
SCALE_COLORS = {68: "#4e79a7", 114: "#59a14f", 219: "#e15759"}
LANDMARK_ORDER = ["landmark_v1", "landmark_m1", "landmark_pcc_precuneus", "landmark_lpfc"]
MODULE_ORDER = ["association_module", "occipital_module", "association_vs_occipital"]
DISTRIBUTED_ORDER = ["principal_gradient", "bipolar_gradient", "uniform_cortical"]


def require_table(path: Path) -> pd.DataFrame:
    """Load a CSV table with a clear rerun hint."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Rerun `python scripts/01_run_primary_analysis.py` first.")
    return pd.read_csv(path)


def metric_label(metric: str) -> str:
    """Human-readable y/x label for a target-difference metric."""
    if metric == "pct_cycle_minus_degree":
        return "Cycle - degree energy (%)"
    return "Cycle - degree energy"


def add_zero_line(ax: plt.Axes, orientation: str) -> None:
    """Draw the no-preference reference line."""
    if orientation == "horizontal":
        ax.axhline(0.0, color="#2f2f2f", linewidth=0.9, zorder=1)
    else:
        ax.axvline(0.0, color="#2f2f2f", linewidth=0.9, zorder=1)


def plot_named_targets(
    ax: plt.Axes,
    summary: pd.DataFrame,
    order: list[str],
    metric: str,
    title: str,
) -> None:
    """Plot median and bootstrap CI for named non-baseline targets."""
    subset = summary[
        (summary["selection_strategy"].eq("static"))
        & (summary["metric"].eq(metric))
        & (summary["target_name"].isin(order))
    ].copy()
    labels = subset.drop_duplicates("target_name").set_index("target_name")["target_label"].to_dict()
    y_positions = {target_name: index for index, target_name in enumerate(order)}
    offsets = {68: -0.22, 114: 0.0, 219: 0.22}
    for scale in SCALES:
        scale_subset = subset[subset["scale"].eq(scale)]
        for _, row in scale_subset.iterrows():
            y = y_positions[row["target_name"]] + offsets[scale]
            ax.errorbar(
                row["median"],
                y,
                xerr=[[row["median"] - row["ci_low"]], [row["ci_high"] - row["median"]]],
                fmt="o",
                markersize=5.8,
                linewidth=1.5,
                capsize=2.5,
                color=SCALE_COLORS[scale],
                label=str(scale) if row["target_name"] == order[0] else None,
                zorder=3,
            )
    add_zero_line(ax, "vertical")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels.get(target_name, target_name) for target_name in order])
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel(metric_label(metric))
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)


def make_spectral_axis_figure(metric: str = "pct_cycle_minus_degree") -> Path:
    """Build the top-to-bottom target-state summary figure."""
    subject_differences = require_table(STATISTICS_DIR / "target_subject_differences.csv")
    named_summary = require_table(STATISTICS_DIR / "target_named_summary.csv")

    static_subjects = subject_differences[subject_differences["selection_strategy"].eq("static")].copy()
    cortical = static_subjects[static_subjects["target_family"].eq("single_cortical_parcel")]
    cortical_distribution = (
        cortical.groupby(["scale", "target_name"], as_index=False)
        .agg(parcel_subject_average=(metric, "mean"))
        .merge(
            cortical[
                [
                    "scale",
                    "target_name",
                    "target_label",
                    "base_region",
                    "region_label",
                ]
            ].drop_duplicates(),
            on=["scale", "target_name"],
            how="left",
        )
    )
    landmarks = (
        static_subjects[static_subjects["target_name"].isin(LANDMARK_ORDER)]
        .groupby(["scale", "target_name", "target_label"], as_index=False)
        .agg(landmark_median=(metric, "median"))
    )

    sns.set_theme(context="paper", style="whitegrid")
    mpl.rcParams.update({"svg.fonttype": "none", "axes.unicode_minus": False})

    fig = plt.figure(figsize=(9.6, 10.4), constrained_layout=True)
    grid = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.45, 1.0, 1.0])
    ax_top = fig.add_subplot(grid[0, 0])
    ax_middle = fig.add_subplot(grid[1, 0])
    ax_bottom = fig.add_subplot(grid[2, 0])

    violin_values = [
        cortical_distribution[cortical_distribution["scale"].eq(scale)]["parcel_subject_average"].to_numpy()
        for scale in SCALES
    ]
    parts = ax_top.violinplot(violin_values, positions=np.arange(len(SCALES)), widths=0.68, showextrema=False)
    for body, scale in zip(parts["bodies"], SCALES):
        body.set_facecolor(SCALE_COLORS[scale])
        body.set_edgecolor("#202020")
        body.set_alpha(0.28)
        body.set_linewidth(0.8)
    medians = [float(np.median(values)) for values in violin_values]
    ax_top.scatter(np.arange(len(SCALES)), medians, color="#202020", s=24, zorder=4, label="Parcel median")

    landmark_offsets = {
        "landmark_v1": -0.24,
        "landmark_m1": -0.08,
        "landmark_pcc_precuneus": 0.08,
        "landmark_lpfc": 0.24,
    }
    short_labels = {
        "landmark_v1": "V1",
        "landmark_m1": "M1",
        "landmark_pcc_precuneus": "PCC/PCu",
        "landmark_lpfc": "lPFC",
    }
    for scale_index, scale in enumerate(SCALES):
        scale_landmarks = landmarks[landmarks["scale"].eq(scale)]
        for _, row in scale_landmarks.iterrows():
            x = scale_index + landmark_offsets[row["target_name"]]
            y = row["landmark_median"]
            ax_top.scatter(x, y, s=44, color="#ffffff", edgecolor="#202020", linewidth=0.9, zorder=5)
            ax_top.text(
                x,
                y,
                short_labels[row["target_name"]],
                ha="center",
                va="bottom",
                fontsize=7.6,
                color="#202020",
                zorder=6,
            )
    add_zero_line(ax_top, "horizontal")
    ax_top.set_xticks(np.arange(len(SCALES)))
    ax_top.set_xticklabels([str(scale) for scale in SCALES])
    ax_top.set_title("Concentrated baseline: cortical single-parcel targets", loc="left", fontsize=12, fontweight="bold")
    ax_top.set_xlabel("Parcellation scale")
    ax_top.set_ylabel(metric_label(metric))
    ax_top.grid(axis="x", visible=False)

    plot_named_targets(ax_middle, named_summary, MODULE_ORDER, metric, "Intermediate: anatomical modules and contrast")
    plot_named_targets(ax_bottom, named_summary, DISTRIBUTED_ORDER, metric, "Distributed: gradients and uniform cortical state")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=SCALE_COLORS[scale], label=str(scale), markersize=6)
        for scale in SCALES
    ]
    ax_middle.legend(handles=handles, title="Scale", frameon=False, loc="lower right", ncols=3)

    output_name = "target_state_spectral_axis_static_pct.svg" if metric == "pct_cycle_minus_degree" else "target_state_spectral_axis_static_raw.svg"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / output_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    """Write target-state summary figures."""
    pct_path = make_spectral_axis_figure("pct_cycle_minus_degree")
    raw_path = make_spectral_axis_figure("cycle_minus_degree")
    print(pct_path)
    print(raw_path)


if __name__ == "__main__":
    main()
