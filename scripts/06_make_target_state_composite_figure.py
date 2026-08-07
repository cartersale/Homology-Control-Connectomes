#!/usr/bin/env python
"""Build composite target-state energy figures across the target-spread axis."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from nilearn import datasets

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hoc.atlas import EXCLUDED_LABELS, SCALE_TO_RESOLUTION, build_label_table
from hoc.config import load_config


SCALES = (68, 114, 219)
K_VALUES = (1, 2, 3, 5)
LANDMARK_TARGETS = ("landmark_v1", "landmark_m1", "landmark_pcc_precuneus", "landmark_lpfc")
MODULE_ROW_TARGETS = ("association_module", "association_vs_occipital")
DISTRIBUTED_ROW_TARGETS = ("principal_gradient", "bipolar_gradient", "uniform_cortical", "occipital_module")
COMBINED_MODULE_DISTRIBUTED_TARGETS = (
    "association_module",
    "association_vs_occipital",
    "occipital_module",
    "uniform_cortical",
    "bipolar_gradient",
    "principal_gradient",
)
LEGEND_TARGETS = LANDMARK_TARGETS + ("association_module", "occipital_module") + (
    "principal_gradient",
    "bipolar_gradient",
    "uniform_cortical",
)
TARGET_LABELS = {
    "landmark_v1": "V1",
    "landmark_m1": "M1",
    "landmark_pcc_precuneus": "PCC/PCu",
    "landmark_lpfc": "lPFC",
    "association_module": "Association",
    "association_vs_occipital": "Assoc-vs-occip",
    "occipital_module": "Occipital",
    "principal_gradient": "Principal",
    "bipolar_gradient": "Bipolar",
    "uniform_cortical": "Uniform",
}
LANDMARK_COLORS = {
    "landmark_v1": "#2c7fb8",
    "landmark_m1": "#f28e2b",
    "landmark_pcc_precuneus": "#59a14f",
    "landmark_lpfc": "#b07aa1",
}
SIGNIFICANCE_COLORS = {
    "cycle": "#2c7fb8",
    "null": "#b8b8b8",
    "degree": "#d7301f",
}
SCALE_COLORS = {68: "#4e79a7", 114: "#59a14f", 219: "#e15759"}
EFFECT_CMAP = LinearSegmentedColormap.from_list(
    "cycle_degree_effect",
    ["#2166ac", "#67a9cf", "#b8b8b8", "#fdae61", "#d7301f"],
)
UNSIGNED_CMAP = LinearSegmentedColormap.from_list(
    "target_warm",
    ["#fff4bc", "#fdae61", "#d7301f"],
)
SIGNED_TARGET_CMAP = LinearSegmentedColormap.from_list(
    "target_signed",
    ["#2166ac", "#67a9cf", "#f7f7f7", "#fdae61", "#d7301f"],
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--scale", default="114", help="Scale to render: 68, 114, 219, or 'all'.")
    parser.add_argument("--stats-only", action="store_true", help="Render the composite without any brain-region maps.")
    parser.add_argument("--pieces", action="store_true", help="Export each figure component as a separate SVG.")
    parser.add_argument("--old-style", action="store_true", help="Export the older violin-plus-forest summary as separate SVG panels.")
    parser.add_argument("--include-per-k", action="store_true", help="Also export per-k supplementary panels.")
    parser.add_argument("--include-subcortical", action="store_true", help="Also export subcortical baseline panels if present.")
    return parser.parse_args()


def requested_scales(scale_text: str) -> list[int]:
    """Resolve scale argument into concrete scales."""
    if scale_text.lower() == "all":
        return list(SCALES)
    scale = int(scale_text)
    if scale not in SCALES:
        raise ValueError(f"Unsupported scale {scale}.")
    return [scale]


def require_csv(path: Path) -> pd.DataFrame:
    """Load a required CSV with a clear rerun hint."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Rerun `python scripts/01_run_primary_analysis.py` first.")
    return pd.read_csv(path)


def load_results(config: dict) -> dict[str, pd.DataFrame]:
    """Load the redesigned target-state result tables."""
    results_root = REPO_ROOT / config["paths"]["results_dir"]
    stats_root = results_root / "statistics"
    return {
        "subject": require_csv(stats_root / "target_subject_differences.csv"),
        "pairwise": require_csv(stats_root / "target_pairwise_differences.csv"),
        "named_summary": require_csv(stats_root / "target_named_summary.csv"),
        "parcel_tests": require_csv(stats_root / "target_single_parcel_tests.csv"),
        "parcel_distribution": require_csv(stats_root / "target_single_parcel_distribution_summary.csv"),
        "metadata": require_csv(results_root / "primary" / "target_metadata.csv"),
    }


def load_targets(config: dict, scale: int) -> dict[str, np.ndarray]:
    """Load saved target vectors for one scale."""
    path = REPO_ROOT / config["paths"]["results_dir"] / "processed" / f"targets_scale-{scale}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Rerun `python scripts/01_run_primary_analysis.py` first.")
    archive = np.load(path)
    return {name: np.asarray(archive[name], dtype=float) for name in archive.files}


def load_geometry(scale: int, atlas_dir: str | Path) -> dict:
    """Load surface geometry and node mapping."""
    fsaverage = datasets.load_fsaverage(mesh="fsaverage5")
    pial = fsaverage["pial"]
    atlas_path = REPO_ROOT / atlas_dir
    resolution = SCALE_TO_RESOLUTION[scale]
    geometry = {}
    centroids = []
    node_index = 0
    for hemi, part in (("L", "left"), ("R", "right")):
        coords = np.asarray(pial.parts[part].coordinates, dtype=float)
        faces = np.asarray(pial.parts[part].faces, dtype=int)
        annot_path = atlas_path / f"atl-Cammoun2012_space-fsaverage5_res-{resolution}_hemi-{hemi}_deterministic.annot"
        vertex_labels, _, names = nib.freesurfer.read_annot(str(annot_path))
        vertex_to_node = np.full(vertex_labels.shape, -1, dtype=int)
        for atlas_index, raw_name in enumerate(names):
            name = raw_name.decode()
            if name in EXCLUDED_LABELS:
                continue
            mask = vertex_labels == atlas_index
            vertex_to_node[mask] = node_index
            if np.any(mask):
                centroid = coords[mask].mean(axis=0)
                centroids.append(
                    {
                        "node_index": node_index,
                        "hemi": hemi,
                        "x": float(centroid[0]),
                        "y": float(centroid[1]),
                    }
                )
            node_index += 1
        geometry[hemi] = {"coords": coords, "faces": faces, "vertex_to_node": vertex_to_node}
    geometry["centroids"] = pd.DataFrame(centroids)
    return geometry


def draw_surface(
    ax: plt.Axes,
    geometry: dict,
    node_values: np.ndarray | None = None,
    *,
    cmap=EFFECT_CMAP,
    norm: Normalize | TwoSlopeNorm | None = None,
    background: str = "#cccccc",
    edgecolor: str = "#777777",
) -> None:
    """Draw a dorsal cortical surface with optional node-value overlay."""
    for hemi in ("L", "R"):
        geom = geometry[hemi]
        coords = geom["coords"]
        faces = geom["faces"]
        polygons = coords[:, [0, 1]][faces]
        base = PolyCollection(
            polygons,
            facecolors=background,
            edgecolors=edgecolor,
            linewidths=0.05,
            rasterized=True,
            zorder=1,
        )
        ax.add_collection(base)
        if node_values is not None:
            vertex_values = np.full(geom["vertex_to_node"].shape, np.nan, dtype=float)
            valid = geom["vertex_to_node"] >= 0
            vertex_values[valid] = node_values[geom["vertex_to_node"][valid]]
            face_vertex_values = vertex_values[faces]
            finite_counts = np.sum(np.isfinite(face_vertex_values), axis=1)
            face_sums = np.nansum(face_vertex_values, axis=1)
            face_values = np.divide(
                face_sums,
                finite_counts,
                out=np.full(face_sums.shape, np.nan, dtype=float),
                where=finite_counts > 0,
            )
            highlight = np.isfinite(face_values) & (np.abs(face_values) > 1.0e-12)
            if np.any(highlight):
                local_norm = norm or Normalize(vmin=float(np.nanmin(face_values)), vmax=float(np.nanmax(face_values)))
                overlay = PolyCollection(
                    polygons[highlight],
                    facecolors=cmap(local_norm(face_values[highlight])),
                    edgecolors="#ffffff",
                    linewidths=0.06,
                    rasterized=True,
                    zorder=2,
                )
                ax.add_collection(overlay)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")


def target_norm(target_values: np.ndarray) -> tuple[Normalize | TwoSlopeNorm, LinearSegmentedColormap]:
    """Return a normalization and colormap for a target vector."""
    if float(np.nanmin(target_values)) < 0.0:
        vmax = float(np.nanmax(np.abs(target_values)))
        return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), SIGNED_TARGET_CMAP
    return Normalize(vmin=0.0, vmax=float(np.nanmax(target_values))), UNSIGNED_CMAP


def landmark_centroids(labels: pd.DataFrame, geometry: dict, target_vector: np.ndarray) -> pd.DataFrame:
    """Return one centroid per hemisphere for a landmark target."""
    support = np.flatnonzero(np.abs(target_vector) > 1.0e-12)
    frame = labels.loc[support, ["node_index", "hemi"]].merge(geometry["centroids"], on=["node_index", "hemi"], how="left")
    return frame.groupby("hemi", as_index=False).agg(x=("x", "mean"), y=("y", "mean"))


def draw_landmark_markers(
    ax: plt.Axes,
    labels: pd.DataFrame,
    geometry: dict,
    targets: dict[str, np.ndarray],
    *,
    annotate: bool,
    size: float = 18,
) -> None:
    """Overlay landmark marker locations."""
    y_offsets = {
        "landmark_v1": -2.8,
        "landmark_m1": 2.7,
        "landmark_pcc_precuneus": 1.5,
        "landmark_lpfc": 2.4,
    }
    for target_name in LANDMARK_TARGETS:
        if target_name not in targets:
            continue
        centroids = landmark_centroids(labels, geometry, targets[target_name])
        ax.scatter(
            centroids["x"],
            centroids["y"],
            s=size,
            facecolor=LANDMARK_COLORS[target_name],
            edgecolor="#202020",
            linewidth=0.35,
            zorder=5,
        )
        if annotate and not centroids.empty:
            row = centroids.iloc[0]
            ax.text(
                row["x"],
                row["y"] + y_offsets[target_name],
                TARGET_LABELS[target_name],
                ha="center",
                va="center",
                fontsize=6.4,
                color="#202020",
                zorder=6,
            )


def parcel_effect_values(results: dict[str, pd.DataFrame], labels: pd.DataFrame, scale: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Return node-wise median parcel effects and parcel test table."""
    subject = results["subject"]
    cortical = subject[
        (subject["scale"].eq(scale))
        & (subject["selection_strategy"].eq("static"))
        & (subject["target_family"].eq("single_cortical_parcel"))
    ].copy()
    medians = cortical.groupby("node_index", as_index=False)["pct_cycle_minus_degree"].median()
    values = np.zeros(len(labels), dtype=float)
    for _, row in medians.iterrows():
        values[int(row["node_index"])] = float(row["pct_cycle_minus_degree"])
    parcel_tests = results["parcel_tests"][
        (results["parcel_tests"]["scale"].eq(scale))
        & (results["parcel_tests"]["selection_strategy"].eq("static"))
        & (results["parcel_tests"]["target_family"].eq("single_cortical_parcel"))
        & (results["parcel_tests"]["metric"].eq("pct_cycle_minus_degree"))
    ].copy()
    return values, parcel_tests


def cortical_parcel_medians(results: dict[str, pd.DataFrame], scale: int) -> np.ndarray:
    """Return subject-median percent effects for cortical single-parcel targets."""
    subject = results["subject"]
    cortical = subject[
        (subject["scale"].eq(scale))
        & (subject["selection_strategy"].eq("static"))
        & (subject["target_family"].eq("single_cortical_parcel"))
    ].copy()
    return cortical.groupby("target_name")["pct_cycle_minus_degree"].median().to_numpy()


def surface_effect_limits(results: dict[str, pd.DataFrame], scale: int) -> tuple[float, float]:
    """Choose robust symmetric limits for signed cortical heatmaps."""
    values = cortical_parcel_medians(results, scale)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -5.0, 5.0
    cap = float(np.nanpercentile(np.abs(values), 90.0))
    cap = float(np.clip(np.ceil(cap / 5.0) * 5.0, 5.0, 20.0))
    return -cap, cap


def compact_parcel_limits(results: dict[str, pd.DataFrame]) -> tuple[float, float]:
    """Choose compact linear limits for cortical single-parcel distributions."""
    values = np.concatenate([cortical_parcel_medians(results, scale) for scale in SCALES])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -5.0, 5.0
    q1, q3 = np.nanpercentile(values, [25.0, 75.0])
    iqr = q3 - q1
    low = min(float(q1 - 2.0 * iqr), -1.0)
    high = max(float(q3 + 2.0 * iqr), 1.0)
    low = float(np.floor(low))
    high = float(np.ceil(high))
    return max(low, -10.0), min(high, 20.0)


def concentrated_panel_limits(results: dict[str, pd.DataFrame]) -> tuple[float, float]:
    """Choose linear limits for compact parcel densities plus landmark outliers."""
    low, high = compact_parcel_limits(results)
    summary = results["named_summary"][
        (results["named_summary"]["selection_strategy"].eq("static"))
        & (results["named_summary"]["metric"].eq("pct_cycle_minus_degree"))
        & (results["named_summary"]["target_name"].isin(LANDMARK_TARGETS))
    ]
    if not summary.empty:
        low = min(low, float(np.floor(summary["median"].min() - 1.0)))
        high = max(high, float(np.ceil(summary["median"].max() + 1.0)))
    return max(low, -10.0), min(high, 25.0)


def figure_limits(results: dict[str, pd.DataFrame], scale: int) -> tuple[float, float]:
    """Choose a clean linear percent range shared by panels within a figure."""
    named = results["named_summary"][
        (results["named_summary"]["scale"].eq(scale))
        & (results["named_summary"]["selection_strategy"].eq("static"))
        & (results["named_summary"]["metric"].eq("pct_cycle_minus_degree"))
    ]
    lower = float(named["ci_low"].min())
    upper = float(named["ci_high"].max())
    lower = np.floor(lower / 5.0) * 5.0
    upper = np.ceil(upper / 5.0) * 5.0
    lower = min(lower, -5.0)
    upper = max(upper, 5.0)
    return float(lower), float(upper)


def stars(p_value: float) -> str:
    """Compact significance stars."""
    if not np.isfinite(p_value) or p_value >= 0.05:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    return "*"


def draw_forest_row(
    ax: plt.Axes,
    results: dict[str, pd.DataFrame],
    scale: int,
    targets: tuple[str, ...],
    xlim: tuple[float, float],
) -> None:
    """Draw one forest/dot-CI row."""
    summary = results["named_summary"][
        (results["named_summary"]["scale"].eq(scale))
        & (results["named_summary"]["selection_strategy"].eq("static"))
        & (results["named_summary"]["metric"].eq("pct_cycle_minus_degree"))
        & (results["named_summary"]["target_name"].isin(targets))
    ].copy()
    summary["order"] = summary["target_name"].map({name: idx for idx, name in enumerate(targets)})
    summary = summary.sort_values("order")
    y = np.arange(len(summary))
    ax.axvline(0.0, color="#343434", linewidth=0.8)
    for yi, (_, row) in zip(y, summary.iterrows()):
        ax.errorbar(
            row["median"],
            yi,
            xerr=[[row["median"] - row["ci_low"]], [row["ci_high"] - row["median"]]],
            fmt="o",
            color="#303030",
            ecolor="#303030",
            markersize=4.5,
            capsize=2.3,
            linewidth=1.2,
            zorder=3,
        )
        star_text = stars(float(row["p_value_holm"]))
        if star_text:
            star_x = min(max(float(row["ci_high"]) + 0.035 * (xlim[1] - xlim[0]), xlim[0]), xlim[1])
            ax.text(star_x, yi, star_text, ha="left", va="center", fontsize=8.5, color="#202020")
    ax.set_yticks(y)
    ax.set_yticklabels([TARGET_LABELS.get(name, name) for name in summary["target_name"]], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.grid(axis="x", color="#d6d6d6", linewidth=0.45)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)


def draw_concentrated_stats_row(
    ax: plt.Axes,
    results: dict[str, pd.DataFrame],
    scale: int,
    xlim: tuple[float, float],
) -> None:
    """Draw cortical parcel distribution plus landmark concentrated targets."""
    subject = results["subject"]
    cortical = subject[
        (subject["scale"].eq(scale))
        & (subject["selection_strategy"].eq("static"))
        & (subject["target_family"].eq("single_cortical_parcel"))
    ].copy()
    parcel_medians = cortical.groupby("target_name")["pct_cycle_minus_degree"].median().to_numpy()
    clipped = np.clip(parcel_medians, xlim[0], xlim[1])
    parts = ax.violinplot([clipped], positions=[0], vert=False, widths=0.55, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#d8d8d8")
        body.set_edgecolor("#555555")
        body.set_alpha(0.9)
        body.set_linewidth(0.5)
    ax.scatter(np.clip(np.median(parcel_medians), xlim[0], xlim[1]), 0, s=18, color="#202020", zorder=4)

    parcel_tests = results["parcel_tests"][
        (results["parcel_tests"]["scale"].eq(scale))
        & (results["parcel_tests"]["selection_strategy"].eq("static"))
        & (results["parcel_tests"]["target_family"].eq("single_cortical_parcel"))
        & (results["parcel_tests"]["metric"].eq("pct_cycle_minus_degree"))
    ].copy()
    if not parcel_tests.empty:
        parcel_tests["category"] = "null"
        parcel_tests.loc[(parcel_tests["reject_bh"]) & (parcel_tests["median_subject_average"] < 0.0), "category"] = "cycle"
        parcel_tests.loc[(parcel_tests["reject_bh"]) & (parcel_tests["median_subject_average"] > 0.0), "category"] = "degree"
        for category, y in (("cycle", -0.28), ("null", -0.36), ("degree", -0.44)):
            subset = parcel_tests[parcel_tests["category"].eq(category)]
            ax.scatter(
                np.clip(subset["median_subject_average"], xlim[0], xlim[1]),
                np.full(len(subset), y),
                s=4,
                color=SIGNIFICANCE_COLORS[category],
                linewidths=0,
                alpha=0.8,
            )

    summary = results["named_summary"][
        (results["named_summary"]["scale"].eq(scale))
        & (results["named_summary"]["selection_strategy"].eq("static"))
        & (results["named_summary"]["metric"].eq("pct_cycle_minus_degree"))
        & (results["named_summary"]["target_name"].isin(LANDMARK_TARGETS))
    ].copy()
    summary["order"] = summary["target_name"].map({name: idx for idx, name in enumerate(LANDMARK_TARGETS)})
    summary = summary.sort_values("order")
    for offset, (_, row) in enumerate(summary.iterrows(), start=1):
        ax.errorbar(
            row["median"],
            offset,
            xerr=[[row["median"] - row["ci_low"]], [row["ci_high"] - row["median"]]],
            fmt="o",
            color=LANDMARK_COLORS[row["target_name"]],
            ecolor="#303030",
            markeredgecolor="#202020",
            markeredgewidth=0.35,
            markersize=4.8,
            capsize=2.2,
            linewidth=1.1,
            zorder=3,
        )
        star_text = stars(float(row["p_value_holm"]))
        if star_text:
            star_x = min(max(float(row["ci_high"]) + 0.035 * (xlim[1] - xlim[0]), xlim[0]), xlim[1])
            ax.text(star_x, offset, star_text, ha="left", va="center", fontsize=8.5, color="#202020")

    ax.axvline(0.0, color="#343434", linewidth=0.8)
    ax.set_xlim(*xlim)
    ax.set_yticks(np.arange(1 + len(LANDMARK_TARGETS)))
    ax.set_yticklabels(["Cortical parcels"] + [TARGET_LABELS[name] for name in LANDMARK_TARGETS], fontsize=8.5)
    ax.set_ylim(-0.65, len(LANDMARK_TARGETS) + 0.45)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#d6d6d6", linewidth=0.45)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)


def draw_parcel_inset(ax: plt.Axes, values: np.ndarray, parcel_tests: pd.DataFrame, xlim: tuple[float, float]) -> None:
    """Draw compact parcel distribution with BH-category rug."""
    finite_values = values[np.isfinite(values)]
    clipped = np.clip(finite_values, xlim[0], xlim[1])
    parts = ax.violinplot([clipped], positions=[0], vert=False, widths=0.55, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#d8d8d8")
        body.set_edgecolor("#555555")
        body.set_alpha(0.9)
        body.set_linewidth(0.5)
    median = float(np.nanmedian(finite_values))
    ax.scatter(np.clip(median, xlim[0], xlim[1]), 0, s=18, color="#202020", zorder=4)
    if not parcel_tests.empty:
        tests = parcel_tests.copy()
        tests["category"] = "null"
        tests.loc[(tests["reject_bh"]) & (tests["median_subject_average"] < 0.0), "category"] = "cycle"
        tests.loc[(tests["reject_bh"]) & (tests["median_subject_average"] > 0.0), "category"] = "degree"
        for category, y in (("cycle", -0.42), ("null", -0.50), ("degree", -0.58)):
            subset = tests[tests["category"].eq(category)]
            ax.scatter(
                np.clip(subset["median_subject_average"], xlim[0], xlim[1]),
                np.full(len(subset), y),
                s=4,
                color=SIGNIFICANCE_COLORS[category],
                linewidths=0,
                alpha=0.85,
            )
    ax.axvline(0.0, color="#343434", linewidth=0.8)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.72, 0.55)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="#d6d6d6", linewidth=0.45)


def add_panel_label(ax: plt.Axes, text: str) -> None:
    """Add small panel label."""
    ax.text(-0.02, 1.05, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, fontweight="bold")


def make_composite_figure(config: dict, results: dict[str, pd.DataFrame], scale: int, *, output_name: str | None = None) -> Path:
    """Create the analysis-only composite figure for one scale."""
    labels = build_label_table(scale=scale, atlas_dir=config["paths"]["atlas_dir"])
    targets = load_targets(config, scale)
    geometry = load_geometry(scale, config["paths"]["atlas_dir"])
    parcel_values, parcel_tests = parcel_effect_values(results, labels, scale)
    xlim = figure_limits(results, scale)
    heatmap_xlim = surface_effect_limits(results, scale)
    parcel_xlim = compact_parcel_limits(results)
    norm = TwoSlopeNorm(vmin=heatmap_xlim[0], vcenter=0.0, vmax=heatmap_xlim[1])

    mpl.rcParams.update({"svg.fonttype": "none", "axes.unicode_minus": False})
    fig = plt.figure(figsize=(8.0, 7.8), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=3,
        ncols=9,
        height_ratios=[3.05, 1.18, 1.42],
        hspace=0.46,
        wspace=0.25,
    )

    ax_surface = fig.add_subplot(grid[0, :7])
    draw_surface(
        ax_surface,
        geometry,
        np.clip(parcel_values, heatmap_xlim[0], heatmap_xlim[1]),
        cmap=EFFECT_CMAP,
        norm=norm,
    )
    draw_landmark_markers(ax_surface, labels, geometry, targets, annotate=True, size=24)
    add_panel_label(ax_surface, "Concentrated baseline")

    ax_inset = fig.add_subplot(grid[0, 7:])
    draw_parcel_inset(ax_inset, parcel_values, parcel_tests, parcel_xlim)
    ax_inset.set_xlabel("Cycle - degree energy (%)", fontsize=8)

    cax = ax_surface.inset_axes([0.78, 0.04, 0.19, 0.035])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=EFFECT_CMAP)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([heatmap_xlim[0], 0.0, heatmap_xlim[1]])
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.set_label("%", fontsize=6, labelpad=-2)

    ax_module = fig.add_subplot(grid[1, :])
    draw_forest_row(ax_module, results, scale, MODULE_ROW_TARGETS, xlim)
    add_panel_label(ax_module, "Intermediate")
    ax_module.set_xlabel("")
    ax_module.set_xticklabels([])

    ax_distributed = fig.add_subplot(grid[2, :])
    draw_forest_row(ax_distributed, results, scale, DISTRIBUTED_ROW_TARGETS, xlim)
    add_panel_label(ax_distributed, "Distributed")
    ax_distributed.set_xlabel("Cycle - degree energy (%)", fontsize=9)

    fig.suptitle(f"Target-state energy across the concentrated-distributed axis | {scale} regions", fontsize=11, y=0.982)
    out_dir = REPO_ROOT / config["paths"]["figures_dir"] / "section_target_state_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / (output_name or f"target_state_axis_composite_scale-{scale}.svg")
    fig.savefig(output, bbox_inches="tight", facecolor="white", dpi=185)
    plt.close(fig)
    return output


def make_stats_only_figure(config: dict, results: dict[str, pd.DataFrame], scale: int) -> Path:
    """Create a composite figure without any brain-region renderings."""
    xlim = figure_limits(results, scale)
    mpl.rcParams.update({"svg.fonttype": "none", "axes.unicode_minus": False})
    fig = plt.figure(figsize=(6.8, 5.6), constrained_layout=True)
    grid = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.55, 0.82, 1.05], hspace=0.14)

    ax_concentrated = fig.add_subplot(grid[0, 0])
    draw_concentrated_stats_row(ax_concentrated, results, scale, xlim)
    add_panel_label(ax_concentrated, "Concentrated")
    ax_concentrated.set_xlabel("")
    ax_concentrated.set_xticklabels([])

    ax_module = fig.add_subplot(grid[1, 0])
    draw_forest_row(ax_module, results, scale, MODULE_ROW_TARGETS, xlim)
    add_panel_label(ax_module, "Intermediate")
    ax_module.set_xlabel("")
    ax_module.set_xticklabels([])

    ax_distributed = fig.add_subplot(grid[2, 0])
    draw_forest_row(ax_distributed, results, scale, DISTRIBUTED_ROW_TARGETS, xlim)
    add_panel_label(ax_distributed, "Distributed")
    ax_distributed.set_xlabel("Cycle - degree energy (%)", fontsize=9)

    fig.suptitle(f"Target-state energy | {scale} regions", fontsize=10.5)
    out_dir = REPO_ROOT / config["paths"]["figures_dir"] / "section_target_state_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"target_state_axis_stats_scale-{scale}.svg"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def piece_output_dir(config: dict, scale: int) -> Path:
    """Directory for manually assembled figure pieces."""
    out_dir = (
        REPO_ROOT
        / config["paths"]["figures_dir"]
        / "section_target_state_selection"
        / f"target_state_axis_pieces_scale-{scale}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def make_surface_heatmap_piece(config: dict, results: dict[str, pd.DataFrame], scale: int) -> Path:
    """Export the cortical single-parcel heatmap as its own SVG."""
    labels = build_label_table(scale=scale, atlas_dir=config["paths"]["atlas_dir"])
    targets = load_targets(config, scale)
    geometry = load_geometry(scale, config["paths"]["atlas_dir"])
    parcel_values, _ = parcel_effect_values(results, labels, scale)
    xlim = surface_effect_limits(results, scale)
    norm = TwoSlopeNorm(vmin=xlim[0], vcenter=0.0, vmax=xlim[1])

    mpl.rcParams.update({"svg.fonttype": "none", "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(5.8, 4.4), facecolor="white")
    draw_surface(
        ax,
        geometry,
        np.clip(parcel_values, xlim[0], xlim[1]),
        cmap=EFFECT_CMAP,
        norm=norm,
        background="#c6c6c6",
        edgecolor="#6f6f6f",
    )
    draw_landmark_markers(ax, labels, geometry, targets, annotate=True, size=24)
    cax = ax.inset_axes([0.72, 0.035, 0.22, 0.04])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=EFFECT_CMAP)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([xlim[0], 0.0, xlim[1]])
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.set_label("%", fontsize=6, labelpad=-2)
    output = piece_output_dir(config, scale) / f"target_state_piece_surface_heatmap_scale-{scale}.svg"
    fig.savefig(output, bbox_inches="tight", facecolor="white", dpi=220)
    plt.close(fig)
    return output


def make_parcel_distribution_piece(config: dict, results: dict[str, pd.DataFrame], scale: int) -> Path:
    """Export the compact cortical parcel distribution inset as its own SVG."""
    labels = build_label_table(scale=scale, atlas_dir=config["paths"]["atlas_dir"])
    parcel_values, parcel_tests = parcel_effect_values(results, labels, scale)
    xlim = compact_parcel_limits(results)
    mpl.rcParams.update({"svg.fonttype": "none", "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(2.35, 1.45), facecolor="white")
    draw_parcel_inset(ax, parcel_values, parcel_tests, xlim)
    ax.set_xlabel("Cycle - degree energy (%)", fontsize=8)
    output = piece_output_dir(config, scale) / f"target_state_piece_cortical_distribution_scale-{scale}.svg"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def make_forest_piece(
    config: dict,
    results: dict[str, pd.DataFrame],
    scale: int,
    targets: tuple[str, ...],
    piece_name: str,
    *,
    xlabel: bool,
) -> Path:
    """Export one forest/dot-CI panel as its own SVG."""
    xlim = figure_limits(results, scale)
    height = 0.52 * len(targets) + 0.62
    mpl.rcParams.update({"svg.fonttype": "none", "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(5.6, height), facecolor="white")
    draw_forest_row(ax, results, scale, targets, xlim)
    if xlabel:
        ax.set_xlabel("Cycle - degree energy (%)", fontsize=9)
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    output = piece_output_dir(config, scale) / f"target_state_piece_{piece_name}_scale-{scale}.svg"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def make_piece_figures(config: dict, results: dict[str, pd.DataFrame], scale: int) -> list[Path]:
    """Export each component of the target-state figure separately."""
    return [
        make_surface_heatmap_piece(config, results, scale),
        make_parcel_distribution_piece(config, results, scale),
        make_forest_piece(config, results, scale, MODULE_ROW_TARGETS, "intermediate_forest", xlabel=False),
        make_forest_piece(config, results, scale, DISTRIBUTED_ROW_TARGETS, "distributed_forest", xlabel=True),
    ]


def old_style_output_dir(config: dict) -> Path:
    """Directory for old-style percent summary panels."""
    out_dir = REPO_ROOT / config["paths"]["figures_dir"] / "section_target_state_selection" / "target_state_axis_old_style_pct"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def scale_positions() -> dict[int, int]:
    """Map scales to categorical x positions."""
    return {scale: index for index, scale in enumerate(SCALES)}


def named_summary_row(results: dict[str, pd.DataFrame], scale: int, target_name: str) -> pd.Series:
    """Fetch one percent summary row for a named target."""
    subset = results["named_summary"][
        (results["named_summary"]["scale"].eq(scale))
        & (results["named_summary"]["selection_strategy"].eq("static"))
        & (results["named_summary"]["metric"].eq("pct_cycle_minus_degree"))
        & (results["named_summary"]["target_name"].eq(target_name))
    ]
    if subset.empty:
        raise ValueError(f"Missing pct summary for {target_name} at scale {scale}.")
    return subset.iloc[0]


def make_old_style_concentrated_panel(config: dict, results: dict[str, pd.DataFrame]) -> Path:
    """Export old-style concentrated baseline violin panel using percent effects."""
    fig, ax = plt.subplots(figsize=(7.4, 3.75), facecolor="white", constrained_layout=True)
    positions = scale_positions()
    ylim = concentrated_panel_limits(results)
    rng = np.random.default_rng(20260619)
    violin_data = []
    for scale in SCALES:
        parcel_medians = cortical_parcel_medians(results, scale)
        in_view = parcel_medians[(parcel_medians >= ylim[0]) & (parcel_medians <= ylim[1])]
        violin_data.append(in_view)
    parts = ax.violinplot(
        violin_data,
        positions=[positions[scale] for scale in SCALES],
        widths=0.7,
        showextrema=False,
    )
    for body, scale in zip(parts["bodies"], SCALES):
        body.set_facecolor(SCALE_COLORS[scale])
        body.set_edgecolor("#303030")
        body.set_alpha(0.24)
        body.set_linewidth(0.75)

    for scale in SCALES:
        x = positions[scale]
        cortical_values = cortical_parcel_medians(results, scale)
        in_view = cortical_values[(cortical_values >= ylim[0]) & (cortical_values <= ylim[1])]
        jitter = rng.normal(0.0, 0.035, size=len(in_view))
        ax.scatter(
            np.full(len(in_view), x) + jitter,
            in_view,
            s=8,
            color="#303030",
            alpha=0.34,
            linewidths=0,
            zorder=3,
        )
        ax.scatter(x, np.median(cortical_values), s=20, color="#202020", zorder=4)
        offsets = {
            "landmark_v1": -0.18,
            "landmark_m1": -0.06,
            "landmark_pcc_precuneus": 0.06,
            "landmark_lpfc": 0.18,
        }
        label_offsets = {
            "landmark_v1": (0, -11),
            "landmark_m1": (-5, 6),
            "landmark_pcc_precuneus": (0, 9),
            "landmark_lpfc": (7, -2),
        }
        for target_name in LANDMARK_TARGETS:
            row = named_summary_row(results, scale, target_name)
            y = float(row["median"])
            ax.scatter(
                x + offsets[target_name],
                y,
                s=24,
                facecolor="white",
                edgecolor="#202020",
                linewidth=0.65,
                zorder=5,
            )
            label = TARGET_LABELS[target_name] + stars(float(row["p_value_holm"]))
            ax.annotate(
                label,
                xy=(x + offsets[target_name], y),
                xytext=label_offsets[target_name],
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=6.4,
                color="#202020",
                zorder=6,
            )

    ax.axhline(0.0, color="#202020", linewidth=0.8)
    ax.set_ylim(*ylim)
    ax.set_xticks([positions[scale] for scale in SCALES])
    ax.set_xticklabels([str(scale) for scale in SCALES])
    ax.set_xlabel("Parcellation scale", fontsize=8)
    ax.set_ylabel("Cycle - degree energy (%)", fontsize=8)
    ax.set_title("Concentrated baseline: cortical single-parcel targets", loc="left", fontsize=9.5, fontweight="bold")
    ax.grid(axis="y", color="#d6d6d6", linewidth=0.55)
    ax.tick_params(labelsize=7.5)
    output = old_style_output_dir(config) / "target_state_old_style_concentrated_pct_compact.svg"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def old_style_xlim(results: dict[str, pd.DataFrame], targets: tuple[str, ...]) -> tuple[float, float]:
    """Choose percent x-limits for old-style forest panels."""
    summary = results["named_summary"][
        (results["named_summary"]["selection_strategy"].eq("static"))
        & (results["named_summary"]["metric"].eq("pct_cycle_minus_degree"))
        & (results["named_summary"]["target_name"].isin(targets))
    ]
    lower = float(np.floor(summary["ci_low"].min() / 5.0) * 5.0)
    upper = float(np.ceil(summary["ci_high"].max() / 5.0) * 5.0)
    return min(lower, -5.0), max(upper, 5.0)


def make_old_style_forest_panel(
    config: dict,
    results: dict[str, pd.DataFrame],
    targets: tuple[str, ...],
    output_stem: str,
    title: str,
) -> Path:
    """Export old-style all-scale forest panel using percent effects."""
    xlim = old_style_xlim(results, targets)
    fig, ax = plt.subplots(figsize=(7.4, 3.15), facecolor="white", constrained_layout=True)
    x_positions = {target_name: idx for idx, target_name in enumerate(targets)}
    scale_offsets = {68: -0.18, 114: 0.0, 219: 0.18}
    ax.axhline(0.0, color="#202020", linewidth=0.85)
    for scale in SCALES:
        for target_name in targets:
            row = named_summary_row(results, scale, target_name)
            x = x_positions[target_name] + scale_offsets[scale]
            median = float(row["median"])
            ax.errorbar(
                x,
                median,
                yerr=[[median - float(row["ci_low"])], [float(row["ci_high"]) - median]],
                fmt="o",
                color=SCALE_COLORS[scale],
                ecolor=SCALE_COLORS[scale],
                markersize=4.2,
                capsize=2.2,
                linewidth=1.1,
                label=str(scale) if target_name == targets[0] else None,
                zorder=3,
            )
            star_text = stars(float(row["p_value_holm"]))
            if star_text:
                star_y = min(float(row["ci_high"]) + 0.03 * (xlim[1] - xlim[0]), xlim[1])
                ax.text(x, star_y, star_text, ha="center", va="bottom", fontsize=8.0, color=SCALE_COLORS[scale])
    ax.set_ylim(*xlim)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([TARGET_LABELS[target] for target in targets], fontsize=8)
    ax.set_ylabel("Cycle - degree energy (%)", fontsize=8)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.grid(axis="y", color="#d6d6d6", linewidth=0.55)
    ax.legend(title="Scale", frameon=False, fontsize=7, title_fontsize=7, loc="upper right", ncols=3)
    ax.tick_params(labelsize=7.5)
    output = old_style_output_dir(config) / f"{output_stem}.svg"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def make_old_style_pct_panels(config: dict, results: dict[str, pd.DataFrame]) -> list[Path]:
    """Export the old-style target-state summary as separate percent SVG panels."""
    return [
        make_old_style_concentrated_panel(config, results),
        make_old_style_forest_panel(
            config,
            results,
            COMBINED_MODULE_DISTRIBUTED_TARGETS,
            "target_state_old_style_module_distributed_pct",
            "Module and distributed targets",
        ),
    ]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> tuple[float, float]:
    """Bootstrap median CI for compact per-k panels."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    low, high = np.percentile(np.median(samples, axis=1), [2.5, 97.5])
    return float(low), float(high)


def make_per_k_figure(config: dict, results: dict[str, pd.DataFrame], scale: int) -> Path:
    """Create supplementary per-k breakdown for cortical baseline and distributed targets."""
    pairwise = results["pairwise"]
    scale_data = pairwise[(pairwise["scale"].eq(scale)) & (pairwise["selection_strategy"].eq("static"))].copy()
    xlim = figure_limits(results, scale)
    rng = np.random.default_rng(20260619 + scale)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)

    cortical = scale_data[scale_data["target_family"].eq("single_cortical_parcel")]
    parcel_by_k = []
    for k in K_VALUES:
        subset = cortical[cortical["k"].eq(k)]
        parcel_by_k.append(subset.groupby("target_name")["pct_cycle_minus_degree"].median().to_numpy())
    parts = axes[0].violinplot(
        [np.clip(vals, xlim[0], xlim[1]) for vals in parcel_by_k],
        positions=np.arange(len(K_VALUES)),
        widths=0.62,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor("#d8d8d8")
        body.set_edgecolor("#555555")
        body.set_linewidth(0.5)
    axes[0].axhline(0.0, color="#343434", linewidth=0.8)
    axes[0].set_xticks(np.arange(len(K_VALUES)))
    axes[0].set_xticklabels([str(k) for k in K_VALUES])
    axes[0].set_ylim(*xlim)
    axes[0].set_title("Cortical single parcels", fontsize=9)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Cycle - degree energy (%)")

    distributed = scale_data[scale_data["target_name"].isin(DISTRIBUTED_ROW_TARGETS)]
    colors = {
        "principal_gradient": "#2166ac",
        "bipolar_gradient": "#67a9cf",
        "uniform_cortical": "#fdae61",
        "occipital_module": "#d7301f",
    }
    axes[1].axhline(0.0, color="#343434", linewidth=0.8)
    for target_name in DISTRIBUTED_ROW_TARGETS:
        target_subset = distributed[distributed["target_name"].eq(target_name)]
        xs, medians, lows, highs = [], [], [], []
        for k in K_VALUES:
            values = target_subset[target_subset["k"].eq(k)]["pct_cycle_minus_degree"].to_numpy()
            xs.append(k)
            medians.append(float(np.median(values)))
            low, high = bootstrap_ci(values, rng)
            lows.append(low)
            highs.append(high)
        axes[1].errorbar(
            xs,
            medians,
            yerr=[np.asarray(medians) - np.asarray(lows), np.asarray(highs) - np.asarray(medians)],
            marker="o",
            markersize=3.5,
            linewidth=1.1,
            capsize=2.0,
            color=colors[target_name],
            label=TARGET_LABELS[target_name],
        )
    axes[1].set_xticks(K_VALUES)
    axes[1].set_ylim(*xlim)
    axes[1].set_title("Distributed targets", fontsize=9)
    axes[1].set_xlabel("k")
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    for ax in axes:
        ax.grid(axis="y", color="#d6d6d6", linewidth=0.45)
        ax.tick_params(labelsize=8)

    out_dir = REPO_ROOT / config["paths"]["figures_dir"] / "section_target_state_selection"
    output = out_dir / f"target_state_axis_per_k_scale-{scale}.svg"
    fig.savefig(output, bbox_inches="tight", facecolor="white", dpi=185)
    plt.close(fig)
    return output


def make_subcortical_figure(config: dict, results: dict[str, pd.DataFrame], scale: int) -> Path | None:
    """Create separate subcortical single-parcel baseline figure when available."""
    subject = results["subject"]
    subcortical = subject[
        (subject["scale"].eq(scale))
        & (subject["selection_strategy"].eq("static"))
        & (subject["target_family"].eq("single_subcortical_parcel"))
    ].copy()
    if subcortical.empty:
        return None
    xlim = figure_limits(results, scale)
    values = subcortical.groupby("target_name")["pct_cycle_minus_degree"].median().to_numpy()
    fig, ax = plt.subplots(figsize=(4.0, 2.3), constrained_layout=True)
    parts = ax.violinplot([np.clip(values, xlim[0], xlim[1])], positions=[0], vert=False, widths=0.6, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#d8d8d8")
        body.set_edgecolor("#555555")
        body.set_linewidth(0.5)
    ax.scatter(np.clip(np.median(values), xlim[0], xlim[1]), 0, s=18, color="#202020")
    ax.axvline(0.0, color="#343434", linewidth=0.8)
    ax.set_xlim(*xlim)
    ax.set_yticks([])
    ax.set_xlabel("Cycle - degree energy (%)")
    ax.set_title(f"Subcortical single-parcel baseline | {scale} regions", fontsize=9)
    output = (
        REPO_ROOT
        / config["paths"]["figures_dir"]
        / "section_target_state_selection"
        / f"target_state_subcortical_baseline_scale-{scale}.svg"
    )
    fig.savefig(output, bbox_inches="tight", facecolor="white", dpi=185)
    plt.close(fig)
    return output


def main() -> None:
    """Generate requested target-state figures."""
    args = parse_args()
    config = load_config(REPO_ROOT / args.config)
    results = load_results(config)
    outputs = []
    if args.old_style:
        outputs.extend(make_old_style_pct_panels(config, results))
        for output in outputs:
            print(output)
        return
    for scale in requested_scales(args.scale):
        if args.pieces:
            outputs.extend(make_piece_figures(config, results, scale))
        elif args.stats_only:
            outputs.append(make_stats_only_figure(config, results, scale))
        else:
            outputs.append(make_composite_figure(config, results, scale))
        if args.include_per_k:
            outputs.append(make_per_k_figure(config, results, scale))
        if args.include_subcortical:
            subcortical_output = make_subcortical_figure(config, results, scale)
            if subcortical_output is not None:
                outputs.append(subcortical_output)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
