#!/usr/bin/env python
"""Make paper-style degree/cycle rank-divergence panels for each atlas scale."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from functools import lru_cache

from nilearn import datasets, plotting
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import nibabel as nib
from matplotlib.collections import PolyCollection

from hoc.atlas import EXCLUDED_LABELS, SCALE_TO_RESOLUTION, surface_values_from_node_scores
from hoc.targets import margulies_principal_gradient

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
ATLAS_DIR = REPO_ROOT / "data" / "external" / "atlas_cache" / "atl-cammoun2012" / "fsaverage5"
FSAVERAGE = datasets.load_fsaverage(mesh="fsaverage5")
# Denser mesh for rendering: parcel labels live on fsaverage5, but the piecewise-constant
# parcel values are upsampled to this mesh by nearest-neighbor on the sphere for crisper surfaces.
HIGHRES_MESH = "fsaverage6"

SCALES = (68, 114, 219)


@lru_cache(maxsize=None)
def _highres_surface():
    return datasets.load_fsaverage(mesh=HIGHRES_MESH)


@lru_cache(maxsize=None)
def _fsaverage5_to_highres_index(hemi: str) -> np.ndarray:
    """Nearest fsaverage5 vertex for each high-resolution vertex, matched on the sphere."""
    part = {"L": "left", "R": "right"}[hemi]
    source = np.asarray(FSAVERAGE["sphere"].parts[part].coordinates, dtype=float)
    target = np.asarray(_highres_surface()["sphere"].parts[part].coordinates, dtype=float)
    _, nearest = cKDTree(source).query(target)
    return nearest


def highres_surface_values(scale: int, node_scores: np.ndarray) -> dict[str, np.ndarray]:
    """Broadcast node scores to fsaverage5 vertices, then upsample to the high-resolution mesh."""
    fs5_values = surface_values_from_node_scores(scale=scale, atlas_dir=ATLAS_DIR, node_scores=node_scores)
    return {hemi: fs5_values[hemi][_fsaverage5_to_highres_index(hemi)] for hemi in ("L", "R")}


# Flat-projection surface rendering, matching the style of the target-state figure
# (Fig. 5): pial surface, colored polygons, gray medial wall. Generalized here from a
# dorsal-only projection to lateral/medial/dorsal views using back-face culling and a
# painter's-algorithm depth sort so the far wall does not bleed through.
FLAT_VIEWS = {
    "L_lateral": {"hemis": ("L",), "depth": 0, "toward": -1, "h": 1, "v": 2, "flip_h": True},
    "L_medial": {"hemis": ("L",), "depth": 0, "toward": +1, "h": 1, "v": 2, "flip_h": False},
    "R_lateral": {"hemis": ("R",), "depth": 0, "toward": +1, "h": 1, "v": 2, "flip_h": False},
    "R_medial": {"hemis": ("R",), "depth": 0, "toward": -1, "h": 1, "v": 2, "flip_h": True},
    "dorsal": {"hemis": ("L", "R"), "depth": 2, "toward": +1, "h": 0, "v": 1, "flip_h": False},
}
MEDIAL_WALL_COLOR = (0.82, 0.82, 0.82, 1.0)


@lru_cache(maxsize=None)
def _highres_vertex_to_node(scale: int) -> tuple[np.ndarray, np.ndarray]:
    """Map each high-resolution vertex to a global node index (-1 for excluded/medial-wall)."""
    resolution = SCALE_TO_RESOLUTION[scale]
    per_hemi = {}
    node_index = 0
    for hemi in ("L", "R"):
        annot = ATLAS_DIR / f"atl-Cammoun2012_space-fsaverage5_res-{resolution}_hemi-{hemi}_deterministic.annot"
        vertex_labels, _, names = nib.freesurfer.read_annot(str(annot))
        mapping = np.full(vertex_labels.shape, -1, dtype=int)
        for atlas_index, raw_name in enumerate(names):
            if raw_name.decode() in EXCLUDED_LABELS:
                continue
            mapping[vertex_labels == atlas_index] = node_index
            node_index += 1
        per_hemi[hemi] = mapping[_fsaverage5_to_highres_index(hemi)]
    return per_hemi["L"], per_hemi["R"]


def _draw_flat_view(ax, scale: int, node_scores: np.ndarray, cmap, view_key: str) -> None:
    """Draw one flat-projected view (one or both hemispheres) into an axis."""
    view = FLAT_VIEWS[view_key]
    v2n = dict(zip(("L", "R"), _highres_vertex_to_node(scale)))
    cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    for hemi in view["hemis"]:
        part = {"L": "left", "R": "right"}[hemi]
        coords = np.asarray(_highres_surface()["pial"].parts[part].coordinates, dtype=float)
        faces = np.asarray(_highres_surface()["pial"].parts[part].faces, dtype=int)
        tri = coords[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        front = normals[:, view["depth"]] * view["toward"] > 0
        tri, front_faces = tri[front], faces[front]
        order = np.argsort(tri[:, :, view["depth"]].mean(axis=1) * view["toward"])
        tri, front_faces = tri[order], front_faces[order]
        horizontal = tri[:, :, view["h"]] * (-1.0 if view["flip_h"] else 1.0)
        vertical = tri[:, :, view["v"]]
        polygons = np.stack([horizontal, vertical], axis=2)
        face_nodes = v2n[hemi][front_faces]
        valid = face_nodes >= 0
        summed = np.where(valid, node_scores[np.clip(face_nodes, 0, None)], 0.0).sum(axis=1)
        counts = valid.sum(axis=1)
        face_value = np.where(counts > 0, summed / np.maximum(counts, 1), np.nan)
        colors = np.where(
            np.isfinite(face_value)[:, None],
            cmap(np.nan_to_num(face_value)),
            np.array(MEDIAL_WALL_COLOR),
        )
        # Edge color matches the face so anti-aliasing does not leave a hairline triangle mesh.
        ax.add_collection(PolyCollection(polygons, facecolors=colors, edgecolors=colors, linewidths=0.2, rasterized=True))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.01)


TOP_N = 15

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }
)

COLORS = {
    "degree_only": "#ef3b2c",
    "cycle_only": "#2b8cbe",
    "overlap": "#7b3294",
    "other": "#4d4d4d",
}

ANATOMICAL_ORDER = ["frontal", "cingulate", "insula", "parietal", "temporal", "occipital", "other"]


def load_scale_data(scale: int) -> tuple[pd.DataFrame, np.ndarray]:
    labels = pd.read_csv(RESULTS_DIR / "processed" / f"atlas_labels_scale-{scale}.csv")
    adjacency = np.load(RESULTS_DIR / "processed" / f"connectomes_scale-{scale}.npz")["cohort_reference"].copy()
    np.fill_diagonal(adjacency, 0.0)
    return labels, adjacency


def mean_rank_table(scale: int) -> pd.DataFrame:
    node_scores = pd.read_csv(RESULTS_DIR / "primary" / "node_scores.csv")
    subset = node_scores[
        (node_scores["scale"] == scale)
        & node_scores["method"].isin(["degree_strength", "cycle_participation"])
    ].copy()
    means = (
        subset.groupby(["node_index", "region_label", "base_region", "hemi", "anatomical_group", "method"], as_index=False)
        .agg(score=("score", "mean"), mean_subject_rank=("rank", "mean"))
    )
    wide = means.pivot_table(
        index=["node_index", "region_label", "base_region", "hemi", "anatomical_group"],
        columns="method",
        values=["score", "mean_subject_rank"],
    )
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    wide["degree_rank"] = wide["score_degree_strength"].rank(ascending=False, method="min").astype(int)
    wide["cycle_rank"] = wide["score_cycle_participation"].rank(ascending=False, method="min").astype(int)
    wide["rank_delta_cycle_minus_degree"] = wide["cycle_rank"] - wide["degree_rank"]
    return wide


def selected_categories(rank_table: pd.DataFrame) -> tuple[set[int], set[int], dict[int, str]]:
    degree_nodes = set(rank_table.nsmallest(TOP_N, "degree_rank")["node_index"].astype(int))
    cycle_nodes = set(rank_table.nsmallest(TOP_N, "cycle_rank")["node_index"].astype(int))
    categories: dict[int, str] = {}
    for node in rank_table["node_index"].astype(int):
        if node in degree_nodes and node in cycle_nodes:
            categories[node] = "overlap"
        elif node in degree_nodes:
            categories[node] = "degree_only"
        elif node in cycle_nodes:
            categories[node] = "cycle_only"
        else:
            categories[node] = "other"
    return degree_nodes, cycle_nodes, categories


def display_label(label: str) -> str:
    text = label.replace("_", " ")
    replacements = {
        "lateralorbitofrontal": "lateral orbitofrontal",
        "medialorbitofrontal": "medial orbitofrontal",
        "rostralanteriorcingulate": "rostral anterior cingulate",
        "caudalanteriorcingulate": "caudal anterior cingulate",
        "postcentral": "postcentral",
        "precentral": "precentral",
        "superiorfrontal": "superior frontal",
        "caudalmiddlefrontal": "caudal middle frontal",
        "rostralmiddlefrontal": "rostral middle frontal",
        "inferiorparietal": "inferior parietal",
        "superiorparietal": "superior parietal",
        "lateraloccipital": "lateral occipital",
        "isthmuscingulate": "isthmus cingulate",
        "posteriorcingulate": "posterior cingulate",
        "parstriangularis": "pars triangularis",
        "parsopercularis": "pars opercularis",
        "parsorbitalis": "pars orbitalis",
        "supramarginal": "supramarginal",
        "precuneus": "precuneus",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def save_lateral_surface(
    scale: int,
    node_values: np.ndarray,
    filename: str,
    cmap: str,
    symmetric_cbar: bool,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Render left/right lateral cortical surfaces for a per-node value vector."""
    surface_scores = surface_values_from_node_scores(scale=scale, atlas_dir=ATLAS_DIR, node_scores=node_values)
    if vmax is None:
        vmax = float(np.nanmax(np.abs(node_values))) if symmetric_cbar else float(np.nanmax(node_values))
    if vmin is None:
        vmin = -vmax if symmetric_cbar else float(np.nanmin(node_values))

    fig = plt.figure(figsize=(7.2, 3.4))
    ax_l = fig.add_subplot(1, 2, 1, projection="3d")
    ax_r = fig.add_subplot(1, 2, 2, projection="3d")
    for ax, hemi, part, show_cbar in ((ax_l, "left", "L", True), (ax_r, "right", "R", False)):
        plotting.plot_surf_stat_map(
            FSAVERAGE["inflated"].parts[hemi],
            surface_scores[part],
            hemi=hemi,
            view="lateral",
            cmap=cmap,
            colorbar=show_cbar,
            symmetric_cbar=symmetric_cbar,
            vmin=vmin,
            vmax=vmax,
            axes=ax,
            title="",
        )
        ax.set_title("")
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_rank_heatmap(scale: int) -> None:
    """Save cortical surface heatmap of topology-vs-degree rank divergence."""
    rank_table = mean_rank_table(scale)
    # Positive values indicate regions ranked higher by topology/cycle than by degree.
    divergence = (rank_table["degree_rank"] - rank_table["cycle_rank"]).to_numpy()
    save_lateral_surface(
        scale,
        divergence,
        f"rank_divergence_scale{scale}_heatmap.svg",
        cmap="coolwarm",
        symmetric_cbar=True,
        vmax=float(int(np.nanmax(np.abs(divergence)))),
    )


def save_score_heatmaps(scale: int) -> None:
    """Composite of mean degree strength and cycle participation on the cortical surface.

    Two rows (degree, cycle) by five flat-projected views (LH lateral, LH medial, RH medial,
    RH lateral, dorsal), rendered in the same pial-projection style as the target-state figure.
    Both rankings share one min--max normalized color scale.
    """
    rank_table = mean_rank_table(scale)
    rows = (("degree_strength", "Degree strength"), ("cycle_participation", "Cycle participation"))
    views = (("L_lateral", "LH lateral"), ("L_medial", "LH medial"), ("R_medial", "RH medial"),
             ("R_lateral", "RH lateral"), ("dorsal", "Dorsal"))
    cmap = "viridis"  # single colormap so the two rankings share one color scale

    fig = plt.figure(figsize=(14.0, 5.8))
    fig.subplots_adjust(left=0.06, right=0.90, top=0.90, bottom=0.02, wspace=0.02, hspace=0.05)
    top_axes = []
    for row_index, (method, row_title) in enumerate(rows):
        scores = rank_table[f"score_{method}"].to_numpy()
        span = float(np.nanmax(scores) - np.nanmin(scores))
        scores = (scores - float(np.nanmin(scores))) / span if span > 0 else np.zeros_like(scores)
        row_axes = []
        for col_index, (view_key, _) in enumerate(views):
            ax = fig.add_subplot(2, len(views), row_index * len(views) + col_index + 1)
            _draw_flat_view(ax, scale, scores, cmap, view_key)
            row_axes.append(ax)
        top_axes = row_axes if row_index == 0 else top_axes
        first = row_axes[0].get_position()
        fig.text(0.045, first.y0 + first.height / 2, row_title, ha="right", va="center", rotation=90, fontsize=12)
    # Single shared colorbar spanning both rows (both maps are on the same [0, 1] scale).
    top = top_axes[-1].get_position()
    bottom = fig.axes[-1].get_position()
    cax = fig.add_axes([0.915, bottom.y0 + 0.05, 0.012, (top.y1 - bottom.y0) - 0.10])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Min--max normalized score", fontsize=10)
    cbar.ax.tick_params(labelsize=9, length=2)
    for ax, (_, view_label) in zip(top_axes, views):
        pos = ax.get_position()
        fig.text(pos.x0 + pos.width / 2, 0.91, view_label, ha="center", va="bottom", fontsize=10)
    fig.savefig(FIGURES_DIR / f"rank_divergence_scale{scale}_score_surfaces.svg", dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def circular_order(labels: pd.DataFrame, rank_table: pd.DataFrame) -> list[int]:
    merged = labels.merge(rank_table[["node_index", "degree_rank"]], on="node_index", how="left")
    group_map = {name: idx for idx, name in enumerate(ANATOMICAL_ORDER)}
    merged["hemi_order"] = merged["hemi"].map({"L": 0, "R": 1}).fillna(9)
    merged["group_order"] = merged["anatomical_group"].map(group_map).fillna(99)
    merged = merged.sort_values(["hemi_order", "group_order", "base_region", "degree_rank", "node_index"])
    return merged["node_index"].astype(int).tolist()


def save_graph_panel(scale: int) -> None:
    labels, adjacency = load_scale_data(scale)
    rank_table = mean_rank_table(scale)
    degree_nodes, cycle_nodes, categories = selected_categories(rank_table)
    selected_nodes = sorted(degree_nodes | cycle_nodes)

    order = [node for node in circular_order(labels, rank_table) if node in selected_nodes]
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(order), endpoint=False)
    positions = {node: np.array([np.cos(angle), np.sin(angle)]) for node, angle in zip(order, angles)}

    selected = np.array(order, dtype=int)
    sub_adj = adjacency[np.ix_(selected, selected)]
    upper = np.triu_indices_from(sub_adj, k=1)
    weights_all = sub_adj[upper]
    positive = np.where(weights_all > 0)[0]
    # Keep only the union-top-15 subgraph. For dense scales this still reads as a compact graph.
    max_edges = min(len(positive), max(35, 3 * len(selected_nodes)))
    keep_local = positive[np.argsort(weights_all[positive])[-max_edges:]]
    edges = [
        (int(selected[upper[0][idx]]), int(selected[upper[1][idx]]), float(weights_all[idx]))
        for idx in keep_local
    ]
    weights = np.array([weight for _, _, weight in edges])
    norm = mpl.colors.Normalize(vmin=float(np.quantile(weights, 0.02)), vmax=float(np.quantile(weights, 0.995)))
    cmap = mpl.colormaps["jet"]
    segments = [[positions[i], positions[j]] for i, j, _ in edges]

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    edge_collection = LineCollection(
        segments,
        colors=cmap(norm(weights)),
        linewidths=0.18 + 1.25 * norm(weights),
        alpha=0.48,
        zorder=1,
    )
    ax.add_collection(edge_collection)

    global_positive = adjacency[np.triu_indices_from(adjacency, k=1)]
    threshold = np.quantile(global_positive[global_positive > 0], 0.92)
    binary_degree = (adjacency >= threshold).sum(axis=1)
    sizes = [36 + 9.0 * np.sqrt(binary_degree[i]) for i in order]
    xy = np.array([positions[i] for i in order])
    node_colors = [COLORS[categories[i]] for i in order]
    node_edges = ["#ffffff" for _ in order]
    node_lw = [0.9 for _ in order]
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=node_colors, edgecolors=node_edges, linewidths=node_lw, zorder=3)

    for node in order:
        x, y = positions[node]
        label = display_label(str(labels.loc[node, "region_label"]))
        ha = "left" if x >= 0 else "right"
        ax.text(1.12 * x, 1.12 * y, label, ha=ha, va="center", fontsize=6.3, color="#333333")

    ax.set_xlim(-1.22, 1.22)
    ax.set_ylim(-1.22, 1.22)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_y = -0.055
    legend_items = [("degree_only", "Degree only"), ("cycle_only", "Cycle only"), ("overlap", "Overlap")]
    for x, (cat, text) in zip([0.18, 0.43, 0.67], legend_items):
        ax.scatter([x], [legend_y], s=42, color=COLORS[cat], edgecolors="white", linewidths=0.6, transform=ax.transAxes, clip_on=False)
        ax.text(x + 0.035, legend_y, text, transform=ax.transAxes, ha="left", va="center", fontsize=8)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
    cbar.set_label("Weight", fontsize=9)
    cbar.ax.tick_params(labelsize=8, length=2)
    fig.savefig(FIGURES_DIR / f"rank_divergence_scale{scale}_graph.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_correlation_panel(scale: int) -> None:
    rank_table = mean_rank_table(scale)
    _, _, categories = selected_categories(rank_table)
    rho, p_value = spearmanr(rank_table["score_degree_strength"], rank_table["score_cycle_participation"])

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for cat in ["other", "degree_only", "cycle_only", "overlap"]:
        subset = rank_table[rank_table["node_index"].map(categories) == cat]
        ax.scatter(
            subset["score_degree_strength"],
            subset["score_cycle_participation"],
            s=24 if cat == "other" else 56,
            c=COLORS[cat],
            edgecolors="white" if cat != "other" else "none",
            linewidths=0.6,
            alpha=0.42 if cat == "other" else 0.92,
    )

    ax.set_xlabel("Degree strength", fontsize=11)
    ax.set_ylabel("Cycle participation", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, color="#e5e5e5", linewidth=0.7)
    ax.text(
        0.04,
        0.96,
        f"Spearman rho = {rho:.2f}\n" + f"p = {p_value:.1e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=3),
    )
    legend_items = [("other", "Other"), ("degree_only", "Degree only"), ("cycle_only", "Cycle only"), ("overlap", "Overlap")]
    for y, (cat, text) in zip([0.22, 0.16, 0.10, 0.04], legend_items):
        ax.scatter([0.72], [y], s=28 if cat == "other" else 44, color=COLORS[cat], edgecolors="white", linewidths=0.6, transform=ax.transAxes, clip_on=False)
        ax.text(0.76, y, text, transform=ax.transAxes, ha="left", va="center", fontsize=8)
    fig.savefig(FIGURES_DIR / f"rank_divergence_scale{scale}_correlation.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_gradient_panel(scale: int) -> None:
    """Relate rank divergence to the principal cortical (Margulies) gradient."""
    labels = pd.read_csv(RESULTS_DIR / "processed" / f"atlas_labels_scale-{scale}.csv")
    gradient = margulies_principal_gradient(labels, str(ATLAS_DIR))
    rank_table = mean_rank_table(scale).sort_values("node_index")
    divergence = (rank_table["degree_rank"] - rank_table["cycle_rank"]).to_numpy()
    degree_score = rank_table["score_degree_strength"].to_numpy()
    cycle_score = rank_table["score_cycle_participation"].to_numpy()

    rho_div, p_div = spearmanr(gradient, divergence)
    rho_deg, _ = spearmanr(gradient, degree_score)
    rho_cyc, _ = spearmanr(gradient, cycle_score)
    _, _, categories = selected_categories(rank_table)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for cat in ["other", "degree_only", "cycle_only", "overlap"]:
        mask = rank_table["node_index"].map(categories).to_numpy() == cat
        ax.scatter(
            gradient[mask],
            divergence[mask],
            s=24 if cat == "other" else 56,
            c=COLORS[cat],
            edgecolors="white" if cat != "other" else "none",
            linewidths=0.6,
            alpha=0.42 if cat == "other" else 0.92,
        )
    coeffs = np.polyfit(gradient, divergence, 1)
    xs = np.linspace(float(gradient.min()), float(gradient.max()), 100)
    ax.plot(xs, np.polyval(coeffs, xs), color="#333333", linewidth=1.2, linestyle="--", zorder=1)
    ax.axhline(0.0, color="#bdbdbd", linewidth=0.8, zorder=0)

    ax.set_xlabel("Principal gradient (unimodal → transmodal)", fontsize=11)
    ax.set_ylabel("Rank divergence (degree − cycle)", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, color="#e5e5e5", linewidth=0.7)
    ax.text(
        0.04,
        0.96,
        f"Divergence: rho = {rho_div:.2f}, p = {p_div:.1e}\n"
        f"Degree: rho = {rho_deg:.2f}\nCycle: rho = {rho_cyc:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=3),
    )
    fig.savefig(FIGURES_DIR / f"rank_divergence_scale{scale}_gradient.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    for scale in SCALES:
        save_rank_heatmap(scale)
        save_score_heatmaps(scale)
        save_graph_panel(scale)
        save_correlation_panel(scale)
        save_gradient_panel(scale)


if __name__ == "__main__":
    main()
