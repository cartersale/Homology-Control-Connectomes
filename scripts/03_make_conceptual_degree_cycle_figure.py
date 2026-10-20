"""Conceptual degree-vs-cycle figure on the Cammoun-68 lateral surface.

Usage: python scripts/03_make_conceptual_degree_cycle_figure.py [--run <name>]

Reads processed/connectomes_scale-68.npz from the run directory (``--run``,
default $HOC_RUN or "main"; ``--run results`` reads the legacy tree) and writes
figures/figure_concept_degree_vs_cycle.png under that run (derived figures for
the legacy tree go to runs/results-derived/).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import nibabel as nib
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation
from nilearn import datasets

from hoc.rankings import strength_scores
from hoc.topology import cycle_participation
from hoc.runs import figures_dir, reader_run_dir


RUN_DIR = reader_run_dir()
FIGURES_DIR = figures_dir(RUN_DIR)
OUTPUT_PATH = FIGURES_DIR / "figure_concept_degree_vs_cycle.png"
ATLAS_DIR = REPO_ROOT / "data/external/atlas_cache/atl-cammoun2012/fsaverage5"
CONNECTOME_PATH = RUN_DIR / "processed" / "connectomes_scale-68.npz"


EXCLUDED_LABELS = {"unknown", "corpuscallosum"}
HEMI_TO_PART = {"L": "left", "R": "right"}
HEMI_OFFSETS = {"L": 0, "R": 34}


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmax, vmin):
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def load_surface_and_nodes() -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Load pial meshes and parcel positions on the visible lateral surface."""
    fsaverage = datasets.load_fsaverage(mesh="fsaverage5")
    pial = fsaverage["pial"]

    mesh_parts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    parcel_positions: list[np.ndarray] = []
    for hemi in ("L", "R"):
        coords = np.asarray(pial.parts[HEMI_TO_PART[hemi]].coordinates, dtype=float)
        faces = np.asarray(pial.parts[HEMI_TO_PART[hemi]].faces, dtype=int)
        mesh_parts[hemi] = (coords, faces)

        annot_path = ATLAS_DIR / f"atl-Cammoun2012_space-fsaverage5_res-033_hemi-{hemi}_deterministic.annot"
        labels, _, names = nib.freesurfer.read_annot(str(annot_path))
        for label_index, raw_name in enumerate(names):
            name = raw_name.decode()
            if name in EXCLUDED_LABELS:
                continue
            parcel_vertices = coords[labels == label_index]
            lateral_index = int(np.argmin(parcel_vertices[:, 0]) if hemi == "L" else np.argmax(parcel_vertices[:, 0]))
            parcel_positions.append(parcel_vertices[lateral_index])
    return mesh_parts, np.asarray(parcel_positions, dtype=float)


def load_reference_connectome() -> np.ndarray:
    return np.asarray(np.load(CONNECTOME_PATH)["cohort_reference"], dtype=float)


def hemisphere_indices(hemi: str) -> np.ndarray:
    start = HEMI_OFFSETS[hemi]
    return np.arange(start, start + 34, dtype=int)


def compute_degree_edges(adjacency: np.ndarray, degree_scores: np.ndarray, hemi: str) -> tuple[list[tuple[int, int]], np.ndarray]:
    nodes = hemisphere_indices(hemi)
    sub_adj = adjacency[np.ix_(nodes, nodes)]
    sub_scores = degree_scores[nodes]
    hubs = nodes[np.argsort(-sub_scores)[:2]]

    selected: set[tuple[int, int]] = set()
    for hub in hubs:
        row = adjacency[hub, nodes]
        neighbors = nodes[np.argsort(-row)]
        kept = 0
        for neighbor in neighbors:
            if neighbor == hub or adjacency[hub, neighbor] <= 0:
                continue
            selected.add(tuple(sorted((int(hub), int(neighbor)))))
            kept += 1
            if kept == 5:
                break

    emphasis = np.zeros(adjacency.shape[0], dtype=float)
    emphasis[hubs] = degree_scores[hubs]
    for i, j in selected:
        emphasis[i] = max(emphasis[i], 0.35 * degree_scores[i])
        emphasis[j] = max(emphasis[j], 0.35 * degree_scores[j])
    return sorted(selected), emphasis


def cycle_edge_weight(adjacency: np.ndarray, cycle_nodes: list[int]) -> float:
    total = 0.0
    for idx, node in enumerate(cycle_nodes):
        nxt = cycle_nodes[(idx + 1) % len(cycle_nodes)]
        total += adjacency[node, nxt]
    return total / len(cycle_nodes)


def compute_cycle_structures(adjacency: np.ndarray, cycle_scores: np.ndarray, hemi: str) -> tuple[list[list[int]], np.ndarray]:
    nodes = hemisphere_indices(hemi)
    sub_adj = adjacency[np.ix_(nodes, nodes)]
    positive = sub_adj[sub_adj > 0]
    threshold = float(np.quantile(positive, 0.88))

    graph = nx.Graph()
    graph.add_nodes_from(nodes.tolist())
    for local_i, global_i in enumerate(nodes):
        for local_j, global_j in enumerate(nodes[local_i + 1 :], start=local_i + 1):
            weight = float(sub_adj[local_i, local_j])
            if weight >= threshold:
                graph.add_edge(int(global_i), int(global_j), weight=weight)

    candidates: list[tuple[float, list[int]]] = []
    for cycle_nodes in nx.cycle_basis(graph):
        if not 3 <= len(cycle_nodes) <= 5:
            continue
        score = float(cycle_scores[cycle_nodes].mean() + cycle_edge_weight(adjacency, cycle_nodes))
        candidates.append((score, cycle_nodes))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[list[int]] = []
    used: set[tuple[int, int]] = set()
    for _, cycle_nodes in candidates:
        cycle_edges = {
            tuple(sorted((cycle_nodes[idx], cycle_nodes[(idx + 1) % len(cycle_nodes)])))
            for idx in range(len(cycle_nodes))
        }
        if cycle_edges & used:
            continue
        selected.append(cycle_nodes)
        used |= cycle_edges
        if len(selected) == 2:
            break

    emphasis = np.zeros(adjacency.shape[0], dtype=float)
    for cycle_nodes in selected:
        emphasis[cycle_nodes] = np.maximum(emphasis[cycle_nodes], cycle_scores[cycle_nodes])
    return selected, emphasis


def setup_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_surface(ax: plt.Axes, coords: np.ndarray, faces: np.ndarray) -> None:
    yz = coords[:, 1:3]
    triangulation = Triangulation(yz[:, 0], yz[:, 1], triangles=faces)
    ax.tripcolor(
        triangulation,
        np.ones(coords.shape[0]),
        shading="flat",
        cmap=ListedColormap(["#eee6d9"]),
        vmin=0.0,
        vmax=1.0,
        edgecolors="none",
        zorder=0,
    )
    ax.triplot(triangulation, color="#ddd3c4", linewidth=0.10, alpha=0.12, zorder=1)


def draw_background_nodes(ax: plt.Axes, positions: np.ndarray, hemi: str) -> None:
    nodes = hemisphere_indices(hemi)
    yz = positions[nodes][:, 1:3]
    ax.scatter(
        yz[:, 0],
        yz[:, 1],
        s=26,
        facecolor="#fbf9f6",
        edgecolor="#9b978f",
        linewidth=0.9,
        zorder=3,
    )


def draw_degree_overlay(
    ax: plt.Axes,
    positions: np.ndarray,
    adjacency: np.ndarray,
    degree_edges: list[tuple[int, int]],
    emphasis: np.ndarray,
) -> None:
    max_weight = max(float(adjacency[i, j]) for i, j in degree_edges)
    for i, j in degree_edges:
        pi = positions[i, 1:3]
        pj = positions[j, 1:3]
        linewidth = 1.2 + 3.0 * float(adjacency[i, j]) / max_weight
        ax.plot([pi[0], pj[0]], [pi[1], pj[1]], color="#d8892d", lw=linewidth, alpha=0.95, zorder=4)

    mask = emphasis > 0
    scale = 38 + 360 * normalize(emphasis)
    yz = positions[mask][:, 1:3]
    ax.scatter(yz[:, 0], yz[:, 1], s=scale[mask], color="#23485a", edgecolors="none", zorder=5)


def draw_cycle_overlay(ax: plt.Axes, positions: np.ndarray, cycles: list[list[int]], emphasis: np.ndarray) -> None:
    for cycle_nodes in cycles:
        yz = positions[cycle_nodes][:, 1:3]
        ax.add_patch(Polygon(yz, closed=True, facecolor="#76b3ae", edgecolor="none", alpha=0.22, zorder=4))
        closed = cycle_nodes + [cycle_nodes[0]]
        yz_closed = positions[closed][:, 1:3]
        ax.plot(yz_closed[:, 0], yz_closed[:, 1], color="#2d9f93", lw=3.0, alpha=0.98, zorder=5)

    mask = emphasis > 0
    scale = 38 + 320 * normalize(emphasis)
    yz = positions[mask][:, 1:3]
    ax.scatter(yz[:, 0], yz[:, 1], s=scale[mask], color="#23485a", edgecolors="none", zorder=6)


def finish_axis_limits(ax: plt.Axes, coords: np.ndarray) -> None:
    yz = coords[:, 1:3]
    ax.set_xlim(yz[:, 0].min() - 6.0, yz[:, 0].max() + 6.0)
    ax.set_ylim(yz[:, 1].min() - 6.0, yz[:, 1].max() + 6.0)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    mesh_parts, positions = load_surface_and_nodes()
    adjacency = load_reference_connectome()
    degree_scores = normalize(strength_scores(adjacency))
    cycle_scores, _ = cycle_participation(adjacency)

    degree_left, degree_emph_left = compute_degree_edges(adjacency, degree_scores, "L")
    degree_right, degree_emph_right = compute_degree_edges(adjacency, degree_scores, "R")
    cycle_left, cycle_emph_left = compute_cycle_structures(adjacency, cycle_scores, "L")
    cycle_right, cycle_emph_right = compute_cycle_structures(adjacency, cycle_scores, "R")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.8), constrained_layout=True)
    setup_axis(axes[0, 0], "Degree (L)")
    setup_axis(axes[0, 1], "Degree (R)")
    setup_axis(axes[1, 0], "Cycle (L)")
    setup_axis(axes[1, 1], "Cycle (R)")

    coords_l, faces_l = mesh_parts["L"]
    coords_r, faces_r = mesh_parts["R"]

    draw_surface(axes[0, 0], coords_l, faces_l)
    draw_background_nodes(axes[0, 0], positions, "L")
    draw_degree_overlay(axes[0, 0], positions, adjacency, degree_left, degree_emph_left)
    finish_axis_limits(axes[0, 0], coords_l)

    draw_surface(axes[0, 1], coords_r, faces_r)
    draw_background_nodes(axes[0, 1], positions, "R")
    draw_degree_overlay(axes[0, 1], positions, adjacency, degree_right, degree_emph_right)
    finish_axis_limits(axes[0, 1], coords_r)

    draw_surface(axes[1, 0], coords_l, faces_l)
    draw_background_nodes(axes[1, 0], positions, "L")
    draw_cycle_overlay(axes[1, 0], positions, cycle_left, cycle_emph_left)
    finish_axis_limits(axes[1, 0], coords_l)

    draw_surface(axes[1, 1], coords_r, faces_r)
    draw_background_nodes(axes[1, 1], positions, "R")
    draw_cycle_overlay(axes[1, 1], positions, cycle_right, cycle_emph_right)
    finish_axis_limits(axes[1, 1], coords_r)

    fig.patch.set_facecolor("white")
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
