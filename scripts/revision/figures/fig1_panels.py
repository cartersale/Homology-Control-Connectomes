"""Figure 1 panels: where inputs enter shapes what they can reach.

Each panel is written as its own SVG with minimal text; labels and lettering
are left to the caption / layout step.

A: the control model. B: hub and spread-out driver sets on the leading
eigenvector v1. C: drivers' loading on v1 and the resulting concentration q1. Brain surfaces are rasterized inside the
SVG so the files stay small, everything else is vector.

Data: cohort-reference connectome at 114 regions, c = 0.1, k = 3 drivers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import networkx as nx
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from nilearn import datasets
from scipy.spatial import Delaunay

from hoc.preprocessing import stabilize_adjacency
from hoc.rankings import strength_scores
from hoc.topology import cycle_participation

SCALE = 114
C = 0.1
K = 3
OUT_DIR = Path("paper/network_neuro_resubmission/figs/fig1")
CONNECTOME_PATH = Path(f"results/processed/connectomes_scale-{SCALE}.npz")
ATLAS_DIR = Path("data/external/atlas_cache/atl-cammoun2012/fsaverage5")
RESOLUTION = {68: "033", 114: "060", 219: "125"}[SCALE]
EXCLUDED_LABELS = {"unknown", "corpuscallosum"}

HUB = "#eb6834"
SPREAD = "#2a78d6"
INK = "#2b2b2b"
MUTED = "#8a8a85"
NODE = "#d9d8d3"
CORTEX = "#ecebe7"
V1_CMAP = LinearSegmentedColormap.from_list("v1", ["#f3f2ee", "#cfcdc6", "#8c8a83", "#3a3a37"])

mpl.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.8,
        "svg.fonttype": "path",
        "savefig.transparent": True,
    }
)


# ----------------------------------------------------------------------------- data


def load_data():
    a0 = np.asarray(np.load(CONNECTOME_PATH)["cohort_reference"], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(a0)
    v1 = eigvecs[:, -1] * np.sign(eigvecs[:, -1].sum())
    strength = strength_scores(a0)
    cycle, _ = cycle_participation(a0)
    hub_set = np.argsort(-strength)[:K]
    spread_set = np.argsort(-cycle)[:K]
    return a0, eigvals, eigvecs, v1, hub_set, spread_set


def gramian(a0: np.ndarray, drivers: np.ndarray) -> np.ndarray:
    """Infinite-horizon Gramian in the eigenbasis of the stabilized system."""
    system, _ = stabilize_adjacency(a0, C)
    lam, u = np.linalg.eigh(system)
    b = np.zeros((a0.shape[0], len(drivers)))
    b[drivers, np.arange(len(drivers))] = 1.0
    ub = u.T @ b
    w_modal = (ub @ ub.T) / -(lam[:, None] + lam[None, :])
    return u @ w_modal @ u.T


def load_surface():
    fsaverage = datasets.load_fsaverage(mesh="fsaverage5")
    meshes, labels_by_hemi = {}, {}
    for hemi, part in (("L", "left"), ("R", "right")):
        coords = np.asarray(fsaverage["pial"].parts[part].coordinates, dtype=float)
        faces = np.asarray(fsaverage["pial"].parts[part].faces, dtype=int)
        meshes[hemi] = (coords, faces)
        annot = ATLAS_DIR / f"atl-Cammoun2012_space-fsaverage5_res-{RESOLUTION}_hemi-{hemi}_deterministic.annot"
        vertex_labels, _, names = nib.freesurfer.read_annot(str(annot))
        node_of_vertex = np.full(vertex_labels.shape, -1)
        offset = 0 if hemi == "L" else int(labels_by_hemi["L"].max()) + 1
        kept = 0
        for label_index, raw in enumerate(names):
            if raw.decode() in EXCLUDED_LABELS:
                continue
            mask = vertex_labels == label_index
            node_of_vertex[mask] = offset + kept
            kept += 1
        labels_by_hemi[hemi] = node_of_vertex
    return meshes, labels_by_hemi


def surface_anchor(meshes, labels_by_hemi, node, view):
    """Screen position of a parcel's most visible vertex in the given view."""
    for hemi, (coords, _) in meshes.items():
        mask = labels_by_hemi[hemi] == node
        if mask.any():
            proj = coords[mask] @ VIEWS[view].T
            return proj[np.argmax(proj[:, 2]), :2]
    raise ValueError(node)


# ------------------------------------------------------------------------ rendering

VIEWS = {
    # (rotation matrix rows = screen x, screen y, depth toward viewer)
    "dorsal": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
    "lateral_L": np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float),
    "medial_L": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float),
}
LIGHT = np.array([-0.3, 0.4, 1.0]) / np.linalg.norm([-0.3, 0.4, 1.0])


def draw_mesh(ax, coords, faces, vertex_rgb, view, alpha=1.0):
    proj = coords @ VIEWS[view].T
    tri = proj[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    facing = normals[:, 2] > 0
    shade = 0.78 + 0.22 * np.clip(normals @ LIGHT, 0, 1)
    rgb = vertex_rgb[faces].mean(axis=1) * shade[:, None]
    order = np.argsort(tri[:, :, 2].mean(axis=1))
    order = order[facing[order]]
    collection = PolyCollection(
        tri[order][:, :, :2],
        facecolors=np.column_stack([np.clip(rgb[order], 0, 1), np.full(order.size, alpha)]),
        edgecolors="none",
        antialiased=False,
        rasterized=True,
    )
    ax.add_collection(collection)
    return proj


def finish_brain_axis(ax, points):
    pad = 4
    ax.set_xlim(points[:, 0].min() - pad, points[:, 0].max() + pad)
    ax.set_ylim(points[:, 1].min() - pad, points[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.svg", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


# ---------------------------------------------------------------------------- panels


def panel_model():
    """Schematic: a few inputs u(t) drive a network x' = Ax + Bu."""
    rng = np.random.default_rng(3)
    points = rng.uniform(-1, 1, (60, 2))
    points = points[np.linalg.norm(points, axis=1) < 1]
    keep = [0]
    for i in range(1, len(points)):
        if np.min(np.linalg.norm(points[keep] - points[i], axis=1)) > 0.42:
            keep.append(i)
    points = points[keep]
    graph = nx.Graph()
    for simplex in Delaunay(points).simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            if np.linalg.norm(points[simplex[a]] - points[simplex[b]]) < 0.8:
                graph.add_edge(int(simplex[a]), int(simplex[b]))
    pos = {n: points[n] for n in graph.nodes}
    drivers = [int(np.argmin(points[:, 0])), int(np.argmax(points[:, 1]))]
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    for i, j in graph.edges:
        ax.plot(*zip(pos[i], pos[j]), color=MUTED, lw=0.9 + 0.9 * rng.random(), alpha=0.55, zorder=1,
                solid_capstyle="round")
    for n in graph.nodes:
        ax.scatter(*pos[n], s=110, color=INK if n in drivers else NODE, edgecolor="white", lw=1.2, zorder=3)

    for d in drivers:
        x, y = pos[d]
        direction = np.array([x, y]) - np.mean(list(pos.values()), axis=0)
        direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0.1 else np.array([0, 1.0])
        start = np.array([x, y]) + 0.42 * direction
        ax.add_patch(
            FancyArrowPatch(start, np.array([x, y]) + 0.09 * direction, arrowstyle="-|>", mutation_scale=10,
                            color=INK, lw=1.2, zorder=2)
        )
        t = np.linspace(0, 1, 60)
        normal = np.array([-direction[1], direction[0]])
        wave = start[None, :] + 0.16 * t[:, None] * direction[None, :] + 0.045 * np.sin(2 * np.pi * 2 * t)[:, None] * normal
        ax.plot(wave[:, 0] + 0.07 * direction[0], wave[:, 1] + 0.07 * direction[1], color=INK, lw=1.1)

    ax.text(0.5, 1.02, r"$\dot{x} = Ax + Bu$", transform=ax.transAxes, ha="center", va="bottom", color=INK,
            fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.18)
    save(fig, "fig1a_model")


def panel_v1_drivers(meshes, labels_by_hemi, v1, drivers, color, name, view="dorsal"):
    """Leading eigenvector v1 on the cortex (grey = low, dark = high) with one driver set on top."""
    norm = mpl.colors.PowerNorm(0.7, 0, float(v1.max()))
    fig, ax = plt.subplots(figsize=(2.2, 2.7))
    points = []
    for hemi in ("L", "R"):
        coords, faces = meshes[hemi]
        node = labels_by_hemi[hemi]
        rgb = V1_CMAP(norm(np.where(node >= 0, v1[np.clip(node, 0, None)], 0.0)))[:, :3]
        rgb[node < 0] = mpl.colors.to_rgb("#dcdbd6")
        points.append(draw_mesh(ax, coords, faces, rgb, view))
    xy = np.array([surface_anchor(meshes, labels_by_hemi, d, view) for d in drivers])
    ax.scatter(xy[:, 0], xy[:, 1], s=90, color=color, edgecolor="white", lw=1.6, zorder=4)
    finish_brain_axis(ax, np.vstack(points))
    save(fig, name)


def panel_v1_colorbar(v1):
    fig, ax = plt.subplots(figsize=(1.3, 0.3))
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.PowerNorm(0.7, 0, float(v1.max())), cmap=V1_CMAP), cax=ax,
                 orientation="horizontal")
    ax.set_xticks([])
    ax.set_xlabel(r"$v_1$", labelpad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, "fig1b_v1_colorbar")


def panel_loading_q1(a0, v1, hub_set, spread_set, n_random=1000, seed=0):
    """Loading on v1 (sum of squared v1 entries over drivers) and q1 for each set; grey tick = random mean."""
    rng = np.random.default_rng(seed)
    random_sets = [rng.choice(a0.shape[0], K, replace=False) for _ in range(n_random)]

    def loading(s):
        return float(np.sum(v1[s] ** 2))

    def q1(s):
        eig = np.linalg.eigvalsh(gramian(a0, s))
        return float(eig[-1] / eig.sum())

    metrics = [
        (loading(hub_set), loading(spread_set), np.mean([loading(s) for s in random_sets])),
        (q1(hub_set), q1(spread_set), np.mean([q1(s) for s in random_sets])),
    ]
    for label, (hub, spread, rand) in zip(("loading on v1", "q1"), metrics):
        print(f"{label}: hub {hub:.2f}, spread {spread:.2f}, random mean {rand:.2f}")

    fig, ax = plt.subplots(figsize=(1.9, 2.0))
    width = 0.32
    for g, (hub, spread, rand) in enumerate(metrics):
        ax.bar(g - width / 2 - 0.02, hub, width=width, color=HUB, edgecolor="none")
        ax.bar(g + width / 2 + 0.02, spread, width=width, color=SPREAD, edgecolor="none")
        ax.plot([g - width - 0.08, g + width + 0.08], [rand, rand], color=INK, lw=1.0, ls=(0, (2, 1.5)))
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$v_1$ loading", r"$q_1$"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=2.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, "fig1c_loading_q1")


def main():
    a0, _, _, v1, hub_set, spread_set = load_data()
    meshes, labels_by_hemi = load_surface()

    panel_model()
    panel_v1_drivers(meshes, labels_by_hemi, v1, hub_set, HUB, "fig1b_v1_hub")
    panel_v1_drivers(meshes, labels_by_hemi, v1, spread_set, SPREAD, "fig1b_v1_spread")
    panel_v1_colorbar(v1)
    panel_loading_q1(a0, v1, hub_set, spread_set)


if __name__ == "__main__":
    main()
