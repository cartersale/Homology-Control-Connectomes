#!/usr/bin/env python
"""Export rasterized SVG brain-region maps for named target states."""

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
from hoc.dataset import load_raw_connectomes
from hoc.preprocessing import preprocess_structural_adjacency
from hoc.targets import (
    DISTRIBUTED_TARGETS,
    LANDMARK_DEFINITIONS,
    MODULE_TARGETS,
    build_targets,
)


NAMED_TARGETS = tuple(LANDMARK_DEFINITIONS) + tuple(MODULE_TARGETS) + tuple(DISTRIBUTED_TARGETS)
DEFAULT_SCALES = (68, 114, 219)

UNSIGNED_CMAP = LinearSegmentedColormap.from_list(
    "target_region_warm",
    ["#fff4bc", "#fdae61", "#d7301f"],
)
SIGNED_CMAP = LinearSegmentedColormap.from_list(
    "target_region_signed",
    ["#2166ac", "#67a9cf", "#f7f7f7", "#fdae61", "#d7301f"],
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml", help="Path to the TOML config file.")
    parser.add_argument(
        "--scale",
        default="219",
        help="Parcellation scale to export: 68, 114, 219, or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to figures/section_target_state_selection/target_region_maps_scale-{scale}.",
    )
    return parser.parse_args()


def requested_scales(scale_text: str) -> list[int]:
    """Resolve the scale toggle into concrete atlas scales."""
    if scale_text.lower() == "all":
        return list(DEFAULT_SCALES)
    scale = int(scale_text)
    if scale not in DEFAULT_SCALES:
        raise ValueError(f"Unsupported scale {scale}. Expected one of {DEFAULT_SCALES} or 'all'.")
    return [scale]


def reference_adjacency(raw_connectomes: dict[int, np.ndarray], scale: int) -> np.ndarray:
    """Build the group-reference adjacency used by target construction."""
    raw_stack = raw_connectomes[scale]
    processed = [
        preprocess_structural_adjacency(raw_stack[:, :, subject_index])
        for subject_index in range(raw_stack.shape[2])
    ]
    return np.mean(np.stack(processed, axis=0), axis=0)


def target_vectors_for_scale(config: dict, scale: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load or rebuild named target vectors for one atlas scale."""
    labels = build_label_table(scale=scale, atlas_dir=config["paths"]["atlas_dir"])
    target_path = REPO_ROOT / config["paths"]["results_dir"] / "processed" / f"targets_scale-{scale}.npz"
    if target_path.exists():
        archive = np.load(target_path)
        if all(target_name in archive.files for target_name in NAMED_TARGETS):
            return labels, {target_name: np.asarray(archive[target_name], dtype=float) for target_name in NAMED_TARGETS}

    raw_connectomes = load_raw_connectomes(config["paths"]["raw_connectomes"])
    targets = build_targets(
        reference_adjacency(raw_connectomes, scale),
        labels,
        config["targets"],
        config["anatomy"],
        config["paths"]["atlas_dir"],
    )
    return labels, {target_name: targets[target_name] for target_name in NAMED_TARGETS}


def load_surface_geometry(scale: int, atlas_dir: str | Path):
    """Load fsaverage geometry and Cammoun annotation mapping for dorsal maps."""
    fsaverage = datasets.load_fsaverage(mesh="fsaverage5")
    pial = fsaverage["pial"]
    atlas_path = REPO_ROOT / atlas_dir
    resolution = SCALE_TO_RESOLUTION[scale]
    geometry = {}
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
            vertex_to_node[vertex_labels == atlas_index] = node_index
            node_index += 1
        geometry[hemi] = {"coords": coords, "faces": faces, "vertex_to_node": vertex_to_node}
    return geometry


def draw_hemi(
    ax: plt.Axes,
    coords: np.ndarray,
    faces: np.ndarray,
    vertex_values: np.ndarray,
    *,
    signed: bool,
    vmax: float,
) -> None:
    """Draw one rasterized dorsal hemisphere with highlighted target regions."""
    polygons = coords[:, [0, 1]][faces]
    base = PolyCollection(
        polygons,
        facecolors="#ededed",
        edgecolors="#9b9b9b",
        linewidths=0.05,
        rasterized=True,
        zorder=1,
    )
    ax.add_collection(base)

    face_values = vertex_values[faces].mean(axis=1)
    highlight = np.abs(face_values) > 1.0e-12
    if np.any(highlight):
        if signed:
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            colors = SIGNED_CMAP(norm(face_values[highlight]))
        else:
            norm = Normalize(vmin=0.0, vmax=vmax)
            colors = UNSIGNED_CMAP(norm(face_values[highlight]))
        overlay = PolyCollection(
            polygons[highlight],
            facecolors=colors,
            edgecolors="#ffffff",
            linewidths=0.06,
            rasterized=True,
            zorder=2,
        )
        ax.add_collection(overlay)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")


def save_target_map(
    *,
    scale: int,
    target_name: str,
    node_values: np.ndarray,
    geometry: dict,
    output_dir: Path,
) -> Path:
    """Save one no-label target map as an SVG with rasterized brain layers."""
    signed = bool(np.nanmin(node_values) < 0.0)
    vmax = float(np.nanmax(np.abs(node_values))) if signed else float(np.nanmax(node_values))
    if vmax <= 0.0:
        raise ValueError(f"Target {target_name} at scale {scale} has no nonzero values.")

    fig, ax = plt.subplots(figsize=(5.8, 4.4), facecolor="white")
    for hemi in ("L", "R"):
        geom = geometry[hemi]
        vertex_values = np.zeros(geom["vertex_to_node"].shape, dtype=float)
        valid = geom["vertex_to_node"] >= 0
        vertex_values[valid] = node_values[geom["vertex_to_node"][valid]]
        draw_hemi(
            ax,
            geom["coords"],
            geom["faces"],
            vertex_values,
            signed=signed,
            vmax=vmax,
        )

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_span = x_max - x_min
    y_span = y_max - y_min
    ax.set_xlim(x_min - 0.04 * x_span, x_max + 0.04 * x_span)
    ax.set_ylim(y_min - 0.04 * y_span, y_max + 0.04 * y_span)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"target_{target_name}_scale-{scale}.svg"
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=220)
    plt.close(fig)
    return out


def export_target_maps(config: dict, scale: int, output_dir: Path | None = None) -> list[Path]:
    """Export all named target-region maps for one scale."""
    mpl.rcParams.update({"svg.fonttype": "none", "axes.unicode_minus": False})
    if output_dir is None:
        output_dir = (
            REPO_ROOT
            / config["paths"]["figures_dir"]
            / "section_target_state_selection"
            / f"target_region_maps_scale-{scale}"
        )
    labels, targets = target_vectors_for_scale(config, scale)
    geometry = load_surface_geometry(scale, config["paths"]["atlas_dir"])
    if len(labels) != next(iter(targets.values())).shape[0]:
        raise ValueError(f"Label/target length mismatch at scale {scale}.")
    return [
        save_target_map(
            scale=scale,
            target_name=target_name,
            node_values=targets[target_name],
            geometry=geometry,
            output_dir=output_dir,
        )
        for target_name in NAMED_TARGETS
    ]


def main() -> None:
    """Run target-region map export."""
    args = parse_args()
    config = load_config(REPO_ROOT / args.config)
    all_outputs = []
    for scale in requested_scales(args.scale):
        output_dir = Path(args.output_dir) if args.output_dir is not None else None
        outputs = export_target_maps(config, scale, output_dir=output_dir)
        all_outputs.extend(outputs)
    for path in all_outputs:
        print(path)


if __name__ == "__main__":
    main()
