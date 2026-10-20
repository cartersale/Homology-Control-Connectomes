#!/usr/bin/env python
"""Build real-data reachable-state cloud assets for control-geometry figures/animation.

Usage: python scripts/08_make_reachable_state_cloud_assets.py [--run <name>] [--scale 114] ...

Reads primary/ tables from the run directory (``--run``, default $HOC_RUN or
"main"; ``--run results`` reads the legacy tree) and writes assets to
<run>/figures/animation_assets/ (runs/results-derived/... for the legacy tree).
"""

from __future__ import annotations

import argparse
import json
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

from hoc.atlas import build_label_table
from hoc.config import load_config
from hoc.control import assemble_gramian, gramian_metrics, single_node_gramians
from hoc.dataset import load_raw_connectomes
from hoc.preprocessing import preprocess_structural_adjacency, stabilize_adjacency
from hoc.runs import config_variant, resolve_run_dir
from hoc.runs import figures_dir as run_figures_dir
from hoc.targets import build_targets


METHODS = ("degree_strength", "cycle_participation")
TARGETS_TO_SHOW = (
    "association_module",
    "principal_gradient",
    "landmark_lpfc",
)
COLORS = {
    "degree_strength": "#e07a2d",
    "cycle_participation": "#2b78b8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--scale", type=int, default=114)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--selection-strategy", default="static")
    parser.add_argument(
        "--subject-id",
        default="auto",
        help="Subject identifier like sub-014, or 'auto' for a representative subject.",
    )
    parser.add_argument("--n-samples", type=int, default=1400)
    parser.add_argument(
        "--run",
        default=None,
        help="Run name under runs/ (default: $HOC_RUN or 'main'; 'results' = legacy tree, read-only).",
    )
    return parser.parse_args()


def parse_node_indices(text: str) -> np.ndarray:
    return np.asarray([int(token) for token in str(text).split(";") if token], dtype=int)


def choose_representative_subject(
    geometry: pd.DataFrame,
    scale: int,
    k: int,
    selection_strategy: str,
) -> str:
    subset = geometry[
        (geometry["scale"] == scale)
        & (geometry["k"] == k)
        & (geometry["selection_strategy"] == selection_strategy)
        & (geometry["method"].isin(METHODS))
    ].copy()
    wide = subset.pivot(
        index="subject_id",
        columns="method",
        values=[
            "log10_average_energy",
            "effective_rank",
            "participation_ratio",
            "log10_condition_number",
        ],
    )
    wide.columns = ["__".join(parts) for parts in wide.columns]
    wide = wide.reset_index()
    metric_columns = [column for column in wide.columns if column != "subject_id"]
    metric_mean = wide[metric_columns].mean(axis=0)
    wide["distance_to_mean"] = ((wide[metric_columns] - metric_mean) ** 2).sum(axis=1)
    return str(wide.nsmallest(1, "distance_to_mean").iloc[0]["subject_id"])


def disk_samples(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    radius = np.sqrt(rng.random(n_samples))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def ellipse_points(gramian_2d: np.ndarray, n_points: int = 400) -> np.ndarray:
    evals, evecs = np.linalg.eigh(gramian_2d)
    transform = evecs @ np.diag(np.sqrt(np.clip(evals, a_min=1.0e-12, a_max=None)))
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    return circle @ transform.T


def projected_cloud(gramian_2d: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    evals, evecs = np.linalg.eigh(gramian_2d)
    transform = evecs @ np.diag(np.sqrt(np.clip(evals, a_min=1.0e-12, a_max=None)))
    return disk_samples(n_samples, rng) @ transform.T


def normalize_projected_gramian(gramian_2d: np.ndarray) -> np.ndarray:
    """Match projected area so the visual emphasizes anisotropy rather than size.

    For a 2D ellipse induced by W, area is proportional to sqrt(det(W)).
    Dividing by sqrt(det(W)) gives unit determinant, so both methods share
    the same projected area while preserving axis ratios and orientation.
    """
    determinant = float(np.linalg.det(gramian_2d))
    scale = np.sqrt(max(determinant, 1.0e-12))
    return gramian_2d / scale


def main() -> None:
    args = parse_args()
    config = load_config(REPO_ROOT / args.config)

    run_dir = resolve_run_dir(args.run, variant=config_variant(config))
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist; pass --run <name> or set $HOC_RUN.")
    figures_dir = run_figures_dir(run_dir, "animation_assets")
    results_dir = run_dir / "primary"


    geometry = pd.read_csv(results_dir / "geometry_metrics.csv")
    driver_sets = pd.read_csv(results_dir / "driver_sets.csv")
    target_energies = pd.read_csv(results_dir / "target_energies.csv")

    subject_id = (
        choose_representative_subject(
            geometry=geometry,
            scale=args.scale,
            k=args.k,
            selection_strategy=args.selection_strategy,
        )
        if args.subject_id == "auto"
        else args.subject_id
    )
    subject_index = int(subject_id.split("-")[1])

    raw_connectomes = load_raw_connectomes(config["paths"]["raw_connectomes"])
    adjacency = preprocess_structural_adjacency(raw_connectomes[args.scale][:, :, subject_index])
    reference = np.mean(
        np.stack(
            [
                preprocess_structural_adjacency(raw_connectomes[args.scale][:, :, idx])
                for idx in range(raw_connectomes[args.scale].shape[2])
            ],
            axis=0,
        ),
        axis=0,
    )
    labels = build_label_table(scale=args.scale, atlas_dir=config["paths"]["atlas_dir"])
    targets = build_targets(reference, labels, config["targets"], config["anatomy"], config["paths"]["atlas_dir"])

    system_matrix, spectral_radius = stabilize_adjacency(adjacency, float(config["control"]["primary_c"]))
    node_gramians = single_node_gramians(system_matrix)
    epsilon = float(config["control"]["primary_epsilon"])

    method_sets: dict[str, np.ndarray] = {}
    method_metrics: dict[str, dict] = {}
    for method in METHODS:
        row = driver_sets[
            (driver_sets["subject_id"] == subject_id)
            & (driver_sets["scale"] == args.scale)
            & (driver_sets["k"] == args.k)
            & (driver_sets["method"] == method)
            & (driver_sets["selection_strategy"] == args.selection_strategy)
        ]
        if row.empty:
            raise ValueError(f"No driver set found for {subject_id}, scale={args.scale}, k={args.k}, method={method}")
        node_indices = parse_node_indices(row.iloc[0]["node_indices"])
        gramian = assemble_gramian(node_gramians, node_indices)
        metrics = gramian_metrics(
            gramian,
            epsilon=epsilon,
            bottom_dim=int(config["control"]["bottom_eigenspace_dim"]),
        )
        method_sets[method] = node_indices
        method_metrics[method] = metrics

    average_regularized = 0.5 * (
        method_metrics["degree_strength"]["regularized_gramian"]
        + method_metrics["cycle_participation"]["regularized_gramian"]
    )
    proj_evals, proj_evecs = np.linalg.eigh(average_regularized)
    projection_basis = proj_evecs[:, -2:]

    rng = np.random.default_rng(20260510)
    output_stem = (
        f"reachable_state_cloud_scale-{args.scale}_k-{args.k}_{args.selection_strategy}_{subject_id}"
    )

    asset_payload: dict[str, np.ndarray] = {
        "projection_basis": projection_basis,
    }
    metadata: dict[str, object] = {
        "subject_id": subject_id,
        "subject_index": subject_index,
        "scale": args.scale,
        "k": args.k,
        "selection_strategy": args.selection_strategy,
        "primary_c": float(config["control"]["primary_c"]),
        "primary_epsilon": epsilon,
        "spectral_radius": spectral_radius,
        "methods": {},
        "targets": {},
    }

    for method in METHODS:
        regularized = method_metrics[method]["regularized_gramian"]
        projected = projection_basis.T @ regularized @ projection_basis
        boundary = ellipse_points(projected)
        cloud = projected_cloud(projected, n_samples=args.n_samples, rng=rng)
        projected_normalized = normalize_projected_gramian(projected)
        boundary_normalized = ellipse_points(projected_normalized)
        cloud_normalized = projected_cloud(projected_normalized, n_samples=args.n_samples, rng=rng)
        asset_payload[f"{method}_boundary"] = boundary
        asset_payload[f"{method}_cloud"] = cloud
        asset_payload[f"{method}_projected_gramian"] = projected
        asset_payload[f"{method}_boundary_normalized"] = boundary_normalized
        asset_payload[f"{method}_cloud_normalized"] = cloud_normalized
        asset_payload[f"{method}_projected_gramian_normalized"] = projected_normalized
        metadata["methods"][method] = {
            "node_indices": method_sets[method].tolist(),
            "region_labels": labels.loc[method_sets[method], "region_label"].tolist(),
            "log10_average_energy": method_metrics[method]["log10_average_energy"],
            "effective_rank": method_metrics[method]["effective_rank"],
            "participation_ratio": method_metrics[method]["participation_ratio"],
            "log10_condition_number": method_metrics[method]["log10_condition_number"],
            "projected_area_scale": float(np.sqrt(max(np.linalg.det(projected), 1.0e-12))),
        }

    target_subset = target_energies[
        (target_energies["subject_id"] == subject_id)
        & (target_energies["scale"] == args.scale)
        & (target_energies["k"] == args.k)
        & (target_energies["selection_strategy"] == args.selection_strategy)
        & (target_energies["method"].isin(METHODS))
        & (target_energies["target_name"].isin(TARGETS_TO_SHOW))
    ].copy()
    target_summary = target_subset.pivot(index="target_name", columns="method", values="target_energy")

    for target_name in TARGETS_TO_SHOW:
        vector = targets[target_name]
        coords = projection_basis.T @ vector
        asset_payload[f"target_{target_name}"] = coords
        degree_energy = float(target_summary.loc[target_name, "degree_strength"])
        cycle_energy = float(target_summary.loc[target_name, "cycle_participation"])
        metadata["targets"][target_name] = {
            "projection": coords.tolist(),
            "degree_energy": degree_energy,
            "cycle_energy": cycle_energy,
            "cycle_minus_degree": cycle_energy - degree_energy,
        }

    np.savez(figures_dir / f"{output_stem}.npz", **asset_payload)
    with open(figures_dir / f"{output_stem}.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.8))
    preview_specs = [
        ("raw", "Raw projected cloud"),
        ("normalized", "Area-matched geometry cloud"),
    ]
    for ax, (mode, title) in zip(axes, preview_specs):
        suffix = "" if mode == "raw" else "_normalized"
        for method in METHODS:
            cloud = asset_payload[f"{method}_cloud{suffix}"]
            boundary = asset_payload[f"{method}_boundary{suffix}"]
            ax.scatter(cloud[:, 0], cloud[:, 1], s=6, alpha=0.12, color=COLORS[method], linewidths=0)
            ax.fill(boundary[:, 0], boundary[:, 1], color=COLORS[method], alpha=0.18)
            ax.plot(boundary[:, 0], boundary[:, 1], color=COLORS[method], linewidth=2.2, label=method.replace("_", " "))

        for target_name in TARGETS_TO_SHOW:
            target_coords = asset_payload[f"target_{target_name}"]
            direction = target_coords / max(np.linalg.norm(target_coords), 1.0e-12)
            ax.arrow(
                0.0,
                0.0,
                0.9 * direction[0],
                0.9 * direction[1],
                width=0.008,
                head_width=0.05,
                head_length=0.08,
                length_includes_head=True,
                color="#333333",
                alpha=0.85,
                zorder=5,
            )
            ax.text(
                1.02 * direction[0],
                1.02 * direction[1],
                target_name.replace("_", " "),
                fontsize=9,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
            )

        ax.axhline(0.0, color="#cccccc", linewidth=0.8)
        ax.axvline(0.0, color="#cccccc", linewidth=0.8)
        ax.set_xlabel("Projected controllable dimension 1")
        ax.set_ylabel("Projected controllable dimension 2")
        ax.set_title(title)
        ax.set_aspect("equal")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle(f"Reachable-state cloud | scale {args.scale}, k={args.k}, {subject_id}", y=0.98)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{output_stem}_preview.svg", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"output_stem": output_stem, "subject_id": subject_id, "directory": str(figures_dir)}, indent=2))


if __name__ == "__main__":
    main()
