#!/usr/bin/env python
"""Evaluate driver-set geometry and target energy after deleting degree hubs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hoc.atlas import build_label_table
from hoc.config import load_config
from hoc.control import assemble_gramian, gramian_metrics, single_node_gramians, target_energy
from hoc.dataset import load_raw_connectomes
from hoc.dataset import build_subject_table
from hoc.runs import RunContext, add_run_arguments
from hoc.io import prepare_results_tree, write_csv
from hoc.preprocessing import preprocess_structural_adjacency, stabilize_adjacency
from hoc.rankings import compute_node_scores, top_k_excluding_nodes, top_k_nodes
from hoc.targets import build_targets, normalize_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml", help="Path to the TOML config file.")
    parser.add_argument("--subject-limit", type=int, default=None, help="Optional cap for smoke tests.")
    add_run_arguments(parser)
    return parser.parse_args()


def load_processed_stack(raw_connectomes: dict[int, np.ndarray], scale: int, n_subjects: int) -> np.ndarray:
    """Preprocess one scale into a subject-major stack."""
    raw_stack = raw_connectomes[scale][:, :, :n_subjects]
    return np.stack(
        [preprocess_structural_adjacency(raw_stack[:, :, subject_index]) for subject_index in range(n_subjects)],
        axis=0,
    )


def surviving_target(target: np.ndarray, keep_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Mask removed coordinates and produce matched intact/lesioned unit targets."""
    masked = np.asarray(target, dtype=float).copy()
    masked[~keep_mask] = 0.0
    retained_norm = float(np.linalg.norm(masked))
    if retained_norm == 0.0:
        return None
    intact_target = normalize_vector(masked)
    lesion_target = normalize_vector(masked[keep_mask])
    return intact_target, lesion_target, retained_norm


def main() -> None:
    args = parse_args()
    config = load_config(REPO_ROOT / args.config)
    ctx = RunContext.from_args("03_hub_lesion", args)
    ctx.require_stage("00_prepare_data")
    results = prepare_results_tree(ctx.root)
    raw_connectomes = load_raw_connectomes(config["paths"]["raw_connectomes"])

    scales = [int(scale) for scale in config["analysis"]["parcellation_scales"]]
    k_values = [int(k) for k in config["analysis"]["top_k_values"]]
    hub_config = config.get(
        "hub_lesion",
        {"hub_counts": [5, 10, 15], "methods": ["degree_strength", "cycle_participation"]},
    )
    hub_counts = [int(h) for h in hub_config["hub_counts"]]
    methods = [str(method) for method in hub_config["methods"]]
    c_value = float(config["control"]["primary_c"])
    epsilon = float(config["control"]["primary_epsilon"])
    bottom_dim = int(config["control"]["bottom_eigenspace_dim"])

    n_subjects = next(iter(raw_connectomes.values())).shape[2]
    if args.subject_limit is not None:
        n_subjects = min(n_subjects, args.subject_limit)
    ctx.begin(build_subject_table(n_subjects)["subject_id"].tolist())

    driver_rows: list[dict] = []
    geometry_rows: list[dict] = []
    target_rows: list[dict] = []

    for scale in tqdm(scales, desc="Scales"):
        labels = build_label_table(scale=scale, atlas_dir=config["paths"]["atlas_dir"])
        processed_stack = load_processed_stack(raw_connectomes, scale, n_subjects)
        reference = np.mean(processed_stack, axis=0)
        targets = build_targets(reference, labels, config["targets"], config["anatomy"], config["paths"]["atlas_dir"])

        for subject_index in tqdm(range(n_subjects), desc=f"Hub lesions {scale}", leave=False):
            subject_id = f"sub-{subject_index:03d}"
            adjacency = processed_stack[subject_index]
            scores, _ = compute_node_scores(adjacency, methods=methods)
            intact_system, _ = stabilize_adjacency(adjacency, c_value)
            intact_node_gramians = single_node_gramians(intact_system)

            for hub_count in hub_counts:
                if hub_count >= adjacency.shape[0]:
                    continue
                hubs = top_k_nodes(scores["degree_strength"], hub_count)
                keep_mask = np.ones(adjacency.shape[0], dtype=bool)
                keep_mask[hubs] = False
                kept_nodes = np.flatnonzero(keep_mask)
                intact_to_lesion = np.full(adjacency.shape[0], -1, dtype=int)
                intact_to_lesion[kept_nodes] = np.arange(kept_nodes.shape[0], dtype=int)
                lesioned_adjacency = adjacency[np.ix_(keep_mask, keep_mask)]
                lesioned_system, _ = stabilize_adjacency(lesioned_adjacency, c_value)
                lesioned_node_gramians = single_node_gramians(lesioned_system)

                matched_targets = {
                    name: surviving_target(target, keep_mask)
                    for name, target in targets.items()
                }
                hub_index_text = ";".join(str(int(index)) for index in hubs)
                hub_label_text = ";".join(labels.loc[hubs, "region_label"].tolist())

                for k in k_values:
                    if k > kept_nodes.shape[0]:
                        continue
                    for method in methods:
                        selected_intact = top_k_excluding_nodes(scores[method], k, hubs)
                        selected_lesioned = intact_to_lesion[selected_intact]
                        intact_gramian = assemble_gramian(intact_node_gramians, selected_intact)
                        lesioned_gramian = assemble_gramian(lesioned_node_gramians, selected_lesioned)
                        intact_metrics = gramian_metrics(intact_gramian, epsilon=epsilon, bottom_dim=bottom_dim)
                        lesioned_metrics = gramian_metrics(lesioned_gramian, epsilon=epsilon, bottom_dim=bottom_dim)

                        common = {
                            "subject_id": subject_id,
                            "scale": scale,
                            "hub_count": hub_count,
                            "k": k,
                            "method": method,
                            "selection_strategy": "intact_rank_nonhub_static",
                        }
                        driver_rows.append(
                            {
                                **common,
                                "n_nodes_intact": adjacency.shape[0],
                                "n_nodes_lesioned": lesioned_adjacency.shape[0],
                                "hub_node_indices": hub_index_text,
                                "hub_region_labels": hub_label_text,
                                "driver_node_indices_intact": ";".join(str(int(index)) for index in selected_intact),
                                "driver_node_indices_lesioned": ";".join(str(int(index)) for index in selected_lesioned),
                                "driver_region_labels": ";".join(labels.loc[selected_intact, "region_label"].tolist()),
                            }
                        )
                        geometry_rows.append(
                            {
                                **common,
                                "intact_log10_average_energy": intact_metrics["log10_average_energy"],
                                "lesioned_log10_average_energy": lesioned_metrics["log10_average_energy"],
                                "degradation_log10_average_energy": (
                                    lesioned_metrics["log10_average_energy"] - intact_metrics["log10_average_energy"]
                                ),
                                "intact_effective_rank": intact_metrics["effective_rank"],
                                "lesioned_effective_rank": lesioned_metrics["effective_rank"],
                                "degradation_effective_rank": (
                                    intact_metrics["effective_rank"] - lesioned_metrics["effective_rank"]
                                ),
                                "intact_participation_ratio": intact_metrics["participation_ratio"],
                                "lesioned_participation_ratio": lesioned_metrics["participation_ratio"],
                                "degradation_participation_ratio": (
                                    intact_metrics["participation_ratio"] - lesioned_metrics["participation_ratio"]
                                ),
                                "intact_condition_number": intact_metrics["condition_number"],
                                "lesioned_condition_number": lesioned_metrics["condition_number"],
                                "intact_log10_condition_number": intact_metrics["log10_condition_number"],
                                "lesioned_log10_condition_number": lesioned_metrics["log10_condition_number"],
                                "degradation_log10_condition_number": (
                                    lesioned_metrics["log10_condition_number"]
                                    - intact_metrics["log10_condition_number"]
                                ),
                            }
                        )
                        for target_name, matched_target in matched_targets.items():
                            if matched_target is None:
                                target_rows.append(
                                    {
                                        **common,
                                        "target_name": target_name,
                                        "target_available": False,
                                        "target_retained_norm": 0.0,
                                        "intact_target_energy": np.nan,
                                        "lesioned_target_energy": np.nan,
                                        "intact_log10_target_energy": np.nan,
                                        "lesioned_log10_target_energy": np.nan,
                                        "degradation_log10_target_energy": np.nan,
                                    }
                                )
                                continue
                            intact_target, lesion_target, retained_norm = matched_target
                            intact_energy = target_energy(intact_gramian, intact_target, epsilon=epsilon)
                            lesion_energy = target_energy(lesioned_gramian, lesion_target, epsilon=epsilon)
                            target_rows.append(
                                {
                                    **common,
                                    "target_name": target_name,
                                    "target_available": True,
                                    "target_retained_norm": retained_norm,
                                    "intact_target_energy": intact_energy,
                                    "lesioned_target_energy": lesion_energy,
                                    "intact_log10_target_energy": float(np.log10(intact_energy)),
                                    "lesioned_log10_target_energy": float(np.log10(lesion_energy)),
                                    "degradation_log10_target_energy": float(np.log10(lesion_energy / intact_energy)),
                                }
                            )

    write_csv(pd.DataFrame(driver_rows), results["hub_lesion"] / "driver_sets.csv")
    write_csv(pd.DataFrame(geometry_rows), results["hub_lesion"] / "geometry_degradation.csv")
    write_csv(pd.DataFrame(target_rows), results["hub_lesion"] / "target_energy_degradation.csv")

    ctx.finish()


if __name__ == "__main__":
    main()
