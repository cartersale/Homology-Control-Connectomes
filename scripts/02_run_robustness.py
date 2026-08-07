#!/usr/bin/env python
"""Run sensitivity and perturbation analyses around the primary results."""

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

from hoc.config import load_config
from hoc.control import assemble_gramian, gramian_metrics, single_node_gramians
from hoc.dataset import load_raw_connectomes
from hoc.io import prepare_results_tree, write_csv
from hoc.landscape import generate_driver_sets, summarize_landscape
from hoc.preprocessing import (
    dropout_perturbation,
    multiplicative_noise_perturbation,
    preprocess_structural_adjacency,
    stabilize_adjacency,
)
from hoc.rankings import compute_node_scores, top_k_nodes
from hoc.utils import make_rng, subject_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml", help="Path to the TOML config file.")
    parser.add_argument("--subject-limit", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument(
        "--skip-perturbations",
        action="store_true",
        help="Skip multiplicative-noise and dropout perturbation sweeps.",
    )
    parser.add_argument(
        "--include-landscape-summary",
        action="store_true",
        help="Also compute sampled/exhaustive landscape summaries for each sensitivity cell.",
    )
    return parser.parse_args()


def load_processed_stack(raw_connectomes: dict[int, np.ndarray], scale: int, n_subjects: int) -> np.ndarray:
    """Preprocess one scale into a subject-major stack."""
    raw_stack = raw_connectomes[scale][:, :, :n_subjects]
    return np.stack(
        [preprocess_structural_adjacency(raw_stack[:, :, subject_index]) for subject_index in range(n_subjects)],
        axis=0,
    )


def summarize_degradation(degradation: np.ndarray) -> dict[str, float]:
    """Summaries for perturbation-induced degradation."""
    return {
        "mean_degradation": float(np.mean(degradation)),
        "variance_degradation": float(np.var(degradation, ddof=1)),
        "worst_decile_degradation": float(np.quantile(degradation, 0.9)),
    }


def robustness_k_values(n_nodes: int, base_k_values: list[int], fractions: list[float]) -> list[int]:
    """Build the requested native and fractional-k robustness grid."""
    values = set(base_k_values)
    values.add(10)
    values.update(max(1, int(round(n_nodes * fraction))) for fraction in fractions)
    return sorted(k for k in values if k < n_nodes)


def metric_row(metrics: dict) -> dict[str, float]:
    """Return the scalar Gramian metrics written to robustness CSVs."""
    return {
        "average_energy": metrics["average_energy"],
        "log10_average_energy": metrics["log10_average_energy"],
        "effective_rank": metrics["effective_rank"],
        "participation_ratio": metrics["participation_ratio"],
        "condition_number": metrics["condition_number"],
        "log10_condition_number": metrics["log10_condition_number"],
        "min_eigenvalue": metrics["min_eigenvalue"],
        "max_eigenvalue": metrics["max_eigenvalue"],
        "log_min_eigenvalue": metrics["log_min_eigenvalue"],
        "eigenvalues_at_floor": metrics["eigenvalues_at_floor"],
        "fraction_eigenvalues_at_floor": metrics["fraction_eigenvalues_at_floor"],
    }


def empty_landscape_summary(thresholds: list[float]) -> dict[str, float]:
    """Return placeholder columns when landscape summaries are not requested."""
    summary = {
        "variance_log10_energy": np.nan,
        "best_energy": np.nan,
        "q05_log10_energy": np.nan,
        "q25_log10_energy": np.nan,
        "q50_log10_energy": np.nan,
        "q75_log10_energy": np.nan,
        "q95_log10_energy": np.nan,
    }
    for threshold in thresholds:
        summary[f"near_optimal_fraction_{int(round(threshold * 100)):02d}pct"] = np.nan
    return summary


def landscape_summary_for(
    node_gramians: np.ndarray,
    n_nodes: int,
    k: int,
    max_sets: int,
    rng: np.random.Generator,
    epsilon: float,
    bottom_dim: int,
    thresholds: list[float],
    apply_floor: bool,
    include_landscape_summary: bool,
) -> tuple[bool | float, dict[str, float]]:
    """Compute optional landscape summaries for a sensitivity cell."""
    if not include_landscape_summary:
        return np.nan, empty_landscape_summary(thresholds)

    candidate_sets, enumerated_exactly = generate_driver_sets(
        n_nodes,
        k,
        max_sets=max_sets,
        rng=rng,
    )
    energies = []
    for driver_set in candidate_sets:
        metrics = gramian_metrics(
            assemble_gramian(node_gramians, driver_set),
            epsilon=epsilon,
            bottom_dim=bottom_dim,
            apply_floor=apply_floor,
        )
        energies.append(metrics["average_energy"])
    return enumerated_exactly, summarize_landscape(np.asarray(energies, dtype=float), thresholds=thresholds)


def main() -> None:
    args = parse_args()
    config = load_config(REPO_ROOT / args.config)
    results = prepare_results_tree(config["paths"]["results_dir"])

    raw_connectomes = load_raw_connectomes(config["paths"]["raw_connectomes"])
    scales = [int(scale) for scale in config["analysis"]["parcellation_scales"]]
    n_subjects = next(iter(raw_connectomes.values())).shape[2]
    if args.subject_limit is not None:
        n_subjects = min(n_subjects, args.subject_limit)

    k_values = [int(k) for k in config["analysis"]["top_k_values"]]
    sensitivity_rows: list[dict] = []
    perturbation_rows: list[dict] = []

    base_seed = int(config["analysis"]["random_seed"])
    primary_methods = ["degree_strength", "cycle_participation"]
    epsilon_values = [float(value) for value in config["control"]["sensitivity_epsilon_values"]]
    c_values = [float(value) for value in config["control"]["sensitivity_c_values"]]
    finite_horizons = [float(value) for value in config["control"]["finite_horizons"]]
    bottom_dim = int(config["control"]["bottom_eigenspace_dim"])
    max_sets = int(config["landscape"]["max_enumerated_sets"])
    thresholds = [float(value) for value in config["landscape"]["near_optimal_thresholds"]]

    for scale in tqdm(scales, desc="Scales"):
        processed_stack = load_processed_stack(raw_connectomes, scale, n_subjects)
        for subject_index in tqdm(range(n_subjects), desc=f"Scale {scale}", leave=False):
            subject_id = f"sub-{subject_index:03d}"
            adjacency = processed_stack[subject_index]
            n_nodes = adjacency.shape[0]
            grid_k_values = robustness_k_values(
                n_nodes,
                k_values,
                [float(fraction) for fraction in config["analysis"]["larger_k_fractions"]],
            )
            node_scores, _ = compute_node_scores(adjacency, methods=["degree_strength", "cycle_participation"])

            for c_value in c_values:
                system_matrix, _ = stabilize_adjacency(adjacency, c_value)
                infinite_node_gramians = single_node_gramians(system_matrix)
                for epsilon in epsilon_values:
                    for k in grid_k_values:
                        candidate_rng = make_rng(subject_seed(base_seed, scale, subject_index, k, int(c_value * 1000)))
                        enumerated_exactly, summary = landscape_summary_for(
                            infinite_node_gramians,
                            n_nodes,
                            k,
                            max_sets,
                            candidate_rng,
                            epsilon,
                            bottom_dim,
                            thresholds,
                            True,
                            args.include_landscape_summary,
                        )
                        for method_name in primary_methods:
                            selected = top_k_nodes(node_scores[method_name], k)
                            metrics = gramian_metrics(
                                assemble_gramian(infinite_node_gramians, selected),
                                epsilon=epsilon,
                                bottom_dim=bottom_dim,
                            )
                            sensitivity_rows.append(
                                {
                                    "subject_id": subject_id,
                                    "scale": scale,
                                    "k": k,
                                    "sensitivity_family": "stabilization_epsilon",
                                    "c_value": c_value,
                                    "epsilon_case": "epsilon_floor",
                                    "epsilon": epsilon,
                                    "horizon": np.nan,
                                    "method": method_name,
                                    "enumerated_exactly": enumerated_exactly,
                                    **metric_row(metrics),
                                    **summary,
                                }
                            )

                for horizon in finite_horizons:
                    finite_node_gramians = single_node_gramians(system_matrix, horizon=horizon)
                    finite_epsilon_cases = [
                        ("epsilon_floor", min(epsilon_values), min(epsilon_values), True),
                        ("epsilon_floor", max(epsilon_values), max(epsilon_values), True),
                        ("no_floor", np.nan, 0.0, False),
                    ]
                    for epsilon_case, epsilon_label, metric_epsilon, apply_floor in finite_epsilon_cases:
                        for k in grid_k_values:
                            candidate_rng = make_rng(subject_seed(base_seed, scale, subject_index, k, int(horizon * 100)))
                            enumerated_exactly, summary = landscape_summary_for(
                                finite_node_gramians,
                                n_nodes,
                                k,
                                max_sets,
                                candidate_rng,
                                metric_epsilon,
                                bottom_dim,
                                thresholds,
                                apply_floor,
                                args.include_landscape_summary,
                            )
                            for method_name in primary_methods:
                                selected = top_k_nodes(node_scores[method_name], k)
                                metrics = gramian_metrics(
                                    assemble_gramian(finite_node_gramians, selected),
                                    epsilon=metric_epsilon,
                                    bottom_dim=bottom_dim,
                                    apply_floor=apply_floor,
                                )
                                sensitivity_rows.append(
                                    {
                                        "subject_id": subject_id,
                                        "scale": scale,
                                        "k": k,
                                        "sensitivity_family": "finite_horizon",
                                        "c_value": c_value,
                                        "epsilon_case": epsilon_case,
                                        "epsilon": epsilon_label,
                                        "horizon": horizon,
                                        "method": method_name,
                                        "enumerated_exactly": enumerated_exactly,
                                        **metric_row(metrics),
                                        **summary,
                                    }
                                )

            if args.skip_perturbations:
                continue

            for method_name in primary_methods:
                selected_by_k = {k: top_k_nodes(node_scores[method_name], k) for k in k_values}
                nominal_system, _ = stabilize_adjacency(adjacency, float(config["control"]["primary_c"]))
                nominal_node_gramians = single_node_gramians(nominal_system)
                for k, selected in selected_by_k.items():
                    nominal_metrics = gramian_metrics(
                        assemble_gramian(nominal_node_gramians, selected),
                        epsilon=float(config["control"]["primary_epsilon"]),
                        bottom_dim=bottom_dim,
                    )
                    nominal_energy = nominal_metrics["log10_average_energy"]

                    for sigma in config["robustness"]["noise_sigmas"]:
                        degradation = []
                        for perturb_index in range(int(config["robustness"]["n_perturbations"])):
                            perturb_rng = make_rng(subject_seed(base_seed, scale, subject_index, k, perturb_index, 1))
                            perturbed = multiplicative_noise_perturbation(adjacency, float(sigma), perturb_rng)
                            system_matrix, _ = stabilize_adjacency(perturbed, float(config["control"]["primary_c"]))
                            metrics = gramian_metrics(
                                assemble_gramian(single_node_gramians(system_matrix), selected),
                                epsilon=float(config["control"]["primary_epsilon"]),
                                bottom_dim=bottom_dim,
                            )
                            degradation.append(metrics["log10_average_energy"] - nominal_energy)
                        perturbation_rows.append(
                            {
                                "subject_id": subject_id,
                                "scale": scale,
                                "k": k,
                                "method": method_name,
                                "perturbation_family": "multiplicative_noise",
                                "level": float(sigma),
                                **summarize_degradation(np.asarray(degradation, dtype=float)),
                            }
                        )

                    for probability in config["robustness"]["dropout_probabilities"]:
                        degradation = []
                        for perturb_index in range(int(config["robustness"]["n_perturbations"])):
                            perturb_rng = make_rng(subject_seed(base_seed, scale, subject_index, k, perturb_index, 2))
                            perturbed = dropout_perturbation(adjacency, float(probability), perturb_rng)
                            system_matrix, _ = stabilize_adjacency(perturbed, float(config["control"]["primary_c"]))
                            metrics = gramian_metrics(
                                assemble_gramian(single_node_gramians(system_matrix), selected),
                                epsilon=float(config["control"]["primary_epsilon"]),
                                bottom_dim=bottom_dim,
                            )
                            degradation.append(metrics["log10_average_energy"] - nominal_energy)
                        perturbation_rows.append(
                            {
                                "subject_id": subject_id,
                                "scale": scale,
                                "k": k,
                                "method": method_name,
                                "perturbation_family": "edge_dropout",
                                "level": float(probability),
                                **summarize_degradation(np.asarray(degradation, dtype=float)),
                            }
                        )

    write_csv(pd.DataFrame(sensitivity_rows), results["robustness"] / "sensitivity_summary.csv")
    if perturbation_rows:
        write_csv(pd.DataFrame(perturbation_rows), results["robustness"] / "perturbation_summary.csv")


if __name__ == "__main__":
    main()
