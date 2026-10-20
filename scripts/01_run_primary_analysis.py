#!/usr/bin/env python
"""Run the primary control, topology, target, and geometry analyses."""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hoc.atlas import build_label_table
from hoc.config import load_config
from hoc.control import assemble_gramian, gramian_metrics, single_node_gramians, subspace_overlap, target_energy
from hoc.dataset import load_raw_connectomes
from hoc.dataset import build_subject_table
from hoc.runs import RunContext, add_run_arguments
from hoc.io import prepare_results_tree, write_csv, write_npz
from hoc.landscape import generate_driver_sets, greedy_extension, summarize_landscape
from hoc.paths import resolve_repo_path
from hoc.preprocessing import preprocess_structural_adjacency, stabilize_adjacency
from hoc.rankings import compute_node_scores, random_top_k_nodes, scores_to_table, top_k_nodes
from hoc.targets import build_target_metadata, build_targets
from hoc.utils import make_rng, subject_seed


TARGET_METHODS = ("degree_strength", "cycle_participation")
NAMED_TARGET_FAMILIES = ("landmark", "module", "distributed")
SINGLE_PARCEL_FAMILIES = ("single_cortical_parcel", "single_subcortical_parcel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml", help="Path to the TOML config file.")
    parser.add_argument("--subject-limit", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument(
        "--skip-landscape",
        action="store_true",
        help="Skip exhaustive/sampled landscape calculations for faster test runs.",
    )
    add_run_arguments(parser)
    return parser.parse_args()


def append_row(frame_rows: list[dict], **kwargs: object) -> None:
    """Append a row dictionary in a compact way."""
    frame_rows.append(dict(kwargs))


def load_processed_stack(scale: int, raw_connectomes: dict[int, np.ndarray], n_subjects: int) -> np.ndarray:
    """Preprocess one scale into a subject-by-node-by-node stack."""
    raw_stack = raw_connectomes[scale][:, :, :n_subjects]
    return np.stack(
        [preprocess_structural_adjacency(raw_stack[:, :, subject_index]) for subject_index in range(n_subjects)],
        axis=0,
    )


def append_target_rows(
    *,
    target_rows: list[dict],
    target_geometry_rows: list[dict],
    subject_id: str,
    scale: int,
    k: int,
    method_name: str,
    selection_strategy: str,
    metrics: dict,
    target_names: list[str],
    target_matrix: np.ndarray,
) -> None:
    """Append vectorized target-energy and target-geometry rows."""
    eigenvalues = metrics["eigenvalues"]
    eigenvectors = metrics["eigenvectors"]
    bottom_basis = metrics["bottom_basis"]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        coefficients = eigenvectors.T @ target_matrix
        coefficient_power = coefficients**2
        directional_energies = np.sum(coefficient_power / eigenvalues[:, None], axis=0)
        hard_overlaps = np.sum((bottom_basis.T @ target_matrix) ** 2, axis=0)
        spectral_supports = np.sum(eigenvalues[:, None] * coefficient_power, axis=0)
        spectral_spreads = 1.0 / np.maximum(np.sum(coefficient_power**2, axis=0), 1.0e-12)
    entropy_terms = np.zeros_like(coefficient_power)
    positive_power = coefficient_power > 0.0
    entropy_terms[positive_power] = coefficient_power[positive_power] * np.log(coefficient_power[positive_power])
    spectral_entropy_spreads = np.exp(-np.sum(entropy_terms, axis=0))
    if not (
        np.all(np.isfinite(directional_energies))
        and np.all(np.isfinite(hard_overlaps))
        and np.all(np.isfinite(spectral_supports))
        and np.all(np.isfinite(spectral_spreads))
        and np.all(np.isfinite(spectral_entropy_spreads))
    ):
        raise FloatingPointError("Non-finite target energy or spectral geometry encountered.")

    for target_index, target_name in enumerate(target_names):
        target_energy_value = float(directional_energies[target_index])
        append_row(
            target_rows,
            subject_id=subject_id,
            scale=scale,
            k=k,
            method=method_name,
            selection_strategy=selection_strategy,
            target_name=target_name,
            target_energy=target_energy_value,
        )
        append_row(
            target_geometry_rows,
            subject_id=subject_id,
            scale=scale,
            k=k,
            method=method_name,
            selection_strategy=selection_strategy,
            target_name=target_name,
            target_energy=target_energy_value,
            hard_overlap=float(hard_overlaps[target_index]),
            spectral_support=float(spectral_supports[target_index]),
            spectral_spread_effective_modes=float(spectral_spreads[target_index]),
            spectral_entropy_effective_modes=float(spectral_entropy_spreads[target_index]),
            bottom_dim=int(bottom_basis.shape[1]),
        )


def bootstrap_ci(values: pd.Series | np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float]:
    """Bootstrap a 95% CI for a median."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan, np.nan
    if array.size == 1:
        return float(array[0]), float(array[0])
    samples = rng.choice(array, size=(n_boot, array.size), replace=True)
    low, high = np.percentile(np.median(samples, axis=1), [2.5, 97.5])
    return float(low), float(high)


def signed_rank_test(values: pd.Series | np.ndarray) -> tuple[float, float]:
    """Run a robust two-sided Wilcoxon signed-rank test against zero."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0 or np.allclose(array, 0.0):
        return np.nan, 1.0
    statistic, p_value = wilcoxon(array, zero_method="wilcox", alternative="two-sided")
    return float(statistic), float(p_value)


def target_difference_tables(
    target_rows: list[dict],
    target_metadata_rows: list[pd.DataFrame],
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Build paired target-energy differences and statistical summaries."""
    targets = pd.DataFrame(target_rows)
    metadata = pd.concat(target_metadata_rows, ignore_index=True)
    paired = (
        targets[targets["method"].isin(TARGET_METHODS)]
        .pivot_table(
            index=["subject_id", "scale", "k", "selection_strategy", "target_name"],
            columns="method",
            values="target_energy",
        )
        .reset_index()
    )
    paired = paired.dropna(subset=list(TARGET_METHODS)).copy()
    paired["degree_energy"] = paired["degree_strength"]
    paired["cycle_energy"] = paired["cycle_participation"]
    paired["cycle_minus_degree"] = paired["cycle_energy"] - paired["degree_energy"]
    paired["pct_cycle_minus_degree"] = 100.0 * paired["cycle_minus_degree"] / paired["degree_energy"]
    paired = paired.drop(columns=list(TARGET_METHODS))
    paired = paired.merge(metadata, on=["scale", "target_name"], how="left")

    subject_average = (
        paired.groupby(
            [
                "subject_id",
                "scale",
                "selection_strategy",
                "target_name",
                "target_label",
                "target_family",
                "target_axis_class",
                "axis_position",
                "node_index",
                "hemi",
                "base_region",
                "region_label",
                "n_support_nodes",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            degree_energy=("degree_energy", "mean"),
            cycle_energy=("cycle_energy", "mean"),
            cycle_minus_degree=("cycle_minus_degree", "mean"),
            pct_cycle_minus_degree=("pct_cycle_minus_degree", "mean"),
        )
    )

    named = subject_average[subject_average["target_family"].isin(NAMED_TARGET_FAMILIES)].copy()
    named_summary_rows = []
    for metric in ("cycle_minus_degree", "pct_cycle_minus_degree"):
        for (scale, selection_strategy, target_name), subset in named.groupby(["scale", "selection_strategy", "target_name"]):
            values = subset[metric]
            ci_low, ci_high = bootstrap_ci(values, rng)
            statistic, p_value = signed_rank_test(values)
            first = subset.iloc[0]
            named_summary_rows.append(
                {
                    "scale": int(scale),
                    "selection_strategy": selection_strategy,
                    "target_name": target_name,
                    "target_label": first["target_label"],
                    "target_family": first["target_family"],
                    "target_axis_class": first["target_axis_class"],
                    "axis_position": int(first["axis_position"]),
                    "metric": metric,
                    "n_subjects": int(subset["subject_id"].nunique()),
                    "median": float(np.median(values)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
    named_summary = pd.DataFrame(named_summary_rows)
    if not named_summary.empty:
        named_summary["p_value_holm"] = np.nan
        named_summary["reject_holm"] = False
        for (metric, scale, selection_strategy), indices in named_summary.groupby(["metric", "scale", "selection_strategy"]).groups.items():
            corrected = multipletests(named_summary.loc[indices, "p_value"], method="holm")[1]
            named_summary.loc[indices, "p_value_holm"] = corrected
            named_summary.loc[indices, "reject_holm"] = corrected < 0.05

    parcel_subject = subject_average[subject_average["target_family"].isin(SINGLE_PARCEL_FAMILIES)].copy()
    parcel_test_rows = []
    for metric in ("cycle_minus_degree", "pct_cycle_minus_degree"):
        for (scale, selection_strategy, family, target_name), subset in parcel_subject.groupby(
            ["scale", "selection_strategy", "target_family", "target_name"]
        ):
            values = subset[metric]
            statistic, p_value = signed_rank_test(values)
            first = subset.iloc[0]
            parcel_test_rows.append(
                {
                    "scale": int(scale),
                    "selection_strategy": selection_strategy,
                    "target_family": family,
                    "target_name": target_name,
                    "target_label": first["target_label"],
                    "node_index": first["node_index"],
                    "hemi": first["hemi"],
                    "base_region": first["base_region"],
                    "region_label": first["region_label"],
                    "metric": metric,
                    "n_subjects": int(subset["subject_id"].nunique()),
                    "mean_subject_average": float(np.mean(values)),
                    "median_subject_average": float(np.median(values)),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
    parcel_tests = pd.DataFrame(parcel_test_rows)
    if not parcel_tests.empty:
        parcel_tests["p_value_bh"] = np.nan
        parcel_tests["reject_bh"] = False
        for (metric, scale, selection_strategy, family), indices in parcel_tests.groupby(
            ["metric", "scale", "selection_strategy", "target_family"]
        ).groups.items():
            corrected = multipletests(parcel_tests.loc[indices, "p_value"], method="fdr_bh")[1]
            parcel_tests.loc[indices, "p_value_bh"] = corrected
            parcel_tests.loc[indices, "reject_bh"] = corrected < 0.05

    parcel_distribution_rows = []
    if not parcel_tests.empty:
        for (metric, scale, selection_strategy, family), subset in parcel_tests.groupby(
            ["metric", "scale", "selection_strategy", "target_family"]
        ):
            values = subset["mean_subject_average"]
            ci_low, ci_high = bootstrap_ci(values, rng)
            negative = (subset["reject_bh"]) & (subset["median_subject_average"] < 0.0)
            positive = (subset["reject_bh"]) & (subset["median_subject_average"] > 0.0)
            n_total = int(len(subset))
            parcel_distribution_rows.append(
                {
                    "scale": int(scale),
                    "selection_strategy": selection_strategy,
                    "target_family": family,
                    "metric": metric,
                    "n_parcels": n_total,
                    "median_parcel_subject_average": float(np.median(values)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_significantly_negative": int(np.count_nonzero(negative)),
                    "n_significantly_positive": int(np.count_nonzero(positive)),
                    "n_null": int(n_total - np.count_nonzero(negative) - np.count_nonzero(positive)),
                    "fraction_significantly_negative": float(np.count_nonzero(negative) / n_total) if n_total else np.nan,
                    "fraction_significantly_positive": float(np.count_nonzero(positive) / n_total) if n_total else np.nan,
                    "fraction_null": float((n_total - np.count_nonzero(negative) - np.count_nonzero(positive)) / n_total)
                    if n_total
                    else np.nan,
                }
            )

    return {
        "target_pairwise_differences": paired,
        "target_subject_differences": subject_average,
        "target_named_summary": named_summary,
        "target_single_parcel_tests": parcel_tests,
        "target_single_parcel_distribution_summary": pd.DataFrame(parcel_distribution_rows),
        "target_metadata": metadata,
    }


def main() -> None:
    args = parse_args()
    config = load_config(REPO_ROOT / args.config)
    ctx = RunContext.from_args("01_primary", args)
    ctx.require_stage("00_prepare_data")
    results = prepare_results_tree(ctx.root)

    analysis_config = config["analysis"]
    control_config = config["control"]
    ranking_config = config["rankings"]
    landscape_config = config["landscape"]
    weighted_inputs_config = config["weighted_inputs"]

    raw_connectomes = load_raw_connectomes(config["paths"]["raw_connectomes"])
    scales = [int(scale) for scale in analysis_config["parcellation_scales"]]
    n_subjects = next(iter(raw_connectomes.values())).shape[2]
    if args.subject_limit is not None:
        n_subjects = min(n_subjects, args.subject_limit)
    ctx.begin(build_subject_table(n_subjects)["subject_id"].tolist())

    primary_methods = [str(method) for method in ranking_config["primary_methods"]]
    secondary_methods = [str(method) for method in ranking_config["secondary_methods"]]
    all_methods = primary_methods + secondary_methods
    k_values = [int(k) for k in analysis_config["top_k_values"]]
    weighted_support_sizes = [int(size) for size in weighted_inputs_config["support_sizes"]]
    rng = make_rng(int(analysis_config["random_seed"]))

    node_score_tables: list[pd.DataFrame] = []
    rank_divergence_rows: list[dict] = []
    driver_set_rows: list[dict] = []
    node_energy_rows: list[dict] = []
    greedy_step_rows: list[dict] = []
    greedy_comparison_rows: list[dict] = []
    landscape_rows: list[dict] = []
    target_rows: list[dict] = []
    target_geometry_rows: list[dict] = []
    target_metadata_tables: list[pd.DataFrame] = []
    geometry_rows: list[dict] = []
    overlap_rows: list[dict] = []
    weighted_rows: list[dict] = []

    atlas_dir = config["paths"]["atlas_dir"]

    for scale in tqdm(scales, desc="Scales"):
        labels = build_label_table(scale=scale, atlas_dir=atlas_dir)
        processed_stack = load_processed_stack(scale, raw_connectomes, n_subjects)
        reference = np.mean(processed_stack, axis=0)
        targets = build_targets(reference, labels, config["targets"], config["anatomy"], config["paths"]["atlas_dir"])
        target_metadata = build_target_metadata(labels, targets)
        target_metadata.insert(0, "scale", scale)
        target_metadata_tables.append(target_metadata)
        target_names = target_metadata["target_name"].tolist()
        target_matrix = np.column_stack([targets[target_name] for target_name in target_names])
        n_target_conditions = n_subjects * len(k_values) * len(TARGET_METHODS)
        target_summary_progress = tqdm(total=n_target_conditions, desc=f"Target summaries {scale}", leave=False)
        write_npz(
            results["processed"] / f"targets_scale-{scale}.npz",
            **{target_name: target_vector for target_name, target_vector in targets.items()},
        )
        write_csv(target_metadata, results["processed"] / f"target_metadata_scale-{scale}.csv")

        for subject_index in tqdm(range(n_subjects), desc=f"Scale {scale}", leave=False):
            subject_id = f"sub-{subject_index:03d}"
            adjacency = processed_stack[subject_index]
            node_scores, _ = compute_node_scores(adjacency=adjacency, methods=all_methods)
            for method_name, scores in node_scores.items():
                node_score_tables.append(
                    scores_to_table(
                        subject_id=subject_id,
                        scale=scale,
                        region_labels=labels,
                        method=method_name,
                        scores=scores,
                    )
                )

            degree_scores = node_scores["degree_strength"]
            topology_scores = node_scores["cycle_participation"]
            rank_correlation = spearmanr(degree_scores, topology_scores).statistic

            system_matrix, spectral_radius = stabilize_adjacency(adjacency, float(control_config["primary_c"]))
            node_gramians = single_node_gramians(system_matrix)
            bottom_dim = int(control_config["bottom_eigenspace_dim"])
            epsilon = float(control_config["primary_epsilon"])

            deterministic_sets: dict[tuple[str, str, int], np.ndarray] = {}
            deterministic_metrics: dict[tuple[str, str, int], dict] = {}
            delta_log10_energies: list[float] = []

            for k in k_values:
                for method_name, scores in node_scores.items():
                    selected = top_k_nodes(scores, k)
                    deterministic_sets[(method_name, "static", k)] = selected
                    gramian = assemble_gramian(node_gramians, selected)
                    metrics = gramian_metrics(gramian, epsilon=epsilon, bottom_dim=bottom_dim)
                    deterministic_metrics[(method_name, "static", k)] = metrics

                    append_row(
                        driver_set_rows,
                        subject_id=subject_id,
                        scale=scale,
                        k=k,
                        method=method_name,
                        selection_strategy="static",
                        node_indices=";".join(str(int(index)) for index in selected),
                        region_labels=";".join(labels.loc[selected, "region_label"].tolist()),
                    )
                    append_row(
                        node_energy_rows,
                        subject_id=subject_id,
                        scale=scale,
                        k=k,
                        method=method_name,
                        selection_strategy="static",
                        random_sample=np.nan,
                        log10_average_energy=metrics["log10_average_energy"],
                        average_energy=metrics["average_energy"],
                    )
                    append_row(
                        geometry_rows,
                        subject_id=subject_id,
                        scale=scale,
                        k=k,
                        method=method_name,
                        selection_strategy="static",
                        log10_average_energy=metrics["log10_average_energy"],
                        effective_rank=metrics["effective_rank"],
                        participation_ratio=metrics["participation_ratio"],
                        log10_condition_number=metrics["log10_condition_number"],
                        log_min_eigenvalue=metrics["log_min_eigenvalue"],
                    )

                for random_sample in range(int(ranking_config["n_random_rankings"])):
                    random_rng = make_rng(subject_seed(int(analysis_config["random_seed"]), scale, subject_index, k, random_sample))
                    selected = random_top_k_nodes(labels.shape[0], k, random_rng)
                    metrics = gramian_metrics(
                        assemble_gramian(node_gramians, selected),
                        epsilon=epsilon,
                        bottom_dim=bottom_dim,
                    )
                    append_row(
                        driver_set_rows,
                        subject_id=subject_id,
                        scale=scale,
                        k=k,
                        method="random",
                        selection_strategy="static",
                        node_indices=";".join(str(int(index)) for index in selected),
                        region_labels=";".join(labels.loc[selected, "region_label"].tolist()),
                    )
                    append_row(
                        node_energy_rows,
                        subject_id=subject_id,
                        scale=scale,
                        k=k,
                        method="random",
                        selection_strategy="static",
                        random_sample=random_sample,
                        log10_average_energy=metrics["log10_average_energy"],
                        average_energy=metrics["average_energy"],
                    )
                    append_row(
                        geometry_rows,
                        subject_id=subject_id,
                        scale=scale,
                        k=k,
                        method="random",
                        selection_strategy="static",
                        log10_average_energy=metrics["log10_average_energy"],
                        effective_rank=metrics["effective_rank"],
                        participation_ratio=metrics["participation_ratio"],
                        log10_condition_number=metrics["log10_condition_number"],
                        log_min_eigenvalue=metrics["log_min_eigenvalue"],
                    )

                greedy_methods = [name for name in primary_methods if name in node_scores]
                for method_name in greedy_methods:
                    greedy_set, step_rows, comparison = greedy_extension(
                        scores=node_scores[method_name],
                        node_gramians=node_gramians,
                        k=k,
                        epsilon=epsilon,
                        candidate_pool=int(landscape_config["greedy_candidate_pool"]),
                        bottom_dim=bottom_dim,
                    )
                    greedy_metrics = gramian_metrics(
                        assemble_gramian(node_gramians, greedy_set),
                        epsilon=epsilon,
                        bottom_dim=bottom_dim,
                    )
                    deterministic_sets[(method_name, "greedy", k)] = greedy_set
                    deterministic_metrics[(method_name, "greedy", k)] = greedy_metrics

                    append_row(
                        driver_set_rows,
                        subject_id=subject_id,
                        scale=scale,
                        k=k,
                        method=method_name,
                        selection_strategy="greedy",
                        node_indices=";".join(str(int(index)) for index in greedy_set),
                        region_labels=";".join(labels.loc[greedy_set, "region_label"].tolist()),
                    )
                    append_row(
                        node_energy_rows,
                        subject_id=subject_id,
                        scale=scale,
                        k=k,
                        method=method_name,
                        selection_strategy="greedy",
                        random_sample=np.nan,
                        log10_average_energy=greedy_metrics["log10_average_energy"],
                        average_energy=greedy_metrics["average_energy"],
                    )
                    append_row(
                        geometry_rows,
                        subject_id=subject_id,
                        scale=scale,
                        k=k,
                        method=method_name,
                        selection_strategy="greedy",
                        log10_average_energy=greedy_metrics["log10_average_energy"],
                        effective_rank=greedy_metrics["effective_rank"],
                        participation_ratio=greedy_metrics["participation_ratio"],
                        log10_condition_number=greedy_metrics["log10_condition_number"],
                        log_min_eigenvalue=greedy_metrics["log_min_eigenvalue"],
                    )
                    append_row(
                        greedy_comparison_rows,
                        subject_id=subject_id,
                        scale=scale,
                        k=k,
                        method=method_name,
                        static_log10_average_energy=comparison["static_log10_average_energy"],
                        greedy_log10_average_energy=comparison["greedy_log10_average_energy"],
                        greedy_minus_static_log10=comparison["greedy_minus_static_log10"],
                    )
                    for row in step_rows:
                        append_row(
                            greedy_step_rows,
                            subject_id=subject_id,
                            scale=scale,
                            k=k,
                            method=method_name,
                            **row,
                        )

                if (
                    ("cycle_participation", "static", k) in deterministic_metrics
                    and ("degree_strength", "static", k) in deterministic_metrics
                ):
                    delta_log10_energies.append(
                        deterministic_metrics[("cycle_participation", "static", k)]["log10_average_energy"]
                        - deterministic_metrics[("degree_strength", "static", k)]["log10_average_energy"]
                    )

                if not args.skip_landscape:
                    landscape_rng = make_rng(subject_seed(int(analysis_config["random_seed"]), scale, subject_index, 999, k))
                    candidate_sets, enumerated_exactly = generate_driver_sets(
                        n_nodes=labels.shape[0],
                        k=k,
                        max_sets=int(landscape_config["max_enumerated_sets"]),
                        rng=landscape_rng,
                    )
                    landscape_energies = []
                    for driver_set in candidate_sets:
                        metrics = gramian_metrics(
                            assemble_gramian(node_gramians, driver_set),
                            epsilon=epsilon,
                            bottom_dim=bottom_dim,
                        )
                        landscape_energies.append(metrics["average_energy"])
                    landscape_energies = np.asarray(landscape_energies, dtype=float)
                    summary = summarize_landscape(
                        energies_linear=landscape_energies,
                        thresholds=[float(value) for value in landscape_config["near_optimal_thresholds"]],
                    )
                    append_row(
                        landscape_rows,
                        subject_id=subject_id,
                        scale=scale,
                        k=k,
                        n_candidate_sets=int(candidate_sets.shape[0]),
                        enumerated_exactly=bool(enumerated_exactly),
                        **summary,
                    )
                    if bool(analysis_config["save_landscape_samples"]):
                        write_npz(
                            results["samples"] / f"{subject_id}_scale-{scale}_k-{k}.npz",
                            candidate_sets=candidate_sets,
                            average_energies=landscape_energies,
                        )

                for method_name in TARGET_METHODS:
                    key = (method_name, "static", k)
                    if key not in deterministic_metrics:
                        continue
                    append_target_rows(
                        target_rows=target_rows,
                        target_geometry_rows=target_geometry_rows,
                        subject_id=subject_id,
                        scale=scale,
                        k=k,
                        method_name=method_name,
                        selection_strategy="static",
                        metrics=deterministic_metrics[key],
                        target_names=target_names,
                        target_matrix=target_matrix,
                    )
                    target_summary_progress.update(1)

                for support_size in weighted_support_sizes:
                    if support_size > labels.shape[0]:
                        continue
                    weight = 1.0 / np.sqrt(support_size)
                    for method_name in ("degree_strength", "cycle_participation"):
                        if method_name not in node_scores:
                            continue
                        selected = top_k_nodes(node_scores[method_name], support_size)
                        weights = np.full(selected.shape[0], weight, dtype=float)
                        gramian = assemble_gramian(node_gramians, selected, node_weights=weights)
                        metrics = gramian_metrics(gramian, epsilon=epsilon, bottom_dim=bottom_dim)
                        append_row(
                            weighted_rows,
                            subject_id=subject_id,
                            scale=scale,
                            support_size=support_size,
                            method=method_name,
                            metric="log10_average_energy",
                            target_name="average",
                            value=metrics["log10_average_energy"],
                        )
                        for target_name, target_vector in targets.items():
                            append_row(
                                weighted_rows,
                                subject_id=subject_id,
                                scale=scale,
                                support_size=support_size,
                                method=method_name,
                                metric="target_energy",
                                target_name=target_name,
                                value=target_energy(gramian=gramian, target=target_vector, epsilon=epsilon),
                            )

                overlap_keys = [
                    key
                    for key in deterministic_metrics
                    if key[2] == k and key[0] in {"degree_strength", "cycle_participation"} and key[1] in {"static", "greedy"}
                ]
                for (method_a, strategy_a, _), (method_b, strategy_b, _) in combinations(overlap_keys, 2):
                    basis_a = deterministic_metrics[(method_a, strategy_a, k)]["bottom_basis"]
                    basis_b = deterministic_metrics[(method_b, strategy_b, k)]["bottom_basis"]
                    append_row(
                        overlap_rows,
                        subject_id=subject_id,
                        scale=scale,
                        k=k,
                        method_a=method_a,
                        strategy_a=strategy_a,
                        method_b=method_b,
                        strategy_b=strategy_b,
                        overlap=subspace_overlap(basis_a, basis_b),
                    )

            append_row(
                rank_divergence_rows,
                subject_id=subject_id,
                scale=scale,
                spearman_degree_vs_topology=rank_correlation,
                mean_delta_log10_energy=float(np.mean(delta_log10_energies)),
                spectral_radius_before_shift=spectral_radius,
            )
        target_summary_progress.close()

    write_csv(pd.concat(node_score_tables, ignore_index=True), results["primary"] / "node_scores.csv")
    write_csv(pd.DataFrame(rank_divergence_rows), results["primary"] / "rank_divergence.csv")
    write_csv(pd.DataFrame(driver_set_rows), results["primary"] / "driver_sets.csv")
    write_csv(pd.DataFrame(node_energy_rows), results["primary"] / "node_level_energy.csv")
    write_csv(pd.DataFrame(greedy_step_rows), results["primary"] / "greedy_steps.csv")
    write_csv(pd.DataFrame(greedy_comparison_rows), results["primary"] / "greedy_comparison.csv")
    write_csv(pd.DataFrame(landscape_rows), results["primary"] / "landscape_summary.csv")
    write_csv(pd.DataFrame(target_rows), results["primary"] / "target_energies.csv")
    write_csv(pd.DataFrame(target_geometry_rows), results["primary"] / "target_geometry.csv")
    target_statistics = target_difference_tables(
        target_rows=target_rows,
        target_metadata_rows=target_metadata_tables,
        rng=make_rng(int(analysis_config["random_seed"]) + 17),
    )
    for table_name, table in target_statistics.items():
        if table_name == "target_metadata":
            write_csv(table, results["primary"] / f"{table_name}.csv")
        else:
            write_csv(table, results["statistics"] / f"{table_name}.csv")
    write_csv(pd.DataFrame(geometry_rows), results["primary"] / "geometry_metrics.csv")
    write_csv(pd.DataFrame(overlap_rows), results["primary"] / "eigenspace_overlap.csv")
    write_csv(pd.DataFrame(weighted_rows), results["primary"] / "weighted_inputs.csv")

    ctx.finish()


if __name__ == "__main__":
    main()
