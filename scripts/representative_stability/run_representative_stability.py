#!/usr/bin/env python
"""Representative-stability test for cycle participation (scheme 2: length-minimized).

Recomputes `CP_i` from length-minimized (geodesic) representatives of the *same*
persistent H1 features that the main pipeline scores from Ripser's cocycles, then
re-runs the downstream selection and control-geometry steps under each scheme.

Three escalating readouts are produced:

1. Score level    -- Spearman correlation of `CP_i` between schemes, per subject.
2. Selection level-- overlap (Jaccard / shared count) of the top-k driver sets.
3. Result level   -- the cycle-minus-degree effective-rank advantage under each
                     scheme, plus a paired test of whether that advantage itself
                     differs between schemes.

This script is intentionally separate from the main `scripts/` pipeline: it reads
the preprocessed connectomes from ``<run>/processed/`` and writes to its own
subdirectory ``<run>/<--output-subdir>/`` without touching any primary output.
The run is chosen with ``--run <name>`` (default $HOC_RUN or "main"); the legacy
``results`` tree is read-only and cannot be used. As in the main pipeline,
``--subject-limit N`` selects the separate smoke-test run ``<name>__nN``. Each
scale is recorded as its own stage in the run manifest; rerunning a finished
scale needs ``--overwrite``.

Example
-------
    python scripts/representative_stability/run_representative_stability.py --run main
    python scripts/representative_stability/run_representative_stability.py \
        --run main --scale 114 --subject-limit 5 --weight-mode hops
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
for extra_path in (str(SRC_ROOT), str(SCRIPT_DIR)):
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from hoc.config import load_config
from hoc.control import assemble_gramian, gramian_metrics, single_node_gramians
from hoc.dataset import load_raw_connectomes
from hoc.io import write_csv
from hoc.paths import ensure_directory
from hoc.preprocessing import preprocess_structural_adjacency, stabilize_adjacency
from hoc.rankings import strength_scores, top_k_nodes
from hoc.runs import RunContext, add_run_arguments, read_manifest
from hoc.utils import make_rng

from representative_schemes import cycle_participation_by_scheme

GEOMETRY_METRICS = (
    "effective_rank",
    "participation_ratio",
    "log10_average_energy",
    "log10_condition_number",
    "log_min_eigenvalue",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/default.toml", help="Path to the TOML config file.")
    parser.add_argument("--scale", type=int, default=114, help="Parcellation scale to test (default: 114).")
    parser.add_argument("--subject-limit", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=None,
        help="Driver-set sizes for the selection and result levels (default: config top_k_values plus 15).",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["ripser", "geodesic", "gudhi"],
        choices=("ripser", "geodesic", "gudhi", "gudhi_geodesic"),
        help="Representative schemes to compare; the first is the baseline.",
    )
    parser.add_argument(
        "--weight-mode",
        choices=("filtration", "hops"),
        default="filtration",
        help="Cycle length used by the geodesic scheme: summed filtration distance or edge count.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("birth", "below_death", "death"),
        default="birth",
        help="Filtration stage the shortest cycle is searched in.",
    )
    parser.add_argument(
        "--output-subdir",
        default="representative_stability",
        help="Subdirectory of the run directory for outputs.",
    )
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples for median CIs.")
    add_run_arguments(parser)
    return parser.parse_args()


def stage_name(args: argparse.Namespace) -> str:
    """Manifest stage name: one stage per output subdirectory and scale."""
    stage = "representative_stability/run_representative_stability"
    if args.output_subdir != "representative_stability":
        stage += f":{args.output_subdir}"
    return stage + f":scale-{args.scale}"


def check_subject_count(run_dir: Path, n_subjects: int, source: Path) -> None:
    """Fail if the connectome stack does not match the run's recorded subject set."""
    manifest = read_manifest(run_dir) or {}
    subjects = manifest.get("subjects")
    if subjects is not None and len(subjects) != n_subjects:
        raise RuntimeError(
            f"{source} holds {n_subjects} subjects but the manifest of {run_dir} lists {len(subjects)}; "
            "the processed connectomes do not belong to this run."
        )


def load_adjacency_stack(config: dict, scale: int, subject_limit: int | None, run_dir: Path) -> np.ndarray:
    """Load preprocessed connectomes, falling back to preprocessing the raw stack."""
    processed_path = run_dir / "processed" / f"connectomes_scale-{scale}.npz"
    if processed_path.exists():
        stack = np.load(processed_path)["adjacency"]
        check_subject_count(run_dir, stack.shape[0], processed_path)
    else:
        raw = load_raw_connectomes(config["paths"]["raw_connectomes"])[scale]
        stack = np.stack(
            [preprocess_structural_adjacency(raw[:, :, index]) for index in range(raw.shape[2])],
            axis=0,
        )
    if subject_limit is not None:
        stack = stack[: min(subject_limit, stack.shape[0])]
    return stack


def jaccard(set_a: np.ndarray, set_b: np.ndarray) -> float:
    """Jaccard index of two driver sets."""
    a, b = set(int(index) for index in set_a), set(int(index) for index in set_b)
    union = a | b
    return float(len(a & b) / len(union)) if union else np.nan


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
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


def signed_rank_test(values: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test against zero."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0 or np.allclose(array, 0.0):
        return np.nan, 1.0
    statistic, p_value = wilcoxon(array, zero_method="wilcox", alternative="two-sided")
    return float(statistic), float(p_value)


def summarize_paired(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    """Median, CI, sign counts, and signed-rank test for a paired difference."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    ci_low, ci_high = bootstrap_median_ci(array, rng, n_boot)
    statistic, p_value = signed_rank_test(array)
    return {
        "n_subjects": int(array.size),
        "mean": float(np.mean(array)) if array.size else np.nan,
        "median": float(np.median(array)) if array.size else np.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "fraction_positive": float(np.mean(array > 0.0)) if array.size else np.nan,
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
    }


def run_subject(
    adjacency: np.ndarray,
    schemes: tuple[str, ...],
    k_values: list[int],
    control_config: dict,
    weight_mode: str,
    threshold_mode: str,
) -> dict[str, list[dict]]:
    """Compute all three readout levels for one subject connectome."""
    scheme_scores, representative_sets, library_comparison = cycle_participation_by_scheme(
        adjacency,
        schemes=schemes,
        weight_mode=weight_mode,
        threshold_mode=threshold_mode,
    )
    degree = strength_scores(adjacency)

    system_matrix, _ = stabilize_adjacency(adjacency, float(control_config["primary_c"]))
    node_gramians = single_node_gramians(system_matrix)
    epsilon = float(control_config["primary_epsilon"])
    bottom_dim = int(control_config["bottom_eigenspace_dim"])

    def geometry_of(selected: np.ndarray) -> dict:
        metrics = gramian_metrics(assemble_gramian(node_gramians, selected), epsilon=epsilon, bottom_dim=bottom_dim)
        return {name: float(metrics[name]) for name in GEOMETRY_METRICS}

    score_rows: list[dict] = []
    node_rows: list[dict] = []
    selection_rows: list[dict] = []
    geometry_rows: list[dict] = []
    diagnostic_rows: list[dict] = []

    for scheme in schemes:
        diagnostic_rows.append(representative_sets[scheme].diagnostics())
        for node_index, score in enumerate(scheme_scores[scheme]):
            node_rows.append({"scheme": scheme, "node_index": node_index, "cycle_participation": float(score)})

    baseline = schemes[0]
    for scheme_a, scheme_b in combinations(schemes, 2):
        correlation = spearmanr(scheme_scores[scheme_a], scheme_scores[scheme_b])
        score_rows.append(
            {
                "scheme_a": scheme_a,
                "scheme_b": scheme_b,
                "spearman": float(correlation.statistic),
                "spearman_p": float(correlation.pvalue),
            }
        )

    # Degree is representative-independent; it is the same comparison arm for every scheme.
    for scheme in schemes:
        correlation = spearmanr(degree, scheme_scores[scheme])
        score_rows.append(
            {
                "scheme_a": "degree_strength",
                "scheme_b": scheme,
                "spearman": float(correlation.statistic),
                "spearman_p": float(correlation.pvalue),
            }
        )

    for k in k_values:
        if k > adjacency.shape[0]:
            continue
        degree_set = top_k_nodes(degree, k)
        degree_geometry = geometry_of(degree_set)
        cycle_sets = {scheme: top_k_nodes(scheme_scores[scheme], k) for scheme in schemes}

        for scheme in schemes:
            if scheme != baseline:
                selection_rows.append(
                    {
                        "k": k,
                        "comparison": f"{baseline}_vs_{scheme}",
                        "kind": "scheme_vs_scheme",
                        "n_shared": int(np.intersect1d(cycle_sets[baseline], cycle_sets[scheme]).size),
                        "jaccard": jaccard(cycle_sets[baseline], cycle_sets[scheme]),
                    }
                )
            selection_rows.append(
                {
                    "k": k,
                    "comparison": f"degree_vs_{scheme}",
                    "kind": "degree_vs_cycle",
                    "n_shared": int(np.intersect1d(degree_set, cycle_sets[scheme]).size),
                    "jaccard": jaccard(degree_set, cycle_sets[scheme]),
                }
            )

            cycle_geometry = geometry_of(cycle_sets[scheme])
            row = {
                "k": k,
                "scheme": scheme,
                "node_indices_cycle": ";".join(str(int(index)) for index in cycle_sets[scheme]),
                "node_indices_degree": ";".join(str(int(index)) for index in degree_set),
            }
            for metric in GEOMETRY_METRICS:
                row[f"cycle_{metric}"] = cycle_geometry[metric]
                row[f"degree_{metric}"] = degree_geometry[metric]
                row[f"cycle_minus_degree_{metric}"] = cycle_geometry[metric] - degree_geometry[metric]
            geometry_rows.append(row)

    return {
        "scores": score_rows,
        "nodes": node_rows,
        "selection": selection_rows,
        "geometry": geometry_rows,
        "diagnostics": diagnostic_rows,
        "library_comparison": [library_comparison] if library_comparison else [],
    }


def summarize_scores(scores: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    """Level 1 summary: distribution of between-scheme score correlations."""
    rows = []
    for (scale, scheme_a, scheme_b), subset in scores.groupby(["scale", "scheme_a", "scheme_b"]):
        values = subset["spearman"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_median_ci(values, rng, n_boot)
        rows.append(
            {
                "scale": int(scale),
                "scheme_a": scheme_a,
                "scheme_b": scheme_b,
                "n_subjects": int(values.size),
                "mean_spearman": float(np.mean(values)),
                "median_spearman": float(np.median(values)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "min_spearman": float(np.min(values)),
                "max_spearman": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def summarize_selection(selection: pd.DataFrame) -> pd.DataFrame:
    """Level 2 summary: driver-set overlap across schemes."""
    return (
        selection.groupby(["scale", "k", "comparison", "kind"], as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            mean_n_shared=("n_shared", "mean"),
            median_n_shared=("n_shared", "median"),
            mean_jaccard=("jaccard", "mean"),
            median_jaccard=("jaccard", "median"),
            fraction_identical=("jaccard", lambda values: float(np.mean(np.isclose(values, 1.0)))),
        )
        .sort_values(["scale", "k", "comparison"])
        .reset_index(drop=True)
    )


def summarize_results(geometry: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    """Level 3 summary: the cycle-minus-degree advantage under each scheme."""
    rows = []
    for (scale, k, scheme), subset in geometry.groupby(["scale", "k", "scheme"]):
        for metric in GEOMETRY_METRICS:
            summary = summarize_paired(subset[f"cycle_minus_degree_{metric}"].to_numpy(dtype=float), rng, n_boot)
            rows.append(
                {
                    "scale": int(scale),
                    "k": int(k),
                    "scheme": scheme,
                    "metric": metric,
                    "median_cycle": float(subset[f"cycle_{metric}"].median()),
                    "median_degree": float(subset[f"degree_{metric}"].median()),
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def summarize_scheme_contrast(geometry: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    """Paired test of whether the advantage itself differs between schemes.

    A representative-robust headline means the per-subject advantage under the
    alternative scheme matches the baseline: this difference is centered near zero.
    """
    rows = []
    schemes = list(dict.fromkeys(geometry["scheme"]))
    baseline = schemes[0]
    for scheme in schemes[1:]:
        for (scale, k), subset in geometry.groupby(["scale", "k"]):
            wide = subset.pivot_table(
                index="subject_id",
                columns="scheme",
                values=[f"cycle_minus_degree_{metric}" for metric in GEOMETRY_METRICS],
            )
            for metric in GEOMETRY_METRICS:
                column = f"cycle_minus_degree_{metric}"
                if (column, baseline) not in wide.columns or (column, scheme) not in wide.columns:
                    continue
                difference = (wide[(column, scheme)] - wide[(column, baseline)]).to_numpy(dtype=float)
                rows.append(
                    {
                        "scale": int(scale),
                        "k": int(k),
                        "baseline_scheme": baseline,
                        "scheme": scheme,
                        "metric": metric,
                        **summarize_paired(difference, rng, n_boot),
                    }
                )
    return pd.DataFrame(rows)


def report(score_summary: pd.DataFrame, selection_summary: pd.DataFrame, result_summary: pd.DataFrame) -> None:
    """Print the three-level readout."""
    print("\n=== Level 1: score stability (Spearman across schemes) ===")
    print(score_summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print("\n=== Level 2: driver-set overlap ===")
    columns = ["scale", "k", "comparison", "mean_n_shared", "mean_jaccard", "fraction_identical"]
    print(selection_summary[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print("\n=== Level 3: cycle-minus-degree effective rank ===")
    effective_rank = result_summary[result_summary["metric"] == "effective_rank"]
    columns = ["scale", "k", "scheme", "median", "ci_low", "ci_high", "fraction_positive", "p_value"]
    print(effective_rank[columns].to_string(index=False, float_format=lambda value: f"{value:.4g}"))


def main() -> None:
    args = parse_args()
    config = load_config(REPO_ROOT / args.config)
    control_config = config["control"]
    ctx = RunContext.from_args(stage_name(args), args)
    output_dir = ctx.root / args.output_subdir

    schemes: tuple[str, ...] = tuple(dict.fromkeys(args.schemes))
    k_values = args.k_values or sorted({*(int(k) for k in config["analysis"]["top_k_values"]), 15})

    stack = load_adjacency_stack(config, args.scale, args.subject_limit, ctx.root)
    ctx.begin()
    ensure_directory(output_dir)
    written: list[Path] = []

    def save(frame: pd.DataFrame, name: str) -> None:
        path = output_dir / name
        write_csv(frame, path)
        written.append(path)

    rng = make_rng(int(config["analysis"]["random_seed"]) + 101)


    score_rows: list[dict] = []
    node_rows: list[dict] = []
    selection_rows: list[dict] = []
    geometry_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    library_rows: list[dict] = []

    for subject_index in tqdm(range(stack.shape[0]), desc=f"Scale {args.scale}"):
        subject_id = f"sub-{subject_index:03d}"
        outputs = run_subject(
            adjacency=stack[subject_index],
            schemes=schemes,
            k_values=k_values,
            control_config=control_config,
            weight_mode=args.weight_mode,
            threshold_mode=args.threshold_mode,
        )
        stamp = {"subject_id": subject_id, "scale": args.scale}
        score_rows.extend({**stamp, **row} for row in outputs["scores"])
        node_rows.extend({**stamp, **row} for row in outputs["nodes"])
        selection_rows.extend({**stamp, **row} for row in outputs["selection"])
        geometry_rows.extend({**stamp, **row} for row in outputs["geometry"])
        diagnostic_rows.extend({**stamp, **row} for row in outputs["diagnostics"])
        library_rows.extend({**stamp, **row} for row in outputs["library_comparison"])

    scores = pd.DataFrame(score_rows)
    nodes = pd.DataFrame(node_rows)
    selection = pd.DataFrame(selection_rows)
    geometry = pd.DataFrame(geometry_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)

    score_summary = summarize_scores(scores[scores["scheme_a"] != "degree_strength"], rng, args.n_boot)
    selection_summary = summarize_selection(selection)
    result_summary = summarize_results(geometry, rng, args.n_boot)
    contrast_summary = summarize_scheme_contrast(geometry, rng, args.n_boot)
    diagnostic_summary = (
        diagnostics.groupby(["scale", "scheme"], as_index=False)
        .mean(numeric_only=True)
        .assign(n_subjects=diagnostics.groupby(["scale", "scheme"])["subject_id"].nunique().to_numpy())
    )

    settings = pd.DataFrame(
        [
            {
                "scale": args.scale,
                "n_subjects": int(stack.shape[0]),
                "schemes": ";".join(schemes),
                "k_values": ";".join(str(k) for k in k_values),
                "weight_mode": args.weight_mode,
                "threshold_mode": args.threshold_mode,
                "primary_c": float(control_config["primary_c"]),
                "primary_epsilon": float(control_config["primary_epsilon"]),
                "bottom_eigenspace_dim": int(control_config["bottom_eigenspace_dim"]),
                "n_boot": args.n_boot,
            }
        ]
    )

    save(nodes, f"node_scores_by_scheme_scale-{args.scale}.csv")
    save(scores, f"score_correlations_scale-{args.scale}.csv")
    save(score_summary, f"score_correlation_summary_scale-{args.scale}.csv")
    save(selection, f"selection_overlap_scale-{args.scale}.csv")
    save(selection_summary, f"selection_overlap_summary_scale-{args.scale}.csv")
    save(geometry, f"geometry_by_scheme_scale-{args.scale}.csv")
    save(result_summary, f"result_level_summary_scale-{args.scale}.csv")
    save(contrast_summary, f"scheme_contrast_summary_scale-{args.scale}.csv")
    save(diagnostics, f"representative_diagnostics_scale-{args.scale}.csv")
    save(diagnostic_summary, f"representative_diagnostics_summary_scale-{args.scale}.csv")
    save(settings, f"run_settings_scale-{args.scale}.csv")
    if library_rows:
        library_comparison = pd.DataFrame(library_rows)
        save(library_comparison, f"library_comparison_scale-{args.scale}.csv")
        print("\n=== Level 0: do Ripser and GUDHI agree? ===")
        print(
            library_comparison.drop(columns=["subject_id"])
            .groupby("scale")
            .mean(numeric_only=True)
            .to_string(float_format=lambda value: f"{value:.4g}")
        )

    report(score_summary, selection_summary, result_summary)
    ctx.finish(written)
    print(f"\nOutputs written to {output_dir}")


if __name__ == "__main__":
    main()
