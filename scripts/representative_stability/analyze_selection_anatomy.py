#!/usr/bin/env python
"""Where do the two representative schemes put their drivers?

The stability run establishes that Ripser-based and length-minimized `CP_i` pick
largely disjoint top-k nodes. Disjoint node *identities* is not the same as
disjoint *anatomy*: two schemes can select different parcels of the same gyrus, or
the same lobe, and still share zero node indices. This script asks whether the
selections land in similar areas, at three granularities:

- anatomical group / lobe and hemisphere composition, with enrichment against the
  parcel-count null,
- region-level agreement (same base region, different subparcel),
- physical proximity on the fsaverage5 surface: chamfer distance between the two
  schemes' driver-set centroids, referenced to a random-set null.

Reads the outputs of `run_representative_stability.py` from
``<run>/<--results-subdir>/`` and writes to ``<run>/<--results-subdir>/anatomy/``;
recomputes no topology. The run is chosen with ``--run <name>`` (default $HOC_RUN
or "main"); the legacy ``results`` tree is read-only and cannot be used. Each
scale is recorded as its own stage in the run manifest; rerunning a finished
scale needs ``--overwrite``.

Example
-------
    python scripts/representative_stability/analyze_selection_anatomy.py --run main
    python scripts/representative_stability/analyze_selection_anatomy.py --run main --k-values 5 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from itertools import combinations

from scipy.stats import chi2_contingency, wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
for extra_path in (str(SRC_ROOT), str(SCRIPT_DIR)):
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from hoc.atlas import EXCLUDED_LABELS, SCALE_TO_RESOLUTION, build_label_table
from hoc.config import load_config
from hoc.io import write_csv
from hoc.paths import ensure_directory, resolve_repo_path
from hoc.runs import RunContext, add_run_arguments
from hoc.utils import make_rng

# Ordering used wherever arms are listed; any scheme absent from the run is skipped.
ARM_ORDER = ("ripser", "geodesic", "gudhi", "gudhi_geodesic", "degree")


def ordered_arms(present: Iterable[str]) -> list[str]:
    """Arms in a stable display order, unknown names appended alphabetically."""
    present = set(present)
    known = [arm for arm in ARM_ORDER if arm in present]
    return known + sorted(present - set(known))


def arm_pairs(arms: Iterable[str]) -> list[tuple[str, str]]:
    """All unordered arm pairs, in display order."""
    return list(combinations(ordered_arms(arms), 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/default.toml", help="Path to the TOML config file.")
    parser.add_argument("--scale", type=int, default=114, help="Parcellation scale to analyze.")
    parser.add_argument(
        "--results-subdir",
        default="representative_stability",
        help="Subdirectory of the run directory holding the stability-run outputs.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=None,
        help="Driver-set sizes to analyze (default: every k present in the stability run).",
    )
    parser.add_argument("--n-null", type=int, default=500, help="Random driver sets per subject for the spatial null.")
    parser.add_argument(
        "--no-surface",
        action="store_true",
        help="Skip the surface-geometry step (drops the spatial-proximity readout).",
    )
    add_run_arguments(parser)
    return parser.parse_args()


def stage_name(args: argparse.Namespace) -> str:
    """Manifest stage name: one stage per input subdirectory and scale."""
    stage = "representative_stability/analyze_selection_anatomy"
    if args.results_subdir != "representative_stability":
        stage += f":{args.results_subdir}"
    return stage + f":scale-{args.scale}"


def parcel_centroids(scale: int, atlas_dir: str | Path) -> np.ndarray:
    """Centroid coordinates of every parcel on the fsaverage5 pial surface.

    Built the same way as the surface figures: vertices are assigned to nodes through
    the Cammoun annotation, then averaged per node.
    """
    from nilearn import datasets

    pial = datasets.load_fsaverage(mesh="fsaverage5")["pial"]
    atlas_path = resolve_repo_path(atlas_dir)
    resolution = SCALE_TO_RESOLUTION[scale]

    centroids: list[np.ndarray] = []
    for hemi, part in (("L", "left"), ("R", "right")):
        coords = np.asarray(pial.parts[part].coordinates, dtype=float)
        annot_path = atlas_path / f"atl-Cammoun2012_space-fsaverage5_res-{resolution}_hemi-{hemi}_deterministic.annot"
        vertex_labels, _, names = nib.freesurfer.read_annot(str(annot_path))
        for atlas_index, raw_name in enumerate(names):
            if raw_name.decode() in EXCLUDED_LABELS:
                continue
            member = vertex_labels == atlas_index
            if not np.any(member):
                raise ValueError(f"Parcel {raw_name.decode()} ({hemi}) has no vertices.")
            centroids.append(coords[member].mean(axis=0))
    return np.vstack(centroids)


def parse_node_indices(value: str) -> np.ndarray:
    """Parse the semicolon-joined node-index strings written by the stability run."""
    text = str(value).strip()
    if not text:
        return np.array([], dtype=int)
    return np.asarray([int(part) for part in text.split(";")], dtype=int)


def load_driver_sets(geometry: pd.DataFrame) -> pd.DataFrame:
    """Reshape the per-scheme geometry table into one row per (subject, k, arm)."""
    rows = []
    for _, record in geometry.iterrows():
        rows.append(
            {
                "subject_id": record["subject_id"],
                "k": int(record["k"]),
                "arm": record["scheme"],
                "nodes": parse_node_indices(record["node_indices_cycle"]),
            }
        )
    degree_rows = (
        geometry.drop_duplicates(subset=["subject_id", "k"])[["subject_id", "k", "node_indices_degree"]]
        .assign(arm="degree")
        .rename(columns={"node_indices_degree": "nodes"})
    )
    for _, record in degree_rows.iterrows():
        rows.append(
            {
                "subject_id": record["subject_id"],
                "k": int(record["k"]),
                "arm": "degree",
                "nodes": parse_node_indices(record["nodes"]),
            }
        )
    return pd.DataFrame(rows)


def chamfer_distance(centroids: np.ndarray, set_a: np.ndarray, set_b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour centroid distance between two driver sets.

    Zero when the sets coincide; small when they pick different parcels of the same
    territory; large when they pick different territories.
    """
    if set_a.size == 0 or set_b.size == 0:
        return np.nan
    pairwise = np.linalg.norm(centroids[set_a][:, None, :] - centroids[set_b][None, :, :], axis=2)
    return float(0.5 * (pairwise.min(axis=1).mean() + pairwise.min(axis=0).mean()))


def dispersion(centroids: np.ndarray, nodes: np.ndarray) -> float:
    """Mean pairwise centroid distance within one driver set."""
    if nodes.size < 2:
        return np.nan
    pairwise = np.linalg.norm(centroids[nodes][:, None, :] - centroids[nodes][None, :, :], axis=2)
    upper = pairwise[np.triu_indices(nodes.size, k=1)]
    return float(upper.mean())


def composition_vector(labels: pd.DataFrame, nodes: np.ndarray, column: str, categories: list[str]) -> np.ndarray:
    """Count how many selected nodes fall in each category."""
    values = labels.loc[nodes, column]
    counts = values.value_counts()
    return np.asarray([float(counts.get(category, 0)) for category in categories])


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Cosine similarity between two composition vectors."""
    norm_a, norm_b = np.linalg.norm(vector_a), np.linalg.norm(vector_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return np.nan
    return float(vector_a @ vector_b / (norm_a * norm_b))


def signed_rank(values: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test against zero."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0 or np.allclose(array, 0.0):
        return np.nan, 1.0
    statistic, p_value = wilcoxon(array, zero_method="wilcox", alternative="two-sided")
    return float(statistic), float(p_value)


def selection_frequency(driver_sets: pd.DataFrame, labels: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    """How often each node is selected, per arm and k."""
    n_nodes = labels.shape[0]
    rows = []
    for k in k_values:
        subset = driver_sets[driver_sets["k"] == k]
        for arm, arm_subset in subset.groupby("arm"):
            counts = np.zeros(n_nodes, dtype=float)
            for nodes in arm_subset["nodes"]:
                counts[nodes] += 1.0
            n_subjects = int(arm_subset.shape[0])
            for node_index in range(n_nodes):
                rows.append(
                    {
                        "k": k,
                        "arm": arm,
                        "node_index": node_index,
                        "region_label": labels.loc[node_index, "region_label"],
                        "base_region": labels.loc[node_index, "base_region"],
                        "anatomical_group": labels.loc[node_index, "anatomical_group"],
                        "hemi": labels.loc[node_index, "hemi"],
                        "n_selected": counts[node_index],
                        "fraction_selected": counts[node_index] / n_subjects if n_subjects else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def group_enrichment(frequency: pd.DataFrame, labels: pd.DataFrame, column: str) -> pd.DataFrame:
    """Observed vs parcel-count-expected share of selections per category."""
    parcel_share = labels[column].value_counts(normalize=True)
    rows = []
    for (k, arm), subset in frequency.groupby(["k", "arm"]):
        total = subset["n_selected"].sum()
        for category, category_subset in subset.groupby(column):
            observed = float(category_subset["n_selected"].sum())
            expected_share = float(parcel_share.get(category, 0.0))
            observed_share = observed / total if total else np.nan
            rows.append(
                {
                    "k": int(k),
                    "arm": arm,
                    column: category,
                    "n_selections": observed,
                    "observed_share": observed_share,
                    "expected_share": expected_share,
                    "enrichment": observed_share / expected_share if expected_share > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def composition_contrast(enrichment: pd.DataFrame, column: str) -> pd.DataFrame:
    """Chi-square test of whether two arms distribute selections differently."""
    rows = []
    for k, subset in enrichment.groupby("k"):
        table = subset.pivot_table(index="arm", columns=column, values="n_selections", fill_value=0.0)
        for arm_a, arm_b in arm_pairs(table.index):
            pair = table.loc[[arm_a, arm_b]].to_numpy(dtype=float)
            pair = pair[:, pair.sum(axis=0) > 0]
            statistic, p_value, degrees_of_freedom, _ = chi2_contingency(pair)
            total = pair.sum()
            rows.append(
                {
                    "k": int(k),
                    "level": column,
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "chi2": float(statistic),
                    "dof": int(degrees_of_freedom),
                    "p_value": float(p_value),
                    "cramers_v": float(np.sqrt(statistic / total)) if total > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def subject_agreement(
    driver_sets: pd.DataFrame,
    labels: pd.DataFrame,
    centroids: np.ndarray | None,
    k_values: list[int],
    rng: np.random.Generator,
    n_null: int,
) -> pd.DataFrame:
    """Per-subject agreement between arms at node, region, lobe, and spatial level."""
    n_nodes = labels.shape[0]
    groups = sorted(labels["anatomical_group"].unique())
    regions = sorted(labels["base_region"].unique())
    hemis = sorted(labels["hemi"].unique())

    indexed = {
        (record["subject_id"], record["k"], record["arm"]): record["nodes"]
        for _, record in driver_sets.iterrows()
    }
    subject_ids = sorted({key[0] for key in indexed})
    pairs = arm_pairs(driver_sets["arm"].unique())

    rows = []
    for k in k_values:
        # One null per (k) reused across subjects: random sets of the same size.
        null_sets = [rng.choice(n_nodes, size=k, replace=False) for _ in range(n_null)]
        for subject_id in subject_ids:
            for arm_a, arm_b in pairs:
                set_a = indexed.get((subject_id, k, arm_a))
                set_b = indexed.get((subject_id, k, arm_b))
                if set_a is None or set_b is None:
                    continue
                shared_nodes = int(np.intersect1d(set_a, set_b).size)
                row = {
                    "k": k,
                    "subject_id": subject_id,
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "n_shared_nodes": shared_nodes,
                    "fraction_shared_nodes": shared_nodes / k,
                    "n_shared_base_regions": int(
                        np.intersect1d(
                            labels.loc[set_a, "base_region"].to_numpy(),
                            labels.loc[set_b, "base_region"].to_numpy(),
                        ).size
                    ),
                    "group_cosine": cosine_similarity(
                        composition_vector(labels, set_a, "anatomical_group", groups),
                        composition_vector(labels, set_b, "anatomical_group", groups),
                    ),
                    "region_cosine": cosine_similarity(
                        composition_vector(labels, set_a, "base_region", regions),
                        composition_vector(labels, set_b, "base_region", regions),
                    ),
                    "hemi_cosine": cosine_similarity(
                        composition_vector(labels, set_a, "hemi", hemis),
                        composition_vector(labels, set_b, "hemi", hemis),
                    ),
                }
                if centroids is not None:
                    observed = chamfer_distance(centroids, set_a, set_b)
                    null_values = np.asarray(
                        [chamfer_distance(centroids, set_a, null_set) for null_set in null_sets],
                        dtype=float,
                    )
                    row.update(
                        {
                            "chamfer_mm": observed,
                            "chamfer_null_mean_mm": float(np.mean(null_values)),
                            "chamfer_minus_null_mm": observed - float(np.mean(null_values)),
                            "chamfer_null_percentile": float(np.mean(null_values <= observed)),
                            "dispersion_a_mm": dispersion(centroids, set_a),
                            "dispersion_b_mm": dispersion(centroids, set_b),
                        }
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_agreement(agreement: pd.DataFrame) -> pd.DataFrame:
    """Cohort medians and tests for each arm pair and k."""
    metrics = [
        column
        for column in (
            "fraction_shared_nodes",
            "n_shared_base_regions",
            "group_cosine",
            "region_cosine",
            "hemi_cosine",
            "chamfer_mm",
            "chamfer_null_mean_mm",
            "chamfer_minus_null_mm",
            "chamfer_null_percentile",
        )
        if column in agreement.columns
    ]
    rows = []
    for (k, arm_a, arm_b), subset in agreement.groupby(["k", "arm_a", "arm_b"]):
        row = {"k": int(k), "arm_a": arm_a, "arm_b": arm_b, "n_subjects": int(subset.shape[0])}
        for metric in metrics:
            row[f"median_{metric}"] = float(subset[metric].median())
        if "chamfer_minus_null_mm" in subset:
            _, p_value = signed_rank(subset["chamfer_minus_null_mm"].to_numpy(dtype=float))
            row["p_chamfer_vs_null"] = p_value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["k", "arm_a", "arm_b"]).reset_index(drop=True)


def score_profile(node_scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Cohort-mean `CP_i` per anatomical group, per scheme (whole-score view)."""
    lookup = labels[["anatomical_group", "hemi"]].rename_axis("node_index").reset_index()
    merged = node_scores.merge(lookup, on="node_index", how="left")
    return (
        merged.groupby(["scheme", "anatomical_group"], as_index=False)
        .agg(
            mean_cycle_participation=("cycle_participation", "mean"),
            median_cycle_participation=("cycle_participation", "median"),
            n_observations=("cycle_participation", "size"),
        )
        .sort_values(["scheme", "mean_cycle_participation"], ascending=[True, False])
        .reset_index(drop=True)
    )


def top_regions(frequency: pd.DataFrame, k: int, n_top: int = 10) -> pd.DataFrame:
    """Most frequently selected parcels per arm at one k."""
    subset = frequency[frequency["k"] == k]
    return (
        subset.sort_values(["arm", "fraction_selected"], ascending=[True, False])
        .groupby("arm")
        .head(n_top)
        .reset_index(drop=True)[["arm", "region_label", "anatomical_group", "fraction_selected"]]
    )


def report(
    group_table: pd.DataFrame,
    contrast: pd.DataFrame,
    agreement_summary: pd.DataFrame,
    profile: pd.DataFrame,
    frequent: pd.DataFrame,
    report_k: int,
) -> None:
    """Print the anatomical readout."""
    formatter = lambda value: f"{value:.3f}"  # noqa: E731

    print(f"\n=== Lobe composition of the top-{report_k} driver sets (enrichment vs parcel-count null) ===")
    wide = (
        group_table[group_table["k"] == report_k]
        .pivot_table(index="anatomical_group", columns="arm", values="enrichment")
        .reindex(columns=ordered_arms(group_table["arm"].unique()))
    )
    print(wide.to_string(float_format=formatter))

    print("\n=== Are the compositions different? (chi-square over lobes) ===")
    print(contrast[contrast["k"] == report_k].to_string(index=False, float_format=lambda value: f"{value:.4g}"))

    print("\n=== Per-subject agreement between arms ===")
    columns = [
        column
        for column in (
            "k",
            "arm_a",
            "arm_b",
            "median_fraction_shared_nodes",
            "median_n_shared_base_regions",
            "median_group_cosine",
            "median_chamfer_mm",
            "median_chamfer_null_mean_mm",
            "p_chamfer_vs_null",
        )
        if column in agreement_summary.columns
    ]
    print(agreement_summary[columns].to_string(index=False, float_format=lambda value: f"{value:.4g}"))

    print("\n=== Cohort-mean CP_i by lobe, per scheme (all nodes, not just selected) ===")
    print(
        profile.pivot_table(index="anatomical_group", columns="scheme", values="mean_cycle_participation").to_string(
            float_format=formatter
        )
    )

    print(f"\n=== Most frequently selected parcels at k = {report_k} ===")
    print(frequent.to_string(index=False, float_format=formatter))


def main() -> None:
    args = parse_args()
    config = load_config(REPO_ROOT / args.config)
    ctx = RunContext.from_args(stage_name(args), args)
    results_dir = ctx.root / args.results_subdir
    output_dir = results_dir / "anatomy"
    ctx.begin()

    geometry = pd.read_csv(results_dir / f"geometry_by_scheme_scale-{args.scale}.csv")
    node_scores = pd.read_csv(results_dir / f"node_scores_by_scheme_scale-{args.scale}.csv")
    labels = build_label_table(scale=args.scale, atlas_dir=config["paths"]["atlas_dir"]).set_index(
        "node_index", drop=False
    )

    k_values = args.k_values or sorted(geometry["k"].unique().tolist())
    driver_sets = load_driver_sets(geometry)
    driver_sets = driver_sets[driver_sets["k"].isin(k_values)]

    centroids = None if args.no_surface else parcel_centroids(args.scale, config["paths"]["atlas_dir"])
    rng = make_rng(int(config["analysis"]["random_seed"]) + 202)

    frequency = selection_frequency(driver_sets, labels, k_values)
    group_table = group_enrichment(frequency, labels, "anatomical_group")
    region_table = group_enrichment(frequency, labels, "base_region")
    hemi_table = group_enrichment(frequency, labels, "hemi")
    contrast = pd.concat(
        [composition_contrast(group_table, "anatomical_group"), composition_contrast(hemi_table, "hemi")],
        ignore_index=True,
    )
    agreement = subject_agreement(driver_sets, labels, centroids, k_values, rng, args.n_null)
    agreement_summary = summarize_agreement(agreement)
    profile = score_profile(node_scores, labels)

    report_k = max(k_values)
    frequent = top_regions(frequency, report_k)

    suffix = f"scale-{args.scale}"
    ensure_directory(output_dir)
    written: list[Path] = []

    def save(frame: pd.DataFrame, name: str) -> None:
        path = output_dir / name
        write_csv(frame, path)
        written.append(path)

    save(frequency, f"selection_frequency_{suffix}.csv")
    save(group_table, f"group_enrichment_{suffix}.csv")
    save(region_table, f"region_enrichment_{suffix}.csv")
    save(hemi_table, f"hemisphere_enrichment_{suffix}.csv")
    save(contrast, f"composition_contrast_{suffix}.csv")
    save(agreement, f"subject_agreement_{suffix}.csv")
    save(agreement_summary, f"subject_agreement_summary_{suffix}.csv")
    save(profile, f"score_profile_by_group_{suffix}.csv")
    save(frequent, f"top_selected_parcels_k-{report_k}_{suffix}.csv")

    report(group_table, contrast, agreement_summary, profile, frequent, report_k)
    ctx.finish(written)
    print(f"\nOutputs written to {output_dir}")



if __name__ == "__main__":
    main()
