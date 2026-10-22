"""Target-state construction for control-energy analyses."""

from __future__ import annotations

from functools import lru_cache

import nibabel as nib
import numpy as np
import pandas as pd

from .atlas import EXCLUDED_LABELS, SCALE_TO_RESOLUTION, region_set_mask
from . import synthetic
from .paths import resolve_repo_path


LANDMARK_DEFINITIONS = {
    "landmark_v1": {
        "label": "V1 / pericalcarine",
        "base_regions": ("pericalcarine",),
        "axis_class": "Concentrated landmark",
        "axis_position": 1,
    },
    "landmark_m1": {
        "label": "M1 / precentral",
        "base_regions": ("precentral",),
        "axis_class": "Concentrated landmark",
        "axis_position": 1,
    },
    "landmark_pcc_precuneus": {
        "label": "Posterior cingulate / precuneus",
        "base_regions": ("posteriorcingulate", "precuneus"),
        "axis_class": "Concentrated landmark",
        "axis_position": 1,
    },
    "landmark_lpfc": {
        "label": "Lateral prefrontal cortex",
        "base_regions": ("rostralmiddlefrontal", "caudalmiddlefrontal"),
        "axis_class": "Concentrated landmark",
        "axis_position": 1,
    },
}

MODULE_TARGETS = {
    "association_module": {
        "label": "Association module",
        "axis_class": "Intermediate module",
        "axis_position": 3,
    },
    "occipital_module": {
        "label": "Occipital module",
        "axis_class": "Intermediate module",
        "axis_position": 3,
    },
    "association_vs_occipital": {
        "label": "Association vs occipital",
        "axis_class": "Intermediate contrast",
        "axis_position": 3,
    },
}

DISTRIBUTED_TARGETS = {
    "principal_gradient": {
        "label": "Principal gradient",
        "axis_class": "Distributed",
        "axis_position": 4,
    },
    "bipolar_gradient": {
        "label": "Bipolar gradient",
        "axis_class": "Distributed",
        "axis_position": 4,
    },
    "uniform_cortical": {
        "label": "Uniform cortical",
        "axis_class": "Distributed",
        "axis_position": 4,
    },
}

SUBCORTICAL_KEYWORDS = (
    "accumbens",
    "amygdala",
    "brainstem",
    "caudate",
    "hippocampus",
    "pallidum",
    "putamen",
    "thalamus",
    "ventraldc",
)


def build_targets(
    reference_adjacency: np.ndarray,
    labels: pd.DataFrame,
    target_config: dict,
    anatomy_config: dict,
    atlas_dir: str,
) -> dict[str, np.ndarray]:
    """Create fixed target states along a concentrated-to-distributed axis."""
    association_mask = region_set_mask(labels, anatomy_config["association_regions"])
    occipital_mask = region_set_mask(labels, anatomy_config["occipital_regions"])
    cortical = cortical_mask(labels)

    targets: dict[str, np.ndarray] = {}

    for target_name, definition in LANDMARK_DEFINITIONS.items():
        targets[target_name] = normalized_mask_vector(base_region_mask(labels, definition["base_regions"]))

    for target_name, target_vector in single_parcel_targets(labels).items():
        targets[target_name] = target_vector

    targets.update(
        {
            "association_module": normalized_mask_vector(association_mask),
            "occipital_module": normalized_mask_vector(occipital_mask),
            "association_vs_occipital": normalized_signed_contrast(association_mask, occipital_mask),
        }
    )

    gradient = margulies_principal_gradient(labels, atlas_dir)
    if np.mean(gradient[association_mask]) < np.mean(gradient[occipital_mask]):
        gradient = -gradient
    targets["principal_gradient"] = normalize_vector(gradient)
    targets["bipolar_gradient"] = normalize_vector(np.sign(gradient))
    targets["uniform_cortical"] = normalized_mask_vector(cortical)
    return targets


def build_target_metadata(labels: pd.DataFrame, targets: dict[str, np.ndarray]) -> pd.DataFrame:
    """Describe target families and anatomical support for downstream summaries."""
    metadata_rows: list[dict[str, object]] = []
    for target_name, definition in LANDMARK_DEFINITIONS.items():
        support = support_labels(labels, targets[target_name])
        metadata_rows.append(
            {
                "target_name": target_name,
                "target_label": definition["label"],
                "target_family": "landmark",
                "target_axis_class": definition["axis_class"],
                "axis_position": definition["axis_position"],
                "node_index": np.nan,
                "hemi": "bilateral",
                "base_region": ";".join(definition["base_regions"]),
                "region_label": ";".join(support),
                "n_support_nodes": len(support),
            }
        )

    for _, row in labels.iterrows():
        node_index = int(row["node_index"])
        family = "single_subcortical_parcel" if is_subcortical_row(row) else "single_cortical_parcel"
        target_name = single_parcel_target_name(node_index, family)
        if target_name not in targets:
            continue
        metadata_rows.append(
            {
                "target_name": target_name,
                "target_label": str(row["region_label"]),
                "target_family": family,
                "target_axis_class": "Single-parcel baseline",
                "axis_position": 2,
                "node_index": node_index,
                "hemi": str(row["hemi"]),
                "base_region": str(row["base_region"]),
                "region_label": str(row["region_label"]),
                "n_support_nodes": 1,
            }
        )

    for target_name, definition in MODULE_TARGETS.items():
        support = support_labels(labels, targets[target_name])
        metadata_rows.append(
            {
                "target_name": target_name,
                "target_label": definition["label"],
                "target_family": "module",
                "target_axis_class": definition["axis_class"],
                "axis_position": definition["axis_position"],
                "node_index": np.nan,
                "hemi": "bilateral",
                "base_region": "",
                "region_label": ";".join(support),
                "n_support_nodes": len(support),
            }
        )

    for target_name, definition in DISTRIBUTED_TARGETS.items():
        support = support_labels(labels, targets[target_name])
        metadata_rows.append(
            {
                "target_name": target_name,
                "target_label": definition["label"],
                "target_family": "distributed",
                "target_axis_class": definition["axis_class"],
                "axis_position": definition["axis_position"],
                "node_index": np.nan,
                "hemi": "bilateral",
                "base_region": "",
                "region_label": ";".join(support),
                "n_support_nodes": len(support),
            }
        )

    return pd.DataFrame(metadata_rows)


def single_parcel_targets(labels: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return one unit target per included atlas parcel."""
    targets = {}
    for _, row in labels.iterrows():
        node_index = int(row["node_index"])
        family = "single_subcortical_parcel" if is_subcortical_row(row) else "single_cortical_parcel"
        targets[single_parcel_target_name(node_index, family)] = normalized_sparse_vector(len(labels), np.array([node_index]))
    return targets


def single_parcel_target_name(node_index: int, family: str) -> str:
    """Create a stable target name for single-parcel baseline targets."""
    prefix = "single_subcortical" if family == "single_subcortical_parcel" else "single_cortical"
    return f"{prefix}_node_{node_index:03d}"


def base_region_mask(labels: pd.DataFrame, base_regions: tuple[str, ...]) -> np.ndarray:
    """Select all parcels whose Cammoun base region is in the requested set."""
    mask = labels["base_region"].isin(base_regions).to_numpy()
    if not np.any(mask):
        raise ValueError(f"Requested landmark regions are absent: {', '.join(base_regions)}")
    hemispheres = set(labels.loc[mask, "hemi"].astype(str))
    if not {"L", "R"}.issubset(hemispheres):
        raise ValueError(f"Requested landmark is not bilateral: {', '.join(base_regions)}")
    return mask


def cortical_mask(labels: pd.DataFrame) -> np.ndarray:
    """Identify cortical parcels, keeping Cammoun-only atlases fully cortical."""
    return np.array([not is_subcortical_row(row) for _, row in labels.iterrows()], dtype=bool)


def is_subcortical_row(row: pd.Series) -> bool:
    """Infer whether an atlas row is subcortical from available label fields."""
    group = str(row.get("anatomical_group", "")).lower()
    fields = " ".join(str(row.get(field, "")).lower() for field in ("atlas_label", "region_label", "base_region"))
    return group == "subcortical" or any(keyword in fields for keyword in SUBCORTICAL_KEYWORDS)


def support_labels(labels: pd.DataFrame, target_vector: np.ndarray) -> list[str]:
    """Return atlas labels with nonzero support in a target vector."""
    support = np.flatnonzero(np.abs(np.asarray(target_vector, dtype=float)) > 1.0e-12)
    return labels.loc[support, "region_label"].astype(str).tolist()


def margulies_principal_gradient(labels: pd.DataFrame, atlas_dir: str) -> np.ndarray:
    """Project the Margulies et al. (2016) principal functional gradient onto atlas parcels.

    The published fcgradient01 map (fsLR-32k) is resampled to fsaverage5 by
    nearest-neighbor matching on the spherical registration and averaged within each
    Cammoun parcel. Unlike the leading nontrivial eigenvector of the structural
    affinity, which is dominated by the interhemispheric split, this yields the genuine
    unimodal-to-transmodal gradient of macroscale cortical organization.
    """
    if synthetic.is_synthetic(atlas_dir):
        return synthetic.gradient(labels)
    from neuromaps.datasets import fetch_annotation

    scale = int(labels["scale"].iloc[0])
    resolution = SCALE_TO_RESOLUTION[scale]
    atlas_path = resolve_repo_path(atlas_dir)
    annotation = fetch_annotation(source="margulies2016", desc="fcgradient01", space="fsLR", den="32k")

    parcel_values: dict[tuple[str, str], float] = {}
    for hemi_index, hemi in enumerate(("L", "R")):
        vertex_gradient = np.asarray(nib.load(str(annotation[hemi_index])).agg_data(), dtype=float)
        gradient_fsaverage5 = vertex_gradient[_fsaverage5_from_fslr_index(hemi)]
        annot_path = atlas_path / f"atl-Cammoun2012_space-fsaverage5_res-{resolution}_hemi-{hemi}_deterministic.annot"
        vertex_labels, _, names = nib.freesurfer.read_annot(str(annot_path))
        for idx, raw_name in enumerate(names):
            name = raw_name.decode()
            if name in EXCLUDED_LABELS:
                continue
            mask = vertex_labels == idx
            parcel_values[(hemi, name)] = float(gradient_fsaverage5[mask].mean()) if mask.any() else 0.0

    return labels.apply(
        lambda row: parcel_values[(str(row["hemi"]), str(row["atlas_label"]))], axis=1
    ).to_numpy(dtype=float)


@lru_cache(maxsize=None)
def _fsaverage5_from_fslr_index(hemi: str) -> np.ndarray:
    """Nearest fsLR-32k vertex for each fsaverage5 vertex, matched on the sphere."""
    from neuromaps.datasets import fetch_atlas
    from scipy.spatial import cKDTree

    source_sphere = _sphere_coordinates(fetch_atlas("fsLR", "32k")["sphere"], hemi)
    target_sphere = _sphere_coordinates(fetch_atlas("fsaverage", "10k")["sphere"], hemi)
    _, nearest = cKDTree(source_sphere).query(target_sphere)
    return nearest


def _sphere_coordinates(sphere_paths, hemi: str) -> np.ndarray:
    """Load vertex coordinates for the requested hemisphere from a set of sphere surfaces."""
    path = next(str(p) for p in sphere_paths if f"hemi-{hemi}" in str(p))
    return np.asarray(nib.load(path).agg_data()[0], dtype=float)


def masked_top_indices(scores: np.ndarray, mask: np.ndarray, count: int) -> np.ndarray:
    """Select the highest-scoring nodes inside a mask."""
    masked_indices = np.where(mask)[0]
    if masked_indices.size == 0:
        raise ValueError("Requested target mask is empty for this atlas scale.")
    local_order = masked_indices[np.argsort(-scores[masked_indices], kind="mergesort")]
    return local_order[:count]


def normalized_mask_vector(mask: np.ndarray) -> np.ndarray:
    """Normalize an indicator vector to unit Euclidean norm."""
    return normalize_vector(mask.astype(float))


def normalized_signed_contrast(positive_mask: np.ndarray, negative_mask: np.ndarray) -> np.ndarray:
    """Build a signed contrast vector and normalize it."""
    vector = positive_mask.astype(float) - negative_mask.astype(float)
    return normalize_vector(vector)


def normalized_sparse_vector(length: int, indices: np.ndarray) -> np.ndarray:
    """Build a sparse unit vector over selected indices."""
    vector = np.zeros(length, dtype=float)
    vector[np.asarray(indices, dtype=int)] = 1.0
    return normalize_vector(vector)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit Euclidean norm."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector target.")
    return np.asarray(vector, dtype=float) / norm
