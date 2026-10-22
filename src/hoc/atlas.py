"""Atlas label handling and anatomical grouping."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from . import synthetic
from .paths import resolve_repo_path


SCALE_TO_RESOLUTION = {
    68: "033",
    114: "060",
    219: "125",
    448: "250",
    1000: "500",
}

EXCLUDED_LABELS = {"unknown", "corpuscallosum"}


def build_label_table(scale: int, atlas_dir: str | Path) -> pd.DataFrame:
    """Load atlas labels for a Cammoun scale from FreeSurfer annotation files."""
    if synthetic.is_synthetic(atlas_dir):
        labels = pd.DataFrame(synthetic.label_rows(scale))
        labels["anatomical_group"] = labels["base_region"].map(classify_region_group)
        return labels
    atlas_path = resolve_repo_path(atlas_dir)
    resolution = SCALE_TO_RESOLUTION[scale]
    rows = []
    node_index = 0
    for hemi in ("L", "R"):
        annot_path = atlas_path / f"atl-Cammoun2012_space-fsaverage5_res-{resolution}_hemi-{hemi}_deterministic.annot"
        _, _, names = nib.freesurfer.read_annot(str(annot_path))
        for raw_name in names:
            name = raw_name.decode()
            if name in EXCLUDED_LABELS:
                continue
            base_name = strip_subparcel_suffix(name)
            rows.append(
                {
                    "node_index": node_index,
                    "scale": scale,
                    "hemi": hemi,
                    "atlas_label": name,
                    "region_label": f"{hemi}_{name}",
                    "base_region": base_name,
                }
            )
            node_index += 1
    labels = pd.DataFrame(rows)
    labels["anatomical_group"] = labels["base_region"].map(classify_region_group)
    return labels


def strip_subparcel_suffix(name: str) -> str:
    """Remove Cammoun subparcel suffixes like '_3'."""
    if "_" not in name:
        return name
    stem, suffix = name.rsplit("_", 1)
    return stem if suffix.isdigit() else name


def classify_region_group(region_name: str) -> str:
    """Assign a coarse anatomical group from region keywords."""
    region_name = region_name.lower()
    # precuneus first: "cuneus" is a substring of it
    if "precuneus" in region_name:
        return "parietal"
    if any(token in region_name for token in ("cuneus", "lingual", "pericalcarine", "lateraloccipital", "occipital")):
        return "occipital"
    if "cingulate" in region_name:
        return "cingulate"
    if "insula" in region_name:
        return "insula"
    if any(token in region_name for token in ("frontal", "orbital", "precentral", "paracentral")):
        return "frontal"
    if any(token in region_name for token in ("parietal", "supramarginal", "precuneus", "postcentral")):
        return "parietal"
    if any(token in region_name for token in ("temporal", "bankssts", "entorhinal", "fusiform", "parahippocampal")):
        return "temporal"
    return "other"


def region_set_mask(labels: pd.DataFrame, base_regions: list[str]) -> np.ndarray:
    """Parcels whose base region is exactly one of ``base_regions``."""
    wanted = {name.lower() for name in base_regions}
    present = set(labels["base_region"].str.lower())
    missing = wanted - present
    if missing:
        raise ValueError(f"Base regions not in the atlas: {', '.join(sorted(missing))}")
    return labels["base_region"].str.lower().isin(wanted).to_numpy()


def surface_values_from_node_scores(
    scale: int,
    atlas_dir: str | Path,
    node_scores: np.ndarray,
) -> dict[str, np.ndarray]:
    """Broadcast node scores onto fsaverage5 annotation vertices."""
    atlas_path = resolve_repo_path(atlas_dir)
    resolution = SCALE_TO_RESOLUTION[scale]
    node_scores = np.asarray(node_scores, dtype=float)
    values_by_hemi: dict[str, np.ndarray] = {}
    score_pointer = 0
    for hemi in ("L", "R"):
        annot_path = atlas_path / f"atl-Cammoun2012_space-fsaverage5_res-{resolution}_hemi-{hemi}_deterministic.annot"
        vertex_labels, _, names = nib.freesurfer.read_annot(str(annot_path))
        included = [name.decode() for name in names if name.decode() not in EXCLUDED_LABELS]
        hemi_scores = node_scores[score_pointer : score_pointer + len(included)]
        mapping = {name: score for name, score in zip(included, hemi_scores)}
        vertex_values = np.zeros_like(vertex_labels, dtype=float)
        for idx, raw_name in enumerate(names):
            name = raw_name.decode()
            score = mapping.get(name, 0.0)
            vertex_values[vertex_labels == idx] = score
        values_by_hemi[hemi] = vertex_values
        score_pointer += len(included)
    return values_by_hemi
