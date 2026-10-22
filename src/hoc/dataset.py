"""Dataset loading and subject-level preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from . import synthetic
from .atlas import build_label_table
from .paths import resolve_repo_path
from .preprocessing import preprocess_structural_adjacency


SCALE_ORDER = [68, 114, 219, 448, 1000]


def load_raw_connectomes(mat_path: str | Path) -> dict[int, np.ndarray]:
    """Load structural connectomes from the Griffa et al. MATLAB release."""
    if synthetic.is_synthetic(mat_path):
        return synthetic.connectomes(mat_path)
    path = resolve_repo_path(mat_path)
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    sc_matrices = mat["connMatrices"].SC
    return {scale: np.asarray(array, dtype=float) for scale, array in zip(SCALE_ORDER, sc_matrices)}


def preprocess_dataset(
    raw_connectomes: dict[int, np.ndarray],
    scales: list[int],
) -> dict[int, np.ndarray]:
    """Preprocess every subject matrix for the requested parcellation scales."""
    processed: dict[int, np.ndarray] = {}
    for scale in scales:
        stack = raw_connectomes[scale]
        processed[scale] = np.stack(
            [preprocess_structural_adjacency(stack[:, :, subject]) for subject in range(stack.shape[2])],
            axis=0,
        )
    return processed


def build_subject_table(n_subjects: int) -> pd.DataFrame:
    """Return a simple subject-index table shared across scales."""
    return pd.DataFrame(
        {
            "subject_index": np.arange(n_subjects, dtype=int),
            "subject_id": [f"sub-{index:03d}" for index in range(n_subjects)],
        }
    )


def build_scale_metadata(scales: list[int], atlas_dir: str | Path) -> pd.DataFrame:
    """Summarize label availability for the requested scales."""
    rows = []
    for scale in scales:
        labels = build_label_table(scale=scale, atlas_dir=atlas_dir)
        rows.append(
            {
                "scale": scale,
                "n_regions": int(labels.shape[0]),
                "n_left": int((labels["hemi"] == "L").sum()),
                "n_right": int((labels["hemi"] == "R").sum()),
            }
        )
    return pd.DataFrame(rows)
