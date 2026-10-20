"""Input-output helpers for tables and arrays."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .paths import ensure_directory, resolve_repo_path


def write_csv(table: pd.DataFrame, path_like: str | Path) -> Path:
    """Write a tidy table as CSV."""
    path = resolve_repo_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def write_npz(path_like: str | Path, **arrays: np.ndarray) -> Path:
    """Write a compressed numpy archive."""
    path = resolve_repo_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def write_pickle(path_like: str | Path, payload: object) -> Path:
    """Write a binary pickle for nested dictionaries."""
    path = resolve_repo_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def prepare_results_tree(results_root: str | Path) -> dict[str, Path]:
    """Create the standard results directory tree."""
    root = ensure_directory(results_root)
    subdirs = {
        "root": root,
        "processed": root / "processed",
        "primary": root / "primary",
        "robustness": root / "robustness",
        "hub_lesion": root / "hub_lesion",
        "samples": root / "primary" / "landscape_samples",
        "metadata": root / "metadata",
        "statistics": root / "statistics",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs
