"""Synthetic stand-ins for the connectome data, atlas labels and gradient map.

Used by tests and CI so the whole pipeline can run without the Griffa data,
the Cammoun annotation files or a neuromaps download. Select it in a config:

    [paths]
    raw_connectomes = "synthetic:subjects=6,seed=7"
    atlas_dir = "synthetic:"

The data are random but fully determined by the spec string, so the outputs are
stable enough to pin in a regression test (tests/golden/).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PREFIX = "synthetic:"
SCALES = (68, 114, 219)

# The 34 Desikan-Killiany cortical regions (Cammoun 2012 base names).
DK_REGIONS = (
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus", "entorhinal",
    "fusiform", "inferiorparietal", "inferiortemporal", "isthmuscingulate", "lateraloccipital",
    "lateralorbitofrontal", "lingual", "medialorbitofrontal", "middletemporal", "parahippocampal",
    "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
    "postcentral", "posteriorcingulate", "precentral", "precuneus", "rostralanteriorcingulate",
    "rostralmiddlefrontal", "superiorfrontal", "superiorparietal", "superiortemporal",
    "supramarginal", "frontalpole", "temporalpole", "transversetemporal", "insula",
)

# Rough unimodal (-1) to transmodal (+1) position, so the synthetic gradient has
# the same sign convention as the Margulies map.
_TRANSMODAL = {
    "pericalcarine": -1.0, "cuneus": -0.9, "lingual": -0.9, "lateraloccipital": -0.8,
    "precentral": -0.7, "postcentral": -0.7, "paracentral": -0.6, "transversetemporal": -0.6,
    "superiortemporal": -0.3, "supramarginal": -0.2, "superiorparietal": -0.2, "fusiform": -0.3,
    "insula": 0.0, "caudalmiddlefrontal": 0.2, "inferiorparietal": 0.4, "precuneus": 0.6,
    "posteriorcingulate": 0.7, "isthmuscingulate": 0.6, "rostralmiddlefrontal": 0.5,
    "superiorfrontal": 0.6, "medialorbitofrontal": 0.8, "frontalpole": 0.7, "parsorbitalis": 0.5,
    "parstriangularis": 0.4, "parsopercularis": 0.2, "lateralorbitofrontal": 0.5,
    "rostralanteriorcingulate": 0.8, "caudalanteriorcingulate": 0.5, "middletemporal": 0.5,
    "inferiortemporal": 0.3, "bankssts": 0.3, "temporalpole": 0.6, "entorhinal": 0.4,
    "parahippocampal": 0.3,
}


def is_synthetic(path_like: str | Path | None) -> bool:
    return isinstance(path_like, str) and path_like.startswith(PREFIX)


def parse_spec(spec: str) -> dict[str, int]:
    body = spec[len(PREFIX):]
    values = {"subjects": 6, "seed": 7}
    for item in filter(None, body.split(",")):
        key, value = item.split("=")
        values[key.strip()] = int(value)
    return values


def _hemisphere_sizes(scale: int) -> tuple[int, int]:
    return scale // 2, scale - scale // 2


def label_rows(scale: int) -> list[dict]:
    """Cammoun-style label rows: base regions split into numbered subparcels."""
    rows = []
    node_index = 0
    for hemi, size in zip(("L", "R"), _hemisphere_sizes(scale)):
        counts = np.full(len(DK_REGIONS), size // len(DK_REGIONS))
        counts[: size % len(DK_REGIONS)] += 1
        for base, count in zip(DK_REGIONS, counts):
            for part in range(1, count + 1):
                name = base if scale == 68 else f"{base}_{part}"
                rows.append(
                    {
                        "node_index": node_index,
                        "scale": scale,
                        "hemi": hemi,
                        "atlas_label": name,
                        "region_label": f"{hemi}_{name}",
                        "base_region": base,
                    }
                )
                node_index += 1
    return rows


def _positions(scale: int, rng: np.random.Generator) -> np.ndarray:
    """Points on two unit hemispheres (x < 0 left, x > 0 right)."""
    left, right = _hemisphere_sizes(scale)
    points = rng.normal(size=(scale, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points[:, 0] = np.abs(points[:, 0]) + 0.3
    points[:left, 0] *= -1.0
    return points


def connectomes(spec: str) -> dict[int, np.ndarray]:
    """Streamline-count-like matrices, shape (n, n, subjects) per scale."""
    params = parse_spec(spec)
    n_subjects, seed = params["subjects"], params["seed"]
    stacks = {}
    for scale in SCALES:
        rng = np.random.default_rng([seed, scale])
        points = _positions(scale, rng)
        distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        same_hemi = np.sign(points[:, None, 0]) == np.sign(points[None, :, 0])
        base = np.exp(-distance / 0.6) * np.where(same_hemi, 1.0, 0.15)
        np.fill_diagonal(base, 0.0)
        support = rng.random(base.shape) < np.clip(3.0 * base, 0.0, 1.0)
        support = np.triu(support, 1)
        support = support | support.T
        stack = np.zeros((scale, scale, n_subjects))
        for subject in range(n_subjects):
            noise = rng.lognormal(mean=0.0, sigma=0.4, size=base.shape)
            keep = rng.random(base.shape) > 0.08
            weights = np.triu(1000.0 * base * noise * support * keep, 1)
            stack[:, :, subject] = np.round(weights + weights.T)
        stacks[scale] = stack
    return stacks


def gradient(labels: pd.DataFrame) -> np.ndarray:
    """A parcel-level stand-in for the Margulies principal gradient."""
    rng = np.random.default_rng(int(labels["scale"].iloc[0]))
    base = labels["base_region"].map(_TRANSMODAL).to_numpy(dtype=float)
    return base + 0.1 * rng.standard_normal(len(labels))
