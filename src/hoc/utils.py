"""Generic utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Construct a reproducible numpy generator."""
    return np.random.default_rng(int(seed))


def subject_seed(base_seed: int, *parts: int) -> int:
    """Create a deterministic derived seed from integer identifiers."""
    seed = int(base_seed)
    for value in parts:
        seed = (seed * 6364136223846793005 + int(value) + 1442695040888963407) % (2**63 - 1)
    return seed


def write_json(path: str | Path, payload: dict) -> None:
    """Write pretty JSON metadata."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
