"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
import tomllib


def load_config(config_path: str | Path) -> dict:
    """Load a TOML configuration file."""
    path = Path(config_path)
    with path.open("rb") as handle:
        return tomllib.load(handle)
