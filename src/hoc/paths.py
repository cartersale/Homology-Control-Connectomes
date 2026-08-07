"""Project path helpers."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like: str | Path) -> Path:
    """Resolve a repo-relative path to an absolute path."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def ensure_directory(path_like: str | Path) -> Path:
    """Create a directory if needed and return its absolute path."""
    path = resolve_repo_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path
