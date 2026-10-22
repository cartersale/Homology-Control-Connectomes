#!/usr/bin/env python
"""Compare two run directories file by file.

    python scripts/compare_runs.py --new canonical-2026-10 --ref results --out baseline/canonical_vs_preprint.md
    python scripts/compare_runs.py --new runs/a --ref runs/b --rtol 1e-12

Tables (.csv, .csv.gz) are matched on their key columns (every column that is
not a float), so row order does not matter; numeric columns report the largest
absolute and relative difference. .npz arrays are compared elementwise; other
files by checksum. Run names are resolved like --run elsewhere ("results" is
the legacy tree); paths work too.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hoc.runs import resolve_run_dir  # noqa: E402

SKIP_PARTS = {"logs", "config", "variants", "__pycache__"}
SKIP_NAMES = {"manifest.json", ".DS_Store"}
TABLE_SUFFIXES = (".csv", ".csv.gz")


@dataclass
class Result:
    path: str
    status: str  # identical | close | differs | only-new | only-ref | error
    detail: str = ""
    columns: list[tuple[str, float, float, int]] = field(default_factory=list)  # name, max_abs, max_rel, n_bad


def run_path(value: str) -> Path:
    path = Path(value)
    if path.exists() and (path.is_absolute() or "/" in value):
        return path.resolve()
    return resolve_run_dir(value)


def files_under(root: Path) -> dict[str, Path]:
    out = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.name in SKIP_NAMES:
            continue
        rel = path.relative_to(root)
        if SKIP_PARTS & set(rel.parts[:-1]):
            continue
        out[rel.as_posix()] = path
    return out


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Integer / float columns that identify a row rather than hold a result.
ID_COLUMNS = {
    "scale", "k", "node_index", "subject_index", "step", "hub_count", "support_size", "random_sample",
    "perturb_index", "draw", "null_index", "rep", "h", "n_hubs", "axis_position",
    "c", "c_value", "eps", "epsilon", "horizon", "T", "level", "sigma", "p_dropout", "fraction",
}


def is_key(name: str, a: pd.Series, b: pd.Series) -> bool:
    """Identifier columns: text labels and known index/parameter columns.

    Text columns that list result sets (``node_indices``, ``*_labels``) are values.
    """
    if name in ID_COLUMNS:
        return True
    if pd.api.types.is_numeric_dtype(a) or pd.api.types.is_numeric_dtype(b):
        return False
    return not (name.endswith(("indices", "labels")) or name.startswith("driver_"))


def key_columns(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    return [c for c in a.columns if c in b.columns and is_key(c, a[c], b[c])]


def with_occurrence(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Number repeated keys (e.g. 100 random sets per subject) in file order."""
    frame = frame.copy()
    frame["__occurrence"] = frame.groupby(keys, dropna=False).cumcount() if keys else np.arange(len(frame))
    return frame


def compare_numeric(x: np.ndarray, y: np.ndarray, rtol: float, atol: float) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    both_nan = np.isnan(x) & np.isnan(y)
    same_inf = np.isinf(x) & np.isinf(y) & (np.sign(x) == np.sign(y))
    ok = both_nan | same_inf
    finite = np.isfinite(x) & np.isfinite(y)
    diff = np.zeros_like(x)
    diff[finite] = np.abs(x[finite] - y[finite])
    rel = np.zeros_like(x)
    rel[finite] = diff[finite] / np.maximum(np.abs(y[finite]), 1e-12)
    close = ok | (finite & (diff <= atol + rtol * np.abs(y)))
    max_abs = float(diff[finite].max()) if finite.any() else 0.0
    max_rel = float(rel[finite & (diff > atol)].max()) if (finite & (diff > atol)).any() else 0.0
    return max_abs, max_rel, int((~close).sum())


def compare_table(rel: str, new: Path, ref: Path, rtol: float, atol: float) -> Result:
    a, b = pd.read_csv(new, low_memory=False), pd.read_csv(ref, low_memory=False)
    notes = []
    if list(a.columns) != list(b.columns):
        added = [c for c in a.columns if c not in b.columns]
        removed = [c for c in b.columns if c not in a.columns]
        notes.append(f"columns +{added} -{removed}" if (added or removed) else "column order differs")
    keys = key_columns(a, b)
    a, b = with_occurrence(a, keys), with_occurrence(b, keys)
    keys = keys + ["__occurrence"]
    for frame in (a, b):
        for c in keys:
            if frame[c].dtype == object:
                frame[c] = frame[c].fillna("<NA>")
    merged = a.merge(b, on=keys, how="outer", suffixes=("", "__ref"), indicator=True)
    only_new = int((merged["_merge"] == "left_only").sum())
    only_ref = int((merged["_merge"] == "right_only").sum())
    if only_new or only_ref:
        shown = ", ".join(k for k in keys if k != "__occurrence")
        notes.append(f"rows: {only_new} only in new, {only_ref} only in ref (matched on {shown})")
    merged = merged[merged["_merge"] == "both"]
    pairs = [(c, merged[c], merged[c + "__ref"]) for c in a.columns if c not in keys and c in b.columns]

    columns = []
    for name, x, y in pairs:
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            max_abs, max_rel, n_bad = compare_numeric(x.to_numpy(), y.to_numpy(), rtol, atol)
        else:
            n_bad = int((x.astype(str).to_numpy() != y.astype(str).to_numpy()).sum())
            max_abs = max_rel = float("nan")
        if n_bad:
            columns.append((name, max_abs, max_rel, n_bad))

    if not notes and not columns:
        return Result(rel, "identical" if sha(new) == sha(ref) else "close")
    return Result(rel, "differs", "; ".join(notes), columns)


def compare_npz(rel: str, new: Path, ref: Path, rtol: float, atol: float) -> Result:
    a, b = np.load(new, allow_pickle=False), np.load(ref, allow_pickle=False)
    notes, columns = [], []
    if set(a.files) != set(b.files):
        notes.append(f"arrays +{sorted(set(a.files) - set(b.files))} -{sorted(set(b.files) - set(a.files))}")
    for name in sorted(set(a.files) & set(b.files)):
        x, y = a[name], b[name]
        if x.shape != y.shape:
            notes.append(f"{name}: shape {x.shape} vs {y.shape}")
        elif np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
            max_abs, max_rel, n_bad = compare_numeric(x.ravel(), y.ravel(), rtol, atol)
            if n_bad:
                columns.append((name, max_abs, max_rel, n_bad))
        elif not np.array_equal(x, y):
            columns.append((name, float("nan"), float("nan"), int(np.sum(x != y))))
    if not notes and not columns:
        return Result(rel, "identical" if sha(new) == sha(ref) else "close")
    return Result(rel, "differs", "; ".join(notes), columns)


def compare(new_root: Path, ref_root: Path, rtol: float, atol: float, only: list[str] | None) -> list[Result]:
    new_files, ref_files = files_under(new_root), files_under(ref_root)
    names = sorted(set(new_files) | set(ref_files))
    if only:
        names = [n for n in names if any(n.startswith(prefix) for prefix in only)]
    results = []
    for rel in names:
        if rel not in ref_files:
            results.append(Result(rel, "only-new"))
            continue
        if rel not in new_files:
            results.append(Result(rel, "only-ref"))
            continue
        new, ref = new_files[rel], ref_files[rel]
        try:
            if sha(new) == sha(ref):
                results.append(Result(rel, "identical"))
            elif rel.endswith(TABLE_SUFFIXES):
                results.append(compare_table(rel, new, ref, rtol, atol))
            elif rel.endswith(".npz"):
                results.append(compare_npz(rel, new, ref, rtol, atol))
            else:
                results.append(Result(rel, "differs", "checksum differs"))
        except Exception as error:  # noqa: BLE001 - report and keep going
            results.append(Result(rel, "error", f"{type(error).__name__}: {error}"))
    return results


def report(results: list[Result], new_root: Path, ref_root: Path, rtol: float, atol: float) -> str:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    lines = [
        f"# Run comparison: `{new_root.name}` vs `{ref_root.name}`",
        "",
        f"Generated by `scripts/compare_runs.py` (rtol {rtol:g}, atol {atol:g}). "
        "\"close\" = not byte-identical but every value within tolerance.",
        "",
        "| Status | Files |",
        "| --- | --- |",
        *[f"| {status} | {counts[status]} |" for status in sorted(counts)],
        "",
        "## Differences",
        "",
    ]
    by_folder: dict[str, list[Result]] = {}
    for r in results:
        if r.status in ("differs", "error", "only-new", "only-ref"):
            by_folder.setdefault("/".join(r.path.split("/")[:-1]) or ".", []).append(r)
    for folder in sorted(by_folder):
        lines += [f"### `{folder}/`", ""]
        for r in by_folder[folder]:
            name = r.path.split("/")[-1]
            if r.status in ("only-new", "only-ref"):
                lines.append(f"- `{name}`: {r.status}")
                continue
            lines.append(f"- `{name}`: {r.detail or r.status}")
            for column, max_abs, max_rel, n_bad in sorted(r.columns, key=lambda c: -c[3])[:12]:
                lines.append(f"  - `{column}`: {n_bad} values differ, max |Δ| {max_abs:.3g}, max rel {max_rel:.3g}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--new", required=True, help="Run name or path.")
    parser.add_argument("--ref", required=True, help="Run name or path ('results' = legacy tree).")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--only", nargs="+", default=None, help="Compare only paths starting with these prefixes.")
    parser.add_argument("--out", type=Path, default=None, help="Write the markdown report here.")
    args = parser.parse_args()

    new_root, ref_root = run_path(args.new), run_path(args.ref)
    results = compare(new_root, ref_root, args.rtol, args.atol, args.only)
    text = report(results, new_root, ref_root, args.rtol, args.atol)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    sys.exit(1 if any(r.status in ("differs", "error") for r in results) else 0)


if __name__ == "__main__":
    main()
