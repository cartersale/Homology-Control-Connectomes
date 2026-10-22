"""Scientific regression test: the full pipeline on synthetic data.

Runs stages 00-03 on config/smoke.toml (synthetic connectomes, labels and
gradient; no downloads) and compares a fingerprint of every output table with
tests/golden/synthetic.json:

- text columns (driver sets, labels, rankings as strings) must match exactly;
- numeric columns are compared through their sum and sum of absolute values
  (rtol 1e-7), which catches any change to a computed value;
- columns set purely by roundoff (no-floor eigenvalue extremes) are left out, so
  macOS/Accelerate and Linux/OpenBLAS give the same fingerprint.

If a change to the numbers is intended, regenerate the golden file with
    uv run python tests/test_synthetic_pipeline.py --update
and explain the diff in the PR.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "tests" / "golden" / "synthetic.json"
CONFIG = "config/smoke.toml"
RUN = "golden"

# Values decided by floating-point roundoff, not by the science.
ROUNDOFF_COLUMNS = {
    ("robustness/sensitivity_summary.csv", "no_floor"): [
        "average_energy",
        "log10_average_energy",
        "condition_number",
        "log10_condition_number",
        "min_eigenvalue",
        "log_min_eigenvalue",
    ],
}
# Counts of eigenvalues at the floor can flip by one when an eigenvalue sits
# within roundoff of it.
LOOSE_COLUMNS = {"eigenvalues_at_floor": 0.02, "fraction_eigenvalues_at_floor": 0.02}
RTOL = 1e-7


def run_pipeline(runs_root: Path) -> Path:
    env = {**os.environ, "HOC_RUNS_ROOT": str(runs_root), "PYTHONWARNINGS": "ignore"}
    cmd = [sys.executable, "scripts/run_all.py", "--config", CONFIG, "--run", RUN]
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True, capture_output=True)
    return runs_root / RUN


def _mask_roundoff(rel: str, frame: pd.DataFrame) -> pd.DataFrame:
    for (path, case), columns in ROUNDOFF_COLUMNS.items():
        if rel == path and "epsilon_case" in frame:
            frame.loc[frame["epsilon_case"] == case, columns] = np.nan
    return frame


def fingerprint(run_dir: Path) -> dict:
    tables = {}
    for path in sorted(run_dir.rglob("*.csv")):
        rel = path.relative_to(run_dir).as_posix()
        if rel.startswith(("metadata/", "config/")):
            continue
        frame = _mask_roundoff(rel, pd.read_csv(path))
        numeric = frame.select_dtypes(include=["number", "bool"])
        text = frame.drop(columns=numeric.columns)
        entry = {
            "rows": len(frame),
            "columns": list(frame.columns),
            "text_sha256": hashlib.sha256(text.to_csv(index=False).encode()).hexdigest(),
            "numeric": {},
        }
        for column in numeric.columns:
            values = numeric[column].astype(float).to_numpy()
            finite = values[np.isfinite(values)]
            entry["numeric"][column] = {
                "sum": float(finite.sum()),
                "abs_sum": float(np.abs(finite).sum()),
                "n_nonfinite": int((~np.isfinite(values)).sum() - np.isnan(values).sum()),
                "n_nan": int(np.isnan(values).sum()),
            }
        tables[rel] = entry
    return tables


def compare(actual: dict, expected: dict) -> list[str]:
    problems = []
    if set(actual) != set(expected):
        problems.append(f"tables differ: +{sorted(set(actual) - set(expected))} -{sorted(set(expected) - set(actual))}")
    for rel in sorted(set(actual) & set(expected)):
        a, e = actual[rel], expected[rel]
        for key in ("rows", "columns", "text_sha256"):
            if a[key] != e[key]:
                problems.append(f"{rel}: {key} differs")
        for column, ev in e["numeric"].items():
            av = a["numeric"].get(column)
            if av is None:
                continue
            rtol = LOOSE_COLUMNS.get(column, RTOL)
            for stat in ("sum", "abs_sum"):
                if not np.isclose(av[stat], ev[stat], rtol=rtol, atol=1e-10):
                    problems.append(f"{rel}:{column} {stat} {av[stat]!r} != {ev[stat]!r}")
            for stat in ("n_nonfinite", "n_nan"):
                if av[stat] != ev[stat]:
                    problems.append(f"{rel}:{column} {stat} {av[stat]} != {ev[stat]}")
    return problems


@pytest.mark.slow
def test_synthetic_pipeline_matches_golden(tmp_path) -> None:
    run_dir = run_pipeline(tmp_path / "runs")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert {name: stage["status"] for name, stage in manifest["stages"].items()} == {
        "00_prepare_data": "complete",
        "01_primary": "complete",
        "02_robustness": "complete",
        "03_hub_lesion": "complete",
    }
    problems = compare(fingerprint(run_dir), json.loads(GOLDEN.read_text()))
    assert not problems, "\n".join(problems[:40])


if __name__ == "__main__":
    import tempfile

    if "--update" not in sys.argv:
        sys.exit("usage: python tests/test_synthetic_pipeline.py --update")
    with tempfile.TemporaryDirectory() as tmp:
        result = fingerprint(run_pipeline(Path(tmp)))
    GOLDEN.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN.relative_to(REPO_ROOT)} ({len(result)} tables)")
