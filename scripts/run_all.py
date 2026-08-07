#!/usr/bin/env python
"""Convenience launcher for the full pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml", help="Path to the TOML config file.")
    parser.add_argument("--subject-limit", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--skip-landscape", action="store_true", help="Forwarded to the primary analysis script.")
    parser.add_argument(
        "--skip-perturbations",
        action="store_true",
        help="Forwarded to the robustness script.",
    )
    return parser.parse_args()


def run_script(script_name: str, extra_args: list[str]) -> None:
    """Run one project script and stream its output."""
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / script_name)] + extra_args
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    common_args = ["--config", args.config]
    if args.subject_limit is not None:
        common_args += ["--subject-limit", str(args.subject_limit)]

    run_script("00_prepare_data.py", common_args)

    primary_args = list(common_args)
    if args.skip_landscape:
        primary_args.append("--skip-landscape")
    run_script("01_run_primary_analysis.py", primary_args)

    robustness_args = list(common_args)
    if args.skip_perturbations:
        robustness_args.append("--skip-perturbations")
    run_script("02_run_robustness.py", robustness_args)
    run_script("03_run_hub_lesion.py", common_args)


if __name__ == "__main__":
    main()
