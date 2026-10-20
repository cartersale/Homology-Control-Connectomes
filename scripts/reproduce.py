#!/usr/bin/env python
"""Run every analysis behind the paper, in dependency order, into one run directory.

    python scripts/reproduce.py --run canonical            # everything
    python scripts/reproduce.py --run canonical --list     # show the steps
    python scripts/reproduce.py --run canonical --only nulls q1
    python scripts/reproduce.py --run canonical --from revision/q1/01_compute_q1

Steps already marked complete in the run's manifest are skipped, so an
interrupted run resumes where it stopped. Each step's output goes to
runs/<name>/logs/<step>.log. BLAS threads are pinned to 1 (src/hoc/threads.py).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import hoc  # noqa: E402,F401 - sets BLAS thread defaults inherited by every step
from hoc.runs import read_manifest, resolve_run_dir  # noqa: E402

FIVE_PCT = "config/landscape_fraction_5pct.toml"
REVISION = "scripts/revision"
REPSTAB = "scripts/representative_stability"
ALL_SCHEMES = ["--schemes", "ripser", "geodesic", "gudhi", "gudhi_geodesic"]


@dataclass(frozen=True)
class Step:
    group: str
    script: str
    stage: str
    args: list[str] = field(default_factory=list)
    variant: str | None = None

    @property
    def name(self) -> str:
        return self.stage


def revision(folder: str, *stems: str) -> list[Step]:
    return [Step(folder, f"{REVISION}/{folder}/{stem}.py", f"revision/{folder}/{stem}") for stem in stems]


def repstab(scale: int, subdir: str = "representative_stability", extra: list[str] | None = None) -> Step:
    stage = "representative_stability/run_representative_stability"
    if subdir != "representative_stability":
        stage += f":{subdir}"
    args = ["--scale", str(scale), *(extra or []), "--output-subdir", subdir]
    return Step("representative_stability", f"{REPSTAB}/run_representative_stability.py", f"{stage}:scale-{scale}", args)


# Order matters: each step's upstream stages come before it (the scripts check this
# themselves with require_stage and refuse to run otherwise).
STEPS: list[Step] = [
    # Preprint pipeline (README "Running the pipeline")
    Step("pipeline", "scripts/00_prepare_data.py", "00_prepare_data"),
    Step("pipeline", "scripts/01_run_primary_analysis.py", "01_primary"),
    Step("pipeline", "scripts/02_run_robustness.py", "02_robustness", ["--skip-perturbations"]),
    Step("pipeline", "scripts/03_run_hub_lesion.py", "03_hub_lesion"),
    Step("pipeline", "scripts/00_prepare_data.py", "00_prepare_data", ["--config", FIVE_PCT], "fraction_5pct"),
    Step(
        "pipeline",
        "scripts/02_run_robustness.py",
        "02_robustness",
        ["--config", FIVE_PCT, "--skip-perturbations", "--include-landscape-summary"],
        "fraction_5pct",
    ),
    # Representative stability (Supplement S4; settings from the legacy run_settings files)
    repstab(114, extra=ALL_SCHEMES),
    repstab(114, "representative_stability_below_death", ["--schemes", "ripser", "geodesic", "--threshold-mode", "below_death"]),
    repstab(68, "revision/stability/existing_script_scale-68", ALL_SCHEMES),
    repstab(219, "revision/stability/existing_script_scale-219", ALL_SCHEMES),
    Step(
        "representative_stability",
        f"{REPSTAB}/analyze_selection_anatomy.py",
        "representative_stability/analyze_selection_anatomy:scale-114",
        ["--scale", "114", "--k-values", "5", "15"],
    ),
    # Revision analyses
    # 03_make_figure reads t4c_perron_piecewise.csv from 05_perron_extras.
    *revision("spectra", "01_compute_spectra", "02_analyze", "04_perron_c_check", "05_perron_extras",
              "03_make_figure", "06_make_spectra_figure"),
    *revision("regime", "00_validate", "01_csweep_perron", "02_nctpy_equivalence", "03_horizon_eps_largek",
              "04_energy_floor", "05_summarize_csweep", "06_perron_diagnostic", "08_perron_mode_sweep",
              "07_report_tables", "09_make_csweep_figure"),
    *revision("modal", "01_compute_modal", "02_analyze_modal", "03_figure"),
    *revision("nulls", "00_validate", "01_run_nulls", "02_analyze", "03_gap_mediators", "04_rayleigh_bound"),
    *revision("q1", "01_compute_q1", "02_tables", "03_decay_collapse", "04_redundancy_geodesic"),
    *revision("mechanism", "01_hub_loading", "02_canonical_graphs", "03_coverage_distance"),
    *revision("targets", "00_reproduce", "01_build_targets", "02_target_energies", "03_target_stats",
              "04_sensitivity", "05_spin_tests", "06_network_enrichment", "07_proximity_controls",
              "08_make_figure", "09_report_tables"),
    *revision("lesion", "01_run_reciprocal_lesion", "02_analyze_reciprocal_lesion"),
    *revision("stability", "run_representatives_all_scales", "run_perturbation_stability",
              "summarize_representatives", "summarize_perturbation"),
    *revision("figures", "fig1_panels", "fig1_layout"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=os.environ.get("HOC_RUN", "main"), help="Run name under runs/.")
    parser.add_argument("--only", nargs="+", default=None, help="Groups or step names to run.")
    parser.add_argument("--from", dest="start", default=None, help="Start at this step (skip earlier ones).")
    parser.add_argument("--list", action="store_true", help="List the steps and their status, then exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    return parser.parse_args()


def step_dir(run: str, step: Step) -> Path:
    return resolve_run_dir(run, variant=step.variant)


def is_complete(run: str, step: Step) -> bool:
    manifest = read_manifest(step_dir(run, step)) or {}
    return manifest.get("stages", {}).get(step.stage, {}).get("status") == "complete"


def selected_steps(args: argparse.Namespace) -> list[Step]:
    steps = STEPS
    if args.start:
        names = [s.name for s in steps]
        if args.start not in names:
            sys.exit(f"unknown step {args.start!r}; see --list")
        steps = steps[names.index(args.start):]
    if args.only:
        wanted = set(args.only)
        steps = [s for s in steps if s.group in wanted or s.name in wanted]
        if not steps:
            sys.exit(f"no steps match {args.only}; see --list")
    return steps


def main() -> None:
    args = parse_args()
    steps = selected_steps(args)

    if args.list:
        for step in steps:
            mark = "done" if is_complete(args.run, step) else "    "
            where = f" [{step.variant}]" if step.variant else ""
            print(f"{mark}  {step.group:<26} {step.name}{where}")
        return

    log_dir = resolve_run_dir(args.run) / "logs"
    timings = []
    for step in steps:
        label = step.name + (f" [{step.variant}]" if step.variant else "")
        if is_complete(args.run, step):
            print(f"skip  {label} (complete)")
            continue
        cmd = [sys.executable, "-u", step.script, *step.args, "--run", args.run]
        print(f"run   {label}\n      {' '.join(cmd[1:])}", flush=True)
        if args.dry_run:
            continue
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / (step.name.replace("/", "__").replace(":", "_") + (f"__{step.variant}" if step.variant else "") + ".log")
        start = time.time()
        with log.open("w") as handle:
            result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        timings.append({"step": label, "seconds": round(elapsed, 1), "returncode": result.returncode})
        (log_dir / "reproduce_timings.json").write_text(json.dumps(timings, indent=1))
        if result.returncode != 0:
            tail = "".join(log.read_text().splitlines(keepends=True)[-25:])
            sys.exit(f"FAILED {label} after {elapsed:.0f}s (log: {log})\n{tail}")
        print(f"      done in {elapsed / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
