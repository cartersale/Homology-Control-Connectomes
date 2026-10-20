#!/usr/bin/env python
"""Build every supplement table from a run directory.

    python scripts/paper/make_tables.py --run canonical-2026-10
    python scripts/paper/make_tables.py --run results --check paper/network_neuro_resubmission/supp_tables

Tables go to <run>/paper/tables/ (for the legacy results/ tree:
runs/results-derived/paper/tables/). ``--check DIR`` compares every generated
file with the file of the same name in DIR and fails on any difference; with
``--run results`` it checks that the ports reproduce the 2026-10-19 draft.
"""

from __future__ import annotations

import argparse
import difflib
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))

from hoc.runs import is_legacy, reader_run_dir, stage_from_argv, writable_root  # noqa: E402

# One module per supplement section; each defines build(res) -> list[Table].
SECTIONS = ["s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s_canonical"]
STAGE = "paper/tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None, help="Run name (default: $HOC_RUN or 'main'; 'results' = legacy tree).")
    parser.add_argument("--overwrite", action="store_true", help="Allow rebuilding a finished stage.")
    parser.add_argument("--only", nargs="+", default=None, choices=SECTIONS, help="Build only these sections.")
    parser.add_argument("--check", type=Path, default=None, help="Compare with the tables in this directory.")
    return parser.parse_args()


def build_all(res: Path, sections: list[str]) -> list:
    tables = []
    for name in sections:
        module = importlib.import_module(name)
        built = module.build(res)
        ids = [t.id for t in built]
        assert len(ids) == len(set(ids)), f"{name}: duplicate table ids"
        tables.extend(built)
    return tables


def write(tables: list, out: Path) -> list[Path]:
    from paper_common import to_longtable

    out.mkdir(parents=True, exist_ok=True)
    (out / "notes").mkdir(exist_ok=True)
    written = []
    for table in tables:
        files = {f"{table.id}.tex": table.tex}
        if table.long:
            files[f"{table.id}_long.tex"] = to_longtable(table.tex, table.long_colspec)
        for name, text in files.items():
            path = out / name
            path.write_text(text.rstrip("\n") + "\n")
            written.append(path)
        if table.notes:
            (out / "notes" / f"{table.id}.txt").write_text(table.notes.strip() + "\n")
    return written


def check(written: list[Path], reference: Path) -> int:
    reference = reference if reference.is_absolute() else REPO_ROOT / reference
    problems = 0
    for path in written:
        ref = reference / path.name
        if not ref.exists():
            print(f"NEW      {path.name} (no reference file)")
            continue
        if ref.read_text() != path.read_text():
            problems += 1
            diff = difflib.unified_diff(
                ref.read_text().splitlines(), path.read_text().splitlines(), "reference", "generated", lineterm="", n=0
            )
            print(f"DIFFERS  {path.name}")
            for line in list(diff)[2:14]:
                print("   ", line)
    names = {p.name for p in written}
    missing = sorted(p.name for p in reference.glob("*.tex") if p.name not in names)
    for name in missing:
        print(f"MISSING  {name} (in reference, not generated)")
    print(f"{len(written)} files generated, {problems} differ, {len(missing)} not generated")
    return problems + len(missing)


def main() -> None:
    args = parse_args()
    argv = ["--run", args.run] if args.run else []
    run_dir = reader_run_dir(argv)
    out = writable_root(run_dir) / "paper" / "tables"
    sections = args.only or SECTIONS

    ctx = None
    if not is_legacy(run_dir) and args.only is None:
        ctx = stage_from_argv(STAGE, argv=argv + (["--overwrite"] if args.overwrite else []))
        for upstream in ("revision/q1/02_tables", "revision/nulls/04_rayleigh_bound", "revision/targets/09_report_tables",
                         "revision/lesion/02_analyze_reciprocal_lesion", "revision/stability/summarize_perturbation",
                         "revision/mechanism/03_coverage_distance", "revision/regime/07_report_tables"):
            ctx.require_stage(upstream)
        ctx.begin()

    written = write(build_all(run_dir / "revision", sections), out)
    print(f"wrote {len(written)} files to {out}")
    if ctx is not None:
        ctx.finish(outputs=written)
    if args.check is not None:
        sys.exit(1 if check(written, args.check) else 0)


if __name__ == "__main__":
    main()
