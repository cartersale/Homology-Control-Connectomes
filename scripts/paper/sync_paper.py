#!/usr/bin/env python
"""Copy a run's generated tables and numbers.tex into the manuscript folder.

    python scripts/paper/sync_paper.py --run canonical-2026-10            # show what would change
    python scripts/paper/sync_paper.py --run canonical-2026-10 --apply    # copy

Copies <run>/paper/tables/*.tex to <paper>/supp_tables/ and <run>/paper/numbers.tex
to <paper>/numbers.tex. The manuscript is not part of this repository.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hoc.runs import reader_run_dir, writable_root  # noqa: E402

DEFAULT_PAPER = REPO_ROOT / "paper" / "network_neuro_resubmission"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None)
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--apply", action="store_true", help="Copy the files (default: only list changes).")
    args = parser.parse_args()

    source = writable_root(reader_run_dir(["--run", args.run] if args.run else [])) / "paper"
    pairs = [(p, args.paper / "supp_tables" / p.name) for p in sorted((source / "tables").glob("*.tex"))]
    pairs.append((source / "numbers.tex", args.paper / "numbers.tex"))
    missing = [str(src) for src, _ in pairs if not src.exists()]
    if missing:
        sys.exit(f"not generated yet: {', '.join(missing)} (run make_tables.py and make_numbers.py)")

    changed = [(src, dst) for src, dst in pairs if not dst.exists() or dst.read_bytes() != src.read_bytes()]
    for src, dst in changed:
        print(f"{'new    ' if not dst.exists() else 'changed'}  {dst.relative_to(args.paper)}")
    print(f"{len(changed)} of {len(pairs)} files {'copied' if args.apply else 'would change'}")
    if args.apply:
        for src, dst in changed:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
