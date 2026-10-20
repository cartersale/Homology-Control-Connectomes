#!/usr/bin/env python
"""Generate nulls and compute network- and set-level statistics (paper/new_idea.md).

One job = (null type, scale, subject). "real" jobs analyze the empirical connectome with
the primary random sets. Each null job generates N_NULLS networks, saves them
(upper triangle, float32) and, per network, recomputes the strength ranking (and cycle
participation at CYCLE_SCALES), draws fresh random sets, and evaluates every
c in C_VALUES both fixed and gap-matched. Chunks are written per job, so the run resumes.

Outputs (results/revision/nulls/):
  chunks/{null}/scale-{N}/sub-XXX_{network,sets}.csv.gz
  networks/{null}/scale-{N}/sub-XXX.npz        (triu float32, shape (N_NULLS, N(N-1)/2))

Usage:
  python 01_run_nulls.py --nulls real strength --scales 68 114 219 --n-jobs 12
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from _common import (
    CYCLE_SCALES,
    N_NULLS,
    N_SUBJECTS,
    OUT_DIR,
    SCALES,
    Spectral,
    analyze_network,
    centroid_distances,
    load_stack,
    make_null,
    null_seed,
    rich_club_level,
)


def chunk_paths(null: str, scale: int, subject: int):
    base = OUT_DIR / "chunks" / null / f"scale-{scale}"
    return base / f"sub-{subject:03d}_network.csv.gz", base / f"sub-{subject:03d}_sets.csv.gz"


def run_job(null: str, scale: int, subject: int, n_nulls: int) -> str:
    net_path, set_path = chunk_paths(null, scale, subject)
    if net_path.exists() and set_path.exists():
        return f"skip {null} {scale} {subject}"
    t0 = time.time()
    real = load_stack(scale)[subject]
    dist = centroid_distances(scale)
    real_gap = Spectral(real).gap
    level = rich_club_level(real)
    with_cycle = scale in CYCLE_SCALES

    net_rows, set_rows = [], []
    if null == "real":
        net, rows = analyze_network(real, real, real_gap, dist, level, "real", -1, scale, subject, with_cycle, {"null_extra": np.nan})
        net_rows.append(net)
        set_rows.extend(rows)
    else:
        iu = np.triu_indices(scale, 1)
        saved = np.empty((n_nulls, iu[0].size), dtype=np.float32)
        for i in range(n_nulls):
            B, extra = make_null(null, real, dist, null_seed(null, scale, subject, i))
            saved[i] = B[iu]
            net, rows = analyze_network(B, real, real_gap, dist, level, null, i, scale, subject, with_cycle, {"null_extra": extra})
            net_rows.append(net)
            set_rows.extend(rows)
        net_dir = OUT_DIR / "networks" / null / f"scale-{scale}"
        net_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(net_dir / f"sub-{subject:03d}.npz", triu=saved)

    net_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(set_rows).to_csv(set_path, index=False)
    pd.DataFrame(net_rows).to_csv(net_path, index=False)  # written last: marks the job complete
    return f"done {null} {scale} {subject} {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nulls", nargs="+", default=["real", "strength", "geometric", "weights"])
    ap.add_argument("--scales", nargs="+", type=int, default=list(SCALES))
    ap.add_argument("--subjects", type=int, default=N_SUBJECTS, help="first n subjects")
    ap.add_argument("--n-nulls", type=int, default=N_NULLS)
    ap.add_argument("--n-jobs", type=int, default=12)
    args = ap.parse_args()

    # biggest jobs first for better load balance
    jobs = [(null, scale, subj) for null in args.nulls for scale in sorted(args.scales, reverse=True) for subj in range(args.subjects)]
    print(f"{len(jobs)} jobs", flush=True)
    for msg in Parallel(n_jobs=args.n_jobs, return_as="generator_unordered", verbose=0)(
        delayed(run_job)(null, scale, subj, args.n_nulls) for null, scale, subj in jobs
    ):
        print(time.strftime("%H:%M:%S"), msg, flush=True)


if __name__ == "__main__":
    main()
