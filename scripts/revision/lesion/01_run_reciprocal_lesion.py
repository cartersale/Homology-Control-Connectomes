#!/usr/bin/env python
"""Reciprocal hub-lesion analysis (Reviewer 2, Major 7).

Lesion types: top-h degree strength, top-h cycle participation, h random nodes (100 draws).
Driver strategies: degree_strength, cycle_participation.  k in {1,2,3,5}, h in {5,10,15}.

Protocol "static" (paper protocol): drivers = top-k surviving nodes by the INTACT ranking; the
same identities are evaluated in the intact and in the restabilized lesioned network (same c).
Protocol "recompute": intact drivers = top-k by intact ranking on the intact network; lesioned
drivers = top-k by the ranking recomputed on the lesioned adjacency.

Size-normalisation references: for every (lesion, k), 100 random k-sets drawn from the
surviving nodes are evaluated in the intact and lesioned networks (static protocol, matched
identities); for the recompute protocol the intact reference is 100 random k-sets over all N
nodes.  Outputs are written to runs/<name>/revision/lesion/.

Run: python scripts/revision/lesion/01_run_reciprocal_lesion.py --run <name> [--overwrite]
(workers default to $HOC_N_JOBS or 12).
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lesion_core import (  # noqa: E402
    BASE_SEED, HUB_COUNTS, K_VALUES, SCALES, STAGE_PREFIX, SpectralSystem, load_stack,
    metrics_from_eigenvalues, out_dir, top_k, top_k_excluding,
)
from hoc.runs import RunContext, add_run_arguments, partial_stage  # noqa: E402
from hoc.rankings import strength_scores  # noqa: E402
from hoc.topology import cycle_participation  # noqa: E402
from hoc.utils import n_jobs, subject_seed  # noqa: E402

N_RANDOM_LESIONS = 100
N_RANDOM_DRIVER_SETS = 100
STRATEGIES = ("degree_strength", "cycle_participation")
KEYS = ("effective_rank", "participation_ratio", "log10_average_energy", "fraction_at_floor",
        "log10_nonfloor_energy", "log10_condition_number")


def rankings(adjacency: np.ndarray) -> dict[str, np.ndarray]:
    return {"degree_strength": strength_scores(adjacency),
            "cycle_participation": cycle_participation(adjacency)[0]}


def pct(value: float, ref: np.ndarray) -> float:
    return float(np.mean(ref < value) + 0.5 * np.mean(ref == value))


def random_sets(rng, pool: np.ndarray, k: int, n: int) -> np.ndarray:
    return np.stack([rng.choice(pool, size=k, replace=False) for _ in range(n)])


def run_subject(task: tuple[int, int, int | None, str]) -> list[dict]:
    scale, subject, n_rand_lesions, root = task
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # spurious Accelerate matmul warnings
    adjacency = load_stack(scale, root)[subject]
    n = adjacency.shape[0]
    scores = rankings(adjacency)
    intact = SpectralSystem(adjacency)

    # intact reference over all N nodes (recompute protocol) and intact top-k sets
    intact_full = {}
    for k in K_VALUES:
        rng = np.random.default_rng(subject_seed(BASE_SEED, 7, scale, subject, k))
        ref = metrics_from_eigenvalues(intact.batch_set_eigenvalues(
            random_sets(rng, np.arange(n), k, N_RANDOM_DRIVER_SETS)))
        for strat in STRATEGIES:
            m = metrics_from_eigenvalues(intact.set_eigenvalues(top_k(scores[strat], k)))
            intact_full[(k, strat)] = (m, {
                "pct_er": pct(m["effective_rank"], ref["effective_rank"]),
                "pct_pr": pct(m["participation_ratio"], ref["participation_ratio"]),
                "ref_er": float(ref["effective_rank"].mean()),
                "ref_pr": float(ref["participation_ratio"].mean())})

    rows: list[dict] = []
    for h in HUB_COUNTS:
        lesions = [("degree", 0, top_k(scores["degree_strength"], h)),
                   ("cycle", 0, top_k(scores["cycle_participation"], h))]
        rng_les = np.random.default_rng(subject_seed(BASE_SEED, 1, scale, subject, h))
        for draw in range(n_rand_lesions):
            lesions.append(("random", draw, rng_les.choice(n, size=h, replace=False)))

        for lesion_type, draw, removed in lesions:
            keep = np.ones(n, dtype=bool)
            keep[removed] = False
            kept = np.flatnonzero(keep)
            to_les = np.full(n, -1)
            to_les[kept] = np.arange(kept.size)
            les_adj = adjacency[np.ix_(keep, keep)]
            lesioned = SpectralSystem(les_adj)  # restabilised with the same c
            les_scores = rankings(les_adj)
            overlap = {s: int(np.isin(top_k(scores[s], h), removed).sum()) for s in STRATEGIES}

            for k in K_VALUES:
                rng = np.random.default_rng(subject_seed(BASE_SEED, 3, scale, subject, h,
                                                         {"degree": 0, "cycle": 1, "random": 2}[lesion_type], draw, k))
                sets = random_sets(rng, kept, k, N_RANDOM_DRIVER_SETS)
                ref_i = metrics_from_eigenvalues(intact.batch_set_eigenvalues(sets))
                ref_l = metrics_from_eigenvalues(lesioned.batch_set_eigenvalues(to_les[sets]))
                for strat in STRATEGIES:
                    base = {"scale": scale, "subject": subject, "h": h, "lesion_type": lesion_type,
                            "draw": draw, "k": k, "strategy": strat, "n_intact": n, "n_lesioned": kept.size,
                            "lesion_hits_topk_degree": overlap["degree_strength"],
                            "lesion_hits_topk_cycle": overlap["cycle_participation"]}
                    # static protocol
                    drivers = top_k_excluding(scores[strat], k, removed)
                    mi = metrics_from_eigenvalues(intact.set_eigenvalues(drivers))
                    ml = metrics_from_eigenvalues(lesioned.set_eigenvalues(to_les[drivers]))
                    row = {**base, "protocol": "static",
                           "drivers_displaced": int(np.isin(top_k(scores[strat], k), removed).sum())}
                    for key in KEYS:
                        row[f"intact_{key}"] = float(mi[key])
                        row[f"lesioned_{key}"] = float(ml[key])
                    row.update({
                        "ref_intact_er": float(ref_i["effective_rank"].mean()),
                        "ref_lesioned_er": float(ref_l["effective_rank"].mean()),
                        "ref_intact_pr": float(ref_i["participation_ratio"].mean()),
                        "ref_lesioned_pr": float(ref_l["participation_ratio"].mean()),
                        "ref_intact_log10E": float(ref_i["log10_average_energy"].mean()),
                        "ref_lesioned_log10E": float(ref_l["log10_average_energy"].mean()),
                        "pct_intact_er": pct(mi["effective_rank"], ref_i["effective_rank"]),
                        "pct_lesioned_er": pct(ml["effective_rank"], ref_l["effective_rank"]),
                        "pct_intact_pr": pct(mi["participation_ratio"], ref_i["participation_ratio"]),
                        "pct_lesioned_pr": pct(ml["participation_ratio"], ref_l["participation_ratio"]),
                    })
                    rows.append(row)
                    # recompute protocol
                    mi2, ri = intact_full[(k, strat)]
                    d2 = top_k(les_scores[strat], k)
                    ml2 = metrics_from_eigenvalues(lesioned.set_eigenvalues(d2))
                    row2 = {**base, "protocol": "recompute", "drivers_displaced": -1}
                    for key in KEYS:
                        row2[f"intact_{key}"] = float(mi2[key])
                        row2[f"lesioned_{key}"] = float(ml2[key])
                    row2.update({
                        "ref_intact_er": ri["ref_er"], "ref_lesioned_er": row["ref_lesioned_er"],
                        "ref_intact_pr": ri["ref_pr"], "ref_lesioned_pr": row["ref_lesioned_pr"],
                        "ref_intact_log10E": np.nan, "ref_lesioned_log10E": row["ref_lesioned_log10E"],
                        "pct_intact_er": ri["pct_er"],
                        "pct_lesioned_er": pct(ml2["effective_rank"], ref_l["effective_rank"]),
                        "pct_intact_pr": ri["pct_pr"],
                        "pct_lesioned_pr": pct(ml2["participation_ratio"], ref_l["participation_ratio"]),
                    })
                    rows.append(row2)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, default=70)
    parser.add_argument("--random-lesions", type=int, default=N_RANDOM_LESIONS)
    parser.add_argument("--workers", type=int, default=n_jobs(12))
    parser.add_argument("--out", default=None, help="default: <run>/revision/lesion/lesion_rows.csv.gz")
    add_run_arguments(parser)
    args = parser.parse_args()
    stage = partial_stage(STAGE_PREFIX + Path(__file__).stem, subjects=(args.subjects, 70))
    ctx = RunContext.from_args(stage, args, config_path="config/default.toml")
    ctx.require_stage("00_prepare_data")
    ctx.begin()
    if args.out is None:
        args.out = str(out_dir(ctx.root) / "lesion_rows.csv.gz")
    tasks = [(s, i, args.random_lesions, str(ctx.root)) for s in reversed(SCALES) for i in range(args.subjects)]
    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for j, part in enumerate(pool.map(run_subject, tasks, chunksize=1)):
            rows.extend(part)
            if j % 10 == 0:
                print(f"{j + 1}/{len(tasks)} done, {time.time() - t0:.0f}s", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows to {out} in {time.time() - t0:.0f}s")
    ctx.finish()


if __name__ == "__main__":
    main()
