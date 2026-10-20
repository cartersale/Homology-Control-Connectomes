#!/usr/bin/env python
"""Task 1: representative analysis at scales 68/114/219, k in {1,2,3,5} (+15).

Per subject x scale, computes CP_i under four variants (Ripser cocycle support,
GUDHI pairing simplices, length-minimized cycles, birth edge only), then for each
driver-set size k:

* overlap and rank agreement with degree strength and with the Ripser ranking;
* ER / PR of the cycle set and of the degree set (primary c = 0.1, eps = 1e-5);
* ER / PR of three nulls:
    - the primary pipeline's 100 uniform random sets (identical seeds),
    - 100 *overlap-matched* random sets sharing exactly as many nodes with the
      degree top-k set as the variant does,
    - 100 *strength-matched* random sets whose members match the variant's nodes'
      strength ranks within +/- 5 % of N.
  The two matched nulls are the direct test of Reviewer 2's "distance from the
  degree set" account: if that account is complete, the variant should sit at
  the 50th percentile of its matched null.

Output: runs/<name>/revision/stability/representatives/per_subject_scale-{S}.csv and
score_agreement_scale-{S}.csv.

Run: python scripts/revision/stability/run_representatives_all_scales.py --run <name> [--subject-limit N] [--overwrite]
(workers default to $HOC_N_JOBS or 12).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from stability_common import (
    STAGE_PREFIX,
    output_dir,
    VARIANTS,
    BASE_SEED,
    EigenGramian,
    jaccard,
    load_stack,
    make_rng,
    overlap_matched_sets,
    percentile_vs,
    spearman,
    strength_matched_sets,
    subject_seed,
    top_k_nodes,
    uniform_random_sets,
    variant_scores,
)
from hoc.runs import RunContext, add_run_arguments, partial_stage  # noqa: E402
from hoc.utils import n_jobs  # noqa: E402

K_VALUES = (1, 2, 3, 5, 15)


def run_subject(args: tuple[int, int, str]) -> tuple[list[dict], list[dict]]:
    scale, subject_index, root = args
    adjacency = load_stack(scale, root)[subject_index]
    n = adjacency.shape[0]
    scores = variant_scores(adjacency)
    degree = scores["degree"]
    strength_rank = np.empty(n, dtype=int)
    strength_rank[np.argsort(-degree, kind="mergesort")] = np.arange(n)
    window = max(2, int(round(0.05 * n)))
    gram = EigenGramian(adjacency)
    subject_id = f"sub-{subject_index:03d}"

    agreement = []
    for variant in VARIANTS:
        agreement.append(
            {
                "scale": scale,
                "subject_id": subject_id,
                "variant": variant,
                "spearman_vs_degree": spearman(scores[variant], degree),
                "spearman_vs_ripser": spearman(scores[variant], scores["ripser"]),
            }
        )

    rows = []
    for k in K_VALUES:
        degree_set = top_k_nodes(degree, k)
        (deg_er,), (deg_pr,) = gram.metrics(degree_set[None])
        rand_er, rand_pr = gram.metrics(uniform_random_sets(n, k, scale, subject_index))
        for v_index, variant in enumerate(VARIANTS):
            chosen = top_k_nodes(scores[variant], k)
            (er,), (pr,) = gram.metrics(chosen[None])
            n_shared = int(np.intersect1d(chosen, degree_set).size)
            rng_o = make_rng(subject_seed(BASE_SEED, scale, subject_index, k, v_index, 7101))
            rng_s = make_rng(subject_seed(BASE_SEED, scale, subject_index, k, v_index, 7102))
            ov_er, ov_pr = gram.metrics(overlap_matched_sets(degree_set, n_shared, n, k, rng_o))
            st_er, st_pr = gram.metrics(strength_matched_sets(chosen, strength_rank, n, rng_s, window))
            rows.append(
                {
                    "scale": scale,
                    "subject_id": subject_id,
                    "k": k,
                    "variant": variant,
                    "nodes": ";".join(map(str, chosen)),
                    "degree_nodes": ";".join(map(str, degree_set)),
                    "n_shared_degree": n_shared,
                    "jaccard_degree": jaccard(chosen, degree_set),
                    "jaccard_ripser": jaccard(chosen, top_k_nodes(scores["ripser"], k)),
                    "mean_strength_rank_pct": float(100.0 * strength_rank[chosen].mean() / (n - 1)),
                    "spearman_vs_degree": spearman(scores[variant], degree),
                    "spearman_vs_ripser": spearman(scores[variant], scores["ripser"]),
                    "er_cycle": er,
                    "er_degree": deg_er,
                    "er_diff_degree": er - deg_er,
                    "pr_cycle": pr,
                    "pr_degree": deg_pr,
                    "pr_diff_degree": pr - deg_pr,
                    "er_random_mean": float(rand_er.mean()),
                    "er_diff_random": er - float(rand_er.mean()),
                    "er_pct_random": percentile_vs(rand_er, er),
                    "degree_er_diff_random": deg_er - float(rand_er.mean()),
                    "degree_er_pct_random": percentile_vs(rand_er, deg_er),
                    "pr_diff_random": pr - float(rand_pr.mean()),
                    "er_overlap_null_mean": float(ov_er.mean()),
                    "er_diff_overlap_null": er - float(ov_er.mean()),
                    "er_pct_overlap_null": percentile_vs(ov_er, er),
                    "pr_diff_overlap_null": pr - float(ov_pr.mean()),
                    "er_strength_null_mean": float(st_er.mean()),
                    "er_diff_strength_null": er - float(st_er.mean()),
                    "er_pct_strength_null": percentile_vs(st_er, er),
                    "pr_diff_strength_null": pr - float(st_pr.mean()),
                    "strength_window": window,
                }
            )
    return rows, agreement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=int, nargs="+", default=[68, 114, 219])
    parser.add_argument("--subject-limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=n_jobs(12))
    add_run_arguments(parser)
    args = parser.parse_args()
    stage = partial_stage(STAGE_PREFIX + Path(__file__).stem, scales=(args.scales, [68, 114, 219]))
    ctx = RunContext.from_args(stage, args, config_path="config/default.toml")
    ctx.require_stage("00_prepare_data")
    ctx.begin()
    out = output_dir(ctx.root) / "representatives"
    out.mkdir(parents=True, exist_ok=True)
    for scale in args.scales:
        n_subjects = load_stack(scale, ctx.root).shape[0]
        if args.subject_limit:
            n_subjects = min(n_subjects, args.subject_limit)
        jobs = [(scale, s, str(ctx.root)) for s in range(n_subjects)]
        rows, agreement = [], []
        with ProcessPoolExecutor(args.workers) as pool:
            for r, a in pool.map(run_subject, jobs):
                rows.extend(r)
                agreement.extend(a)
        pd.DataFrame(rows).to_csv(out / f"per_subject_scale-{scale}.csv", index=False)
        pd.DataFrame(agreement).to_csv(out / f"score_agreement_scale-{scale}.csv", index=False)
        print(f"scale {scale}: {len(rows)} rows")
    ctx.finish()


if __name__ == "__main__":
    main()
