#!/usr/bin/env python
"""Task 2: edge-weight perturbation stability of cycle-participation rankings.

For every subject x scale, the preprocessed connectome A0 is perturbed with
``hoc.preprocessing.multiplicative_noise_perturbation`` (sigma in 0.05/0.10/0.20)
and ``hoc.preprocessing.dropout_perturbation`` (p in 0.05/0.10/0.20), 50 draws per
level (config ``robustness.n_perturbations``). After multiplicative noise a few
weights can exceed 1, which would make the distance 1 - w negative; such matrices
are divided by their maximum (the pipeline's own max-normalization). That is an
increasing affine map of the Rips filtration, so it leaves the persistence pairing
and the min-max normalized CP_i unchanged and only keeps distances in [0, 1].

On each perturbed matrix all rankings are recomputed from scratch and compared with
the unperturbed ones (Spearman, top-k Jaccard); the cycle-minus-degree ER/PR
difference is recomputed on the perturbed network with the perturbed rankings.

Output: runs/<name>/revision/stability/perturbation/per_draw_scale-{S}.csv and
unperturbed_scores_scale-{S}.npz.

Run: python scripts/revision/stability/run_perturbation_stability.py --run <name> [--subject-limit N] [--overwrite]
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
    BASE_SEED,
    CONFIG,
    VARIANTS,
    EigenGramian,
    jaccard,
    load_stack,
    make_rng,
    spearman,
    subject_seed,
    top_k_nodes,
    variant_scores,
)
from hoc.preprocessing import dropout_perturbation, multiplicative_noise_perturbation  # noqa: E402
from hoc.runs import RunContext, add_run_arguments, partial_stage  # noqa: E402
from hoc.utils import n_jobs  # noqa: E402

K_VALUES = (1, 2, 3, 5, 15)
RANKINGS = (*VARIANTS, "degree")
FAMILIES = {
    "noise": (1, [float(x) for x in CONFIG["robustness"]["noise_sigmas"]], multiplicative_noise_perturbation),
    "dropout": (2, [float(x) for x in CONFIG["robustness"]["dropout_probabilities"]], dropout_perturbation),
}


def perturb(adjacency, family, level, rng):
    _, _, fn = FAMILIES[family]
    out = fn(adjacency, level, rng)
    peak = float(out.max())
    return (out / peak, True) if peak > 1.0 else (out, False)


def run_subject(args):
    scale, subject_index, n_draws, root = args
    adjacency = load_stack(scale, root)[subject_index]
    base = variant_scores(adjacency)
    base_sets = {r: {k: top_k_nodes(base[r], k) for k in K_VALUES} for r in RANKINGS}
    rows = []
    for family, (family_id, levels, _) in FAMILIES.items():
        for level_index, level in enumerate(levels):
            for draw in range(n_draws):
                rng = make_rng(subject_seed(BASE_SEED, scale, subject_index, family_id, level_index, draw, 7201))
                perturbed, rescaled = perturb(adjacency, family, level, rng)
                scores = variant_scores(perturbed)
                gram = EigenGramian(perturbed)
                sets = {r: {k: top_k_nodes(scores[r], k) for k in K_VALUES} for r in RANKINGS}
                metrics = {r: {k: gram.metrics(sets[r][k][None]) for k in K_VALUES} for r in RANKINGS}
                for r in RANKINGS:
                    row = {
                        "scale": scale,
                        "subject_id": f"sub-{subject_index:03d}",
                        "family": family,
                        "level": level,
                        "draw": draw,
                        "rescaled": rescaled,
                        "ranking": r,
                        "spearman_vs_unperturbed": spearman(scores[r], base[r]),
                    }
                    for k in K_VALUES:
                        row[f"jaccard_top{k}"] = jaccard(sets[r][k], base_sets[r][k])
                        if r != "degree":
                            row[f"er_diff_degree_k{k}"] = float(metrics[r][k][0][0] - metrics["degree"][k][0][0])
                            row[f"pr_diff_degree_k{k}"] = float(metrics[r][k][1][0] - metrics["degree"][k][1][0])
                    rows.append(row)
    return subject_index, base, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=int, nargs="+", default=[68, 114, 219])
    parser.add_argument("--subject-limit", type=int, default=None)
    parser.add_argument("--n-draws", type=int, default=int(CONFIG["robustness"]["n_perturbations"]))
    parser.add_argument("--workers", type=int, default=n_jobs(12))
    add_run_arguments(parser)
    args = parser.parse_args()
    stage = partial_stage(STAGE_PREFIX + Path(__file__).stem, scales=(args.scales, [68, 114, 219]))
    ctx = RunContext.from_args(stage, args, config_path="config/default.toml")
    ctx.require_stage("00_prepare_data")
    ctx.begin()
    out = output_dir(ctx.root) / "perturbation"
    out.mkdir(parents=True, exist_ok=True)
    for scale in args.scales:
        n_subjects = load_stack(scale, ctx.root).shape[0]
        if args.subject_limit:
            n_subjects = min(n_subjects, args.subject_limit)
        rows, base_scores = [], {}
        with ProcessPoolExecutor(args.workers) as pool:
            for subject_index, base, r in pool.map(run_subject, [(scale, s, args.n_draws, str(ctx.root)) for s in range(n_subjects)]):
                rows.extend(r)
                for name, values in base.items():
                    base_scores.setdefault(name, {})[subject_index] = values
                print(f"scale {scale} subject {subject_index} done", flush=True)
        pd.DataFrame(rows).to_csv(out / f"per_draw_scale-{scale}.csv", index=False)
        np.savez(
            out / f"unperturbed_scores_scale-{scale}.npz",
            **{name: np.stack([d[i] for i in sorted(d)]) for name, d in base_scores.items()},
        )
        print(f"scale {scale}: {len(rows)} rows", flush=True)
    ctx.finish()


if __name__ == "__main__":
    main()
