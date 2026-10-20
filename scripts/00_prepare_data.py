#!/usr/bin/env python
"""Preprocess structural connectomes and atlas metadata."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hoc.config import load_config
from hoc.dataset import build_scale_metadata, build_subject_table, load_raw_connectomes
from hoc.atlas import build_label_table
from hoc.runs import RunContext, add_run_arguments
from hoc.io import prepare_results_tree, write_csv, write_npz
from hoc.preprocessing import preprocess_structural_adjacency
from hoc.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml", help="Path to the TOML config file.")
    parser.add_argument("--subject-limit", type=int, default=None, help="Optional cap for smoke tests.")
    add_run_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(REPO_ROOT / args.config)
    ctx = RunContext.from_args("00_prepare_data", args)
    results = prepare_results_tree(ctx.root)

    raw_connectomes = load_raw_connectomes(config["paths"]["raw_connectomes"])
    scales = [int(scale) for scale in config["analysis"]["parcellation_scales"]]
    atlas_dir = config["paths"]["atlas_dir"]

    n_subjects = next(iter(raw_connectomes.values())).shape[2]
    if args.subject_limit is not None:
        n_subjects = min(n_subjects, args.subject_limit)
    ctx.begin(build_subject_table(n_subjects)["subject_id"].tolist())

    write_csv(build_subject_table(n_subjects), results["metadata"] / "subjects.csv")
    write_csv(build_scale_metadata(scales, atlas_dir), results["metadata"] / "scale_metadata.csv")

    config_snapshot = results["metadata"] / "config_snapshot.toml"
    shutil.copy2(REPO_ROOT / args.config, config_snapshot)
    write_json(results["metadata"] / "config_summary.json", config)

    for scale in tqdm(scales, desc="Scales"):
        stack = raw_connectomes[scale][:, :, :n_subjects]
        processed_stack = np.stack(
            [
                preprocess_structural_adjacency(stack[:, :, subject_index])
                for subject_index in tqdm(range(n_subjects), desc=f"Scale {scale}", leave=False)
            ],
            axis=0,
        )
        cohort_reference = np.mean(processed_stack, axis=0)
        labels = build_label_table(scale=scale, atlas_dir=atlas_dir)
        write_npz(
            results["processed"] / f"connectomes_scale-{scale}.npz",
            adjacency=processed_stack,
            cohort_reference=cohort_reference,
        )
        write_csv(labels, results["processed"] / f"atlas_labels_scale-{scale}.csv")

    ctx.finish()


if __name__ == "__main__":
    main()
