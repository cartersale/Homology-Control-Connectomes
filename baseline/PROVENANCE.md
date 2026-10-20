# Provenance of the preprint and revision-draft outputs

Recorded 2026-10-20 (issue #1). This file describes the `results/`, `figures/` and
supplement-table files **as they exist at the time of recording**, before any
cleanup. Checksums of every file are in `MANIFEST.sha256`; checksums of the
inputs are in `inputs.sha256`; the Python environment is in
`environment-preprint.txt`.

## Summary

`results/` is **not the output of a single run**. Files were written between
2026-04-23 and 2026-10-19 by several partial runs, from code that was only
committed later (the first commit is 2026-08-07). The only config record,
`results/metadata/config_snapshot.toml`, comes from the first run (2026-04-23)
and does not describe the later runs. This is why some numbers did not
reproduce during review.

The canonical rerun in issue #3 replaces this tree. Until then, treat these files
as a record of what was reported, not as reproducible outputs.

## Code versions

| Tag / commit | Date | What it is |
| --- | --- | --- |
| `v0.1.0-preprint` = `72e27b6` | 2026-08-07 | Code published with arXiv:2608.03181. Committed **after** the preprint results were computed, so it is the best available match, not a verified one. |
| `revision-draft-2026-10-19` | 2026-10-20 | `scripts/revision/` exactly as it produced `results/revision/`, plus the supplement-table builders rescued from a temporary folder (`scripts/revision/paper/_rescued_2026-10-19/`). |

## Environment

Every result was computed in the base conda environment `/opt/anaconda3`
(Python 3.12.2, macOS arm64, numpy with Apple Accelerate). Key versions: numpy
2.1.3, scipy 1.15.2, ripser 0.6.14, persim 0.3.8, bctpy 0.6.1, networkx 3.4.2,
neuromaps 0.0.7, netneurotools 0.3.0, nilearn 0.13.1, pandas 2.2.3, gudhi 3.13.0,
statsmodels 0.14.4, joblib 1.4.2. No BLAS thread pinning, except
`OMP_NUM_THREADS=1` in some revision scripts.

## Inputs

| Input | Source | Checksum |
| --- | --- | --- |
| `data/raw/Individual_Connectomes.mat` | Griffa et al. structural connectomes (70 subjects, 5 Lausanne scales) | `inputs.sha256` |
| Cammoun 2012 fsaverage5 `.annot` (5 resolutions × 2 hemispheres) | netneurotools `fetch_cammoun2012` naming; read locally, never fetched by the code | `inputs.sha256` |
| Yeo 2011 7/17-network fsaverage5 `.annot` | CBIG; used only by `scripts/revision/targets/` | `inputs.sha256` |
| Margulies 2016 `fcgradient01/02` (fsLR 32k), fsLR 32k and fsaverage 10k atlases | downloaded by neuromaps into `$NEUROMAPS_DATA` (default `~/neuromaps-data`) | `inputs.sha256` |

## Results folders

Dates are file modification times.

| Folder | Written | Producer | Notes |
| --- | --- | --- | --- |
| `metadata/` | 2026-04-23 | `scripts/00_prepare_data.py` | Snapshot of the **first** run only: `larger_k_fractions = [0.05, 0.10]`, `sensitivity_epsilon_values = [1e-6, 1e-5, 1e-4]`, no `[hub_lesion]` block. Does not match the current `config/default.toml` or the later runs. |
| `processed/connectomes_*`, `atlas_labels_*` | 2026-04-23 | `00_prepare_data.py` | |
| `processed/targets_*`, `target_metadata_*` | 2026-07-10 | `01_run_primary_analysis.py` (`build_targets`) | Rewritten in the July run, after the Margulies gradient registration fix. |
| `primary/*.csv` | 2026-07-10 23:42 | `01_run_primary_analysis.py` | One run: 70 subjects, scales 68/114/219, k ∈ {1,2,3,5}, 10 ranking methods. Config at the time was not recorded. |
| `primary/landscape_samples/` | 2026-06-18 | `01_run_primary_analysis.py` | From an **earlier** run than the CSVs beside them (the July run did not rewrite them). |
| `statistics/target_*` (5 files) | 2026-07-10 | `01_run_primary_analysis.py` | Written by script 01, not by a notebook as the README says. |
| `statistics/primary_statistics.csv`, `target_energy_summary.csv` | 2026-04-24 | **No writer in the committed code** | Stale; do not cite. |
| `robustness/sensitivity_summary.csv` | 2026-06-23 | `02_run_robustness.py` | Families `finite_horizon` and `stabilization_epsilon` only (no perturbation arm). c ∈ {0.01, 1.0}; ε grid 1e-8…1e-2 (7 values); T ∈ {1, 2.5, 5}; k includes the larger-k values 6–44. Matches neither the April snapshot nor a single current config. |
| `hub_lesion/` | 2026-05-27 | `03_run_hub_lesion.py`; `*_strategy_comparison.csv` rewritten by notebook 07 | hub counts 5/10/15. |
| `results_fraction_5pct/` | 2026-05-14 | `02_run_robustness.py` with `config/landscape_fraction_5pct.toml` | Only `robustness/sensitivity_summary.csv` (ε = 1e-5, k = 3/6/11). `metadata/` is empty. The config says `results_dir = "results_fraction_5pct"` (repo root), but the files are under `results/`, and notebook 03 reads the root path. |
| `representative_stability/`, `representative_stability_below_death/` | 2026-08-02 | `scripts/representative_stability/` | |
| `lumpability/` | 2026-08-27 | **No writer in the repository** | Exploratory C. elegans analysis, not part of the paper; the modules were removed (kept in the local archive). |
| `paper/` | 2026-08-06/07 | — | PDFs of the preprint and abstracts; not outputs. |
| `revision/spectra`, `regime`, `modal`, `targets`, `lesion`, `stability` | 2026-09-25 … 09-28 | `scripts/revision/<name>/` | `stability/existing_script_scale-*` and `reproduce_existing_114` were made by running `scripts/representative_stability/` by hand. |
| `revision/nulls` | 2026-09-29 … 10-19 | `scripts/revision/nulls/` | `04_rayleigh_bound.py` is not in `run_all.sh`. |
| `revision/q1`, `mechanism` | 2026-10-01 | `scripts/revision/q1/`, `mechanism/` | |
| `figures/` | 2026-04-24 … 09-28 | notebooks and `scripts/*_make_*.py` | `figures/formatted_figs/parcellations.png` has no generating script. |

## Supplement tables (`paper/network_neuro_resubmission/supp_tables/`)

All were built on 2026-10-19 by the scripts now under
`scripts/revision/paper/_rescued_2026-10-19/` (map in that folder's README),
reading the `results/revision/` files above. The split into per-table files had
no script, and `S_canonical.tex` was written by hand; its values match
`results/revision/mechanism/canonical_summary.csv`.

## Known issues in the revision draft (to fix from the canonical run)

| Where | Text | Saved result |
| --- | --- | --- |
| main.tex L188 | strength-preserving null at 114: 0.75, Holm p = 0.001 | `nulls/inference.csv` (k = avg): 0.7447, p = 6.1e-4. The 0.001 is from the avg2+ row (p = 9.2e-4, mean 0.707). |
| main.tex L117 | random single regions 95%, 96%, 96% | 114 value is 0.9546 → 95% |
| main.tex L119 | CI (0.68–0.98) at 219 | `ci_lo` = 0.669 → 0.67 |
| supp_mat.tex L315 | 3.36 for cycle-participation sets | 3.36 is the static degree-lesion row of S11_5; the intact cycle set is 3.27 |
| main.tex L135 | leading-mode shares; L_S–q1 ρ | Computed only by scratch `a1.py`/`a2.py`. Pools all families incl. matched sets, unlike S7_4. |
| main.tex L190 | r_W and R2/R3 shares | Computed only inside `make_s8.py` |
| Fig. 1 caption | q1 0.78 / 0.51 / 0.36; "strongest 7%" | Printed to stdout by `fig1_panels.py`; "7%" has no source |
| main.tex L245 | single isolated region | No source found |
| Old submission | Table S2 | Matches no saved run (reviewer 2, point 3) |

Determinism: `bct.community_louvain` in `src/hoc/rankings.py:48` is unseeded,
so `participation_coefficient` scores and their driver sets are not reproducible
run to run.
