# Homology-Control-Connectomes

Analysis code for:

> Carter Sale, Marco Coraggio, Mengsen Zhang, Michael J. Richardson.
> *Persistent Homology Broadens the Controllable Subspace in Human Structural Connectomes.*
> arXiv:2608.03181 — https://arxiv.org/abs/2608.03181

The code applies network control theory to human structural connectomes: 70 subjects,
at Cammoun parcellation scales 68, 114 and 219. For each connectome it:

- ranks candidate driver nodes by many criteria, including degree strength and
  participation in persistent `H1` cycles (persistent homology);
- builds controllability Gramians for the resulting driver sets;
- measures the energy and geometry of the controllable subspace (effective rank,
  conditioning, eigenvalue spectra);
- repeats the comparison under sensitivity sweeps, null networks, hub lesions and
  alternative target states.

This README explains how the repository is organised and how to run it. The paper
itself covers the methods in full and the results.

## Contents

- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data](#data)
- [How a run works](#how-a-run-works)
- [Reproducing everything](#reproducing-everything)
- [Pipeline stages](#pipeline-stages)
- [Revision analyses](#revision-analyses)
- [Paper tables and numbers](#paper-tables-and-numbers)
- [Tests](#tests)
- [Configuration](#configuration)
- [Conventions in the code](#conventions-in-the-code)
- [Citation](#citation)

## Quick start

```bash
git clone https://github.com/cartersale/Homology-Control-Connectomes.git
cd Homology-Control-Connectomes
uv sync --frozen --all-extras          # exact locked environment (Python 3.12)
uv run pytest                          # unit, numerical and synthetic regression tests (~30 s)
uv run python scripts/run_all.py --config config/smoke.toml --run smoke
                                       # whole main pipeline on synthetic data (~20 s, no downloads)
```

None of this needs the real dataset. To run the real analyses, get the [data](#data)
first and then follow [Reproducing everything](#reproducing-everything).

## Repository layout

```text
config/                        Analysis parameters (TOML)
  default.toml                   primary configuration
  landscape_fraction_5pct.toml   variant: driver sets of 5 % of N
  smoke.toml                     synthetic data, small grids (tests and CI)
src/hoc/                       Library: all core calculations live here
scripts/
  00–03_*.py                     main pipeline stages
  03–12_make_*.py                figure scripts for the main pipeline
  run_all.py                     runs stages 00–03 in order
  reproduce.py                   runs every analysis behind the paper, resumably
  compare_runs.py                file-by-file numerical diff of two runs
  representative_stability/      check of cycle-representative choice
  revision/                      analyses added in the journal revision (one folder per topic)
  paper/                         builds supplement tables and number macros from a run
notebooks/                     Statistics and figures for the main pipeline (read a run)
tests/                         pytest suite; tests/golden/ holds the pinned reference values
baseline/                      Checksums and provenance of the preprint-era outputs
animations/                    Manim sources for the explanatory animations
```

The following are **not** tracked in git:
- `data/` holds the inputs you supply.
- `runs/` holds one folder per analysis run.

The pre-cleanup `results/` and `figures/` trees are kept locally, read-only, as a baseline;
`baseline/README.md` describes them.

### `src/hoc/`

| Module | Contents |
| --- | --- |
| `config.py`, `paths.py` | TOML config loading, repo-relative paths |
| `runs.py` | Run directories, `manifest.json`, overwrite protection, stage dependencies |
| `threads.py` | Pins BLAS threads (default 1) so results don't depend on thread count |
| `dataset.py` | Loads the `.mat` connectomes, preprocesses each subject, builds subject/scale tables |
| `preprocessing.py` | Adjacency normalisation, spectral stabilisation, noise and dropout perturbations |
| `atlas.py` | Cammoun label tables, anatomical grouping, exact region-set masks, surface projection |
| `control.py` | Single-node and assembled Gramians (infinite and finite horizon), geometry metrics, target energy |
| `topology.py` | Adjacency → distance transform, `H1` persistence, cycle participation and variants, simplicial participation |
| `rankings.py` | Every node-ranking criterion, top-*k* selection, degree-matched and random baselines |
| `landscape.py` | Driver-set enumeration and sampling, near-optimal-set summaries, seeded greedy extension |
| `targets.py` | Named and single-parcel target states, including the Margulies principal gradient |
| `synthetic.py` | Seeded synthetic connectomes, labels and gradient for tests and the smoke config |
| `io.py`, `utils.py` | Table and array writers, seeded RNG helpers (`subject_seed`) |
| `dong.py` | Pilot stochastic-tracking module. **Not used by the paper pipeline**; kept for reference and tested in `tests/test_dong.py` |

## Installation

The environment is locked with [uv](https://docs.astral.sh/uv/). `uv.lock` pins every
package. The packages that affect the numbers (numpy, scipy, ripser, gudhi, bctpy,
networkx, neuromaps, …) are held at the versions that produced the published results.

```bash
uv sync --frozen --all-extras    # everything, including gudhi and netneurotools
uv sync --frozen                 # core pipeline only
```

Extras: `gudhi` (second persistence library for the representative check),
`revision` (gudhi and netneurotools, used by the revision analyses), `animation` (Manim),
`dev` (pytest, ruff). Run commands with `uv run …`, or activate `.venv/`.

Without uv, `python -m pip install -e ".[revision,dev]"` on Python 3.12 works. But it
resolves versions fresh, so numbers may differ in the last digits from a locked run.

## Data

The raw data is not redistributed. Every file the analyses read is listed, with its
SHA-256, in `baseline/inputs.sha256`.

| What | Where it goes | How to get it |
| --- | --- | --- |
| Structural connectomes (Griffa et al., 70 subjects) | `data/raw/Individual_Connectomes.mat` | Download from Zenodo: Griffa, Alemán-Gómez & Hagmann (2019), [doi:10.5281/zenodo.2872624](https://doi.org/10.5281/zenodo.2872624) (701 MB, CC BY 4.0, MD5 `87ef63debc0cbe44afc1e6cf9ab9717e`). The code reads the `connMatrices.SC` stack. |
| Cammoun 2012 atlas, fsaverage5 `.annot` files | `data/external/atlas_cache/atl-cammoun2012/fsaverage5/` | `uv run python -c "from netneurotools.datasets import fetch_cammoun2012; fetch_cammoun2012('fsaverage5', data_dir='data/external/atlas_cache')"` |
| Yeo 2011 7- and 17-network `.annot` files (revision target analyses only) | `data/external/yeo2011_fsaverage5/` | From the [CBIG repository](https://github.com/ThomasYeoLab/CBIG): `stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/fsaverage5/label/` |
| Margulies et al. (2016) gradients and fsLR/fsaverage surfaces | `$NEUROMAPS_DATA` (default `~/neuromaps-data`) | Downloaded on first use by `neuromaps` (see below) |
| fsaverage5 / fsaverage6 meshes for surface figures | nilearn's data folder | Downloaded on first use by `nilearn` |

**Margulies gradient.** The principal-gradient target needs network access the first
time it is built, in stage `01_primary` and in the revision target scripts:
- `neuromaps` downloads the `margulies2016` `fcgradient01` and `fcgradient02` maps
  (fsLR 32k), plus the fsLR 32k and fsaverage 10k atlases used to resample them;
- the total is about 10 MB, cached under `$NEUROMAPS_DATA`.

Set `NEUROMAPS_DATA` to keep the cache elsewhere. To fetch everything ahead of time,
e.g. before working offline or on a cluster node:

```bash
uv run python -c "
from neuromaps.datasets import fetch_annotation, fetch_atlas
for g in ('fcgradient01', 'fcgradient02'):
    fetch_annotation(source='margulies2016', desc=g, space='fsLR', den='32k')
fetch_atlas('fsLR', '32k'); fetch_atlas('fsaverage', '10k')"
```

Check that your inputs match the ones used for the paper:

```bash
sed "s#\$NEUROMAPS_DATA#${NEUROMAPS_DATA:-$HOME/neuromaps-data}#" baseline/inputs.sha256 | shasum -a 256 -c --quiet
```

## How a run works

Every script writes into a **run directory**, `runs/<name>/`.

- **The name** comes from `--run`, then `$HOC_RUN`, then `main`.
- **Smaller runs:** a run with `--subject-limit N` always goes to `runs/<name>__nN/`, so a
  quick test can't overwrite or mix with a full run.
- **Variants:** configuration variants go to `runs/<name>/variants/<variant>/`. For example,
  the 5 % config goes to `variants/fraction_5pct/`.
- **Manifest:** `runs/<name>/manifest.json` records the config and its SHA-256, the subject
  list and the input checksums. For every stage it also records the git commit (and whether
  the tree was dirty), the command line, package versions, BLAS thread pools, start and end
  times, and the checksum of every output.
- **Protection:** a stage refuses to run if it would overwrite a finished stage, or mix
  configs or subject sets in one run, unless you pass `--overwrite`.
- **Dependencies:** stages check that the stages they depend on have finished in the same
  run.
- **Readers:** notebooks and figure scripts read the run named by `--run` / `$HOC_RUN`, and
  write figures to `runs/<name>/figures/`.
- **Legacy tree:** `--run results` reads the legacy `results/` tree for comparison. It is
  read-only, and anything derived from it goes to `runs/results-derived/`.

Environment variables:

| Variable | Effect |
| --- | --- |
| `HOC_RUN` | Default run name |
| `HOC_RUNS_ROOT` | Where run directories live (default `runs/`) |
| `HOC_N_JOBS` | Worker processes for parallel steps (default per script, usually 10–12) |
| `HOC_BLAS_THREADS` | BLAS threads per process (default 1) |
| `NEUROMAPS_DATA` | neuromaps download cache |
| `HOC_PAPER_DIR` | Local manuscript folder, used only by `scripts/paper/` checks |

## Reproducing everything

`scripts/reproduce.py` runs every analysis behind the paper, in dependency order, into one
run:

```bash
uv run python scripts/reproduce.py --run canonical --list      # the 60 steps and which are done
uv run python scripts/reproduce.py --run canonical             # run (or resume) all of them
uv run python scripts/reproduce.py --run canonical --only nulls lesion
uv run python scripts/reproduce.py --run canonical --from revision/q1/01_compute_q1
```

- **Resuming:** finished steps are skipped, so an interrupted run carries on where it
  stopped. The two longest steps (`revision/nulls/01_run_nulls` and
  `revision/lesion/01_run_reciprocal_lesion`) also save their work per subject, so they
  resume mid-step.
- **Logs:** each step logs to `runs/<name>/logs/`, and the timings go to
  `logs/reproduce_timings.json`.
- **Order:** the main pipeline (00–03), then the 5 % variant, the representative-stability
  runs, and the revision analyses folder by folder.

Approximate times on a 12-core laptop:
- stages 00–03: about 1.5 h, mostly `01_primary`;
- the 5 % variant: 30 min;
- most revision folders: minutes each.

The null-network step is by far the longest, close to a day, followed by the lesion step.
Run it plugged in (`caffeinate -i` on macOS), or on a workstation.

To check one run against another, use `compare_runs.py`. It matches table rows on their
identifying columns and reports the largest absolute and relative difference per column:

```bash
uv run python scripts/compare_runs.py --new canonical --ref results --out comparison.md
```

## Pipeline stages

The main pipeline is four scripts. `scripts/run_all.py` runs them in order. Every script
takes `--config` and `--run`.

| Stage | Script | Does |
| --- | --- | --- |
| `00_prepare_data` | `scripts/00_prepare_data.py` | Normalise connectomes, build atlas label tables and target metadata |
| `01_primary` | `scripts/01_run_primary_analysis.py` | Node scores, driver sets, energies, geometry, target energies, landscape sampling, greedy comparison |
| `02_robustness` | `scripts/02_run_robustness.py` | Sweeps over `c`, `ε`, horizon and larger `k`; noise and dropout perturbations |
| `03_hub_lesion` | `scripts/03_run_hub_lesion.py` | Delete degree hubs and compare with matched non-hub deletions |

Useful flags:
- `--subject-limit N` caps the cohort;
- `--skip-landscape` skips set sampling in stage 01;
- `--skip-perturbations` skips the noise and dropout arm of stage 02.

The 5 %-of-N variant needs only stages 00 and 02:

```bash
uv run python scripts/00_prepare_data.py   --config config/landscape_fraction_5pct.toml --run canonical
uv run python scripts/02_run_robustness.py --config config/landscape_fraction_5pct.toml --run canonical \
    --skip-perturbations --include-landscape-summary
```

Outputs under `runs/<name>/`:

| Folder | Contents |
| --- | --- |
| `processed/` | Preprocessed connectome stacks (`.npz`), label tables, target vectors, per scale |
| `primary/` | Node scores, driver sets, node-level energy, rank divergence, landscape and greedy summaries, target energies, geometry metrics, eigenspace overlap |
| `robustness/` | `sensitivity_summary` across the `c`, `ε`, horizon, larger-`k` and perturbation arms |
| `hub_lesion/` | Intact-vs-lesioned geometry and target energy |
| `statistics/` | Inferential tables written by stage 01 |
| `metadata/` | Subject and scale tables |
| `representative_stability/` | Output of the representative check (below) |
| `revision/<topic>/` | Revision analyses (below) |
| `paper/` | Generated supplement tables and number macros (below) |
| `figures/` | Figures from notebooks and figure scripts |

**After the pipeline: figures and notebooks**, in this order. Each one reads a finished
run, given by `--run`, or by `$HOC_RUN` for notebooks.

1. **Stages 00–03 and the 5 % variant.** These must have finished.
2. **`scripts/08_make_reachable_state_cloud_assets.py`.** This exports the assets that the
   Manim scene `scripts/09_reachable_state_cloud_scene.py` renders.
3. **Notebooks `01`–`08`.** They run the statistics and draw the main-text figures.
   Notebooks 03 and 08 also read the 5 % variant.

   | Notebook | Produces |
   | --- | --- |
   | `01_degree_cycle_rankings` | Degree-vs-cycle ranking divergence |
   | `02_rank_divergence_control_energy` | Rank divergence against relative control energy |
   | `03_landscape_degeneracy` | Driver-set landscape and near-optimal degeneracy |
   | `04_static_vs_greedy_comparison` | Static ranking vs seeded greedy selection |
   | `05_control_geometry` | Effective rank, condition number, eigenspace overlap |
   | `06_target_state_selection` | Target-state energy statistics |
   | `07_hub_lesion_degradation` | Degradation after hub deletion |
   | `08_robustness_summary` | Sensitivity-sweep summary |

4. **The remaining figure scripts**, in any order:
   - `03_make_conceptual_degree_cycle_figure`
   - `06_make_target_*`
   - `07_make_rank_divergence_panels`, which downloads nilearn's `fsaverage6`
   - `10_make_geometry_separate_figure`
   - `11_make_target_forest`
   - `12_make_target_distribution`

## Revision analyses

`scripts/revision/` holds the analyses added for the journal revision, one folder per
topic. Each folder's scripts are numbered in the order they run. Each script declares the
stages it needs and writes to `runs/<name>/revision/<topic>/`. `reproduce.py` runs them in
this order:

| Folder | Topic |
| --- | --- |
| `spectra/` | Full Gramian spectra, random and matched baselines, Perron-vector loading |
| `regime/` | Validation against `hoc.control` and nctpy; sweeps over `c`, horizon and `ε`; the energy floor |
| `modal/` | Set-level Gramian concentration vs single-node modal and average controllability |
| `nulls/` | Strength-, geometry- and weight-preserving null networks, and what drives the spectral gap |
| `q1/` | Set tables led by leading-eigenvalue share (`q1`); decay collapse; redundancy |
| `mechanism/` | Why degree-selected hubs load the leading eigenvector; canonical graphs; coverage distance |
| `targets/` | Yeo-network and principal-gradient targets: energies, statistics, spin tests, enrichment, proximity controls |
| `lesion/` | Reciprocal hub lesions (degree and cycle hubs, static and recomputed drivers) |
| `stability/` | Cycle representatives at every scale; stability of rankings under edge-weight perturbations |
| `figures/` | Figure 1 panels and layout |

**Representative stability** (`scripts/representative_stability/`). Cycle participation
is computed from Ripser's `H1` cocycle representatives, and those are one valid choice of
representative among many for the same feature. These scripts recompute the analysis with:
- length-minimised (geodesic) representatives;
- GUDHI's representatives.

They compare the result with the Ripser baseline at three levels: score correlation,
driver-set overlap, and the effective-rank difference. Method and options are in that
folder's `README.md`.

## Paper tables and numbers

The manuscript is not part of this repository. The values it quotes are generated from a
run:

```bash
uv run python scripts/paper/make_tables.py  --run canonical   # supplement tables → runs/canonical/paper/tables/
uv run python scripts/paper/make_numbers.py --run canonical   # numbers.tex, key_results.json, numbers.md
```

- **Tables:** `make_tables.py` has one module per supplement section (`s4.py` … `s11.py`,
  `s_canonical.py`). It writes each table as a `.tex` file and its description to
  `notes/`.
- **Numbers:** `make_numbers.py` writes every number quoted in the text as a LaTeX macro in
  `numbers.tex`. The file `numbers.md` lists each macro, the file and filter it comes from,
  and the draft line where it is quoted.
- **Checks:** `--check` compares the output with a local manuscript folder (`--paper` or
  `$HOC_PAPER_DIR`).
- **Copying:** `scripts/paper/sync_paper.py` copies the tables and macros into that folder.
  By default it only lists what it would copy; `--apply` copies them.

## Tests

```bash
uv run pytest                 # everything except tests that need the real data
uv run pytest -m data         # tests that read a real run (paper numbers)
```

What the tests cover:

- **Closed-form and numerical checks** of the core calculations:
  - Gramians: diagonal closed forms, Lyapunov residual, finite-horizon convergence, the
    eigenvalue floor;
  - geometry metrics on known spectra;
  - persistent homology on graphs with known cycles;
  - ranking ties and seeds;
  - landscape enumeration;
  - target masks.
- **`test_runs.py`:** run directories and overwrite protection.
- **`test_synthetic_pipeline.py`:** runs the full pipeline on the synthetic config and
  compares 25 output tables with `tests/golden/synthetic.json`, so any change in a number
  shows up. After an intended change, regenerate the golden file with
  `uv run python tests/test_synthetic_pipeline.py --update` and explain the change in the
  PR.
- **`test_paper_numbers.py`** (marker `data`): compares a run's `key_results.json` with
  `tests/golden/key_results.json`.

## Configuration

All parameters are in `config/default.toml`. Each run copies the config into
`runs/<name>/config/` and records its hash in the manifest. Main values:

| Parameter | Value | Meaning |
| --- | --- | --- |
| `primary_c` | `0.1` | Stabilisation shift of the continuous-time system (`0.01` and `1.0` in sensitivity runs) |
| `primary_epsilon` | `1e-5` | Eigenvalue floor for the geometry metrics (`1e-8` … `1e-2` in sensitivity runs) |
| `finite_horizons` | `1.0, 2.5, 5.0` | Finite control horizons, alongside the infinite horizon |
| `top_k_values` | `1, 2, 3, 5` | Driver-set sizes; `larger_k_fractions` adds 5, 10 and 20 % of N |
| `n_random_rankings` | `100` | Random driver sets per subject and `k` |
| `parcellation_scales` | `68, 114, 219` | Cammoun scales |
| `random_seed` | `20260423` | Combined with `subject_seed()` for per-subject seeds |

The `[dong]` block configures `src/hoc/dong.py` and is not used by the paper pipeline.

## Conventions in the code

- **Distance for persistent homology.**
  - The transform is `D_ij = 1 − A_ij` for present edges, `1` for absent edges, and `0`
    on the diagonal.
  - `H1` persistence comes from a Vietoris–Rips filtration of `D` (Ripser).
  - A node's cycle participation is the summed persistence of the `H1` representatives
    whose support contains it. It is min–max normalised within each connectome.
- **Ranking ties** are broken by node index (lower index first). All randomness is seeded:
  - `subject_seed()` derives a seed from the config seed and integer identifiers, such as
    scale and subject;
  - Louvain (for the participation coefficient) uses a fixed seed.
- **Anatomical masks** for the association and occipital targets are exact sets of
  Desikan–Killiany base-region names, listed in `[anatomy]` in the config. A name missing
  from the atlas is an error.
- **Principal gradient.** The Margulies et al. (2016) `fcgradient01` map is resampled from
  fsLR 32k to fsaverage5 through the spherical registration, then averaged within each
  Cammoun parcel.
- **BLAS threads** default to one per process (`src/hoc/threads.py`), so results don't
  depend on the machine's thread count. Parallelism comes from worker processes
  (`HOC_N_JOBS`).

## Citation

```bibtex
@article{sale2026persistent,
  title   = {Persistent Homology Broadens the Controllable Subspace in Human Structural Connectomes},
  author  = {Sale, Carter and Coraggio, Marco and Zhang, Mengsen and Richardson, Michael J.},
  journal = {arXiv preprint arXiv:2608.03181},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.03181}
}
```

If you use the connectome data, also cite Griffa, Alemán-Gómez & Hagmann (2019),
doi:10.5281/zenodo.2872624.

Correspondence: carter.sale@mq.edu.au
