# Persistent Homology Broadens the Controllable Subspace in Human Structural Connectomes

Analysis code accompanying the preprint:

> Carter Sale, Marco Coraggio, Mengsen Zhang, Michael J. Richardson.
> *Persistent Homology Broadens the Controllable Subspace in Human Structural Connectomes.*
> arXiv:2608.03181 — https://arxiv.org/abs/2608.03181

Network control theory applied to structural connectomes usually ranks candidate
driver nodes by connectivity strength and scores them with a scalar control energy.
This repository implements the alternative used in the paper: ranking nodes by the
**persistent topological cycles they participate in**, and comparing the two criteria
across 70 human structural connectomes at three parcellation scales (68, 114, 219).

The headline result the code reproduces: topology- and degree-informed driver sets
reach nearly identical scalar control energy (~0.2 % apart), but the *geometry* of the
controllable subspace differs substantially — topology-informed sets spread
controllability over more dimensions and yield better-conditioned controllability
matrices, an advantage that survives hub removal and translates into different classes
of reachable target states.

## Repository layout

```text
config/
  default.toml                    All analysis parameters (the primary configuration)
  landscape_fraction_5pct.toml    Variant config for the 5 %-of-N driver-set sweep
src/hoc/                          Reusable library — all analysis logic lives here
scripts/                          Heavy analyses and figure generation (write to results/, figures/)
scripts/representative_stability/ Standalone robustness check on cycle representatives
notebooks/                        Inferential statistics and paper figures (read results/, write figures/)
animations/                       Manim scene sources for the explanatory animations
tests/                            Unit tests for the numerical core
```

Directories that are **not** tracked in git: `data/` (inputs you supply) and
`runs/` (one folder per analysis run, including its figures). The pre-cleanup
`results/` and `figures/` trees are kept locally as a read-only baseline
(`baseline/PROVENANCE.md`).

### `src/hoc/`

| Module | Contents |
| --- | --- |
| `paths.py` | Repo-root resolution and directory creation |
| `config.py` | TOML config loading |
| `io.py` | CSV / NPZ / pickle writers and the `results/` tree scaffold |
| `utils.py` | Seeded RNG helpers (`subject_seed`), JSON writing |
| `dataset.py` | Raw `.mat` connectome loading, per-subject preprocessing, subject/scale metadata tables |
| `preprocessing.py` | Adjacency normalization, spectral stabilization, noise and dropout perturbations |
| `atlas.py` | Cammoun label tables, region/lobe grouping, keyword masks, node-score → surface projection |
| `control.py` | Single-node and assembled controllability Gramians, matrix exponential, geometry metrics, target energy, bottom-eigenspace overlap |
| `topology.py` | Adjacency → distance transform, `H1` persistence, cycle participation and its variants, simplicial participation |
| `rankings.py` | All node-ranking heuristics, top-*k* selection, degree-matched and random baselines |
| `landscape.py` | Driver-set enumeration/sampling, near-optimal-set summaries, seeded-greedy extension |
| `targets.py` | Construction of the named and single-parcel target states |
| `dong.py` | Pilot module for a Dong-style stochastic-tracking side analysis. **Not part of the paper pipeline** — no runner script uses it; kept for reference and covered by `tests/test_dong.py`. |

## Installation

Python ≥ 3.11.

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[gudhi]"      # second-library arm of the stability check
python -m pip install -e ".[animation]"  # Manim scenes
python -m pip install -e ".[dev]"        # pytest
```

If you would rather not install the package, the scripts add `src/` to `sys.path` at
runtime, so they run from a checkout as-is.

## Data you need to supply

The raw dataset is not redistributed here. Place it as:

```text
data/raw/Individual_Connectomes.mat
data/external/atlas_cache/atl-cammoun2012/fsaverage5/*.annot
```

- The `.mat` file is expected to contain the `connMatrices.SC` stack from the Griffa
  et al. release. Scales `68`, `114`, and `219` are used by default.
- The Cammoun 2012 atlas cache is downloaded by `neuromaps`/`netneurotools` into
  `data/external/atlas_cache/`. Surface plotting uses the `fsaverage5` mesh packaged
  with `nilearn`, so no geometry is fetched at analysis time.

## Running the pipeline

Everything is driven by `config/default.toml`; every script takes `--config`.

**Run directories.** Each run writes into its own folder, `runs/<name>/`. The name
comes from `--run`, then `$HOC_RUN`, then `main`.
- A `--subject-limit N` run always goes to `runs/<name>__nN/`, so it can't overwrite
  a full run.
- The 5 % variant goes to `runs/<name>/variants/fraction_5pct/`.
- Each run folder has a `manifest.json` recording the config hash, subjects, input
  checksums and, for each stage, the git commit, package versions and output
  checksums.
- A stage won't overwrite a finished stage, or mix configs or subject sets, unless
  you pass `--overwrite`.
- Notebooks and figure scripts read the run named by `$HOC_RUN` / `--run`, and write
  figures to `runs/<name>/figures/`.
- `--run results` reads the pre-cleanup `results/` tree (read-only; see
  `baseline/PROVENANCE.md`).

```bash
python scripts/00_prepare_data.py          # normalize connectomes, build atlas + target metadata
python scripts/01_run_primary_analysis.py  # node scores, driver sets, energy, geometry, targets
python scripts/02_run_robustness.py        # c / epsilon / horizon sweeps and perturbations
python scripts/03_run_hub_lesion.py        # degree-hub deletion with matched non-hub controls
```

Or in sequence:

```bash
python scripts/run_all.py
```

Fast smoke test on a subset of the cohort:

```bash
python scripts/run_all.py --subject-limit 2 --skip-landscape --skip-perturbations
```

The 5 %-of-N driver-set variant reported alongside the primary analysis. It is a
trimmed configuration — no `k ∈ {1,2,3,5}` arm, no perturbations, and no
`[hub_lesion]` block — so run the first three scripts directly rather than
`run_all.py`, and note that it writes to `runs/<name>/variants/fraction_5pct/`:

```bash
python scripts/00_prepare_data.py         --config config/landscape_fraction_5pct.toml
python scripts/01_run_primary_analysis.py --config config/landscape_fraction_5pct.toml
python scripts/02_run_robustness.py       --config config/landscape_fraction_5pct.toml
```

Useful flags: `--subject-limit N` (cap the cohort), `--skip-landscape` (skip set
sampling in the primary run), `--skip-perturbations` (skip the noise/dropout arm of
the robustness run).

### Figure scripts

Standalone panels that do not belong to a notebook are generated by the numbered
`scripts/*_make_*.py` files, e.g.:

```bash
python scripts/07_make_rank_divergence_panels.py     # per-scale degree/cycle divergence panels
python scripts/06_make_target_region_maps.py         # surface maps for named targets
python scripts/10_make_geometry_separate_figure.py   # degree and cycle geometry as separate lines
python scripts/11_make_target_forest.py              # forest plot of target energy differences
python scripts/12_make_target_distribution.py        # single-parcel target distribution violins
```

`scripts/08_make_reachable_state_cloud_assets.py` exports the real-data assets that
`scripts/09_reachable_state_cloud_scene.py` (a Manim scene) renders.

### Notebooks

The notebooks do **not** rerun the heavy analyses — they read what the scripts wrote
into `results/`, run the inferential statistics, and render the paper figures into
`figures/section_*/`.

| Notebook | Produces |
| --- | --- |
| `01_degree_cycle_rankings.ipynb` | Degree-vs-cycle ranking divergence panels |
| `02_rank_divergence_control_energy.ipynb` | Rank divergence against relative control energy |
| `03_landscape_degeneracy.ipynb` | Driver-set landscape sampling and near-optimal degeneracy |
| `04_static_vs_greedy_comparison.ipynb` | Static ranking vs seeded-greedy selection |
| `05_control_geometry.ipynb` | Effective rank, condition number, eigenspace overlap |
| `06_target_state_selection.ipynb` | Target-state energy statistics (`results/statistics/`) |
| `07_hub_lesion_degradation.ipynb` | Geometry and target-energy degradation after hub deletion |
| `08_robustness_summary.ipynb` | Sensitivity-sweep summary |

## Outputs

Written under `runs/<name>/` (the pre-cleanup `results/` tree has the same layout):

| Path | Contents |
| --- | --- |
| `processed/` | Preprocessed connectome stacks (`.npz`), atlas label tables, target vectors and metadata, per scale |
| `primary/` | `node_scores`, `driver_sets`, `node_level_energy`, `rank_divergence`, `landscape_summary`, `greedy_comparison`/`greedy_steps`, `target_energies`, `target_geometry`, `geometry_metrics`, `eigenspace_overlap`, `weighted_inputs` |
| `primary/landscape_samples/` | Sampled candidate-set energies (when `save_landscape_samples` is on) |
| `robustness/` | `sensitivity_summary` across `c`, `epsilon`, horizon, larger-`k`, and perturbation arms |
| `hub_lesion/` | Intact-vs-lesioned geometry and target-energy degradation with matched strategy comparisons |
| `statistics/` | Notebook-generated inferential tables (target summaries, pairwise and subject-level differences, Holm-corrected tests) |
| `metadata/` | `subjects.csv`, `scale_metadata.csv`, and a frozen `config_snapshot.toml` for the run |
| `representative_stability/` | Output of the representative-stability check (see below) |

Figures land in `figures/`, grouped into `section_*` subdirectories matching the
manuscript sections.

## Configuration

All parameters live in `config/default.toml`. Each run copies the config into
`runs/<name>/config/` and records its SHA-256 in `runs/<name>/manifest.json`. The primary values:

| Parameter | Value | Rationale |
| --- | --- | --- |
| `primary_c` | `0.1` | Between the two sensitivity values (`0.01`, `1.0`); comfortably stable continuous-time system without swamping the native adjacency structure |
| `primary_epsilon` | `1e-5` | Middle of the Tikhonov sweep (`1e-8`…`1e-2`); keeps ill-conditioned small-driver Gramians usable |
| `finite_horizons` | `1.0, 2.5, 5.0` | Short, intermediate, and longer windows without redundancy against the infinite-horizon case |
| `n_random_rankings` | `100` | Stable null estimate at manageable cost |
| `larger_k_fractions` | `0.05, 0.10, 0.20` | Checks that small-`k` conclusions persist beyond the primary `k ∈ {1,2,3,5}` |
| `parcellation_scales` | `68, 114, 219` | The first three Cammoun scales |
| `random_seed` | `20260423` | Combined with `subject_seed()` for per-subject reproducibility |

The `[dong]` block configures the pilot module in `src/hoc/dong.py` and is unused by
the paper pipeline.

## Methodological choices worth knowing

### Cycle participation

`H1` persistent homology on a Vietoris–Rips filtration of the transformed distance
matrix:

- `D_ij = 1 - A_ij` for present edges,
- `D_ij = 1` for absent edges,
- `D_ii = 0`.

Node participation `CP_i` sums feature persistence over the `H1` representatives whose
support contains node *i*, min–max normalized within each connectome before ranking.

### Supplementary ranking methods

Beyond degree strength, weighted betweenness, and cycle participation, the pipeline
implements eigenvector centrality, participation coefficient, degree-normalized cycle
participation, cycle criticality, and triangle/tetrahedron participation. The
participation coefficient uses a weighted Louvain partition estimated separately per
subject connectome.

### Target-state construction

The dataset ships no system-assignment table, so the manuscript's structured targets
are operationalized from atlas labels — transparently, and all in
[`src/hoc/targets.py`](src/hoc/targets.py):

- association masks from frontal, parietal, precuneus, cingulate, and insular keywords;
- occipital masks from occipital, cuneus, lingual, lateral-occipital, and pericalcarine keywords;
- the sparse frontocingulate target from the top six topology-scoring nodes inside the
  frontocingulate mask of the intact cohort-average reference;
- the principal gradient from the published Margulies et al. (2016) `fcgradient01` map
  (fsLR-32k), resampled to `fsaverage5` by nearest-neighbour matching on the spherical
  registration and averaged within each Cammoun parcel. This is used in preference to
  the leading nontrivial eigenvector of the structural affinity, which is dominated by
  the interhemispheric split rather than the unimodal-to-transmodal axis. Fetching the
  map requires network access via `neuromaps`.

Keyword masks are matched by `hoc.atlas.classify_region_group`, which matches `cuneus`
before `precuneus` — precuneus parcels are therefore grouped as occipital rather than
parietal in the anatomical tables. This affects grouping labels only, not any control
result.

## Representative stability

`CP_i` is computed from Ripser's `H1` cocycle representatives, which are one valid
choice among many for the same homology class.
[`scripts/representative_stability/`](scripts/representative_stability/) tests whether
the headline result survives a different representative of the *same* features, using
three schemes (Ripser cocycles, length-minimized geodesic cycles, and GUDHI's pairing
simplices) and a three-level readout: score correlation, driver-set overlap, and the
cycle-minus-degree effective-rank difference itself.

```bash
python scripts/representative_stability/run_representative_stability.py
python scripts/representative_stability/analyze_selection_anatomy.py --k-values 5 15
```

Short answer: the advantage holds in sign and significance under all three schemes for
every `k ≥ 2`, but its magnitude varies by a factor of 2–10; Ripser and GUDHI agree on
the barcode *and* the birth simplex of every feature, so the variation is about the
convention for turning a feature into a node set, not about the persistence software.
Full method and numbers in that directory's
[`README.md`](scripts/representative_stability/README.md) and
[`REPORT.md`](scripts/representative_stability/REPORT.md); the motivating plan is in
[`comper.md`](comper.md).

## Tests

```bash
pytest                                       # core numerical tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest scripts/representative_stability/
```

`tests/` covers preprocessing, Gramian additivity and assembly, ranking helpers, and
target normalization. For end-to-end checks, `--subject-limit` makes a full pipeline
run cheap.

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

Correspondence: carter.sale@mq.edu.au
