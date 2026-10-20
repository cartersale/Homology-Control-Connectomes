# Baseline record (preprint + revision draft)

Frozen record of the outputs as they were before the reproducibility cleanup.
Do not edit these files; later runs are compared against them.

| File | Contents |
| --- | --- |
| `PROVENANCE.md` | Which code, config and data produced each results folder, and the known issues |
| `MANIFEST.sha256` | SHA-256 of every file under `results/`, `figures/` and `paper/network_neuro_resubmission/supp_tables/` |
| `inputs.sha256` | SHA-256 of the raw connectomes, atlas files and neuromaps downloads (`$NEUROMAPS_DATA` = `~/neuromaps-data` by default) |
| `environment-preprint.txt` | `pip freeze` and `conda list --export` of the environment that produced the results |

## Local archive

The files themselves (`results/`, `figures/`, `supp_tables/`, `scripts/revision/`,
`config/`, and the removed `src/hoc/celegans.py` and `randi.py`) are archived off
the repository. Raw data is not included, only hashed.

- File: `~/Research/archives/hoc-baseline-2026-10-20.tar.zst` (978 MB)
- SHA-256: `ba6d94ccebce88053add708657bd03e887d3b979bbd06bd96ecb750b85121f02`

## Checking

```bash
shasum -a 256 -c baseline/MANIFEST.sha256 --quiet
sed "s#\$NEUROMAPS_DATA#${NEUROMAPS_DATA:-$HOME/neuromaps-data}#" baseline/inputs.sha256 | shasum -a 256 -c --quiet
```
