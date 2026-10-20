#!/usr/bin/env python
"""Spin tests for parcelwise correlations (R2 minor; R1 Minor 6).

Rank divergence = degree rank - cycle rank from subject-averaged scores (as scripts/07, positive = cycle priority).
Correlated (Spearman) with: Margulies G1 (legacy repo projection -> reproduces paper values; and correctly
registered projection), Margulies G2 (registered). Also degree vs cycle participation (subject-averaged scores).
Nulls: 10,000 spherical rotations of parcel centroids on the fsaverage5 sphere, reassigned with the Vasa (2018)
method and, secondarily, the original Alexander-Bloch (2018) method (neuromaps.nulls.spins.gen_spinsamples),
seed 20260423. Two-sided p = (1 + #|rho_null| >= |rho_obs|) / (1 + n_spins). The rank-divergence map is spun.

Run: python scripts/revision/targets/05_spin_tests.py --run <name> [--overwrite]  (reads/writes runs/<name>/)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from neuromaps.nulls.spins import gen_spinsamples
from scipy.stats import rankdata, spearmanr

import common as C

ctx = C.start_stage(__file__, ["00_prepare_data", "01_primary"])

warnings.filterwarnings("ignore")
N_SPIN = 10000


def rank_table(scale: int) -> pd.DataFrame:
    ns = pd.read_csv(C.ROOT / "primary" / "node_scores.csv", usecols=["scale", "node_index", "method", "score"])
    ns = ns[(ns.scale == scale) & ns.method.isin(C.METHODS)]
    w = ns.groupby(["node_index", "method"]).score.mean().unstack("method").sort_index()
    w["degree_rank"] = w.degree_strength.rank(ascending=False, method="min")
    w["cycle_rank"] = w.cycle_participation.rank(ascending=False, method="min")
    w["divergence"] = w.degree_rank - w.cycle_rank
    return w


def spin_p(x: np.ndarray, y: np.ndarray, spins: np.ndarray) -> tuple[float, float, float]:
    rx, ry = rankdata(x), rankdata(y)
    obs = np.corrcoef(rx, ry)[0, 1]
    null = np.array([np.corrcoef(rx[spins[:, i]], ry)[0, 1] for i in range(spins.shape[1])])
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + spins.shape[1])
    return float(obs), float(p), float(np.percentile(np.abs(null), 95))


rows = []
maps_out = []
for scale in C.SCALES:
    lab = C.labels(scale)
    coords = C.centroids(scale, "sphere")
    hemiid = (lab.hemi == "R").astype(int).to_numpy()
    spins = {m: gen_spinsamples(coords, hemiid, n_rotate=N_SPIN, method=m, seed=C.SEED) for m in ("vasa", "original")}
    rt = rank_table(scale)
    maps = {
        "G1_legacy_projection": C.margulies_parcel(scale, 1, registered=False),
        "G1_registered": C.margulies_parcel(scale, 1),
        "G2_registered": C.margulies_parcel(scale, 2),
    }
    div = rt.divergence.to_numpy()
    tests = [("rank_divergence", name, div, m) for name, m in maps.items()]
    tests.append(("degree_strength", "cycle_participation", rt.degree_strength.to_numpy(), rt.cycle_participation.to_numpy()))
    tests += [("degree_strength", name, rt.degree_strength.to_numpy(), m) for name, m in maps.items() if "legacy" not in name]
    tests += [("cycle_participation", name, rt.cycle_participation.to_numpy(), m) for name, m in maps.items() if "legacy" not in name]
    for xname, yname, x, y in tests:
        r = spearmanr(x, y)
        row = dict(scale=scale, x=xname, y=yname, spearman_rho=r.statistic, parametric_p=r.pvalue, n_parcels=len(x), n_spins=N_SPIN)
        for m, sp in spins.items():
            _, p, q95 = spin_p(x, y, sp)
            row[f"p_spin_{m}"] = p
            row[f"null_abs_rho_95_{m}"] = q95
        rows.append(row)
    maps_out.append(pd.DataFrame(dict(scale=scale, node_index=rt.index, region_label=lab.region_label, degree_mean=rt.degree_strength.values,
                                      cycle_mean=rt.cycle_participation.values, rank_divergence=div, **maps)))
    print("done", scale)
res = pd.DataFrame(rows)
res.to_csv(C.OUT / "spin_tests.csv", index=False)
pd.concat(maps_out).to_csv(C.OUT / "parcel_maps_for_spin.csv", index=False)
print(res.round(4).to_string())
ctx.finish()
