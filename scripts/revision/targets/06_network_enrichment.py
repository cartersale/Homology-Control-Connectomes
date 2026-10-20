#!/usr/bin/env python
"""Yeo-7 network enrichment of rank divergence (degree rank - cycle rank; positive = cycle priority).

Statistic: mean divergence of parcels assigned to each network; null from 10,000 Vasa spins (seed 20260423);
two-sided p = (1 + #|null - mean(null)| >= |obs - mean(null)|)/(1 + n). Also counts of each network among the
top-15 cycle-only and degree-only parcels.

Run: python scripts/revision/targets/06_network_enrichment.py --run <name> [--overwrite]  (reads/writes runs/<name>/)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from neuromaps.nulls.spins import gen_spinsamples

import common as C

ctx = C.start_stage(__file__, ["00_prepare_data", C.stage("01_build_targets"), C.stage("05_spin_tests")])

warnings.filterwarnings("ignore")
maps = pd.read_csv(C.OUT / "parcel_maps_for_spin.csv")
assign = pd.read_csv(C.OUT / "parcel_network_assignment.csv")
rows, top_rows = [], []
for scale in C.SCALES:
    m = maps[maps.scale == scale].sort_values("node_index")
    a = assign[assign.scale == scale].sort_values("node_index")
    lab = C.labels(scale)
    spins = gen_spinsamples(C.centroids(scale, "sphere"), (lab.hemi == "R").astype(int).to_numpy(), n_rotate=10000, method="vasa", seed=C.SEED)
    div = m.rank_divergence.to_numpy()
    net = a.yeo7_name.to_numpy()
    for name in C.YEO7_NAMES.values():
        mask = net == name
        if not mask.any():
            continue
        obs = div[mask].mean()
        null = np.array([div[spins[:, i]][mask].mean() for i in range(spins.shape[1])])
        p = (1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) / (1 + len(null))
        rows.append(dict(scale=scale, network=name, n_parcels=int(mask.sum()), mean_rank_divergence=obs, null_mean=null.mean(), p_spin_vasa=p))
    deg_rank = m.degree_mean.rank(ascending=False).to_numpy()
    cyc_rank = m.cycle_mean.rank(ascending=False).to_numpy()
    top_d, top_c = deg_rank <= 15, cyc_rank <= 15
    for cat, mask in (("cycle_only", top_c & ~top_d), ("degree_only", top_d & ~top_c), ("both", top_c & top_d)):
        top_rows.append(dict(scale=scale, category=cat, n=int(mask.sum()), **pd.Series(net[mask]).value_counts().to_dict(),
                             parcels=";".join(m.region_label.to_numpy()[mask])))
res = pd.DataFrame(rows)
res.to_csv(C.OUT / "rank_divergence_network_enrichment.csv", index=False)
top = pd.DataFrame(top_rows).fillna(0)
top.to_csv(C.OUT / "top15_network_composition.csv", index=False)
print(res.round(4).to_string())
print(top.drop(columns="parcels").to_string())
ctx.finish()
