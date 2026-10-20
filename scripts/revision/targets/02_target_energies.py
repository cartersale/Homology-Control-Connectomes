#!/usr/bin/env python
"""Target energies with corrected targets, driver-target overlap, driver-excluded targets, and distance.

Per subject x scale x k x method (static top-k degree_strength / cycle_participation, saved driver sets):
  energy_full      x*^T W_eps^{-1} x*  (c = 0.1, eps = 1e-5)
  energy_excl      same with x* zeroed at the union of BOTH methods' driver nodes (same k), renormalized
  mass_on_drivers  sum_{i in S} x_i^2          (fraction of target mass on this method's drivers)
  n_drivers_in     |S intersect supp(x)|
  dist_nearest     sum_i x_i^2 * min_{s in S} ||c_i - c_s||   (mm; fsaverage5 midthickness parcel centroids)
  dist_nearest_excl same on the driver-excluded target
  coupling         sum_i x_i^2 * sum_{s in S} A0[i, s]  (structural connection strength of target mass to drivers)
Output: target_energies_long.csv.gz
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import common as C

warnings.filterwarnings("ignore", category=RuntimeWarning)

lookup = C.driver_lookup()
frames = []
for scale in C.SCALES:
    tg = np.load(C.OUT / f"targets_scale-{scale}.npz")
    names = list(tg.files)
    T = np.column_stack([tg[n] for n in names])
    T2 = T**2
    n = T.shape[0]
    D = cdist(C.centroids(scale, "mid"), C.centroids(scale, "mid"))
    stack = C.adjacency_stack(scale)
    rows = []
    for s in range(stack.shape[0]):
        sid = f"sub-{s:03d}"
        A0 = stack[s]
        _, V, den = C.system_eig(A0, C.PRIMARY_C)
        for k in C.K_VALUES:
            sets = {m: lookup[(sid, scale, k, m)] for m in C.METHODS}
            union = np.union1d(sets[C.METHODS[0]], sets[C.METHODS[1]])
            Tex = T.copy()
            Tex[union, :] = 0.0
            norms = np.linalg.norm(Tex, axis=0)
            valid = norms > 1e-12
            Tex[:, valid] /= norms[valid]
            Tex[:, ~valid] = np.nan
            Tex2 = Tex**2
            for m, S in sets.items():
                ev, U = C.regularized_eig(C.gramian(V, den, S), C.PRIMARY_EPS)
                e_full = C.energies(ev, U, T)
                e_ex = np.full(len(names), np.nan)
                e_ex[valid] = C.energies(ev, U, Tex[:, valid])
                dmin = D[:, S].min(axis=1)
                coup = A0[:, S].sum(axis=1)
                rows.append(pd.DataFrame(dict(
                    scale=scale, subject_id=sid, k=k, method=m, target_name=names,
                    energy_full=e_full, energy_excl=e_ex,
                    mass_on_drivers=T2[S].sum(axis=0),
                    n_drivers_in=(np.abs(T[S]) > 1e-12).sum(axis=0),
                    mass_on_union=T2[union].sum(axis=0),
                    dist_nearest=dmin @ T2, dist_nearest_excl=np.where(valid, dmin @ np.nan_to_num(Tex2), np.nan),
                    coupling=coup @ T2, coupling_excl=np.where(valid, coup @ np.nan_to_num(Tex2), np.nan),
                )))
    frames.append(pd.concat(rows))
    print("done", scale)

out = pd.concat(frames, ignore_index=True)
cat = pd.read_csv(C.OUT / "target_catalog.csv")[["scale", "target_name", "family"]]
out = out.merge(cat, on=["scale", "target_name"], how="left")
out.to_csv(C.OUT / "target_energies_long.csv.gz", index=False)
print(out.shape)
