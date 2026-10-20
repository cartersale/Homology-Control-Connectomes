#!/usr/bin/env python
"""epsilon and c sensitivity of key target results (network, keyword modules, landmarks, distributed).

Grid: (c=0.1, eps in {1e-8, 1e-5, 1e-2}) and (eps=1e-5, c in {0.01, 1.0}). Driver sets unchanged
(rankings do not depend on c or eps). Full and driver-excluded targets. Also records the fraction of each
target's energy contributed by eigen-directions at the eps floor.

Run: python scripts/revision/targets/04_sensitivity.py --run <name> [--overwrite]  (reads/writes runs/<name>/)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import common as C

ctx = C.start_stage(__file__, ["00_prepare_data", "01_primary", C.stage("01_build_targets")])

warnings.filterwarnings("ignore", category=RuntimeWarning)
GRID = [(0.1, 1e-8), (0.1, 1e-5), (0.1, 1e-2), (0.01, 1e-5), (1.0, 1e-5)]
FAMS = {"landmark", "keyword_module", "yeo7_network", "distributed", "distributed_supp"}
cat = pd.read_csv(C.OUT / "target_catalog.csv")
lookup = C.driver_lookup()
rows = []
for scale in C.SCALES:
    tg = np.load(C.OUT / f"targets_scale-{scale}.npz")
    fam = cat[cat.scale == scale].set_index("target_name").family
    names = [n for n in tg.files if fam[n] in FAMS]
    T = np.column_stack([tg[n] for n in names])
    stack = C.adjacency_stack(scale)
    for s in range(stack.shape[0]):
        sid = f"sub-{s:03d}"
        for c in sorted({g[0] for g in GRID}):
            _, V, den = C.system_eig(stack[s], c)
            for k in C.K_VALUES:
                sets = {m: lookup[(sid, scale, k, m)] for m in C.METHODS}
                union = np.union1d(*sets.values())
                Tex = T.copy()
                Tex[union] = 0
                Tex /= np.linalg.norm(Tex, axis=0)
                for m, S in sets.items():
                    W = C.gramian(V, den, S)
                    for cc, eps in GRID:
                        if cc != c:
                            continue
                        ev, U = C.regularized_eig(W, eps)
                        rows.append(pd.DataFrame(dict(scale=scale, subject_id=sid, c=c, eps=eps, k=k, method=m, target_name=names,
                                                      energy_full=C.energies(ev, U, T), energy_excl=C.energies(ev, U, Tex),
                                                      floor_frac=C.floor_fraction(ev, U, T, eps))))
    print("done", scale)
E = pd.concat(rows, ignore_index=True)
E["family"] = E.merge(cat, on=["scale", "target_name"], how="left").family.values
E.to_csv(C.OUT / "sensitivity_long.csv.gz", index=False)

rng = np.random.default_rng(C.SEED)
out = []
for variant, col in (("full", "energy_full"), ("drivers_excluded", "energy_excl")):
    w = E.pivot_table(index=["scale", "c", "eps", "family", "target_name", "subject_id", "k"], columns="method", values=col).reset_index()
    w["pct"] = 100 * (w.cycle_participation - w.degree_strength) / w.degree_strength
    sub = w.groupby(["scale", "c", "eps", "family", "target_name", "subject_id"], as_index=False).pct.mean()
    for key, g in sub.groupby(["scale", "c", "eps", "family", "target_name"]):
        out.append(dict(zip(["scale", "c", "eps", "family", "target_name"], key), variant=variant, **C.summarize(g.pct.to_numpy(), rng)))
out = C.holm(pd.DataFrame(out), ["variant", "scale", "c", "eps", "family"])
ff = E.groupby(["scale", "c", "eps", "family", "target_name", "method"]).floor_frac.median().unstack("method").reset_index()
ff.columns = [str(x) for x in ff.columns]
ff = ff.rename(columns={"degree_strength": "floor_frac_deg", "cycle_participation": "floor_frac_cyc"})
out = out.merge(ff, on=["scale", "c", "eps", "family", "target_name"], how="left")
out.to_csv(C.OUT / "sensitivity_summary.csv", index=False)
key = ["net_Visual", "kw_occipital_module", "net_Default", "net_VentAttn", "net_DorsAttn", "uniform_cortical", "principal_gradient"]
print(out[(out.variant == "full") & out.target_name.isin(key)].pivot_table(index=["target_name", "scale"], columns=["c", "eps"], values="median_pct").round(2).to_string())
ctx.finish()
