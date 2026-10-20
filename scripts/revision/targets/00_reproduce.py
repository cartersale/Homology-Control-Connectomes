#!/usr/bin/env python
"""Reproduce saved primary target results with the legacy (keyword-substring) targets.

Checks (1) raw target energies vs results/primary/target_energies.csv and
(2) named-target medians / Wilcoxon p vs results/statistics/target_named_summary.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

rows = []
lookup = C.driver_lookup()
for scale in C.SCALES:
    tg = C.legacy_targets(scale)
    names = [n for n in tg if not n.startswith("single")]
    T = np.column_stack([tg[n] for n in names])
    stack = C.adjacency_stack(scale)
    for s in range(stack.shape[0]):
        sid = f"sub-{s:03d}"
        _, V, den = C.system_eig(stack[s], C.PRIMARY_C)
        for k in C.K_VALUES:
            for m in C.METHODS:
                ev, U = C.regularized_eig(C.gramian(V, den, lookup[(sid, scale, k, m)]), C.PRIMARY_EPS)
                for n, e in zip(names, C.energies(ev, U, T)):
                    rows.append(dict(scale=scale, subject_id=sid, k=k, method=m, target_name=n, energy=e))
E = pd.DataFrame(rows)

saved = pd.read_csv(C.REPO / "results/primary/target_energies.csv")
saved = saved[saved.selection_strategy == "static"]
m = E.merge(saved, on=["scale", "subject_id", "k", "method", "target_name"])
rel = np.abs(m.energy - m.target_energy) / m.target_energy
print(f"raw energy match: n={len(m)}, max rel err={rel.max():.2e}, median rel err={rel.median():.2e}")

sub = C.pct_subject_table(E, ["scale", "target_name"])
rng = np.random.default_rng(C.SEED)
out = []
for (scale, t), g in sub.groupby(["scale", "target_name"]):
    out.append(dict(scale=scale, target_name=t, **C.summarize(g.pct.to_numpy(), rng)))
out = pd.DataFrame(out)
ref = pd.read_csv(C.REPO / "results/statistics/target_named_summary.csv")
ref = ref[ref.metric == "pct_cycle_minus_degree"][["scale", "target_name", "median", "p_value"]]
cmp = out.merge(ref, on=["scale", "target_name"])
cmp["abs_diff_median"] = (cmp.median_pct - cmp["median"]).abs()
cmp["p_ratio"] = cmp.wilcoxon_p / cmp.p_value
print(cmp[["scale", "target_name", "median_pct", "median", "wilcoxon_p", "p_value", "abs_diff_median"]].to_string())
C.OUT.mkdir(parents=True, exist_ok=True)
cmp.to_csv(C.OUT / "reproduction_check.csv", index=False)
pd.DataFrame([dict(n_rows=len(m), max_rel_err=rel.max(), median_rel_err=rel.median(),
                   max_abs_median_diff=cmp.abs_diff_median.max(),
                   max_p_log10_ratio=np.abs(np.log10(cmp.p_ratio)).max())]).to_csv(C.OUT / "reproduction_energy_match.csv", index=False)
