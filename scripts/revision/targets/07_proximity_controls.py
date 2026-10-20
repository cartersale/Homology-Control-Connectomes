#!/usr/bin/env python
"""Proximity controls for named targets (c = 0.1, eps = 1e-5, full targets).

(a) no-overlap subset: keep (subject, k) cases in which NEITHER method has a driver inside the target support,
    average pct over the remaining k per subject, summarize across subjects (targets with >= 20 subjects).
(b) equal-distance intercept: median (quantile q=0.5) regression across subjects of pct on d_dist
    (cycle - degree nearest-driver distance, mm, k-averaged); intercept = predicted pct when both driver sets are
    equally far from the target. Bootstrap 95% CI (1000 resamples of subjects).

Run: python scripts/revision/targets/07_proximity_controls.py --run <name> [--overwrite]  (reads/writes runs/<name>/)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import common as C

ctx = C.start_stage(__file__, [C.stage("02_target_energies")])

warnings.filterwarnings("ignore")
L = pd.read_csv(C.OUT / "target_energies_long.csv.gz")
L = L[L.family.isin(["landmark", "keyword_module", "yeo7_network", "distributed", "legacy"])]
rng = np.random.default_rng(C.SEED)
w = L.pivot_table(index=["scale", "family", "target_name", "subject_id", "k"], columns="method",
                  values=["energy_full", "n_drivers_in", "dist_nearest"]).reset_index()
w.columns = ["_".join([c for c in col if c]) for col in w.columns]
w["pct"] = 100 * (w.energy_full_cycle_participation - w.energy_full_degree_strength) / w.energy_full_degree_strength
w["d_dist"] = w.dist_nearest_cycle_participation - w.dist_nearest_degree_strength
w["no_overlap"] = (w.n_drivers_in_cycle_participation == 0) & (w.n_drivers_in_degree_strength == 0)

rows = []
for key, g in w.groupby(["scale", "family", "target_name"]):
    row = dict(zip(["scale", "family", "target_name"], key))
    no = g[g.no_overlap].groupby("subject_id").pct.mean()
    row["no_overlap_frac_cases"] = float(g.no_overlap.mean())
    if len(no) >= 20:
        s = C.summarize(no.to_numpy(), rng)
        row.update({f"noov_{k}": v for k, v in s.items()})
    sub = g.groupby("subject_id")[["pct", "d_dist"]].mean().reset_index()
    if sub.d_dist.std() > 0:
        fit = smf.quantreg("pct ~ d_dist", sub).fit(q=0.5)
        boots = []
        for _ in range(1000):
            b = sub.sample(len(sub), replace=True, random_state=int(rng.integers(1 << 31)))
            try:
                boots.append(smf.quantreg("pct ~ d_dist", b).fit(q=0.5).params.Intercept)
            except Exception:
                pass
        row.update(eqdist_intercept=float(fit.params.Intercept), eqdist_ci_low=float(np.percentile(boots, 2.5)),
                   eqdist_ci_high=float(np.percentile(boots, 97.5)), slope_pct_per_mm=float(fit.params.d_dist))
    rows.append(row)
out = pd.DataFrame(rows)
if "noov_wilcoxon_p" in out:
    ok = out.noov_wilcoxon_p.notna()
    out.loc[ok, "noov_p_holm"] = C.holm(out[ok].rename(columns={"noov_wilcoxon_p": "wilcoxon_p"}), ["scale", "family"]).p_holm.values
out.to_csv(C.OUT / "proximity_controls.csv", index=False)
cols = ["scale", "target_name", "no_overlap_frac_cases", "noov_n", "noov_median_pct", "noov_ci_low", "noov_ci_high", "noov_p_holm",
        "eqdist_intercept", "eqdist_ci_low", "eqdist_ci_high", "slope_pct_per_mm"]
print(out[cols].round(3).to_string())
ctx.finish()
