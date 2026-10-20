#!/usr/bin/env python
"""Statistics for corrected targets, driver overlap, driver exclusion and distance.

Convention: per subject, pct = 100 (E_cycle - E_degree)/E_degree averaged over k in {1,2,3,5};
median across 70 subjects, bootstrap 95% CI (2000, seed 20260423), n negative / positive,
two-sided Wilcoxon, rank-biserial r (positive = degree advantage), Holm within scale x family x variant.

Run: python scripts/revision/targets/03_target_stats.py --run <name> [--overwrite]  (reads/writes runs/<name>/)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

import common as C

ctx = C.start_stage(__file__, [C.stage("02_target_energies")])

warnings.filterwarnings("ignore", category=RuntimeWarning)
L = pd.read_csv(C.OUT / "target_energies_long.csv.gz")
rng = np.random.default_rng(C.SEED)
NAMED = ["landmark", "keyword_module", "yeo7_network", "yeo17_network", "distributed", "distributed_supp", "legacy"]


def per_subject(df: pd.DataFrame, energy_col: str) -> pd.DataFrame:
    w = df.pivot_table(index=["scale", "family", "target_name", "subject_id", "k"], columns="method",
                       values=[energy_col, "dist_nearest", "dist_nearest_excl", "coupling", "coupling_excl"]).reset_index()
    w.columns = ["_".join([c for c in col if c]) for col in w.columns]
    e_c, e_d = w[f"{energy_col}_cycle_participation"], w[f"{energy_col}_degree_strength"]
    w["pct"] = 100 * (e_c - e_d) / e_d
    sfx = "_excl" if energy_col.endswith("excl") else ""
    w["d_dist"] = w[f"dist_nearest{sfx}_cycle_participation"] - w[f"dist_nearest{sfx}_degree_strength"]
    w["d_coupling"] = w[f"coupling{sfx}_cycle_participation"] - w[f"coupling{sfx}_degree_strength"]
    return w.groupby(["scale", "family", "target_name", "subject_id"], as_index=False)[["pct", "d_dist", "d_coupling"]].mean()


# ------------------------------------------------------------------ named targets: full and driver-excluded
named_rows = []
subj_tables = {}
for variant, col in (("full", "energy_full"), ("drivers_excluded", "energy_excl")):
    sub = per_subject(L[L.family.isin(NAMED)], col)
    subj_tables[variant] = sub
    for (scale, fam, t), g in sub.groupby(["scale", "family", "target_name"]):
        v = g.pct.dropna().to_numpy()
        if v.size < 10:
            continue
        r_dist = spearmanr(g.pct, g.d_dist, nan_policy="omit")
        r_coup = spearmanr(g.pct, g.d_coupling, nan_policy="omit")
        named_rows.append(dict(variant=variant, scale=scale, family=fam, target_name=t, **C.summarize(v, rng),
                               rho_pct_vs_ddist=r_dist.statistic, p_pct_vs_ddist=r_dist.pvalue,
                               rho_pct_vs_dcoupling=r_coup.statistic, p_pct_vs_dcoupling=r_coup.pvalue,
                               median_d_dist_mm=float(g.d_dist.median()),
                               n_subj_cycle_closer=int((g.d_dist < 0).sum())))
named = C.holm(pd.DataFrame(named_rows), ["variant", "scale", "family"])
named.to_csv(C.OUT / "named_target_summary.csv", index=False)
pd.concat([t.assign(variant=v) for v, t in subj_tables.items()]).to_csv(C.OUT / "named_target_subject_pct.csv.gz", index=False)

# ------------------------------------------------------------------ single-parcel baseline: full and driver-excluded
sp_rows, sp_dist_rows, sp_parcel = [], [], []
for variant, col in (("full", "energy_full"), ("drivers_excluded", "energy_excl")):
    sub = per_subject(L[L.family == "single_cortical_parcel"], col)
    for scale, gs in sub.groupby("scale"):
        par = []
        for t, g in gs.groupby("target_name"):
            v = g.pct.dropna().to_numpy()
            p = wilcoxon(v).pvalue if v.size >= 10 and np.any(v != 0) else np.nan
            par.append(dict(variant=variant, scale=scale, target_name=t, n_subjects=v.size, mean_pct=np.mean(v) if v.size else np.nan,
                            median_pct=np.median(v) if v.size else np.nan, p=p,
                            mean_d_dist=g.d_dist.mean(), mean_d_coupling=g.d_coupling.mean()))
        par = pd.DataFrame(par)
        ok = par.p.notna()
        par.loc[ok, "p_bh"] = multipletests(par.loc[ok, "p"], method="fdr_bh")[1]
        sp_parcel.append(par)
        boots = np.median(rng.choice(par.mean_pct.dropna(), size=(2000, ok.sum())), axis=1)
        sig = par.p_bh < 0.05
        rd = spearmanr(par.mean_pct, par.mean_d_dist, nan_policy="omit")
        rc = spearmanr(par.mean_pct, par.mean_d_coupling, nan_policy="omit")
        # within-subject (pooled over subjects): Spearman of per-parcel pct vs per-parcel d_dist, then median across subjects
        within = gs.groupby("subject_id").apply(lambda x: spearmanr(x.pct, x.d_dist, nan_policy="omit").statistic)
        within_c = gs.groupby("subject_id").apply(lambda x: spearmanr(x.pct, x.d_coupling, nan_policy="omit").statistic)
        sp_rows.append(dict(variant=variant, scale=scale, n_parcels=int(ok.sum()),
                            median_parcel_mean_pct=float(par.mean_pct.median()),
                            ci_low=float(np.percentile(boots, 2.5)), ci_high=float(np.percentile(boots, 97.5)),
                            n_sig_negative=int((sig & (par.median_pct < 0)).sum()), n_sig_positive=int((sig & (par.median_pct > 0)).sum()),
                            rho_parcels_pct_vs_ddist=rd.statistic, p_parcels_pct_vs_ddist=rd.pvalue,
                            rho_parcels_pct_vs_dcoupling=rc.statistic, p_parcels_pct_vs_dcoupling=rc.pvalue,
                            median_within_subject_rho_ddist=float(within.median()),
                            n_subj_within_rho_ddist_positive=int((within > 0).sum()),
                            median_within_subject_rho_dcoupling=float(within_c.median()),
                            n_subj_within_rho_dcoupling_negative=int((within_c < 0).sum())))
pd.DataFrame(sp_rows).to_csv(C.OUT / "single_parcel_summary.csv", index=False)
pd.concat(sp_parcel).to_csv(C.OUT / "single_parcel_tests.csv", index=False)

# ------------------------------------------------------------------ driver-target overlap
ov = L[L.family.isin(NAMED)].copy()
ov["any_driver_in"] = ov.n_drivers_in > 0
overlap_k = ov.groupby(["scale", "family", "target_name", "method", "k"], as_index=False).agg(
    mean_mass_on_drivers=("mass_on_drivers", "mean"), mean_n_drivers_in=("n_drivers_in", "mean"),
    frac_subjects_any_driver_in=("any_driver_in", "mean"), mean_mass_on_union=("mass_on_union", "mean"))
overlap_k.to_csv(C.OUT / "driver_target_overlap_by_k.csv", index=False)
overlap = ov.groupby(["scale", "family", "target_name", "method"], as_index=False).agg(
    mean_mass_on_drivers=("mass_on_drivers", "mean"), mean_n_drivers_in=("n_drivers_in", "mean"),
    frac_any_driver_in=("any_driver_in", "mean"))
ow = overlap.pivot_table(index=["scale", "family", "target_name"], columns="method",
                         values=["mean_mass_on_drivers", "mean_n_drivers_in", "frac_any_driver_in"]).reset_index()
ow.columns = ["_".join([c for c in col if c]).replace("degree_strength", "deg").replace("cycle_participation", "cyc") for col in ow.columns]
ow.to_csv(C.OUT / "driver_target_overlap.csv", index=False)
# single parcels: fraction of (subject, k) in which the parcel is a driver, by method
spo = L[L.family == "single_cortical_parcel"].groupby(["scale", "method"]).n_drivers_in.mean()
spo.to_csv(C.OUT / "single_parcel_driver_rate.csv")

# also the old-vs-corrected paired overlap of the paper's occipital module
print(named[named.family.isin(["keyword_module", "yeo7_network", "legacy", "distributed", "landmark"])]
      [["variant", "scale", "target_name", "median_pct", "ci_low", "ci_high", "n_negative", "n_positive", "p_holm", "rank_biserial", "rho_pct_vs_ddist"]]
      .round(4).to_string())
print(pd.DataFrame(sp_rows).round(4).to_string())
ctx.finish()
