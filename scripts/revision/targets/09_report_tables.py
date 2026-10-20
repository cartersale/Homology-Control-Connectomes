#!/usr/bin/env python
"""Render markdown tables used in REPORT.md -> results/revision/targets/_tables.md"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

O = C.OUT
named = pd.read_csv(O / "named_target_summary.csv")
sens = pd.read_csv(O / "sensitivity_summary.csv")
prox = pd.read_csv(O / "proximity_controls.csv")
spin = pd.read_csv(O / "spin_tests.csv")
sp = pd.read_csv(O / "single_parcel_summary.csv")
ov = pd.read_csv(O / "driver_target_overlap.csv")
enr = pd.read_csv(O / "rank_divergence_network_enrichment.csv")
masks = pd.read_csv(O / "target_mask_parcels.csv")


def p_fmt(p):
    return "<1e-4" if p < 1e-4 else f"{p:.3g}"


def cell(r):
    star = "*" if r.p_holm < 0.05 else ""
    return f"{r.median_pct:+.1f} [{r.ci_low:+.1f}, {r.ci_high:+.1f}] {r.n_negative}/{r.n_positive} r={r.rank_biserial:+.2f} p={p_fmt(r.p_holm)}{star}"


ORDER = ["landmark_v1", "landmark_m1", "landmark_pcc_precuneus", "landmark_lpfc",
         "legacy_occipital_module", "kw_occipital_module", "kw_association_module", "legacy_association_vs_occipital", "kw_association_vs_occipital",
         "net_Visual", "net_SomMot", "net_DorsAttn", "net_VentAttn", "net_Limbic", "net_FrontoParietal", "net_Default",
         "net_unimodal_VisSomMot", "net_transmodal_DefFP", "net_transmodal_vs_unimodal",
         "legacy_principal_gradient", "principal_gradient", "legacy_bipolar_gradient", "bipolar_gradient", "uniform_cortical", "second_gradient"]
out = []
for variant in ("full", "drivers_excluded"):
    out.append(f"\n### Named targets, {variant} (median % [95% CI], n neg/pos of 70, rank-biserial, Holm p within scale x family)\n")
    out.append("| target | 68 | 114 | 219 |\n|---|---|---|---|")
    for t in ORDER:
        cells = []
        for s in C.SCALES:
            r = named[(named.variant == variant) & (named.target_name == t) & (named.scale == s)]
            cells.append(cell(r.iloc[0]) if len(r) else "")
        out.append(f"| {t} | " + " | ".join(cells) + " |")

out.append("\n### Yeo-17 network targets (full; supplementary)\n")
out.append("| target | 68 | 114 | 219 |\n|---|---|---|---|")
for t in sorted(named[named.family == "yeo17_network"].target_name.unique()):
    cells = []
    for s in C.SCALES:
        r = named[(named.variant == "full") & (named.target_name == t) & (named.scale == s)]
        cells.append(f"{r.iloc[0].median_pct:+.1f} ({r.iloc[0].n_negative}/{r.iloc[0].n_positive}){'*' if r.iloc[0].p_holm < .05 else ''}" if len(r) else "-")
    out.append(f"| {t} | " + " | ".join(cells) + " |")

out.append("\n### Single-cortical-parcel baseline\n")
out.append("| variant | scale | median parcel % [CI] | sig. neg / pos (BH) of N | Spearman(parcel %, delta dist) | Spearman(parcel %, delta coupling) | median within-subject rho (dist) |\n|---|---|---|---|---|---|---|")
for r in sp.itertuples():
    out.append(f"| {r.variant} | {r.scale} | {r.median_parcel_mean_pct:+.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}] | {r.n_sig_negative} / {r.n_sig_positive} of {r.n_parcels} | "
               f"{r.rho_parcels_pct_vs_ddist:+.2f} (p={p_fmt(r.p_parcels_pct_vs_ddist)}) | {r.rho_parcels_pct_vs_dcoupling:+.2f} (p={p_fmt(r.p_parcels_pct_vs_dcoupling)}) | "
               f"{r.median_within_subject_rho_ddist:+.2f} ({r.n_subj_within_rho_ddist_positive}/70 > 0) |")

out.append("\n### Driver-target overlap (mean over subjects and k; cyc / deg)\n")
out.append("| target | scale | frac. (subject,k) with >=1 driver inside | mean # drivers inside | mean target mass on drivers |\n|---|---|---|---|---|")
for t in ORDER:
    for s in C.SCALES:
        r = ov[(ov.target_name == t) & (ov.scale == s)]
        if len(r) and not t.startswith("legacy_p") and "gradient" not in t:
            r = r.iloc[0]
            out.append(f"| {t} | {s} | {r.frac_any_driver_in_cyc:.2f} / {r.frac_any_driver_in_deg:.2f} | {r.mean_n_drivers_in_cyc:.2f} / {r.mean_n_drivers_in_deg:.2f} | {r.mean_mass_on_drivers_cyc:.3f} / {r.mean_mass_on_drivers_deg:.3f} |")

out.append("\n### Distance / coupling vs energy difference across subjects (full targets)\n")
out.append("| target | scale | median delta dist (cyc-deg, mm) | # subj cycle closer | Spearman(pct, delta dist) | Spearman(pct, delta coupling) |\n|---|---|---|---|---|---|")
for t in ORDER:
    for s in C.SCALES:
        r = named[(named.variant == "full") & (named.target_name == t) & (named.scale == s)]
        if len(r):
            r = r.iloc[0]
            out.append(f"| {t} | {s} | {r.median_d_dist_mm:+.1f} | {r.n_subj_cycle_closer} | {r.rho_pct_vs_ddist:+.2f} (p={p_fmt(r.p_pct_vs_ddist)}) | {r.rho_pct_vs_dcoupling:+.2f} (p={p_fmt(r.p_pct_vs_dcoupling)}) |")

out.append("\n### Proximity controls (full targets)\n")
out.append("| target | scale | frac. cases no driver inside | no-overlap subset: median % [CI] (n subj) Holm p | equal-distance intercept % [CI] |\n|---|---|---|---|---|")
for t in ORDER:
    for s in C.SCALES:
        r = prox[(prox.target_name == t) & (prox.scale == s)]
        if not len(r):
            continue
        r = r.iloc[0]
        no = f"{r.noov_median_pct:+.1f} [{r.noov_ci_low:+.1f}, {r.noov_ci_high:+.1f}] ({int(r.noov_n)}) p={p_fmt(r.noov_p_holm)}" if pd.notna(r.get("noov_n")) else "n/a (<20 subj)"
        out.append(f"| {t} | {s} | {r.no_overlap_frac_cases:.2f} | {no} | {r.eqdist_intercept:+.1f} [{r.eqdist_ci_low:+.1f}, {r.eqdist_ci_high:+.1f}] |")

KEY = ["net_Visual", "kw_occipital_module", "net_SomMot", "net_DorsAttn", "net_VentAttn", "net_Limbic", "net_FrontoParietal", "net_Default",
       "net_transmodal_vs_unimodal", "kw_association_module", "kw_association_vs_occipital", "uniform_cortical", "principal_gradient", "bipolar_gradient", "landmark_v1"]
conds = [(0.1, 1e-8), (0.1, 1e-5), (0.1, 1e-2), (0.01, 1e-5), (1.0, 1e-5)]
for variant in ("full", "drivers_excluded"):
    out.append(f"\n### eps / c sensitivity, {variant} (median %, * Holm p<.05)\n")
    out.append("| target | scale | " + " | ".join(f"c={c}, eps={e:.0e}" for c, e in conds) + " |\n|---|---|" + "---|" * len(conds))
    for t in KEY:
        for s in C.SCALES:
            cells = []
            for c, e in conds:
                r = sens[(sens.variant == variant) & (sens.target_name == t) & (sens.scale == s) & np.isclose(sens.c, c) & np.isclose(sens.eps, e)].iloc[0]
                cells.append(f"{r.median_pct:+.1f}{'*' if r.p_holm < .05 else ''}")
            out.append(f"| {t} | {s} | " + " | ".join(cells) + " |")
ff = sens[(sens.variant == "full") & (sens.c == 0.1)].groupby(["scale", "eps"])[["floor_frac_deg", "floor_frac_cyc"]].median().reset_index()
out.append("\n### Fraction of target energy from eigen-directions at the eps floor (median over targets/subjects/k, c = 0.1)\n")
out.append("| scale | eps | degree | cycle |\n|---|---|---|---|")
for r in ff.itertuples():
    out.append(f"| {r.scale} | {r.eps:.0e} | {r.floor_frac_deg:.3f} | {r.floor_frac_cyc:.3f} |")

out.append("\n### Spin tests (10,000 spins; Vasa primary, Alexander-Bloch 'original' secondary)\n")
out.append("| scale | x | y | Spearman rho | parametric p | p_spin (Vasa) | p_spin (A-B) |\n|---|---|---|---|---|---|---|")
for r in spin.itertuples():
    out.append(f"| {r.scale} | {r.x} | {r.y} | {r.spearman_rho:+.3f} | {r.parametric_p:.2g} | {r.p_spin_vasa:.4f} | {r.p_spin_original:.4f} |")

out.append("\n### Rank divergence by Yeo-7 network (mean degree rank - cycle rank; + = cycle priority; Vasa spin p, uncorrected)\n")
out.append("| network | 68 | 114 | 219 |\n|---|---|---|---|")
for n in C.YEO7_NAMES.values():
    cells = []
    for s in C.SCALES:
        r = enr[(enr.network == n) & (enr.scale == s)]
        cells.append(f"{r.iloc[0].mean_rank_divergence:+.1f} (n={r.iloc[0].n_parcels}, p={r.iloc[0].p_spin_vasa:.3f})" if len(r) else "-")
    out.append(f"| {n} | " + " | ".join(cells) + " |")

out.append("\n### Exact parcels per mask\n")
for t in ["legacy_occipital_module", "kw_occipital_module", "kw_association_module", "net_Visual", "net_SomMot", "net_DorsAttn", "net_VentAttn",
          "net_Limbic", "net_FrontoParietal", "net_Default", "landmark_v1", "landmark_m1", "landmark_pcc_precuneus", "landmark_lpfc"]:
    for s in C.SCALES:
        r = masks[(masks.target_name == t) & (masks.scale == s)].iloc[0]
        out.append(f"- **{t} @ {s}** (n={r.n_positive}): {r.positive_parcels.replace(';', ', ')}")
(O / "_tables.md").write_text("\n".join(out) + "\n")
print("ok")
