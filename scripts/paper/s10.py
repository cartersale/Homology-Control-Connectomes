"""Supplement S10: alternative control targets (S10_1-S10_11).

Port of scripts/revision/paper/_rescued_2026-10-19/scripts/make_s10_s11.py (S10 part).
All sources are in <run>/revision/targets/. The occipital module is the corrected mask
(kw_occipital_module, no precuneus); the legacy masks / misregistered G1 appear only in rows
explicitly labelled "legacy".

Change from the rescued builder: S10_8 (floor fraction) no longer uses values transcribed by hand
from REPORT.md; the degree/cycle columns are computed from sensitivity_summary.csv exactly as
scripts/revision/targets/09_report_tables.py computes the REPORT.md table.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

from paper_common import SCALES, Table, np, pd
from s10_s11_fmt import f2, pf, tex_esc

COMMON = """Settings (targets REPORT.md sec. 3): model A = A0 - (lmax + c)I, infinite-horizon Gramian, saved static top-k
degree_strength / cycle_participation driver sets, W_eps = W + eps*I with eigenvalues floored at eps; E = x*' W_eps^-1 x*.
Primary c = 0.1, eps = 1e-5, k in {1,2,3,5}, 70 subjects. pct = 100 (E_cyc - E_deg)/E_deg per subject, averaged over k.
Negative pct = cycle sets cheaper. Medians across subjects; two-sided Wilcoxon; Holm within scale x target family x variant
(families: landmark 4, keyword_module 3, yeo7_network 10, distributed {G1, G1 bipolar, uniform},
distributed_supp {G2, G2 bipolar}, legacy 4). p values exact from the CSV, 2 s.f."""

LABEL = OrderedDict([
    ("net_Visual", "Yeo Visual"), ("net_SomMot", "Yeo SomMot"), ("net_DorsAttn", "Yeo DorsAttn"),
    ("net_VentAttn", "Yeo VentAttn"), ("net_Limbic", "Yeo Limbic"), ("net_FrontoParietal", "Yeo FrontoParietal"),
    ("net_Default", "Yeo Default"), ("net_unimodal_VisSomMot", "Unimodal (Vis+SomMot)"),
    ("net_transmodal_DefFP", "Transmodal (Def+FP)"), ("net_transmodal_vs_unimodal", "Transmodal $-$ unimodal"),
    ("kw_occipital_module", "Occipital module (corrected)"), ("kw_association_module", "Association module"),
    ("kw_association_vs_occipital", "Association $-$ occipital (corr.)"),
    ("legacy_occipital_module", "Occipital module (legacy, +precuneus)"),
    ("legacy_association_vs_occipital", "Association $-$ occipital (legacy)"),
    ("landmark_v1", "V1 (pericalcarine)"), ("landmark_m1", "M1 (precentral)"),
    ("landmark_pcc_precuneus", "PCC/precuneus"), ("landmark_lpfc", "lPFC"),
    ("principal_gradient", "G1 (registered)"), ("bipolar_gradient", "G1 bipolar (registered)"),
    ("legacy_principal_gradient", "G1 (legacy, misregistered)"),
    ("second_gradient", "G2 (registered)"), ("uniform_cortical", "Uniform cortical"),
])
GROUPS = [
    ("Yeo-7 networks", ["net_Visual", "net_SomMot", "net_DorsAttn", "net_VentAttn", "net_Limbic",
                        "net_FrontoParietal", "net_Default", "net_unimodal_VisSomMot", "net_transmodal_DefFP",
                        "net_transmodal_vs_unimodal"]),
    ("Keyword modules", ["kw_occipital_module", "kw_association_module", "kw_association_vs_occipital",
                         "legacy_occipital_module", "legacy_association_vs_occipital"]),
    ("Landmarks", ["landmark_v1", "landmark_m1", "landmark_pcc_precuneus", "landmark_lpfc"]),
    ("Gradients / distributed", ["principal_gradient", "bipolar_gradient", "legacy_principal_gradient",
                                 "second_gradient", "uniform_cortical"]),
]
EPS_LABELS = ("1e-08", "1e-05", "1e-02")


def _end(lines: list[str]) -> str:
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"])


def named_table(nts: pd.DataFrame, variant: str) -> str:
    d = nts[nts.variant == variant].set_index(["target_name", "scale"])
    L = ["\\begin{tabular}{l" + "rcr" * 3 + "}", "\\toprule",
         " & \\multicolumn{3}{c}{68} & \\multicolumn{3}{c}{114} & \\multicolumn{3}{c}{219} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
         "Target & Median \\% & $n_-/n_+$ & $p_H$ & Median \\% & $n_-/n_+$ & $p_H$ & Median \\% & $n_-/n_+$ & $p_H$ \\\\",
         "\\midrule"]
    for gi, (g, tg) in enumerate(GROUPS):
        if gi:
            L.append("\\midrule")
        L.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{g}}}}} \\\\")
        for t in tg:
            cells = []
            for s in SCALES:
                r = d.loc[(t, s)]
                cells += [f2(r.median_pct), f"{int(r.n_negative)}/{int(r.n_positive)}", pf(r.p_holm)]
            L.append(LABEL[t] + " & " + " & ".join(cells) + " \\\\")
    return _end(L)


def overlap_table(ov: pd.DataFrame) -> str:
    ov = ov.set_index(["target_name", "scale"])
    targets = ["net_Visual", "net_SomMot", "net_DorsAttn", "net_VentAttn", "net_Limbic", "net_FrontoParietal",
               "net_Default", "kw_occipital_module", "kw_association_module", "landmark_v1", "landmark_m1",
               "landmark_pcc_precuneus", "landmark_lpfc"]
    L = ["\\begin{tabular}{lccc}", "\\toprule", "Target & 68 & 114 & 219 \\\\", "\\midrule"]
    for t in targets:
        L.append(LABEL[t] + " & " + " & ".join(
            f"{ov.loc[(t, s)].frac_any_driver_in_cyc:.2f} / {ov.loc[(t, s)].frac_any_driver_in_deg:.2f}" for s in SCALES)
            + " \\\\")
    return _end(L)


def proximity_table(nts: pd.DataFrame, pc: pd.DataFrame) -> str:
    pc = pc.set_index(["target_name", "scale"])
    fu = nts[nts.variant == "full"].set_index(["target_name", "scale"])
    targets = ["net_Visual", "kw_occipital_module", "net_DorsAttn", "net_VentAttn", "net_Default", "net_SomMot",
               "kw_association_module", "landmark_v1", "principal_gradient", "second_gradient"]
    L = ["\\begin{tabular}{llrcrrcc}", "\\toprule",
         "Target & Scale & Med.\\ $\\Delta$dist (mm) & Cyc.\\ closer & $\\rho$(pct,$\\Delta$dist) & "
         "$\\rho$(pct,$\\Delta$coupl.) & Equal-dist.\\ intercept \\% & No-overlap median \\% ($n$; $p_H$) \\\\",
         "\\midrule"]
    for i, t in enumerate(targets):
        if i:
            L.append("\\addlinespace")
        for s in SCALES:
            r = fu.loc[(t, s)]
            n_subj = int(r.n)
            if (t, s) not in pc.index:  # proximity_controls.csv has no rows for G2
                q = pd.Series(dict(eqdist_intercept=np.nan, eqdist_ci_low=np.nan, eqdist_ci_high=np.nan))
                no = "not computed"
            else:
                q = pc.loc[(t, s)]
                if pd.isna(q.noov_median_pct):
                    no = "n/a ($<$20 subj.)"
                else:
                    no = f"{f2(q.noov_median_pct)} ({int(q.noov_n)}; {pf(q.noov_p_holm)})"
            L.append(f"{LABEL[t] if s == 68 else ''} & {s} & {f2(r.median_d_dist_mm)} & "
                     f"{int(r.n_subj_cycle_closer)}/{n_subj} & "
                     f"{f2(r.rho_pct_vs_ddist)} & {f2(r.rho_pct_vs_dcoupling)} & "
                     f"{f2(q.eqdist_intercept)} [{f2(q.eqdist_ci_low)}, {f2(q.eqdist_ci_high)}] & {no} \\\\")
    return _end(L)


def single_parcel_table(sp: pd.DataFrame) -> str:
    L = ["\\begin{tabular}{llrcccc}", "\\toprule",
         "Variant & Scale & Median parcel \\% [95\\% CI] & Sig.\\ neg./pos.\\ (BH) & $\\rho$(pct,$\\Delta$dist) & "
         "$\\rho$(pct,$\\Delta$coupl.) & Within-subj.\\ $\\rho$(dist) \\\\",
         "\\midrule"]
    for _, r in sp.iterrows():
        L.append(f"{'Full' if r.variant == 'full' else 'Drivers excl.'} & {r.scale} & "
                 f"{f2(r.median_parcel_mean_pct)} [{f2(r.ci_low)}, {f2(r.ci_high)}] & "
                 f"{int(r.n_sig_negative)}/{int(r.n_sig_positive)} of {int(r.n_parcels)} & "
                 f"{f2(r.rho_parcels_pct_vs_ddist)} & {f2(r.rho_parcels_pct_vs_dcoupling)} & "
                 f"{f2(r.median_within_subject_rho_ddist)} ({int(r.n_subj_within_rho_ddist_positive)}/70 $>0$) \\\\")
    return _end(L)


def sens_table(ss: pd.DataFrame, variant: str) -> str:
    conds = [(0.1, 1e-8), (0.1, 1e-5), (0.1, 1e-2), (0.01, 1e-5), (1.0, 1e-5)]
    targets = ["net_Visual", "kw_occipital_module", "net_DorsAttn", "net_VentAttn", "net_Default", "net_SomMot",
               "kw_association_module", "kw_association_vs_occipital", "net_transmodal_vs_unimodal", "landmark_v1",
               "principal_gradient", "second_gradient", "uniform_cortical"]
    d = ss[ss.variant == variant]
    L = ["\\begin{tabular}{llrrrrr}", "\\toprule",
         " & & \\multicolumn{3}{c}{$c=0.1$} & \\multicolumn{2}{c}{$\\varepsilon=10^{-5}$} \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}",
         "Target & Scale & $\\varepsilon=10^{-8}$ & $10^{-5}$ & $10^{-2}$ & $c=0.01$ & $c=1$ \\\\", "\\midrule"]
    for i, t in enumerate(targets):
        if i:
            L.append("\\addlinespace")
        for s in SCALES:
            cells = []
            for c, e in conds:
                x = d[(d.target_name == t) & (d.scale == s) & np.isclose(d.c, c) & np.isclose(d.eps, e)]
                assert len(x) == 1, (variant, t, s, c, e)
                r = x.iloc[0]
                cells.append(f2(r.median_pct) + ("$^*$" if r.p_holm < 0.05 else ""))
            L.append(f"{LABEL[t] if s == 68 else ''} & {s} & " + " & ".join(cells) + " \\\\")
    return _end(L)


def floor_table(ss: pd.DataFrame, sl: pd.DataFrame) -> str:
    # Degree/cycle columns: median over targets of the per-target floor_frac medians (over subjects x k),
    # as in scripts/revision/targets/09_report_tables.py (REPORT.md sec. 5 table). These were previously
    # typed in by hand from REPORT.md.
    ff = (ss[(ss.variant == "full") & np.isclose(ss.c, 0.1)]
          .groupby(["scale", "eps"])[["floor_frac_deg", "floor_frac_cyc"]].median())
    # Last column: per-target median over subjects x k x method at c = 0.1, eps = 1e-5; min--max over targets.
    sl = sl[np.isclose(sl.c, 0.1) & np.isclose(sl.eps, 1e-5)]
    tmed = sl.groupby(["scale", "target_name"]).floor_frac.median().groupby("scale").agg(["min", "max"])
    L = ["\\begin{tabular}{llccc}", "\\toprule",
         "Scale & $\\varepsilon$ & Degree sets & Cycle sets & Range of per-target medians ($\\varepsilon=10^{-5}$) \\\\",
         "\\midrule"]
    for s in SCALES:
        for e in EPS_LABELS:
            r = ff.loc[(s, float(e))]
            rng = f"{tmed.loc[s, 'min']:.2f}--{tmed.loc[s, 'max']:.2f}" if e == "1e-05" else ""
            L.append(f"{s if e == '1e-08' else ''} & $10^{{{int(e.split('e')[1])}}}$ & "
                     f"{r.floor_frac_deg:.2f} & {r.floor_frac_cyc:.2f} & {rng} \\\\")
    return _end(L)


def spin_table(sp: pd.DataFrame) -> str:
    names = {"rank_divergence": "Rank divergence", "degree_strength": "Degree strength",
             "cycle_participation": "Cycle participation", "G1_legacy_projection": "G1 (legacy, misregistered)",
             "G1_registered": "G1 (registered)", "G2_registered": "G2 (registered)"}
    L = ["\\begin{tabular}{lllrccc}", "\\toprule",
         "Scale & $x$ & $y$ & Spearman $\\rho$ & $p_{\\mathrm{param}}$ & $p_{\\mathrm{spin}}$ (Vasa) & "
         "$p_{\\mathrm{spin}}$ (A--B) \\\\", "\\midrule"]
    for s in SCALES:
        if s != 68:
            L.append("\\addlinespace")
        for _, r in sp[sp.scale == s].iterrows():
            L.append(f"{s} & {names[r.x]} & {names[r.y]} & {f2(r.spearman_rho)} & {pf(r.parametric_p)} & "
                     f"{r.p_spin_vasa:.4f} & {r.p_spin_original:.4f} \\\\")
    return _end(L)


def rankdiv_table(rd: pd.DataFrame) -> str:
    rd = rd.set_index(["network", "scale"])
    L = ["\\begin{tabular}{lccc}", "\\toprule", "Network & 68 & 114 & 219 \\\\", "\\midrule"]
    for n in ["Visual", "SomMot", "DorsAttn", "VentAttn", "Limbic", "FrontoParietal", "Default"]:
        L.append(n + " & " + " & ".join(
            f"{f2(rd.loc[(n, s)].mean_rank_divergence)} ($n$={int(rd.loc[(n, s)].n_parcels)}; "
            f"$p$={rd.loc[(n, s)].p_spin_vasa:.2f})" for s in SCALES) + " \\\\")
    return _end(L)


def compact(lst) -> str:
    """'L_lingual_1;L_lingual_2;R_lingual_1' -> 'lingual (L1,2; R1)'."""
    if not isinstance(lst, str) or not lst:
        return ""
    groups: OrderedDict = OrderedDict()
    for p in lst.split(";"):
        m = re.match(r"([LR])_([a-z]+)(?:_(\d+))?$", p)
        assert m, p
        h, base, idx = m.group(1), m.group(2), m.group(3)
        groups.setdefault(base, {"L": [], "R": []})[h].append(idx)
    parts = []
    for base, hh in groups.items():
        segs = []
        for h in "LR":
            ids = hh[h]
            if not ids:
                continue
            segs.append(h if ids == [None] else h + ",".join(ids))
        parts.append(f"{base} ({'; '.join(segs)})")
    return ", ".join(parts)


def parcel_table(mp: pd.DataFrame) -> str:
    targets = ["kw_occipital_module", "legacy_occipital_module", "kw_association_module", "net_Visual", "net_SomMot",
               "net_DorsAttn", "net_VentAttn", "net_Limbic", "net_FrontoParietal", "net_Default", "landmark_v1",
               "landmark_m1", "landmark_pcc_precuneus", "landmark_lpfc"]
    L = ["\\begin{tabular}{llr p{12cm}}", "\\toprule", "Target & Scale & $n$ & Parcels \\\\", "\\midrule"]
    for i, t in enumerate(targets):
        if i:
            L.append("\\addlinespace")
        for s in SCALES:
            x = mp[(mp.target_name == t) & (mp.scale == s)]
            assert len(x) == 1, (t, s)
            r = x.iloc[0]
            L.append(f"{LABEL[t] if s == 68 else ''} & {s} & {int(r.n_positive)} & "
                     f"{tex_esc(compact(r.positive_parcels))} \\\\")
    return _end(L)


def build(res: Path) -> list[Table]:
    T = res / "targets"
    nts = pd.read_csv(T / "named_target_summary.csv")
    ss = pd.read_csv(T / "sensitivity_summary.csv")
    src = "Source: results/revision/targets/"

    named_note = f"""{COMMON}
{src}named_target_summary.csv (variant = {{v}}; columns median_pct, n_negative, n_positive, p_holm).
"Drivers excluded" = x* zeroed on the union of both methods' driver sets for that subject/k, renormalized, both energies
recomputed. Legacy rows (old occipital incl. precuneus; legacy association-vs-occipital; misregistered G1) are included for
comparison; Holm for them is within the "legacy" family. Bootstrap CIs and rank-biserial r are in the CSV (not tabulated).
Yeo-17 targets (family yeo17_network) are in the CSV, not tabulated. At 68 regions DorsAttn = 2 parcels and
FrontoParietal = 3 parcels; treat them as near-landmarks."""

    return [
        Table("S10_1", named_table(nts, "full"),
              "Named targets, full. pct = 100(E_cyc-E_deg)/E_deg, k-averaged; c=0.1, eps=1e-5.\n"
              + named_note.replace("{v}", "full")),
        Table("S10_2", named_table(nts, "drivers_excluded"),
              "Named targets, union of both methods' drivers zeroed and target renormalized.\n"
              + named_note.replace("{v}", "drivers_excluded")),
        Table("S10_3", overlap_table(pd.read_csv(T / "driver_target_overlap.csv")),
              f"""Driver-target overlap: fraction of subject x k cases with >= 1 driver inside supp(x*); cycle / degree.
{src}driver_target_overlap.csv (frac_any_driver_in_cyc / frac_any_driver_in_deg). Mean mass on drivers and mean number of
drivers inside are in that CSV (and per k in driver_target_overlap_by_k.csv)."""),
        Table("S10_4", proximity_table(nts, pd.read_csv(T / "proximity_controls.csv")),
              f"""Proximity. dDist = cyc - deg target-mass-weighted distance to nearest driver (mm), k-avg; rho across 70 subjects
(Spearman); intercept = median-regression pct at dDist = 0 [bootstrap 95% CI, 1,000 resamples]; no-overlap = subset of
(subject, k) with no driver of either method in the target (Holm p; reported only when >= 20 subjects contribute).
{src}named_target_summary.csv (full; median_d_dist_mm, n_subj_cycle_closer, rho_pct_vs_ddist, rho_pct_vs_dcoupling) and
proximity_controls.csv (eqdist_*, noov_*). Distance D(S,x) = sum_i x_i^2 min_s ||c_i - c_s|| (mm) on fsaverage5 midthickness
centroids; coupling = sum_i x_i^2 sum_(s in S) A0[i,s].
MISSING: proximity_controls.csv has no rows for second_gradient (G2) -> shown as "--" / "not computed"."""),
        Table("S10_5", single_parcel_table(pd.read_csv(T / "single_parcel_summary.csv")),
              f"""Single-parcel targets. {src}single_parcel_summary.csv (BH within scale). The 68-scale full CI is wide because
pct is unbounded above."""),
        Table("S10_6", sens_table(ss, "full"),
              f"""eps/c sensitivity, full targets; median pct, * = Holm p < 0.05 (Holm within scale x family x condition).
{src}sensitivity_summary.csv (variant = full; median_pct, p_holm). Conditions: eps in {{1e-8, 1e-5, 1e-2}} at c = 0.1;
c in {{0.01, 1}} at eps = 1e-5. Driver sets fixed.""", long=True),
        Table("S10_7", sens_table(ss, "drivers_excluded"),
              f"""eps/c sensitivity, drivers excluded; median pct, * = Holm p < 0.05 (Holm within scale x family x condition).
{src}sensitivity_summary.csv (variant = drivers_excluded; median_pct, p_holm). Conditions as in S10_6.""", long=True),
        Table("S10_8", floor_table(ss, pd.read_csv(T / "sensitivity_long.csv.gz")),
              f"""Fraction of target energy carried by eigen-directions at the eps floor (c = 0.1).
Degree / Cycle sets: {src}sensitivity_summary.csv (variant = full, c = 0.1; columns floor_frac_deg / floor_frac_cyc =
per-target median of floor_frac over subjects x k, written by scripts/revision/targets/04_sensitivity.py), median over the 22
targets per (scale, eps) -- the computation of the REPORT.md sec. 5 table in 09_report_tables.py. (The rescued builder had
these values typed in from REPORT.md; the computed values agree to the 3 decimals shown there.)
Range of per-target medians: {src}sensitivity_long.csv.gz (c = 0.1, eps = 1e-5), median over subjects x k x method of
floor_frac per target, then min--max across targets."""),
        Table("S10_9", spin_table(pd.read_csv(T / "spin_tests.csv")),
              f"""Spin tests (10,000 rotations; Vasa primary, Alexander-Bloch "original" secondary; p floor 1e-4); two-sided p.
{src}spin_tests.csv. Rank divergence = degree rank - cycle rank of subject-averaged scores (+ = cycle priority).
G1 (legacy) = the first version's misregistered projection; G1/G2 (registered) use the corrected fsLR->fsaverage5 map."""),
        Table("S10_10", rankdiv_table(pd.read_csv(T / "rank_divergence_network_enrichment.csv")),
              f"""Mean rank divergence (degree rank - cycle rank; + = cycle priority) by Yeo-7 network; n parcels; Vasa spin p
(uncorrected). {src}rank_divergence_network_enrichment.csv."""),
        Table("S10_11", parcel_table(pd.read_csv(T / "target_mask_parcels.csv")),
              f"""Exact parcels per mask. Format: base region (L idx; R idx); 'L'/'R' alone = unsubdivided parcel at scale 68.
{src}target_mask_parcels.csv, positive support only (none of the listed masks has negative support; signed contrasts and
gradients are not listed). Yeo-7 parcel assignment = majority vote of fsaverage5 vertices (medial wall excluded)
(parcel_network_assignment.csv).""", long=True, long_colspec="llr p{10.3cm}"),
    ]
