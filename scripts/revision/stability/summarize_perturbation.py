#!/usr/bin/env python
"""Summaries and supplement tables for Task 2 (edge-weight perturbation stability)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from stability_common import OUTPUT_DIR, VARIANTS

SCALES = (68, 114, 219)
K_ER = (1, 2, 3, 5)
IN = OUTPUT_DIR / "perturbation"
OUT = OUTPUT_DIR / "tables"
REP = OUTPUT_DIR / "representatives"
LEVEL_LABEL = {"noise": r"$\sigma$", "dropout": "$p$"}
SHORT = {"ripser": "Ripser CP", "gudhi": "GUDHI pairing", "geodesic": "Length-min.", "birth_edge": "Birth edge", "degree": "Degree"}


def load() -> pd.DataFrame:
    frames = [pd.read_csv(IN / f"per_draw_scale-{s}.csv") for s in SCALES if (IN / f"per_draw_scale-{s}.csv").exists()]
    draws = pd.concat(frames, ignore_index=True)
    complete = draws.groupby("scale").subject_id.nunique()
    return draws[draws.scale.isin(complete[complete >= 70].index)]


def between_subject_reference() -> pd.DataFrame:
    rows = []
    for scale in SCALES:
        path = IN / f"unperturbed_scores_scale-{scale}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        if data["degree"].shape[0] < 70:
            continue
        for name in (*VARIANTS, "degree"):
            m = data[name]
            rho = spearmanr(m.T).statistic
            off = rho[np.triu_indices_from(rho, 1)]
            rows.append({"scale": scale, "ranking": name, "between_subject_spearman_median": float(np.median(off))})
    return pd.DataFrame(rows)


def rank_stability(draws: pd.DataFrame) -> pd.DataFrame:
    subj = draws.groupby(["scale", "family", "level", "ranking", "subject_id"], as_index=False)[
        ["spearman_vs_unperturbed", "jaccard_top5", "jaccard_top15", "jaccard_top1", "jaccard_top2", "jaccard_top3"]
    ].mean()
    agg = subj.groupby(["scale", "family", "level", "ranking"]).agg(
        rho_median=("spearman_vs_unperturbed", "median"),
        rho_q25=("spearman_vs_unperturbed", lambda s: s.quantile(0.25)),
        rho_q75=("spearman_vs_unperturbed", lambda s: s.quantile(0.75)),
        j1=("jaccard_top1", "mean"),
        j5=("jaccard_top5", "mean"),
        j5_median=("jaccard_top5", "median"),
        j15=("jaccard_top15", "mean"),
        j15_median=("jaccard_top15", "median"),
    ).reset_index()
    return agg


def er_persistence(draws: pd.DataFrame) -> pd.DataFrame:
    base = pd.concat([pd.read_csv(REP / f"per_subject_scale-{s}.csv") for s in SCALES])
    rows = []
    cyc = draws[draws.ranking != "degree"]
    for (scale, family, level, ranking), sub in cyc.groupby(["scale", "family", "level", "ranking"]):
        for k in K_ER:
            for metric in ("er", "pr"):
                col = f"{metric}_diff_degree_k{k}"
                per_subject = sub.groupby("subject_id")[col].mean()
                b = base[(base.scale == scale) & (base.variant == ranking) & (base.k == k)].set_index("subject_id")[f"{metric}_diff_degree"]
                v = per_subject.to_numpy()
                p = float(wilcoxon(v).pvalue) if not np.allclose(v, 0) else 1.0
                rows.append({
                    "scale": scale, "family": family, "level": level, "variant": ranking, "k": k, "metric": metric,
                    "unperturbed_mean": float(b.mean()), "unperturbed_n_positive": int((b > 0).sum()),
                    "perturbed_mean": float(v.mean()), "perturbed_median": float(np.median(v)),
                    "n_subjects_positive": int((v > 0).sum()), "n_subjects": int(v.size),
                    "fraction_draws_positive": float((sub[col] > 0).mean()), "wilcoxon_p": p,
                })
    return pd.DataFrame(rows)


def ptex(p: float) -> str:
    if p >= 0.001:
        return f"{p:.3f}"
    mant, exp = f"{p:.1e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def tex_stability(st: pd.DataFrame, rankings=("ripser", "degree")) -> str:
    lines = [r"\begin{table}[h]", r"\centering", r"\small",
             r"\caption{\textbf{Stability of node rankings under edge-weight perturbation.} Each subject's connectome was perturbed "
             r"50 times per level by multiplicative Gaussian noise on existing edges (s.d.\ $\sigma$) or by dropping each edge with "
             r"probability $p$, and rankings were recomputed from scratch. Entries: median across 70 subjects of the per-subject mean "
             r"Spearman $\rho$ with the unperturbed ranking, and mean Jaccard overlap of the top-5 and top-15 sets. Degree strength is "
             r"shown for comparison.}",
             r"\label{tab:perturb_rank}", r"\begin{tabular}{llcccccc}", r"\toprule",
             r"& & \multicolumn{3}{c}{Cycle participation} & \multicolumn{3}{c}{Degree strength} \\ \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Scale & Perturbation & $\rho$ & $J_5$ & $J_{15}$ & $\rho$ & $J_5$ & $J_{15}$ \\", r"\midrule"]
    for scale in SCALES:
        first = True
        for family in ("noise", "dropout"):
            for level in sorted(st[st.family == family].level.unique()):
                cells = []
                for r in rankings:
                    x = st[(st.scale == scale) & (st.family == family) & (st.level == level) & (st.ranking == r)]
                    if x.empty:
                        cells += ["--"] * 3
                        continue
                    x = x.iloc[0]
                    cells += [f"{x.rho_median:.2f}", f"{x.j5:.2f}", f"{x.j15:.2f}"]
                lab = f"{'Noise' if family == 'noise' else 'Dropout'} {LEVEL_LABEL[family]}$={level:.2f}$"
                lines.append(f"{scale if first else ''} & {lab} & " + " & ".join(cells) + r" \\")
                first = False
        if scale != SCALES[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def tex_er(er: pd.DataFrame, variant="ripser") -> str:
    lines = [r"\begin{table}[h]", r"\centering", r"\small",
             r"\caption{\textbf{Cycle-minus-degree effective rank under edge-weight perturbation.} Both rankings and the Gramian were "
             r"recomputed on each perturbed connectome. Entries: mean across 70 subjects of the per-subject mean difference over 50 "
             r"perturbations (subjects with a positive mean, of 70). The unperturbed row repeats the main-analysis value.}",
             r"\label{tab:perturb_er}", r"\begin{tabular}{llcccc}", r"\toprule",
             r"Scale & Condition & $k=1$ & $k=2$ & $k=3$ & $k=5$ \\", r"\midrule"]
    e = er[(er.variant == variant) & (er.metric == "er")]
    for scale in SCALES:
        s = e[e.scale == scale]
        if s.empty:
            continue
        cells = [f"{s[s.k == k].iloc[0].unperturbed_mean:+.2f} ({s[s.k == k].iloc[0].unperturbed_n_positive})" for k in K_ER]
        lines.append(f"{scale} & Unperturbed & " + " & ".join(cells) + r" \\")
        for family in ("noise", "dropout"):
            for level in sorted(s[s.family == family].level.unique()):
                cells = []
                for k in K_ER:
                    x = s[(s.family == family) & (s.level == level) & (s.k == k)].iloc[0]
                    cells.append(f"{x.perturbed_mean:+.2f} ({x.n_subjects_positive})")
                lab = f"{'Noise' if family == 'noise' else 'Dropout'} {LEVEL_LABEL[family]}$={level:.2f}$"
                lines.append(f" & {lab} & " + " & ".join(cells) + r" \\")
        if scale != SCALES[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    draws = load()
    st = rank_stability(draws)
    er = er_persistence(draws)
    ref = between_subject_reference()
    rescaled = draws[draws.ranking == "degree"].groupby(["scale", "family", "level"]).rescaled.mean().reset_index()
    st.to_csv(OUT / "perturb_rank_stability.csv", index=False)
    er.to_csv(OUT / "perturb_er_persistence.csv", index=False)
    ref.to_csv(OUT / "perturb_between_subject_reference.csv", index=False)
    rescaled.to_csv(OUT / "perturb_rescaled_fraction.csv", index=False)
    (OUT / "perturbation_tables.tex").write_text(tex_stability(st) + "\n\n" + tex_er(er) + "\n")
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 30)
    print(st.round(3).to_string(index=False))
    print(er[er.metric == "er"].round(3).to_string(index=False))
    print(ref.round(3).to_string(index=False))
    print(rescaled.to_string(index=False))


if __name__ == "__main__":
    main()
