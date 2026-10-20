#!/usr/bin/env python
"""P4: does the c-dependence collapse across scales when c is replaced by the decay ratio
d = c / (c + lambda1 - lambda2)?

For every subject and scale, on a 25-point log grid of c in [10^-2.5, 10^0.5], compute q1 and ER of
the degree set, the cycle-participation set (saved primary ranks) and the subject's 100 primary
random sets (k in {1,2,3,5}; k-averaged as in the paper). Curves of interest per subject:
  cd_q1 = q1(cycle) - q1(degree), cd_er = ER(cycle) - ER(degree), dr_q1 = q1(degree) - mean q1(random)
Collapse tests, on each axis (log c vs log d):
  1. dispersion: per-subject curves are interpolated onto a common log-axis grid (the range shared
     by all subjects); D = mean over the grid of the between-scale variance of the three scale-mean
     curves, divided by the overall variance of all subjects' curve values. Smaller D = tighter collapse.
  2. per-subject sign-flip points (first zero crossing) on each axis; spread across scales
     (ratio of max to min scale median, and Kruskal-Wallis across scales).
  3. dispersion on matched windows: the shared decay-ratio range [d_lo, d_hi] vs the c window it maps
     to at the median gap (c = d * gap / (1 - d)), so both axes cover the same regime.
  4. within-scale individual differences: Spearman rho across subjects between the gap and each curve
     at fixed c (if the regime is set by c relative to the gap, larger gaps should mean larger effects).
  5. the same, as partial Spearman rho controlling for mean strength (and for mean strength + density
     + strength CV), since the gap covaries with overall connectivity.
Outputs: results/revision/q1/decay_curves.csv.gz, decay_collapse.csv, decay_flips.csv,
decay_flip_summary.csv, decay_gap_correlations.csv, decay_gap_partial_correlations.csv,
fig_decay_collapse.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kruskal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nulls"))
from _common import REPO_ROOT, Spectral, load_stack, random_driver_sets, spectrum_metrics  # noqa: E402

OUT = REPO_ROOT / "results" / "revision" / "q1"
SCALES = (68, 114, 219)
K_VALUES = (1, 2, 3, 5)
C_GRID = np.logspace(-2.5, 0.5, 25)
CURVES = ("cd_q1", "cd_er", "dr_q1")


def load_orders(scale: int) -> dict:
    t = pd.read_csv(REPO_ROOT / "results" / "primary" / "node_scores.csv", usecols=["subject_id", "scale", "node_index", "method", "rank"])
    t = t[(t["scale"] == scale) & t["method"].isin(["degree_strength", "cycle_participation"])]
    return {key: g.sort_values("rank")["node_index"].to_numpy(int) for key, g in t.groupby(["subject_id", "method"])}


def job(scale, si, A, orders):
    sid = f"sub-{si:03d}"
    spec = Spectral(A)
    rows = []
    sets_k = {k: [orders[(sid, "degree_strength")][:k], orders[(sid, "cycle_participation")][:k]] + random_driver_sets("real", -1, scale, si, k) for k in K_VALUES}
    for c in C_GRID:
        kernel = spec.kernel(c)
        acc = {"q1_deg": 0.0, "q1_cyc": 0.0, "q1_rand": 0.0, "er_deg": 0.0, "er_cyc": 0.0}
        for k in K_VALUES:
            w, _, _ = spec.set_spectra(kernel, sets_k[k])
            m = spectrum_metrics(w)
            acc["q1_deg"] += m["q1"][0] / len(K_VALUES)
            acc["q1_cyc"] += m["q1"][1] / len(K_VALUES)
            acc["q1_rand"] += m["q1"][2:].mean() / len(K_VALUES)
            acc["er_deg"] += m["er"][0] / len(K_VALUES)
            acc["er_cyc"] += m["er"][1] / len(K_VALUES)
        rows.append({"subject_id": sid, "scale": scale, "c": c, "gap": spec.gap, "decay_ratio": c / (c + spec.gap), **acc,
                     "cd_q1": acc["q1_cyc"] - acc["q1_deg"], "cd_er": acc["er_cyc"] - acc["er_deg"], "dr_q1": acc["q1_deg"] - acc["q1_rand"]})
    return rows


def first_crossing(x, y):
    s = np.sign(y)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if idx.size == 0:
        return np.nan
    i = idx[0]
    return x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])


def collapse_stats(df: pd.DataFrame, axis: str, curve: str, lo: float | None = None, hi: float | None = None) -> dict:
    df = df.assign(lx=np.log10(df[axis]))
    lo = df.groupby(["scale", "subject_id"])["lx"].min().max() if lo is None else np.log10(lo)
    hi = df.groupby(["scale", "subject_id"])["lx"].max().min() if hi is None else np.log10(hi)
    grid = np.linspace(lo, hi, 40)
    per = {s: [] for s in SCALES}
    for (s, sid), g in df.groupby(["scale", "subject_id"]):
        g = g.sort_values("lx")
        per[s].append(np.interp(grid, g["lx"], g[curve]))
    means = np.array([np.mean(per[s], axis=0) for s in SCALES])
    allv = np.concatenate([np.ravel(per[s]) for s in SCALES])
    return {"axis": axis, "curve": curve, "grid_lo": 10**lo, "grid_hi": 10**hi,
            "between_scale_var": float(means.var(axis=0).mean()), "total_var": float(allv.var()),
            "D": float(means.var(axis=0).mean() / allv.var())}


def main():
    path = OUT / "decay_curves.csv.gz"
    if path.exists():
        df = pd.read_csv(path)
    else:
        rows = []
        for scale in SCALES:
            stack = load_stack(scale)
            orders = load_orders(scale)
            out = Parallel(n_jobs=12)(delayed(job)(scale, si, stack[si], orders) for si in range(stack.shape[0]))
            rows += [r for o in out for r in o]
            print(scale, "done", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

    coll = [collapse_stats(df, ax, cv) for cv in CURVES for ax in ("c", "decay_ratio")]
    d_lo, d_hi = coll[1]["grid_lo"], coll[1]["grid_hi"]
    gap_med = df.groupby(["scale", "subject_id"])["gap"].first().median()
    c_lo, c_hi = d_lo * gap_med / (1 - d_lo), d_hi * gap_med / (1 - d_hi)
    for cv in CURVES:
        for ax, lo, hi in (("c", c_lo, c_hi), ("decay_ratio", d_lo, d_hi)):
            coll.append({**collapse_stats(df, ax, cv, lo, hi), "axis": f"{ax} (matched window)"})
    coll = pd.DataFrame(coll)
    coll.to_csv(OUT / "decay_collapse.csv", index=False)
    print(coll.round(4).to_string(index=False))

    flips = []
    for (s, sid), g in df.groupby(["scale", "subject_id"]):
        g = g.sort_values("c")
        for cv in ("cd_q1", "cd_er"):
            fc = first_crossing(np.log10(g["c"].to_numpy()), g[cv].to_numpy())
            fd = first_crossing(np.log10(g["decay_ratio"].to_numpy()), g[cv].to_numpy())
            flips.append({"scale": s, "subject_id": sid, "curve": cv, "flip_c": 10**fc, "flip_decay": 10**fd, "gap": g["gap"].iloc[0]})
    flips = pd.DataFrame(flips)
    flips.to_csv(OUT / "decay_flips.csv", index=False)
    rows = []
    for cv, g in flips.groupby("curve"):
        for ax in ("flip_c", "flip_decay"):
            med = g.groupby("scale")[ax].median()
            n = g.groupby("scale")[ax].apply(lambda x: x.notna().sum())
            groups = [x.dropna().to_numpy() for _, x in g.groupby("scale")[ax]]
            p = kruskal(*groups).pvalue if all(len(x) > 2 for x in groups) else np.nan
            rows.append({"curve": cv, "axis": ax, **{f"median_{s}": med.get(s) for s in SCALES}, **{f"n_flip_{s}": n.get(s) for s in SCALES},
                         "max_over_min": med.max() / med.min(), "kruskal_p": p})
    fsum = pd.DataFrame(rows)
    fsum.to_csv(OUT / "decay_flip_summary.csv", index=False)
    print(fsum.round(3).to_string(index=False))

    from scipy.stats import spearmanr

    corr = []
    for (s, c), g in df.groupby(["scale", "c"]):
        for cv in CURVES:
            r = spearmanr(g["gap"], g[cv])
            corr.append({"scale": s, "c": c, "curve": cv, "rho_gap": r.statistic, "p": r.pvalue})
    corr = pd.DataFrame(corr)
    corr.to_csv(OUT / "decay_gap_correlations.csv", index=False)

    # 5. partial Spearman rho(gap, curve | controls): ranks residualized on the controls' ranks (OLS)
    from scipy.stats import t as t_dist

    net = []
    for s in SCALES:
        stack = load_stack(s)
        for si, A in enumerate(stack):
            strength = A.sum(axis=0)
            net.append({"scale": s, "subject_id": f"sub-{si:03d}", "mean_strength": strength.mean(),
                        "density": np.count_nonzero(np.triu(A, 1)) / (s * (s - 1) / 2), "strength_cv": strength.std() / strength.mean()})
    net = pd.DataFrame(net)
    control_sets = {"mean_strength": ["mean_strength"], "strength+density+cv": ["mean_strength", "density", "strength_cv"]}
    part = []
    for (s, c), g in df.merge(net, on=["scale", "subject_id"]).groupby(["scale", "c"]):
        for label, ctrl in control_sets.items():
            Z = np.column_stack([np.ones(len(g))] + [g[x].rank().to_numpy() for x in ctrl])
            resid = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]  # noqa: E731
            rg = resid(g["gap"].rank().to_numpy())
            for cv in CURVES:
                r = float(np.corrcoef(rg, resid(g[cv].rank().to_numpy()))[0, 1])
                dof = len(g) - 2 - len(ctrl)
                tval = r * np.sqrt(dof / (1 - r**2))
                part.append({"scale": s, "c": c, "curve": cv, "controls": label, "partial_rho_gap": r, "p": 2 * t_dist.sf(abs(tval), dof)})
    part = pd.DataFrame(part)
    part.to_csv(OUT / "decay_gap_partial_correlations.csv", index=False)
    gaps = df.groupby(["scale", "subject_id"])["gap"].first().reset_index()
    ctrl_corr = net.merge(gaps, on=["scale", "subject_id"]).groupby("scale")[["gap", "mean_strength", "density", "strength_cv"]].corr(method="spearman").xs("gap", level=1)
    print("Spearman rho of gap with controls:\n", ctrl_corr.round(2).to_string())
    print(part[np.isclose(part["c"], 0.1, rtol=0.05)].round(3).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {68: "#1b6ca8", 114: "#d1495b", 219: "#2e933c"}
    fig, axes = plt.subplots(len(CURVES), 2, figsize=(9, 9), sharey="row")
    labels = {"cd_q1": "q1(cycle) − q1(degree)", "cd_er": "ER(cycle) − ER(degree)", "dr_q1": "q1(degree) − q1(random)"}
    for i, cv in enumerate(CURVES):
        for j, ax_name in enumerate(("c", "decay_ratio")):
            ax = axes[i, j]
            for s in SCALES:
                g = df[df["scale"] == s]
                m = g.groupby("c")[[ax_name, cv]].mean() if ax_name == "c" else g.groupby("c")[[ax_name, cv]].mean()
                ax.plot(m[ax_name], m[cv], color=colors[s], lw=2, label=f"{s} regions")
            ax.axhline(0, color="0.5", lw=0.8)
            ax.set_xscale("log")
            D = coll[(coll["curve"] == cv) & (coll["axis"] == f"{ax_name} (matched window)")]["D"].iloc[0]
            ax.set_title(f"{labels[cv]} vs {'c' if ax_name == 'c' else 'c/(c+λ1−λ2)'}  (D matched = {D:.3f})", fontsize=9)
        axes[i, 0].set_ylabel(labels[cv])
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[-1, 0].set_xlabel("c")
    axes[-1, 1].set_xlabel("decay ratio c/(c + λ1 − λ2)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_decay_collapse.png", dpi=150)


if __name__ == "__main__":
    main()
