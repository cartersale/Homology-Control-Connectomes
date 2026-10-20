#!/usr/bin/env python
"""Extras for Task 4: piecewise Spearman (Perron loading <= / > 0.05) and subject-level tests
of the Perron share of W (v^T W v / tr W), degree vs cycle vs random, from perron_c_check.csv.

Reads and writes runs/<name>/revision/spectra/ (choose the run with ``--run <name>``)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hoc.runs import stage_from_argv  # noqa: E402
from importlib import import_module
paired_stats = import_module("02_analyze").paired_stats
ctx = stage_from_argv("revision/spectra/05_perron_extras")
ctx.require_stage("revision/spectra/04_perron_c_check")
ctx.begin()
OUT = ctx.root / "revision" / "spectra"
d = pd.read_csv(OUT / "perron_c_check.csv")
rows = []
for (c, sc, k), g in d.groupby(["c", "scale", "k"]):
    lo, hi = g[g.perron_loading <= 0.05], g[g.perron_loading > 0.05]
    rows.append({"c": c, "scale": sc, "k": k, "n_low": len(lo), "rho_low": spearmanr(lo.perron_loading, lo.effective_rank).statistic,
                 "n_high": len(hi), "rho_high": spearmanr(hi.perron_loading, hi.effective_rank).statistic if len(hi) > 10 else np.nan})
t = pd.DataFrame(rows); t.to_csv(OUT / "tables/t4c_perron_piecewise.csv", index=False); print(t.round(3).to_string())
key = ["subject_id", "scale", "k"]
rows = []
for c, gc in d.groupby("c"):
    det = gc[gc["sample"] == -1].pivot_table(index=key, columns="family", values="perron_share_W")
    rnd = gc[gc.family == "random"].groupby(key).perron_share_W.mean()
    for name, diff in (("degree - cycle", det.degree_strength - det.cycle_participation),
                       ("cycle - random", det.cycle_participation - rnd), ("degree - random", det.degree_strength - rnd)):
        s = diff.groupby(["subject_id", "scale"]).mean()
        for sc in (68, 114, 219):
            rows.append({"c": c, "comparison": name, "scale": sc, **paired_stats(s.xs(sc, level="scale").to_numpy())})
t = pd.DataFrame(rows)
t["p_holm"] = np.nan
for _, idx in t.groupby(["c", "scale"]).groups.items():
    t.loc[idx, "p_holm"] = multipletests(t.loc[idx, "p"], method="holm")[1]
t.to_csv(OUT / "tables/t4d_perron_share_W_tests.csv", index=False); print(t.round(3).to_string())
ctx.finish()
