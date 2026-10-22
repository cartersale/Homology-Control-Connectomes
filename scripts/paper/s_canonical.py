"""Supplement S_canonical: hub concentration in canonical model networks vs connectomes.

The draft's S_canonical.tex was typed by hand from
results/revision/mechanism/canonical_summary.csv (written by
scripts/revision/mechanism/02_canonical_graphs.py); this builder regenerates it.
"""

from __future__ import annotations

from pathlib import Path

from paper_common import SCALES, Table, pd, tabular

MODELS = [
    ("ER", "Random (Erd\\H{o}s--R\\'enyi)"),
    ("BA", "Scale-free (Barab\\'asi--Albert)"),
    ("WS", "Small-world (Watts--Strogatz)"),
    ("SBM", "Modular (stochastic block)"),
    ("RGG", "Spatial (random geometric)"),
    ("CONN", "Human connectomes"),
]


def build(res: Path) -> list[Table]:
    cs = pd.read_csv(res / "mechanism" / "canonical_summary.csv", dtype={"kgrp": str})
    cs = cs[(cs.weighting == "weighted") & (cs.c == 0.1)]
    rows = []
    for model, label in MODELS:
        r = [label]
        for scale in SCALES:
            sub = cs[(cs.model == model) & (cs.scale == scale)]
            x1 = sub[sub.kgrp == "1"]
            x2 = sub[sub.kgrp == "2+"]
            assert len(x1) == 1 and len(x2) == 1, (model, scale)
            x1, x2 = x1.iloc[0], x2.iloc[0]
            r.append(f"{x2.gap:.2f}")
            for x in (x1, x2):
                r.append(f"{x.deg_minus_rand_q1:+.2f} ({x.deg_q1_pct:.2f})")
        rows.append(r)
    tab = tabular("l" + "ccc" * 3,
                  [" & \\multicolumn{3}{c}{68} & \\multicolumn{3}{c}{114} & \\multicolumn{3}{c}{219} \\\\",
                   "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
                   "Network & Gap & $k=1$ & $k\\geq2$ & Gap & $k=1$ & $k\\geq2$ & Gap & $k=1$ & $k\\geq2$ \\\\"], rows)
    note = """S_canonical. Hub concentration (q1) of degree-strength driver sets in weighted canonical model networks and in the
connectomes (c = 0.1, T = inf; q1 of the raw Gramian, kernel of scripts/revision/nulls/nulls_common.py Spectral). Previously typed by hand; now generated.
Source: results/revision/mechanism/canonical_summary.csv (written by scripts/revision/mechanism/02_canonical_graphs.py;
filters: weighting == 'weighted', c == 0.1). Models ER, BA, WS, SBM (7 blocks), RGG: 50 graphs per model and scale, density matched
to the scale, edge weights drawn from the pooled empirical weights; CONN = the connectomes with no zero-strength node
(69 of 70 at each scale in canonical_graphs.csv.gz), so the human row is not the 70-subject primary analysis.
Gap = column gap (mean spectral gap lambda1 - lambda2 over graphs; identical in the k = 1 and k >= 2 rows, taken from the 2+ row).
k = 1 cells: deg_minus_rand_q1 (mean over graphs of degree-set q1 minus mean q1 of 100 random sets), with deg_q1_pct in parentheses
(mean over graphs of the fraction of the 100 random sets with q1 below the degree set, ties 1/2; 0-1), from the kgrp == '1' row.
k >= 2 cells: the same columns from the kgrp == '2+' row (mean over graphs and k in {2,3,5}).
The binary-weighted versions (weighting == 'binary') are in the CSV but not tabulated: their gaps are in edge-count units (3-22),
so c = 0.1 is not regime-matched to the connectomes (see results/revision/mechanism/REPORT.md section 2)."""
    return [Table("S_canonical", tab, note)]
