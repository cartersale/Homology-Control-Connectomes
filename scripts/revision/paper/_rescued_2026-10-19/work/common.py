"""Read-only helpers for building Supplement tables. Never writes into the repo."""
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/cartersale/Research/projects/higher_order_connectomes")
RES = REPO / "results" / "revision"
OUT = Path("/private/tmp/claude-501/-Users-cartersale-Research-projects-higher-order-connectomes/"
           "cea7c7b4-c39c-4a27-8881-388281d3a80d/scratchpad/tables")
SCALES = (68, 114, 219)

LABEL = {
    "degree_strength": "Degree strength", "degree": "Degree strength",
    "eigenvector_centrality": "Eigenvector centrality",
    "betweenness": "Betweenness",
    "participation_coefficient": "Participation coefficient",
    "triangle_participation": "Triangle participation",
    "tetrahedron_participation": "Tetrahedron participation",
    "cycle_participation": "Cycle participation",
    "cycle_resid": "Cycle part.\\ (degree-residualized)",
    "degree_normalized_cycle_participation": "Degree-normalized cycle part.",
    "cycle_criticality": "Cycle criticality",
    "top_avg_ctrl": "Top-$k$ average controllability",
    "top_modal_gu": "Top-$k$ modal controllability (Gu)",
    "top_modal_ct": "Top-$k$ modal controllability (CT)",
    "low_perron": "Lowest Perron loading",
    "random": "Random (100/subject)", "random_mean": "Random (mean of 100)",
    "degree_matched": "Degree-matched random",
    "spatial_matched": "Spatially matched random",
    "degree_spatial_matched": "Degree+spatially matched random",
    "ripser": "Ripser cocycle", "gudhi": "GUDHI pairing", "geodesic": "Length-minimized",
    "birth_edge": "Birth edge",
}


def sgn(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:+.{d}f}"
    if s in ("-0." + "0" * d,):
        s = "-0." + "0" * d  # keep sign of tiny negatives explicit
    return s.replace("-", "$-$") if False else s


def num(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{d}f}"


def ptex(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p >= 0.995:
        return "1"
    if p >= 0.01:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def npos(n, tot=70):
    return f"{int(round(n))}/{tot}"


def tabular(colspec, header_lines, rows, midrules=()):
    """rows: list of lists of strings, or the string 'MIDRULE' / ('MULTI', text)."""
    out = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    out += header_lines
    out.append("\\midrule")
    for r in rows:
        if r == "MIDRULE":
            out.append("\\midrule")
        elif isinstance(r, tuple) and r[0] == "MULTI":
            out.append(f"\\multicolumn{{{r[2]}}}{{l}}{{\\textit{{{r[1]}}}}} \\\\")
        else:
            out.append(" & ".join(r) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out)


def block(title, note, tab):
    """A commented title + plain-text note, then the tabular."""
    lines = [f"% ==== {title} ===="]
    lines += ["% " + l for l in note.strip().splitlines()]
    return "\n".join(lines) + "\n" + tab + "\n"
