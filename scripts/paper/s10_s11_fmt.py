"""Number formatting shared by the S10 (targets) and S11 (lesion) builders.

Ported unchanged from scripts/revision/paper/_rescued_2026-10-19/scripts/make_s10_s11.py
(these differ from paper_common.ptex / sgn: 2 significant digits for p, and $-$ for minus signs).
"""

from __future__ import annotations

import pandas as pd


def pf(p) -> str:
    """p-value: 2 decimals for p >= 0.1, 2 s.f. for 0.01 <= p < 0.1, else m.m x 10^e."""
    if pd.isna(p):
        return "--"
    if p >= 0.01:
        return f"{p:.2g}" if p < 0.1 else f"{p:.2f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def f2(x, sign: bool = True) -> str:
    """Two decimals (signed by default), with a math minus."""
    if pd.isna(x):
        return "--"
    s = f"{x:+.2f}" if sign else f"{x:.2f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def tex_esc(s: str) -> str:
    return s.replace("_", "\\_")
