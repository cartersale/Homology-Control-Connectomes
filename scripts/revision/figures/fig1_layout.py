"""Compose the Figure 1 panel SVGs into a full-width layout mock-up (7 in = 504 pt).

Run: python scripts/revision/figures/fig1_layout.py --run <name> [--overwrite]
Reads the panels fig1_panels.py wrote to runs/<name>/paper/figs/fig1/ and writes
fig1_layout.svg next to them.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hoc.runs import stage_from_argv  # noqa: E402

STAGE = "revision/figures/fig1_layout"
# Set from the run directory in main(): <run>/paper/figs/fig1 (same folder fig1_panels.py writes).
FIG_DIR: Path | None = None
OUT_PATH: Path | None = None
WIDTH = 504.0
INK = "#2b2b2b"
MUTED = "#6f6f6a"

# (panel file, x, y, width) in pt; height follows each panel's aspect ratio.
PLACEMENTS = [
    ("fig1a_model", 0, 36, 120),
    ("fig1b_v1_hub", 130, 30, 118),
    ("fig1b_v1_spread", 250, 30, 118),
    ("fig1b_v1_colorbar", 214, 178, 72),
    ("fig1c_loading_q1", 382, 38, 122),
]
LETTERS = [("A", 0, 12), ("B", 130, 12), ("C", 380, 12)]
CAPTIONS = [("hub set", 189, 28), ("spread-out set", 309, 28)]
HEIGHT = 212.0


def panel_group(name: str, x: float, y: float, width: float) -> str:
    svg = (FIG_DIR / f"{name}.svg").read_text()
    view_box = [float(v) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
    scale = width / view_box[2]
    body = svg[svg.index(">", svg.index("<svg")) + 1 : svg.rindex("</svg>")]
    # Prefix ids so clip paths and glyph definitions from different panels do not collide.
    body = re.sub(r'id="([^"]+)"', rf'id="{name}-\1"', body)
    body = re.sub(r"url\(#([^)]+)\)", rf"url(#{name}-\1)", body)
    body = re.sub(r'href="#([^"]+)"', rf'href="#{name}-\1"', body)
    return f'<g transform="translate({x:.2f} {y:.2f}) scale({scale:.5f})">{body}</g>'


def text(label: str, x: float, y: float, size: float, color: str, weight: str = "normal", anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Helvetica, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{label}</text>'
    )


def main() -> None:
    global FIG_DIR, OUT_PATH
    ctx = stage_from_argv(STAGE)
    ctx.require_stage("revision/figures/fig1_panels")
    ctx.begin()
    FIG_DIR = ctx.root / "paper" / "figs" / "fig1"
    OUT_PATH = FIG_DIR / "fig1_layout.svg"
    parts = [panel_group(*placement) for placement in PLACEMENTS]
    parts += [text(label, x, y, 11, INK, "bold") for label, x, y in LETTERS]
    parts += [text(label, x, y, 7.5, MUTED, anchor="middle") for label, x, y in CAPTIONS]
    OUT_PATH.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{WIDTH}pt" height="{HEIGHT}pt" viewBox="0 0 {WIDTH} {HEIGHT}">\n'
        + "\n".join(parts)
        + "\n</svg>\n"
    )
    print(OUT_PATH)
    ctx.finish()


if __name__ == "__main__":
    main()
