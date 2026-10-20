"""Paper-number regression test (needs a finished run; not run in CI).

    HOC_RUN=canonical-2026-10 uv run pytest -m data tests/test_paper_numbers.py

Compares <run>/paper/key_results.json (written by scripts/paper/make_numbers.py)
with tests/golden/key_results.json: every number must print the same and agree
to rtol 1e-9. Regenerate the golden file only together with an explained
compare_runs diff (see CONTRIBUTING.md).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hoc.runs import resolve_run_dir

GOLDEN = Path(__file__).parent / "golden" / "key_results.json"


@pytest.mark.data
def test_paper_numbers_match_golden() -> None:
    run_dir = resolve_run_dir(None)
    path = run_dir / "paper" / "key_results.json"
    if not path.exists():
        pytest.skip(f"{path} not found; run scripts/reproduce.py and scripts/paper/make_numbers.py first")
    actual, expected = json.loads(path.read_text()), json.loads(GOLDEN.read_text())
    assert set(actual) == set(expected), f"macros differ: +{sorted(set(actual) - set(expected))} -{sorted(set(expected) - set(actual))}"
    problems = []
    for name, want in expected.items():
        got = actual[name]
        if got["text"] != want["text"]:
            problems.append(f"\\{name}: prints {got['text']!r}, golden {want['text']!r}")
        a, b = got["value"], want["value"]
        if not (a is None and b is None) and not (math.isnan(a) and math.isnan(b)) and not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"\\{name}: {a!r} vs golden {b!r}")
    assert not problems, "\n".join(problems[:50])
