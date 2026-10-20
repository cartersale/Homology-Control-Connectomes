#!/usr/bin/env python
"""Dependence of the cycle-degree spectral difference on the stabilization margin c.

For every subject, scale, c and k in {1,2,3,5}, and for the static degree and cycle sets:
  - effective rank of the regularized Gramian W_eps (primary pipeline definition)
  - q1 = lambda_1(W_eps) / tr W_eps (spectral concentration; as in Fig. 2D)
  - v1' W v1, with v1 the Perron vector of A0 (analytically L_S / (2c), L_S = sum_{i in S} v1_i^2)
  - effective rank after projecting out v1: W_perp = P W P, P = I - v1 v1'

Uses the eigenbasis Gramian from _common.Spectral (validated against the Lyapunov route).
In that basis W is M = K o (U_S' U_S); v1' W v1 is the Perron diagonal entry of M, and the
spectrum of P W P is the spectrum of M with the Perron row/column removed, plus one zero.

Writes results/revision/regime/perron_mode_sweep.csv (per subject x scale x c x k x method).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import K_VALUES, OUT_DIR, SCALES, Spectral, load_ranks, load_stack, metrics_from_eigs  # noqa: E402

C_VALUES = (0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 2.0)
EPSILON = 1.0e-5
METHODS = ("degree_strength", "cycle_participation")


def spectral_rows(spec: Spectral, c: float, driver_set: np.ndarray) -> dict:
    kernel = spec.kernel(c)
    m = spec.gramian_eigbasis(kernel, driver_set)
    p = int(np.argmax(spec.lam))  # Perron index in the eigenbasis
    full = np.linalg.eigvalsh(m)
    keep = np.delete(np.arange(m.shape[0]), p)
    perp = np.concatenate([[0.0], np.linalg.eigvalsh(m[np.ix_(keep, keep)])])
    metrics = metrics_from_eigs(np.vstack([full, perp]), EPSILON)
    ev = np.clip(full + EPSILON, EPSILON, None)
    loading = float(np.sum(spec.perron[np.asarray(driver_set, dtype=int)] ** 2))
    return {
        "effective_rank": metrics["effective_rank"][0],
        "effective_rank_perp1": metrics["effective_rank"][1],
        "q1": ev.max() / ev.sum(),
        "perron_energy": float(m[p, p]),
        "perron_energy_analytic": loading / (2.0 * c),
        "perron_loading": loading,
        "trace_w": float(np.trace(m)),
    }


def main() -> None:
    rows = []
    for scale in SCALES:
        stack = load_stack(scale)
        ranks = load_ranks(scale)
        subjects = sorted({s for s, _ in ranks})
        for si, subject_id in enumerate(subjects):
            spec = Spectral(stack[si])
            for method in METHODS:
                order = ranks[(subject_id, method)]
                for k in K_VALUES:
                    for c in C_VALUES:
                        rows.append({"subject_id": subject_id, "scale": scale, "method": method, "k": k, "c": c, **spectral_rows(spec, c, order[:k])})
        print("done scale", scale, flush=True)
    df = pd.DataFrame(rows)
    err = np.max(np.abs(df.perron_energy - df.perron_energy_analytic) / df.perron_energy_analytic)
    print(f"max relative error, v1'Wv1 vs L/(2c): {err:.2e}")
    df.to_csv(OUT_DIR / "perron_mode_sweep.csv", index=False)
    print("saved", OUT_DIR / "perron_mode_sweep.csv")


if __name__ == "__main__":
    main()
