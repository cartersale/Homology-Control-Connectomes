#!/usr/bin/env python
"""Revision (R2 Majors 1 & 5): Gramian spectra, random/matched baselines, Perron loading.

Recomputes, at the primary setting (c = 0.1, eps = 1e-5, infinite horizon), the
regularized controllability Gramian W_eps for every static top-k driver set of every
saved ranking, the 100 primary random sets (regenerated with the primary seeds), and
new matched control sets:

* ``degree_matched``         100 sets per cycle set; each cycle node is replaced by a node
                             drawn from its nearest-strength neighbours (pool of
                             max(5, round(0.05 N)) nodes, cycle-set nodes excluded).
* ``spatial_matched``        100 uniformly drawn k-sets (from 20 000 candidates) whose mean
                             pairwise centroid distance is closest to the cycle set's (k >= 2).
* ``degree_spatial_matched`` 100 of 5 000 degree-matched draws closest in spatial spread.
* ``cycle_resid``            top-k of the residual of cycle participation regressed (OLS,
                             with intercept) on strength within subject.

Outputs (results/revision/spectra/):
  set_metrics.csv                     one row per driver set
  spectra_scale-{N}.npz               sorted (descending) eigenvalue spectra of W_eps
  centroids_scale-{N}.csv             parcel centroids (fsaverage5 pial, mean vertex)
  verification.json                   reproduction checks against results/primary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hoc.atlas import EXCLUDED_LABELS, SCALE_TO_RESOLUTION, build_label_table  # noqa: E402
from hoc.config import load_config  # noqa: E402
from hoc.control import assemble_gramian, gramian_metrics, single_node_gramians  # noqa: E402
from hoc.preprocessing import stabilize_adjacency  # noqa: E402
from hoc.rankings import random_top_k_nodes, top_k_nodes  # noqa: E402
from hoc.utils import make_rng, subject_seed  # noqa: E402

OUT = REPO_ROOT / "results" / "revision" / "spectra"
METHODS = [
    "degree_strength",
    "eigenvector_centrality",
    "betweenness",
    "cycle_participation",
    "participation_coefficient",
    "triangle_participation",
    "tetrahedron_participation",
    "degree_normalized_cycle_participation",
    "cycle_criticality",
]
# Stream tags for new random draws (primary random sets use subject_seed(seed, scale, subj, k, sample)).
TAG_DEGREE = 7101
TAG_SPATIAL = 7102
TAG_DEGSPAT = 7103
N_MATCHED = 100
N_SPATIAL_CANDIDATES = 20000
N_DEGSPAT_CANDIDATES = 5000


def parcel_centroids(scale: int, atlas_dir: Path) -> pd.DataFrame:
    """Mean pial-vertex coordinate per parcel on fsaverage5, in pipeline node order."""
    from nilearn import datasets, surface

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    rows = []
    res = SCALE_TO_RESOLUTION[scale]
    for hemi, key in (("L", "pial_left"), ("R", "pial_right")):
        coords = np.asarray(surface.load_surf_mesh(fs[key]).coordinates)
        annot = atlas_dir / f"atl-Cammoun2012_space-fsaverage5_res-{res}_hemi-{hemi}_deterministic.annot"
        vlabels, _, names = nib.freesurfer.read_annot(str(annot))
        for idx, raw in enumerate(names):
            name = raw.decode()
            if name in EXCLUDED_LABELS:
                continue
            mask = vlabels == idx
            c = coords[mask].mean(axis=0) if mask.any() else np.full(3, np.nan)
            rows.append({"region_label": f"{hemi}_{name}", "n_vertices": int(mask.sum()), "x": c[0], "y": c[1], "z": c[2]})
    return pd.DataFrame(rows)


def mean_pairwise(dist: np.ndarray, nodes: np.ndarray) -> float:
    if len(nodes) < 2:
        return np.nan
    sub = dist[np.ix_(nodes, nodes)]
    iu = np.triu_indices(len(nodes), 1)
    return float(sub[iu].mean())


def degree_matched_draw(cycle_set, strength, excluded, n_pool, rng):
    chosen: list[int] = []
    for node in rng.permutation(cycle_set):
        order = np.argsort(np.abs(strength - strength[node]), kind="mergesort")
        pool = [int(j) for j in order if j not in excluded and j not in chosen][:n_pool]
        chosen.append(int(rng.choice(pool)))
    return np.asarray(chosen, dtype=int)


def set_row(metrics, nodes, strength, strength_pct, perron_sq, dist, eps):
    ev = metrics["eigenvalues"]
    raw_w = metrics["raw_eigenvalues"] - eps  # eigenvalues of W itself
    return {
        "effective_rank": metrics["effective_rank"],
        "participation_ratio": metrics["participation_ratio"],
        "log10_average_energy": metrics["log10_average_energy"],
        "lambda_min": float(ev[0]),
        "lambda_max": float(ev[-1]),
        "log10_lambda_min": float(np.log10(ev[0])),
        "log10_lambda_max": float(np.log10(ev[-1])),
        "log10_condition_number": metrics["log10_condition_number"],
        "lambda_min_equals_eps": bool(np.isclose(ev[0], eps, rtol=1e-9, atol=0.0)),
        "frac_at_floor": metrics["fraction_eigenvalues_at_floor"],
        "frac_W_below_eps": float(np.mean(raw_w < eps)),
        "frac_W_below_0p1eps": float(np.mean(raw_w < 0.1 * eps)),
        "top1_share": float(ev[-1] / ev.sum()),
        "perron_loading": float(perron_sq[nodes].sum()),
        "mean_strength": float(strength[nodes].mean()),
        "mean_strength_pct": float(strength_pct[nodes].mean()),
        "mean_pairwise_dist": mean_pairwise(dist, nodes),
        "nodes": ";".join(map(str, nodes)),
    }


def run_subject(scale, subj, adjacency, scores, dist, cfg):
    seed = int(cfg["analysis"]["random_seed"])
    c = float(cfg["control"]["primary_c"])
    eps = float(cfg["control"]["primary_epsilon"])
    bottom = int(cfg["control"]["bottom_eigenspace_dim"])
    k_values = [int(k) for k in cfg["analysis"]["top_k_values"]]
    n_random = int(cfg["rankings"]["n_random_rankings"])
    n = adjacency.shape[0]
    system, _ = stabilize_adjacency(adjacency, c)
    grams = single_node_gramians(system)

    evals, evecs = np.linalg.eigh(adjacency)
    v = evecs[:, -1]
    perron_sq = v**2  # unit norm; sign irrelevant
    strength = scores["degree_strength"]
    strength_pct = (np.argsort(np.argsort(strength)) + 0.5) / n  # 0..1, 1 = highest strength
    cp = scores["cycle_participation"]
    X = np.column_stack([np.ones(n), strength])
    beta, *_ = np.linalg.lstsq(X, cp, rcond=None)
    resid = cp - X @ beta
    all_scores = dict(scores)
    all_scores["cycle_resid"] = resid
    n_pool = max(5, int(round(0.05 * n)))

    rows = []
    spec_methods = {m: np.zeros((len(k_values), n)) for m in all_scores}
    spec_random = np.zeros((len(k_values), n_random, n))
    rawW = {m: np.zeros((len(k_values), n)) for m in ("degree_strength", "cycle_participation")}
    rawW_random = np.zeros((len(k_values), n_random, n), dtype=np.float32)
    spec_matched = {f: np.zeros((len(k_values), n)) for f in ("degree_matched", "spatial_matched", "degree_spatial_matched")}

    def evaluate(nodes):
        return gramian_metrics(assemble_gramian(grams, nodes), epsilon=eps, bottom_dim=bottom)

    base = {"subject_id": f"sub-{subj:03d}", "scale": scale, "n_nodes": n}
    for ki, k in enumerate(k_values):
        for m, s in all_scores.items():
            nodes = top_k_nodes(s, k)
            met = evaluate(nodes)
            spec_methods[m][ki] = met["eigenvalues"][::-1]
            if m in rawW:
                rawW[m][ki] = (met["raw_eigenvalues"] - eps)[::-1]
            rows.append({**base, "k": k, "family": m, "sample": -1, **set_row(met, nodes, strength, strength_pct, perron_sq, dist, eps)})
        for r in range(n_random):
            nodes = random_top_k_nodes(n, k, make_rng(subject_seed(seed, scale, subj, k, r)))
            met = evaluate(nodes)
            spec_random[ki, r] = met["eigenvalues"][::-1]
            rawW_random[ki, r] = (met["raw_eigenvalues"] - eps)[::-1]
            rows.append({**base, "k": k, "family": "random", "sample": r, **set_row(met, nodes, strength, strength_pct, perron_sq, dist, eps)})

        cyc = top_k_nodes(cp, k)
        excluded = set(int(x) for x in cyc)
        # (a) degree-matched
        rng = make_rng(subject_seed(seed, scale, subj, k, TAG_DEGREE))
        acc = np.zeros(n)
        for r in range(N_MATCHED):
            nodes = degree_matched_draw(cyc, strength, excluded, n_pool, rng)
            met = evaluate(nodes)
            acc += met["eigenvalues"][::-1]
            rows.append({**base, "k": k, "family": "degree_matched", "sample": r, **set_row(met, nodes, strength, strength_pct, perron_sq, dist, eps)})
        spec_matched["degree_matched"][ki] = acc / N_MATCHED
        # (b) spatially matched (mean pairwise centroid distance); undefined for k = 1
        if k >= 2:
            d_cyc = mean_pairwise(dist, cyc)
            rng = make_rng(subject_seed(seed, scale, subj, k, TAG_SPATIAL))
            cands = np.array([rng.permutation(n)[:k] for _ in range(N_SPATIAL_CANDIDATES)])
            cands = cands[[set(int(q) for q in cc) != excluded for cc in cands]]
            d = np.array([mean_pairwise(dist, cc) for cc in cands])
            pick = np.argsort(np.abs(d - d_cyc), kind="mergesort")[:N_MATCHED]
            acc = np.zeros(n)
            for r, idx in enumerate(pick):
                nodes = cands[idx]
                met = evaluate(nodes)
                acc += met["eigenvalues"][::-1]
                rows.append({**base, "k": k, "family": "spatial_matched", "sample": r, **set_row(met, nodes, strength, strength_pct, perron_sq, dist, eps)})
            spec_matched["spatial_matched"][ki] = acc / N_MATCHED
            # (c) degree + spatial
            rng = make_rng(subject_seed(seed, scale, subj, k, TAG_DEGSPAT))
            cands = np.array([degree_matched_draw(cyc, strength, excluded, n_pool, rng) for _ in range(N_DEGSPAT_CANDIDATES)])
            d = np.array([mean_pairwise(dist, cc) for cc in cands])
            pick = np.argsort(np.abs(d - d_cyc), kind="mergesort")[:N_MATCHED]
            acc = np.zeros(n)
            for r, idx in enumerate(pick):
                nodes = cands[idx]
                met = evaluate(nodes)
                acc += met["eigenvalues"][::-1]
                rows.append({**base, "k": k, "family": "degree_spatial_matched", "sample": r, **set_row(met, nodes, strength, strength_pct, perron_sq, dist, eps)})
            spec_matched["degree_spatial_matched"][ki] = acc / N_MATCHED
        else:
            spec_matched["spatial_matched"][ki] = np.nan
            spec_matched["degree_spatial_matched"][ki] = np.nan

    extra = {
        "perron_eigenvalue": float(evals[-1]),
        "cp_strength_spearman": float(pd.Series(cp).corr(pd.Series(strength), method="spearman")),
        "resid_strength_spearman": float(pd.Series(resid).corr(pd.Series(strength), method="spearman")),
        "resid_cp_spearman": float(pd.Series(resid).corr(pd.Series(cp), method="spearman")),
    }
    return subj, rows, spec_methods, spec_random, rawW, rawW_random, spec_matched, extra


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.toml")
    ap.add_argument("--subject-limit", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    OUT.mkdir(parents=True, exist_ok=True)
    atlas_dir = REPO_ROOT / cfg["paths"]["atlas_dir"]
    k_values = [int(k) for k in cfg["analysis"]["top_k_values"]]

    print("Loading node scores ...", flush=True)
    ns = pd.read_csv(REPO_ROOT / "results/primary/node_scores.csv", usecols=["subject_id", "scale", "node_index", "method", "score"])
    all_rows = []
    extras = []
    verification = {}
    for scale in [int(s) for s in cfg["analysis"]["parcellation_scales"]]:
        labels = build_label_table(scale, atlas_dir)
        cen = parcel_centroids(scale, atlas_dir)
        assert (cen["region_label"].to_numpy() == labels["region_label"].to_numpy()).all()
        cen.insert(0, "node_index", np.arange(len(cen)))
        cen.to_csv(OUT / f"centroids_scale-{scale}.csv", index=False)
        xyz = cen[["x", "y", "z"]].to_numpy()
        dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)

        stack = np.load(REPO_ROOT / f"results/processed/connectomes_scale-{scale}.npz")["adjacency"]
        n_subj = stack.shape[0] if args.subject_limit is None else min(stack.shape[0], args.subject_limit)
        sub = ns[ns.scale == scale]
        score_maps = {}
        for (sid, m), g in sub.groupby(["subject_id", "method"]):
            score_maps.setdefault(sid, {})[m] = g.sort_values("node_index")["score"].to_numpy()

        print(f"Scale {scale}: {n_subj} subjects", flush=True)
        results = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(run_subject)(scale, s, stack[s], {m: score_maps[f"sub-{s:03d}"][m] for m in METHODS}, dist, cfg)
            for s in range(n_subj)
        )
        results.sort(key=lambda r: r[0])
        n = stack.shape[1]
        methods_all = METHODS + ["cycle_resid"]
        np.savez_compressed(
            OUT / f"spectra_scale-{scale}.npz",
            k_values=np.array(k_values),
            methods=np.array(methods_all),
            **{f"method__{m}": np.stack([r[2][m] for r in results]) for m in methods_all},
            random=np.stack([r[3] for r in results]),
            rawW__degree_strength=np.stack([r[4]["degree_strength"] for r in results]),
            rawW__cycle_participation=np.stack([r[4]["cycle_participation"] for r in results]),
            rawW__random=np.stack([r[5] for r in results]),
            **{f"matchedmean__{f}": np.stack([r[6][f] for r in results]) for f in ("degree_matched", "spatial_matched", "degree_spatial_matched")},
            epsilon=float(cfg["control"]["primary_epsilon"]),
            c=float(cfg["control"]["primary_c"]),
        )
        for r in results:
            all_rows.extend(r[1])
            extras.append({"scale": scale, "subject_id": f"sub-{r[0]:03d}", **r[7]})

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "set_metrics.csv", index=False)
    pd.DataFrame(extras).to_csv(OUT / "subject_extras.csv", index=False)

    # ---- Verification against saved primary outputs ----
    geo = pd.read_csv(REPO_ROOT / "results/primary/geometry_metrics.csv")
    geo = geo[geo.selection_strategy == "static"].copy()
    geo["sample"] = geo.groupby(["subject_id", "scale", "k", "method"]).cumcount()
    geo.loc[geo.method != "random", "sample"] = -1
    mine = df[df.family.isin(METHODS + ["random"])].rename(columns={"family": "method"})
    merged = mine.merge(geo, on=["subject_id", "scale", "k", "method", "sample"], suffixes=("", "_saved"))
    for col in ("effective_rank", "participation_ratio", "log10_condition_number", "log10_average_energy"):
        verification[f"max_abs_diff_{col}"] = float(np.nanmax(np.abs(merged[col] - merged[f"{col}_saved"])))
    verification["n_rows_compared"] = int(len(merged))
    ds = pd.read_csv(REPO_ROOT / "results/primary/driver_sets.csv")
    ds = ds[(ds.selection_strategy == "static")].copy()
    ds["sample"] = ds.groupby(["subject_id", "scale", "k", "method"]).cumcount()
    ds.loc[ds.method != "random", "sample"] = -1
    mm = mine.merge(ds[["subject_id", "scale", "k", "method", "sample", "node_indices"]], on=["subject_id", "scale", "k", "method", "sample"])
    same = [set(a.split(";")) == set(str(b).split(";")) for a, b in zip(mm["nodes"], mm["node_indices"])]
    verification["driver_sets_identical_fraction"] = float(np.mean(same))
    verification["driver_sets_compared"] = int(len(mm))
    piv = mine[mine["sample"] == -1].pivot_table(index=["subject_id", "scale", "k"], columns="method", values="effective_rank")
    diff = (piv["cycle_participation"] - piv["degree_strength"]).groupby(["subject_id", "scale"]).mean().groupby("scale")
    verification["cycle_minus_degree_ER_mean_over_k"] = {int(s): round(float(v.mean()), 4) for s, v in diff}
    verification["cycle_minus_degree_ER_npos"] = {int(s): int((v > 0).sum()) for s, v in diff}
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
