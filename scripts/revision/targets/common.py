"""Shared helpers for the revision target analyses (Reviewer 2 Major 6, R2 minor, R1 Minor 6).

Reuses the exact primary pipeline:
  A = A0 - (lambda_max(A0) + c) I            (hoc.preprocessing.stabilize_adjacency)
  W = infinite-horizon controllability Gramian (Lyapunov A W + W A^T = -B B^T)
  W_eps = W + eps I, eigenvalues floored at eps (hoc.control.gramian_metrics)
  E_target = x*^T W_eps^{-1} x*
Driver sets are the saved static top-k sets (results/primary/driver_sets.csv).

Because A is symmetric, the Lyapunov solution is computed in closed form in the eigenbasis
of A (W~_ij = (V^T B B^T V)_ij / -(l_i + l_j)); this is verified against the saved primary
target energies in 00_reproduce.py.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from hoc.atlas import EXCLUDED_LABELS, SCALE_TO_RESOLUTION  # noqa: E402
from hoc.preprocessing import stabilize_adjacency  # noqa: E402
from hoc.targets import _fsaverage5_from_fslr_index  # noqa: E402

OUT = REPO / "results" / "revision" / "targets"
OUT.mkdir(parents=True, exist_ok=True)
PROCESSED = REPO / "results" / "processed"
ATLAS_DIR = REPO / "data" / "external" / "atlas_cache" / "atl-cammoun2012" / "fsaverage5"
YEO_DIR = REPO / "data" / "external" / "yeo2011_fsaverage5"
SCALES = (68, 114, 219)
K_VALUES = (1, 2, 3, 5)
METHODS = ("degree_strength", "cycle_participation")
SEED = 20260423
PRIMARY_C = 0.1
PRIMARY_EPS = 1.0e-5

YEO7_NAMES = {1: "Visual", 2: "SomMot", 3: "DorsAttn", 4: "VentAttn", 5: "Limbic", 6: "FrontoParietal", 7: "Default"}

# Corrected keyword masks: exact Desikan-Killiany base-region membership (no substring matching).
ASSOCIATION_BASE = (
    "superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "lateralorbitofrontal",
    "medialorbitofrontal", "frontalpole", "superiorparietal", "inferiorparietal", "precuneus",
    "rostralanteriorcingulate", "caudalanteriorcingulate", "posteriorcingulate", "isthmuscingulate", "insula",
)
OCCIPITAL_BASE = ("cuneus", "lateraloccipital", "lingual", "pericalcarine")
LANDMARKS = {
    "landmark_v1": ("pericalcarine",),
    "landmark_m1": ("precentral",),
    "landmark_pcc_precuneus": ("posteriorcingulate", "precuneus"),
    "landmark_lpfc": ("rostralmiddlefrontal", "caudalmiddlefrontal"),
}


# ----------------------------------------------------------------------------- data
def labels(scale: int) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / f"atlas_labels_scale-{scale}.csv")


def adjacency_stack(scale: int) -> np.ndarray:
    return np.load(PROCESSED / f"connectomes_scale-{scale}.npz")["adjacency"]


@lru_cache(maxsize=None)
def driver_sets() -> pd.DataFrame:
    d = pd.read_csv(REPO / "results" / "primary" / "driver_sets.csv")
    d = d[(d.selection_strategy == "static") & d.method.isin(METHODS)].copy()
    d["nodes"] = d.node_indices.astype(str).map(lambda s: np.array([int(x) for x in s.split(";")]))
    return d


def driver_lookup() -> dict:
    d = driver_sets()
    return {(r.subject_id, int(r.scale), int(r.k), r.method): r.nodes for r in d.itertuples()}


def legacy_targets(scale: int) -> dict[str, np.ndarray]:
    t = np.load(PROCESSED / f"targets_scale-{scale}.npz")
    return {k: t[k] for k in t.files}


# ----------------------------------------------------------------------------- control
def system_eig(adjacency: np.ndarray, c: float):
    A, _ = stabilize_adjacency(adjacency, c)
    A = 0.5 * (A + A.T)
    lam, V = np.linalg.eigh(A)
    denom = -(lam[:, None] + lam[None, :])
    return lam, V, denom


def gramian(V: np.ndarray, denom: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    Vs = V[np.asarray(nodes, dtype=int), :]
    Wt = (Vs.T @ Vs) / denom
    W = V @ Wt @ V.T
    return 0.5 * (W + W.T)


def regularized_eig(W: np.ndarray, eps: float):
    """Eigen-decomposition of W + eps I with the eps floor used in hoc.control.gramian_metrics."""
    ev, U = np.linalg.eigh(W + eps * np.eye(W.shape[0]))
    ev = np.clip(ev, eps, None)
    return ev, U


def energies(ev: np.ndarray, U: np.ndarray, T: np.ndarray) -> np.ndarray:
    """E = x^T W_eps^{-1} x for each column of T."""
    coef = U.T @ T
    return np.sum(coef**2 / ev[:, None], axis=0)


def floor_fraction(ev: np.ndarray, U: np.ndarray, T: np.ndarray, eps: float) -> np.ndarray:
    """Fraction of each target's energy contributed by eigen-directions at (<=1.01x) the floor."""
    coef2 = (U.T @ T) ** 2
    at = ev <= eps * 1.01
    tot = np.sum(coef2 / ev[:, None], axis=0)
    return np.sum(coef2[at] / ev[at, None], axis=0) / tot


# ----------------------------------------------------------------------------- atlas geometry
def _annot(scale: int, hemi: str):
    res = SCALE_TO_RESOLUTION[scale]
    return nib.freesurfer.read_annot(str(ATLAS_DIR / f"atl-Cammoun2012_space-fsaverage5_res-{res}_hemi-{hemi}_deterministic.annot"))


@lru_cache(maxsize=None)
def parcel_vertices(scale: int) -> list[tuple[str, np.ndarray]]:
    """(hemi, vertex index array) per node, in node_index order (matches hoc.atlas.build_label_table)."""
    out = []
    lab = labels(scale)
    for hemi in ("L", "R"):
        vl, _, names = _annot(scale, hemi)
        for idx, raw in enumerate(names):
            name = raw.decode()
            if name in EXCLUDED_LABELS:
                continue
            out.append((hemi, name, np.flatnonzero(vl == idx)))
    assert [f"{h}_{n}" for h, n, _ in out] == lab.region_label.tolist()
    return [(h, v) for h, _, v in out]


def _surf(kind: str, hemi: str) -> np.ndarray:
    from neuromaps.datasets import fetch_atlas

    paths = fetch_atlas("fsaverage", "10k")[kind]
    p = next(str(x) for x in paths if f"hemi-{hemi}" in str(x))
    return np.asarray(nib.load(p).agg_data()[0], dtype=float)


@lru_cache(maxsize=None)
def centroids(scale: int, kind: str = "mid") -> np.ndarray:
    """Parcel centroids: mean vertex coordinate. kind='mid' = (white+pial)/2 fsaverage5; 'sphere' = sphere, re-projected."""
    rows = []
    for hemi, v in parcel_vertices(scale):
        if kind == "sphere":
            xyz = _surf("sphere", hemi)[v].mean(0)
            xyz = 100.0 * xyz / np.linalg.norm(xyz)
        else:
            xyz = (0.5 * (_surf("white", hemi) + _surf("pial", hemi)))[v].mean(0)
        rows.append(xyz)
    return np.asarray(rows)


def yeo_assignment(scale: int, n_networks: int = 7) -> pd.DataFrame:
    """Majority vote of fsaverage5 vertices (medial-wall vertices excluded from the vote)."""
    lab = labels(scale)
    rows = []
    annots = {h: nib.freesurfer.read_annot(str(YEO_DIR / f"{'lh' if h == 'L' else 'rh'}.Yeo2011_{n_networks}Networks_N1000.annot"))[0]
              for h in ("L", "R")}
    for node, (hemi, v) in enumerate(parcel_vertices(scale)):
        y = annots[hemi][v]
        net = y[y > 0]
        counts = np.bincount(net, minlength=n_networks + 1)
        winner = int(np.argmax(counts)) if net.size else 0
        rows.append(
            {
                "node_index": node,
                "region_label": lab.region_label[node],
                f"yeo{n_networks}_id": winner,
                f"yeo{n_networks}_purity": float(counts[winner] / net.size) if net.size else np.nan,
                f"yeo{n_networks}_medialwall_frac": float(np.mean(y == 0)),
                "n_vertices": int(v.size),
            }
        )
    df = pd.DataFrame(rows)
    if n_networks == 7:
        df["yeo7_name"] = df.yeo7_id.map(YEO7_NAMES).fillna("MedialWall")
    return df


NM_ATLAS = Path.home() / "neuromaps-data" / "atlases"


@lru_cache(maxsize=None)
def _registered_nn(hemi: str) -> np.ndarray:
    """Nearest non-medial-wall fsLR-32k vertex for each fsaverage5 vertex, using the fsLR sphere
    *registered to fsaverage* (tpl-fsLR_space-fsaverage_den-32k sphere, HCP/neuromaps), so both meshes
    share the fsaverage spherical coordinate system."""
    from scipy.spatial import cKDTree

    src = nib.load(str(NM_ATLAS / "fsLR" / f"tpl-fsLR_space-fsaverage_den-32k_hemi-{hemi}_sphere.surf.gii")).agg_data()[0]
    trg = nib.load(str(NM_ATLAS / "fsaverage" / f"tpl-fsaverage_den-10k_hemi-{hemi}_sphere.surf.gii")).agg_data()[0]
    cortex = np.asarray(nib.load(str(NM_ATLAS / "fsLR" / f"tpl-fsLR_den-32k_hemi-{hemi}_desc-nomedialwall_dparc.label.gii")).agg_data()) > 0
    src = src / np.linalg.norm(src, axis=1, keepdims=True)
    trg = trg / np.linalg.norm(trg, axis=1, keepdims=True)
    idx = np.flatnonzero(cortex)
    _, nn = cKDTree(src[idx]).query(trg)
    return idx[nn]


@lru_cache(maxsize=None)
def margulies_vertex(gradient: int, registered: bool = True) -> dict[str, np.ndarray]:
    """Margulies (2016) gradient on fsaverage5 vertices.

    registered=True  : correct fsLR->fsaverage registration sphere (used for all revision analyses).
    registered=False : legacy repo projection (hoc.targets._fsaverage5_from_fslr_index), which matches the
                       *native* fsLR sphere to the fsaverage sphere without registration.
    """
    from neuromaps.datasets import fetch_annotation

    ann = fetch_annotation(source="margulies2016", desc=f"fcgradient0{gradient}", space="fsLR", den="32k")
    out = {}
    for i, hemi in enumerate(("L", "R")):
        vg = np.asarray(nib.load(str(ann[i])).agg_data(), dtype=float)
        out[hemi] = vg[_registered_nn(hemi)] if registered else vg[_fsaverage5_from_fslr_index(hemi)]
    return out


def margulies_parcel(scale: int, gradient: int, registered: bool = True) -> np.ndarray:
    """Parcel mean of the vertex-resampled map (same averaging as hoc.targets.margulies_principal_gradient)."""
    vg = margulies_vertex(gradient, registered)
    return np.array([vg[h][v].mean() if v.size else 0.0 for h, v in parcel_vertices(scale)])


# ----------------------------------------------------------------------------- stats
def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / np.linalg.norm(x)


def rank_biserial(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x != 0)]
    if x.size == 0:
        return np.nan
    r = rankdata(np.abs(x))
    wp, wm = r[x > 0].sum(), r[x < 0].sum()
    return float((wp - wm) / (wp + wm))


def summarize(values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> dict:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    boots = np.median(rng.choice(v, size=(n_boot, v.size), replace=True), axis=1)
    p = wilcoxon(v, zero_method="wilcox", alternative="two-sided").pvalue if np.any(v != 0) else 1.0
    return {
        "n": int(v.size),
        "median_pct": float(np.median(v)),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "n_negative": int(np.sum(v < 0)),
        "n_positive": int(np.sum(v > 0)),
        "wilcoxon_p": float(p),
        "rank_biserial": rank_biserial(v),
    }


def holm(df: pd.DataFrame, by: list[str], p: str = "wilcoxon_p", out: str = "p_holm") -> pd.DataFrame:
    df = df.copy()
    df[out] = np.nan
    for _, idx in df.groupby(by).groups.items():
        df.loc[idx, out] = multipletests(df.loc[idx, p], method="holm")[1]
    return df


def pct_subject_table(E: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """E has columns keys + [subject_id, k, method, energy]; returns per-subject k-averaged pct difference."""
    w = E.pivot_table(index=keys + ["subject_id", "k"], columns="method", values="energy").reset_index()
    w["pct"] = 100.0 * (w.cycle_participation - w.degree_strength) / w.degree_strength
    return w.groupby(keys + ["subject_id"], as_index=False).agg(pct=("pct", "mean"),
                                                                E_degree=("degree_strength", "mean"),
                                                                E_cycle=("cycle_participation", "mean"))
