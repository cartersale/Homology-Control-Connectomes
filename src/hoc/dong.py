"""Pilot Dong-style stochastic tracking helpers.

The functions here intentionally separate fitted-dynamics topology from the
older structural-connectome analyses.  They implement a lightweight,
auditable pilot path: fit linear stochastic dynamics from ROI time series,
score topology on transformed coupling matrices, and evaluate driver sets
with a covariance-tracking proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import solve_continuous_lyapunov
from scipy.stats import pearsonr

from .control import assemble_gramian, single_node_gramians
from .rankings import stable_descending_ranks, top_k_nodes
from .topology import cycle_participation, min_max_normalize


@dataclass(frozen=True)
class FittedDynamics:
    """Container for one fitted continuous-time stochastic model."""

    system_matrix: np.ndarray
    residual_covariance: np.ndarray
    discrete_matrix: np.ndarray
    spectral_abscissa_before_shift: float
    stabilization_shift: float


def run_datalad_get(paths: list[Path], dataset_root: Path) -> None:
    """Fetch git-annex payloads for a small selected file list."""
    if not paths:
        return
    relative_paths = [str(path.relative_to(dataset_root)) for path in paths]
    datalad = shutil.which("datalad")
    if datalad is not None:
        subprocess.run([datalad, "get", *relative_paths], cwd=dataset_root, check=True)
        return
    git_annex = shutil.which("git-annex")
    if git_annex is not None:
        subprocess.run([git_annex, "get", *relative_paths], cwd=dataset_root, check=True)
        return
    raise RuntimeError(
        "Cannot fetch DataLad/git-annex payloads because neither 'datalad' nor "
        "'git-annex' is installed or available on PATH. Install DataLad, for example "
        "`brew install datalad` on macOS or `python -m pip install datalad` plus "
        "git-annex, then rerun with --fetch-payloads. You can also fetch files "
        "outside this script and rerun without --fetch-payloads."
    )


def zscore_timeseries(timeseries: np.ndarray) -> np.ndarray:
    """Column-wise z-score with finite-value cleanup."""
    values = np.asarray(timeseries, dtype=float)
    values = np.where(np.isfinite(values), values, np.nan)
    means = np.nanmean(values, axis=0)
    stds = np.nanstd(values, axis=0, ddof=1)
    stds = np.where(stds > 0.0, stds, 1.0)
    zscored = (values - means) / stds
    return np.nan_to_num(zscored, nan=0.0, posinf=0.0, neginf=0.0)


def fit_linear_stochastic_dynamics(
    timeseries: np.ndarray,
    dt: float = 1.0,
    ridge: float = 1.0e-3,
    stability_margin: float = 1.0e-2,
) -> FittedDynamics:
    """Fit x[t+1] = A_d x[t] + noise and convert to stable continuous time."""
    x = zscore_timeseries(timeseries)
    if x.shape[0] < 3:
        raise ValueError("At least three retained time points are required to fit dynamics.")
    predictors = x[:-1]
    responses = x[1:]
    n_nodes = x.shape[1]
    gram = predictors.T @ predictors + ridge * np.eye(n_nodes)
    discrete = np.linalg.solve(gram, predictors.T @ responses).T
    residuals = responses - predictors @ discrete.T
    residual_covariance = np.cov(residuals, rowvar=False)
    residual_covariance = regularize_covariance(residual_covariance, epsilon=ridge)

    continuous = (discrete - np.eye(n_nodes)) / float(dt)
    spectral_abscissa = float(np.max(np.real(np.linalg.eigvals(continuous))))
    shift = max(0.0, spectral_abscissa + stability_margin)
    if shift > 0.0:
        continuous = continuous - shift * np.eye(n_nodes)
    return FittedDynamics(
        system_matrix=continuous,
        residual_covariance=residual_covariance,
        discrete_matrix=discrete,
        spectral_abscissa_before_shift=spectral_abscissa,
        stabilization_shift=shift,
    )


def regularize_covariance(covariance: np.ndarray, epsilon: float = 1.0e-6) -> np.ndarray:
    """Return a symmetric positive-definite covariance approximation."""
    cov = 0.5 * (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, a_min=epsilon, a_max=None)
    return 0.5 * ((eigenvectors * eigenvalues) @ eigenvectors.T + (eigenvectors * eigenvalues) @ eigenvectors.T)


def stationary_covariance(system_matrix: np.ndarray, noise_covariance: np.ndarray) -> np.ndarray:
    """Solve the continuous-time stationary covariance Lyapunov equation."""
    covariance = solve_continuous_lyapunov(system_matrix, -noise_covariance)
    return regularize_covariance(covariance)


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    """Convert covariance to a finite correlation matrix."""
    diagonal = np.sqrt(np.maximum(np.diag(covariance), 1.0e-12))
    correlation = covariance / np.outer(diagonal, diagonal)
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def upper_triangle_correlation(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    """Pearson correlation between upper-triangular entries."""
    if matrix_a.shape[0] < 2:
        return np.nan
    upper = np.triu_indices(matrix_a.shape[0], k=1)
    values_a = np.asarray(matrix_a[upper], dtype=float)
    values_b = np.asarray(matrix_b[upper], dtype=float)
    if np.isclose(np.std(values_a), 0.0) or np.isclose(np.std(values_b), 0.0):
        return np.nan
    return float(pearsonr(values_a, values_b).statistic)


def model_qc_metrics(timeseries: np.ndarray, fitted: FittedDynamics) -> dict[str, float]:
    """Compute fit-quality diagnostics for the pilot stochastic model."""
    x = zscore_timeseries(timeseries)
    empirical_cov = regularize_covariance(np.cov(x, rowvar=False))
    empirical_fc = covariance_to_correlation(empirical_cov)
    model_cov = stationary_covariance(fitted.system_matrix, fitted.residual_covariance)
    model_fc = covariance_to_correlation(model_cov)

    empirical_lagged = (x[:-1].T @ x[1:]) / max(1, x.shape[0] - 1)
    model_lagged = empirical_cov @ fitted.discrete_matrix.T
    covariance_condition = float(np.linalg.cond(fitted.residual_covariance))
    return {
        "n_timepoints": int(x.shape[0]),
        "n_nodes": int(x.shape[1]),
        "spectral_abscissa_before_shift": fitted.spectral_abscissa_before_shift,
        "stabilization_shift": fitted.stabilization_shift,
        "is_stable": bool(np.max(np.real(np.linalg.eigvals(fitted.system_matrix))) < 0.0),
        "residual_covariance_condition": covariance_condition,
        "empirical_model_fc_correlation": upper_triangle_correlation(empirical_fc, model_fc),
        "empirical_model_lagged_covariance_correlation": upper_triangle_correlation(empirical_lagged, model_lagged),
    }


def transform_coupling_for_topology(system_matrix: np.ndarray, transform: str) -> np.ndarray:
    """Map a possibly signed/asymmetric fitted A to nonnegative symmetric weights."""
    matrix = np.asarray(system_matrix, dtype=float).copy()
    np.fill_diagonal(matrix, 0.0)
    if transform == "A_sym":
        weights = 0.5 * (np.abs(matrix) + np.abs(matrix.T))
    elif transform == "A_pos":
        positive = np.clip(matrix, a_min=0.0, a_max=None)
        weights = 0.5 * (positive + positive.T)
    elif transform == "A_abs":
        weights = np.abs(matrix)
        weights = 0.5 * (weights + weights.T)
    else:
        raise ValueError(f"Unknown fitted-A topology transform: {transform}")
    np.fill_diagonal(weights, 0.0)
    max_value = float(np.max(weights))
    if max_value > 0.0:
        weights = weights / max_value
    return weights


def weighted_clustering_scores(weights: np.ndarray) -> np.ndarray:
    """Weighted clustering coefficient as a node score."""
    graph = nx.from_numpy_array(np.asarray(weights, dtype=float))
    scores = nx.clustering(graph, weight="weight")
    return np.asarray([scores[node] for node in range(weights.shape[0])], dtype=float)


def residualized_cycle_scores(cycle_scores: np.ndarray, degree_scores: np.ndarray, strength_scores: np.ndarray) -> np.ndarray:
    """Residualize cycle participation against degree and weighted strength."""
    design = np.column_stack(
        [
            np.ones_like(cycle_scores, dtype=float),
            np.asarray(degree_scores, dtype=float),
            np.asarray(strength_scores, dtype=float),
        ]
    )
    coefficients = np.linalg.lstsq(design, np.asarray(cycle_scores, dtype=float), rcond=None)[0]
    residuals = np.asarray(cycle_scores, dtype=float) - design @ coefficients
    return min_max_normalize(residuals)


def fitted_a_node_scores(system_matrix: np.ndarray, node_gramians: np.ndarray, transform: str) -> pd.DataFrame:
    """Compute node scores used for fitted-A topology analyses."""
    weights = transform_coupling_for_topology(system_matrix, transform)
    binary_degree = np.sum(weights > 0.0, axis=1).astype(float)
    strength = np.sum(weights, axis=1)
    cycle, _ = cycle_participation(weights)
    residual_cycle = residualized_cycle_scores(cycle, binary_degree, strength)
    clustering = weighted_clustering_scores(weights)
    avg_control = np.asarray([float(np.trace(node_gramians[index])) for index in range(weights.shape[0])], dtype=float)
    table = pd.DataFrame(
        {
            "node_index": np.arange(weights.shape[0]),
            "topology_transform": transform,
            "degree": binary_degree,
            "strength": strength,
            "clustering": clustering,
            "average_controllability": avg_control,
            "cycle_participation": cycle,
            "cycle_residual": residual_cycle,
        }
    )
    for column in [
        "degree",
        "strength",
        "clustering",
        "average_controllability",
        "cycle_participation",
        "cycle_residual",
    ]:
        table[f"{column}_rank"] = stable_descending_ranks(table[column].to_numpy())
    return table


def gaussian_kl(target_covariance: np.ndarray, candidate_covariance: np.ndarray) -> float:
    """KL(target || candidate) for zero-mean multivariate Gaussians."""
    target = regularize_covariance(target_covariance)
    candidate = regularize_covariance(candidate_covariance)
    n_nodes = target.shape[0]
    sign_target, logdet_target = np.linalg.slogdet(target)
    sign_candidate, logdet_candidate = np.linalg.slogdet(candidate)
    if sign_target <= 0 or sign_candidate <= 0:
        return np.nan
    solved = np.linalg.solve(candidate, target)
    return float(0.5 * (np.trace(solved) - n_nodes + logdet_candidate - logdet_target))


def controlled_covariance_proxy(
    patient_covariance: np.ndarray,
    target_covariance: np.ndarray,
    gramian: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Approximate covariance after tracking control through a driver-set Gramian.

    The control influence operator C = W(W + eps I)^-1 moves the patient
    covariance toward the healthy target in controllable directions while
    leaving weakly controllable directions close to the patient baseline.
    """
    n_nodes = gramian.shape[0]
    regularized = 0.5 * (gramian + gramian.T) + epsilon * np.eye(n_nodes)
    influence = np.linalg.solve(regularized, gramian).T
    delta = target_covariance - patient_covariance
    controlled = patient_covariance + influence @ delta @ influence.T
    return regularize_covariance(controlled, epsilon=epsilon)


def tracking_set_metrics(
    patient_covariance: np.ndarray,
    target_covariance: np.ndarray,
    node_gramians: np.ndarray,
    node_indices: np.ndarray,
    epsilon: float,
) -> dict[str, float]:
    """Evaluate a driver set with energy and KL tracking summaries."""
    gramian = assemble_gramian(node_gramians, node_indices)
    regularized = 0.5 * (gramian + gramian.T) + epsilon * np.eye(gramian.shape[0])
    tracking_energy = float(np.trace(np.linalg.solve(regularized, target_covariance)))
    controlled = controlled_covariance_proxy(patient_covariance, target_covariance, gramian, epsilon=epsilon)
    uncontrolled_kl = gaussian_kl(target_covariance, patient_covariance)
    controlled_kl = gaussian_kl(target_covariance, controlled)
    baseline_error = np.abs(np.diag(patient_covariance) - np.diag(target_covariance))
    controlled_error = np.abs(np.diag(controlled) - np.diag(target_covariance))
    improved = controlled_error < baseline_error
    return {
        "tracking_energy": tracking_energy,
        "log10_tracking_energy": float(np.log10(max(tracking_energy, epsilon))),
        "uncontrolled_kl": uncontrolled_kl,
        "controlled_kl": controlled_kl,
        "kl_reduction": float(uncontrolled_kl - controlled_kl),
        "fraction_nodes_improved": float(np.mean(improved)),
        "worst_decile_node_error": float(np.quantile(controlled_error, 0.9)),
        "node_error_variance": float(np.var(controlled_error, ddof=1)) if controlled_error.size > 1 else 0.0,
    }


def single_node_tracking_energy(
    patient_covariance: np.ndarray,
    target_covariance: np.ndarray,
    node_gramians: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Compute single-node tracking-energy oracle scores."""
    energies = np.zeros(node_gramians.shape[0], dtype=float)
    for node_index in range(node_gramians.shape[0]):
        energies[node_index] = tracking_set_metrics(
            patient_covariance,
            target_covariance,
            node_gramians,
            np.asarray([node_index], dtype=int),
            epsilon,
        )["tracking_energy"]
    return energies


def degree_matched_high_cycle_nodes(cycle_scores: np.ndarray, degree_scores: np.ndarray, k: int, n_bins: int = 5) -> np.ndarray:
    """Select high-cycle nodes while spreading choices across degree bins."""
    frame = pd.DataFrame({"node": np.arange(len(cycle_scores)), "cycle": cycle_scores, "degree": degree_scores})
    try:
        frame["degree_bin"] = pd.qcut(frame["degree"].rank(method="first"), q=min(n_bins, len(frame)), labels=False)
    except ValueError:
        frame["degree_bin"] = 0
    chosen: list[int] = []
    while len(chosen) < k:
        progressed = False
        for _, group in frame.groupby("degree_bin", sort=True):
            candidates = group[~group["node"].isin(chosen)].sort_values("cycle", ascending=False)
            if candidates.empty:
                continue
            chosen.append(int(candidates.iloc[0]["node"]))
            progressed = True
            if len(chosen) == k:
                break
        if not progressed:
            break
    if len(chosen) < k:
        fallback = [int(node) for node in top_k_nodes(cycle_scores, len(cycle_scores)) if int(node) not in chosen]
        chosen.extend(fallback[: k - len(chosen)])
    return np.asarray(chosen[:k], dtype=int)


def target_aware_greedy_set(
    seed_node: int,
    k: int,
    candidate_nodes: np.ndarray,
    patient_covariance: np.ndarray,
    target_covariance: np.ndarray,
    node_gramians: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, list[dict]]:
    """Greedy set selection with a fixed seed and KL objective."""
    selected = [int(seed_node)]
    candidate_nodes = np.asarray(candidate_nodes, dtype=int)
    rows: list[dict] = []
    seed_metrics = tracking_set_metrics(patient_covariance, target_covariance, node_gramians, np.asarray(selected), epsilon)
    rows.append({"step": 1, "added_node": int(seed_node), **seed_metrics})
    for step in range(2, k + 1):
        best_node = None
        best_metrics = None
        for candidate in candidate_nodes:
            candidate_int = int(candidate)
            if candidate_int in selected:
                continue
            proposal = np.asarray(selected + [candidate_int], dtype=int)
            metrics = tracking_set_metrics(patient_covariance, target_covariance, node_gramians, proposal, epsilon)
            if best_metrics is None or metrics["controlled_kl"] < best_metrics["controlled_kl"]:
                best_node = candidate_int
                best_metrics = metrics
        if best_node is None or best_metrics is None:
            break
        selected.append(best_node)
        rows.append({"step": step, "added_node": best_node, **best_metrics})
    return np.asarray(selected, dtype=int), rows


def attenuate_nodes(system_matrix: np.ndarray, nodes: np.ndarray, factor: float) -> np.ndarray:
    """Attenuate incoming and outgoing coupling for selected nodes."""
    attenuated = np.asarray(system_matrix, dtype=float).copy()
    diagonal = np.diag(attenuated).copy()
    nodes = np.asarray(nodes, dtype=int)
    attenuated[nodes, :] *= factor
    attenuated[:, nodes] *= factor
    np.fill_diagonal(attenuated, diagonal)
    return attenuated
