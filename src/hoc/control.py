"""Controllability Gramians and control-energy summaries."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov


def single_node_gramians(
    system_matrix: np.ndarray,
    horizon: float | None = None,
) -> np.ndarray:
    """Precompute one Gramian per canonical input node."""
    n_nodes = system_matrix.shape[0]
    infinite_gramians = np.zeros((n_nodes, n_nodes, n_nodes), dtype=float)

    for node_index in range(n_nodes):
        basis = np.zeros(n_nodes, dtype=float)
        basis[node_index] = 1.0
        q_matrix = np.outer(basis, basis)
        gramian = solve_continuous_lyapunov(system_matrix, -q_matrix)
        infinite_gramians[node_index] = 0.5 * (gramian + gramian.T)

    if horizon is None:
        return infinite_gramians

    exp_at = matrix_exponential(system_matrix, horizon)
    finite_gramians = np.zeros_like(infinite_gramians)
    for node_index in range(n_nodes):
        tail = np.einsum("ij,jk,lk->il", exp_at, infinite_gramians[node_index], exp_at)
        finite_gramians[node_index] = 0.5 * (infinite_gramians[node_index] - tail + (infinite_gramians[node_index] - tail).T)
    return finite_gramians


def matrix_exponential(system_matrix: np.ndarray, horizon: float) -> np.ndarray:
    """Compute exp(A T), exploiting symmetry when available."""
    scaled = np.asarray(system_matrix, dtype=float) * float(horizon)
    if np.allclose(scaled, scaled.T, atol=1.0e-10):
        eigenvalues, eigenvectors = np.linalg.eigh(scaled)
        return np.einsum("ik,k,jk->ij", eigenvectors, np.exp(eigenvalues), eigenvectors)
    return expm(scaled)


def assemble_gramian(
    node_gramians: np.ndarray,
    node_indices: np.ndarray,
    node_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble a set Gramian from single-node Gramians."""
    node_indices = np.asarray(node_indices, dtype=int)
    if node_weights is None:
        weights = np.ones(node_indices.shape[0], dtype=float)
    else:
        weights = np.asarray(node_weights, dtype=float)
    weighted = node_gramians[node_indices] * (weights[:, None, None] ** 2)
    gramian = np.sum(weighted, axis=0)
    return 0.5 * (gramian + gramian.T)


def gramian_metrics(
    gramian: np.ndarray,
    epsilon: float,
    bottom_dim: int = 5,
    apply_floor: bool = True,
) -> dict:
    """Compute scalar and geometric summaries from a regularized Gramian."""
    n_nodes = gramian.shape[0]
    symmetric = 0.5 * (gramian + gramian.T)
    regularized = symmetric + float(epsilon) * np.eye(n_nodes) if apply_floor else symmetric
    raw_eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    if apply_floor:
        floor_threshold = float(epsilon)
        floor_tolerance = max(np.finfo(float).eps, abs(floor_threshold) * 1.0e-8)
        at_floor = raw_eigenvalues <= floor_threshold + floor_tolerance
        eigenvalues = np.clip(raw_eigenvalues, a_min=floor_threshold, a_max=None)
    else:
        at_floor = np.zeros_like(raw_eigenvalues, dtype=bool)
        eigenvalues = np.clip(raw_eigenvalues, a_min=0.0, a_max=None)

    eigenvalue_sum = float(np.sum(eigenvalues))
    positive = eigenvalues[eigenvalues > 0.0]
    if eigenvalue_sum > 0.0:
        probabilities = positive / eigenvalue_sum
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        participation_ratio = float((eigenvalue_sum**2) / np.sum(eigenvalues**2))
    else:
        effective_rank = 0.0
        participation_ratio = 0.0

    min_eigenvalue = float(eigenvalues[0])
    max_eigenvalue = float(eigenvalues[-1])
    if min_eigenvalue > 0.0:
        condition_number = float(max_eigenvalue / min_eigenvalue)
        average_energy = float(np.sum(1.0 / eigenvalues))
    else:
        condition_number = np.inf
        average_energy = np.inf
    hard_basis = eigenvectors[:, : min(bottom_dim, n_nodes)]
    return {
        "regularized_gramian": regularized,
        "eigenvalues": eigenvalues,
        "raw_eigenvalues": raw_eigenvalues,
        "eigenvectors": eigenvectors,
        "bottom_basis": hard_basis,
        "log10_average_energy": float(np.log10(average_energy)),
        "average_energy": average_energy,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "condition_number": condition_number,
        "log10_condition_number": float(np.log10(condition_number)),
        "min_eigenvalue": min_eigenvalue,
        "max_eigenvalue": max_eigenvalue,
        "log_min_eigenvalue": float(np.log(min_eigenvalue)) if min_eigenvalue > 0.0 else -np.inf,
        "eigenvalues_at_floor": int(np.count_nonzero(at_floor)),
        "fraction_eigenvalues_at_floor": float(np.mean(at_floor)),
    }


def target_energy(gramian: np.ndarray, target: np.ndarray, epsilon: float) -> float:
    """Compute minimum energy for a target from the regularized Gramian."""
    regularized = 0.5 * (gramian + gramian.T) + epsilon * np.eye(gramian.shape[0])
    solution = np.linalg.solve(regularized, target)
    return float(target.T @ solution)


def subspace_overlap(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Compare two bottom-eigenspace projectors."""
    cross_gram = basis_a.T @ basis_b
    numerator = np.linalg.norm(cross_gram, ord="fro")
    return float(numerator / np.sqrt(basis_a.shape[1]))
