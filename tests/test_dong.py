from __future__ import annotations

import numpy as np

from hoc.control import single_node_gramians
from hoc.dong import (
    attenuate_nodes,
    controlled_covariance_proxy,
    fit_linear_stochastic_dynamics,
    fitted_a_node_scores,
    gaussian_kl,
    single_node_tracking_energy,
    stationary_covariance,
    target_aware_greedy_set,
    transform_coupling_for_topology,
)


def synthetic_timeseries(n_timepoints: int = 80, n_nodes: int = 6) -> np.ndarray:
    rng = np.random.default_rng(123)
    dynamics = 0.65 * np.eye(n_nodes)
    dynamics += rng.normal(scale=0.04, size=(n_nodes, n_nodes))
    values = np.zeros((n_timepoints, n_nodes), dtype=float)
    for index in range(1, n_timepoints):
        values[index] = dynamics @ values[index - 1] + rng.normal(scale=0.5, size=n_nodes)
    return values


def test_fit_linear_stochastic_dynamics_returns_stable_model() -> None:
    fitted = fit_linear_stochastic_dynamics(synthetic_timeseries(), ridge=1.0e-3, stability_margin=1.0e-2)
    eigenvalues = np.linalg.eigvals(fitted.system_matrix)
    assert np.max(np.real(eigenvalues)) < 0.0
    assert fitted.residual_covariance.shape == fitted.system_matrix.shape
    assert np.all(np.linalg.eigvalsh(fitted.residual_covariance) > 0.0)


def test_fitted_a_topology_transforms_are_nonnegative_symmetric_and_zero_diagonal() -> None:
    matrix = np.array(
        [
            [-1.0, 0.2, -0.4],
            [0.8, -1.1, 0.0],
            [-0.3, 0.5, -0.9],
        ]
    )
    for transform in ("A_sym", "A_pos", "A_abs"):
        weights = transform_coupling_for_topology(matrix, transform)
        assert np.all(weights >= 0.0)
        assert np.allclose(weights, weights.T)
        assert np.allclose(np.diag(weights), 0.0)


def test_fitted_a_node_scores_include_residualized_cycle_and_average_controllability() -> None:
    fitted = fit_linear_stochastic_dynamics(synthetic_timeseries(n_nodes=5))
    gramians = single_node_gramians(fitted.system_matrix)
    scores = fitted_a_node_scores(fitted.system_matrix, gramians, "A_sym")
    assert {"cycle_participation", "cycle_residual", "average_controllability"}.issubset(scores.columns)
    assert scores.shape[0] == 5
    assert np.all(np.isfinite(scores["average_controllability"]))


def test_tracking_metrics_are_finite_and_greedy_respects_k() -> None:
    fitted = fit_linear_stochastic_dynamics(synthetic_timeseries(n_nodes=5))
    gramians = single_node_gramians(fitted.system_matrix)
    patient_cov = stationary_covariance(fitted.system_matrix, fitted.residual_covariance)
    target_cov = patient_cov + 0.1 * np.eye(5)
    energies = single_node_tracking_energy(patient_cov, target_cov, gramians, epsilon=1.0e-5)
    assert np.all(np.isfinite(energies))
    nodes, rows = target_aware_greedy_set(
        seed_node=0,
        k=3,
        candidate_nodes=np.arange(5),
        patient_covariance=patient_cov,
        target_covariance=target_cov,
        node_gramians=gramians,
        epsilon=1.0e-5,
    )
    assert len(nodes) == 3
    assert len(rows) == 3
    assert np.all(np.isfinite([row["controlled_kl"] for row in rows]))


def test_covariance_kl_and_control_proxy_are_finite() -> None:
    patient = np.eye(4)
    target = 1.2 * np.eye(4)
    gramian = 0.5 * np.eye(4)
    controlled = controlled_covariance_proxy(patient, target, gramian, epsilon=1.0e-5)
    assert np.isfinite(gaussian_kl(target, patient))
    assert np.isfinite(gaussian_kl(target, controlled))
    assert np.all(np.linalg.eigvalsh(controlled) > 0.0)


def test_attenuate_nodes_preserves_diagonal_and_reduces_coupling() -> None:
    matrix = np.array(
        [
            [-2.0, 1.0, 0.5],
            [0.3, -1.5, 0.2],
            [0.4, 0.6, -1.0],
        ]
    )
    attenuated = attenuate_nodes(matrix, np.array([1]), factor=0.25)
    assert np.allclose(np.diag(attenuated), np.diag(matrix))
    assert np.isclose(attenuated[0, 1], 0.25)
    assert np.isclose(attenuated[1, 0], 0.075)
