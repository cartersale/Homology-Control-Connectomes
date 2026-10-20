"""Gramians, geometry metrics and target energy against closed forms."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad_vec

from hoc.control import (
    assemble_gramian,
    gramian_metrics,
    matrix_exponential,
    single_node_gramians,
    target_energy,
)
from hoc.preprocessing import stabilize_adjacency


def random_stable_system(n: int, seed: int = 0, c: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    adjacency = rng.random((n, n))
    adjacency = 0.5 * (adjacency + adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    system, _ = stabilize_adjacency(adjacency / adjacency.max(), c)
    return system


def test_diagonal_system_closed_form() -> None:
    rates = np.array([0.5, 1.0, 2.0, 4.0])
    system = -np.diag(rates)
    gramians = single_node_gramians(system)
    for j, a in enumerate(rates):
        expected = np.zeros((4, 4))
        expected[j, j] = 1.0 / (2.0 * a)
        np.testing.assert_allclose(gramians[j], expected, atol=1e-14)

    horizon = 0.7
    finite = single_node_gramians(system, horizon=horizon)
    for j, a in enumerate(rates):
        assert finite[j][j, j] == pytest.approx((1.0 - np.exp(-2.0 * a * horizon)) / (2.0 * a), rel=1e-12)


def test_lyapunov_residual_is_zero() -> None:
    system = random_stable_system(12)
    gramians = single_node_gramians(system)
    for j in range(12):
        b = np.zeros((12, 1))
        b[j] = 1.0
        residual = system @ gramians[j] + gramians[j] @ system.T + b @ b.T
        assert np.abs(residual).max() < 1e-12
        np.testing.assert_allclose(gramians[j], gramians[j].T, atol=0)
        assert np.linalg.eigvalsh(gramians[j]).min() > -1e-12


def test_finite_horizon_matches_integral_and_converges() -> None:
    system = random_stable_system(8, seed=1)
    horizon = 1.5
    finite = single_node_gramians(system, horizon=horizon)
    j = 3

    def integrand(t: float) -> np.ndarray:
        column = matrix_exponential(system, t)[:, j]
        return np.outer(column, column)

    integral, _ = quad_vec(integrand, 0.0, horizon, epsabs=1e-13, epsrel=1e-11)
    np.testing.assert_allclose(finite[j], integral, rtol=1e-8, atol=1e-12)

    infinite = single_node_gramians(system)
    long = single_node_gramians(system, horizon=500.0)
    np.testing.assert_allclose(long, infinite, atol=1e-12)


def test_matrix_exponential_symmetric_path_matches_expm() -> None:
    from scipy.linalg import expm

    system = random_stable_system(10, seed=2)
    np.testing.assert_allclose(matrix_exponential(system, 2.0), expm(2.0 * system), rtol=1e-10, atol=1e-13)


def test_set_gramian_is_sum_with_weights() -> None:
    gramians = single_node_gramians(random_stable_system(6, seed=3))
    weighted = assemble_gramian(gramians, np.array([1, 4]), node_weights=np.array([2.0, 0.5]))
    np.testing.assert_allclose(weighted, 4.0 * gramians[1] + 0.25 * gramians[4], atol=1e-15)


def test_metrics_on_known_spectrum() -> None:
    gramian = np.diag([4.0, 2.0, 1.0, 1.0])
    m = gramian_metrics(gramian, epsilon=0.0, apply_floor=False)
    # p = (1/2, 1/4, 1/8, 1/8): entropy = 1.75 ln 2
    assert m["effective_rank"] == pytest.approx(2.0**1.75, rel=1e-12)
    assert m["participation_ratio"] == pytest.approx(64.0 / 22.0, rel=1e-12)
    assert m["condition_number"] == pytest.approx(4.0)
    assert m["average_energy"] == pytest.approx(0.25 + 0.5 + 1.0 + 1.0)
    assert m["eigenvalues_at_floor"] == 0


def test_floor_sets_condition_number() -> None:
    eps = 1e-3
    gramian = np.diag([1.0, 0.0, 0.0])
    m = gramian_metrics(gramian, epsilon=eps)
    assert m["min_eigenvalue"] == pytest.approx(eps)
    assert m["condition_number"] == pytest.approx((1.0 + eps) / eps)
    assert m["eigenvalues_at_floor"] == 2
    assert m["fraction_eigenvalues_at_floor"] == pytest.approx(2 / 3)
    assert m["log10_average_energy"] == pytest.approx(np.log10(2 / eps + 1 / (1 + eps)))


def test_target_energy_closed_form() -> None:
    gramian = np.diag([2.0, 0.5, 1.0])
    target = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    eps = 1e-6
    expected = 0.5 / (2.0 + eps) + 0.5 / (0.5 + eps)
    assert target_energy(gramian, target, eps) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("c", [0.01, 0.1, 1.0])
def test_stabilization_spectral_abscissa(c: float) -> None:
    system = random_stable_system(15, seed=4, c=c)
    assert np.max(np.linalg.eigvalsh(system)) == pytest.approx(-c, abs=1e-12)
