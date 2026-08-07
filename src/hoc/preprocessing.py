"""Adjacency preprocessing, stabilization, and perturbations."""

from __future__ import annotations

import numpy as np


def preprocess_structural_adjacency(adjacency_raw: np.ndarray) -> np.ndarray:
    """Symmetrize, zero the diagonal, log-transform, and max-normalize."""
    adjacency_raw = np.asarray(adjacency_raw, dtype=float)
    adjacency_sym = 0.5 * (adjacency_raw + adjacency_raw.T)
    np.fill_diagonal(adjacency_sym, 0.0)
    adjacency_log = np.log1p(np.clip(adjacency_sym, a_min=0.0, a_max=None))
    max_value = float(adjacency_log.max())
    if max_value <= 0.0:
        return adjacency_log
    return adjacency_log / max_value


def stabilize_adjacency(adjacency: np.ndarray, c: float) -> tuple[np.ndarray, float]:
    """Shift the spectrum into the left half-plane."""
    spectral_radius = float(np.max(np.real(np.linalg.eigvals(adjacency))))
    stabilized = adjacency - np.eye(adjacency.shape[0]) * (spectral_radius + c)
    return stabilized, spectral_radius


def multiplicative_noise_perturbation(
    adjacency: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply multiplicative Gaussian noise to nonzero edges."""
    mask = adjacency > 0
    factors = rng.normal(loc=1.0, scale=sigma, size=adjacency.shape)
    perturbed = adjacency.copy()
    perturbed[mask] *= factors[mask]
    perturbed = np.clip(perturbed, a_min=0.0, a_max=None)
    perturbed = 0.5 * (perturbed + perturbed.T)
    np.fill_diagonal(perturbed, 0.0)
    return perturbed


def dropout_perturbation(
    adjacency: np.ndarray,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly remove existing edges with probability p."""
    upper = np.triu(adjacency > 0, k=1)
    drop_mask = rng.random(adjacency.shape) < p
    drop_upper = np.logical_and(upper, drop_mask)
    keep = adjacency.copy()
    keep[drop_upper] = 0.0
    keep[(drop_upper.T)] = 0.0
    return keep
