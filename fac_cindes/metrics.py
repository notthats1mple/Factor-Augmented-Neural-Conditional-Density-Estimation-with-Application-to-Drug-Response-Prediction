"""Density recovery and evaluation metrics."""

from __future__ import annotations

import numpy as np

from fac_cindes.data import true_conditional_density
from fac_cindes.models import CINDESDensityEstimator


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """Small NumPy logsumexp implementation to avoid an extra SciPy dependency."""

    max_a = np.max(a, axis=axis, keepdims=True)
    stable = np.exp(a - max_a)
    summed = np.sum(stable, axis=axis, keepdims=True)
    out = max_a + np.log(summed)
    return np.squeeze(out, axis=axis)


def _log_trapezoid_weights(y_grid: np.ndarray) -> np.ndarray:
    """Log integration weights for the trapezoidal rule on a grid."""

    y_grid = np.asarray(y_grid, dtype=np.float64)
    if y_grid.ndim != 1 or y_grid.shape[0] < 2:
        raise ValueError("y_grid must be a one-dimensional grid with at least two points.")

    deltas = np.diff(y_grid)
    if not np.all(deltas > 0):
        raise ValueError("y_grid must be strictly increasing.")

    weights = np.empty_like(y_grid)
    weights[0] = 0.5 * deltas[0]
    weights[-1] = 0.5 * deltas[-1]
    if y_grid.shape[0] > 2:
        weights[1:-1] = 0.5 * (deltas[:-1] + deltas[1:])
    return np.log(weights)


def recovered_log_density_grid(
    estimator: CINDESDensityEstimator,
    Z: np.ndarray,
    y_grid: np.ndarray,
    z_batch_size: int = 64,
) -> np.ndarray:
    """Recover normalized log densities on a grid."""

    logits = estimator.predict_logits_on_grid(Z, y_grid, z_batch_size=z_batch_size)
    log_q = estimator.reference.logpdf(y_grid)[None, :]
    log_unnormalized = log_q + logits
    log_weights = _log_trapezoid_weights(y_grid)[None, :]
    log_norm = _logsumexp(log_unnormalized + log_weights, axis=1)
    return log_unnormalized - log_norm[:, None]


def recovered_density_grid(
    estimator: CINDESDensityEstimator,
    Z: np.ndarray,
    y_grid: np.ndarray,
    z_batch_size: int = 64,
) -> np.ndarray:
    """Recover normalized densities on a grid."""

    return np.exp(recovered_log_density_grid(estimator, Z, y_grid, z_batch_size=z_batch_size))


def negative_log_likelihood(
    estimator: CINDESDensityEstimator,
    Z_test: np.ndarray,
    Y_test: np.ndarray,
    y_grid: np.ndarray,
    z_batch_size: int = 64,
) -> float:
    """Average test negative log-likelihood under the recovered density."""

    Z_test = np.asarray(Z_test, dtype=np.float64)
    Y_test = np.asarray(Y_test, dtype=np.float64)

    logits_y = estimator.predict_logits(Z_test, Y_test)
    log_q_y = estimator.reference.logpdf(Y_test)

    logits_grid = estimator.predict_logits_on_grid(Z_test, y_grid, z_batch_size=z_batch_size)
    log_q_grid = estimator.reference.logpdf(y_grid)[None, :]
    log_weights = _log_trapezoid_weights(y_grid)[None, :]
    log_norm = _logsumexp(log_q_grid + logits_grid + log_weights, axis=1)

    log_density = log_q_y + logits_y - log_norm
    return float(-np.mean(log_density))


def integrated_total_variation(
    estimator: CINDESDensityEstimator,
    Z_test: np.ndarray,
    F_test: np.ndarray,
    y_grid: np.ndarray,
    z_batch_size: int = 64,
) -> float:
    """Average integrated TV distance int |p_hat - p_true| dy over test points."""

    estimated = recovered_density_grid(estimator, Z_test, y_grid, z_batch_size=z_batch_size)
    truth = true_conditional_density(F_test, y_grid)
    tv_by_obs = np.trapz(np.abs(estimated - truth), y_grid, axis=1)
    return float(np.mean(tv_by_obs))


def evaluate_nll_and_tv(
    estimator: CINDESDensityEstimator,
    Z_test: np.ndarray,
    Y_test: np.ndarray,
    F_test: np.ndarray,
    y_grid: np.ndarray,
    z_batch_size: int = 64,
) -> tuple[float, float]:
    """Compute NLL and TV = int |p_hat - p_true| dy using one grid pass."""

    Z_test = np.asarray(Z_test, dtype=np.float64)
    Y_test = np.asarray(Y_test, dtype=np.float64)

    logits_grid = estimator.predict_logits_on_grid(Z_test, y_grid, z_batch_size=z_batch_size)
    log_q_grid = estimator.reference.logpdf(y_grid)[None, :]
    log_weights = _log_trapezoid_weights(y_grid)[None, :]
    log_unnormalized_grid = log_q_grid + logits_grid
    log_norm = _logsumexp(log_unnormalized_grid + log_weights, axis=1)

    logits_y = estimator.predict_logits(Z_test, Y_test)
    log_q_y = estimator.reference.logpdf(Y_test)
    log_density_y = log_q_y + logits_y - log_norm
    nll = float(-np.mean(log_density_y))

    estimated = np.exp(log_unnormalized_grid - log_norm[:, None])
    truth = true_conditional_density(F_test, y_grid)
    tv_by_obs = np.trapz(np.abs(estimated - truth), y_grid, axis=1)
    tv = float(np.mean(tv_by_obs))
    return nll, tv


def aligned_factor_mse(F_true: np.ndarray, F_hat: np.ndarray) -> float:
    """Least-squares aligned factor MSE: mean((F_hat A - F_true)^2)."""

    alignment = fit_factor_alignment(F_true, F_hat)
    return aligned_factor_mse_with_alignment(F_true, F_hat, alignment)


def fit_factor_alignment(F_true_train: np.ndarray, F_hat_train: np.ndarray) -> np.ndarray:
    """Fit A = argmin_A ||F_hat_train A - F_true_train||_F."""

    F_true_train = np.asarray(F_true_train, dtype=np.float64)
    F_hat_train = np.asarray(F_hat_train, dtype=np.float64)
    if F_true_train.shape != F_hat_train.shape:
        raise ValueError("F_true_train and F_hat_train must have the same shape.")

    alignment, *_ = np.linalg.lstsq(F_hat_train, F_true_train, rcond=None)
    return alignment.astype(np.float64)


def aligned_factor_mse_with_alignment(
    F_true: np.ndarray,
    F_hat: np.ndarray,
    alignment: np.ndarray,
) -> float:
    """Aligned factor MSE using a pre-fitted alignment matrix."""

    F_true = np.asarray(F_true, dtype=np.float64)
    F_hat = np.asarray(F_hat, dtype=np.float64)
    alignment = np.asarray(alignment, dtype=np.float64)
    if F_true.shape != F_hat.shape:
        raise ValueError("F_true and F_hat must have the same shape.")
    if alignment.shape != (F_hat.shape[1], F_true.shape[1]):
        raise ValueError("alignment has incompatible shape.")

    F_aligned = np.dot(F_hat, alignment)
    return float(np.mean((F_aligned - F_true) ** 2))


def density_grid_integrals(density_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """Check numerical integrals of densities represented on a y-grid."""

    return np.trapz(density_grid, y_grid, axis=1)
