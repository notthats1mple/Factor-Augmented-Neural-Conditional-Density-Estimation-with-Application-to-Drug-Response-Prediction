"""Data generation utilities for factor-augmented CINDES simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class FactorSample:
    """One simulated sample from the factor model."""

    F: np.ndarray
    X: np.ndarray
    Y: np.ndarray


@dataclass(frozen=True)
class TrainTestData:
    """Train/test split generated with the same loading matrix."""

    train: FactorSample
    test: FactorSample
    loading: np.ndarray


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""

    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def true_mixture_parameters(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return pi(F), mu1(F), mu2(F), sigma1, sigma2 for the DGP."""

    if F.shape[1] < 2:
        raise ValueError("The mixture DGP requires at least two latent factors.")

    f1 = F[:, 0]
    f2 = F[:, 1]
    pi = sigmoid(f1)
    mu1 = f1 + 0.5 * f2**2
    mu2 = -f1 + 0.5 * np.sin(f2)
    return pi, mu1, mu2, 0.5, 0.7


def gaussian_pdf(y: np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
    """Evaluate N(mean, sigma^2) densities with broadcasting."""

    z = (y - mean) / sigma
    return np.exp(-0.5 * z**2) / (sigma * np.sqrt(2.0 * np.pi))


def true_conditional_density(F: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """Evaluate the true conditional density f0(y | F) on a y-grid.

    Returns an array with shape (n_observations, n_grid).
    """

    pi, mu1, mu2, sigma1, sigma2 = true_mixture_parameters(F)
    y = y_grid[None, :]
    density1 = gaussian_pdf(y, mu1[:, None], sigma1)
    density2 = gaussian_pdf(y, mu2[:, None], sigma2)
    return pi[:, None] * density1 + (1.0 - pi[:, None]) * density2


def sample_y_given_f(F: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw Y from the two-component Gaussian mixture conditional on F."""

    pi, mu1, mu2, sigma1, sigma2 = true_mixture_parameters(F)
    choose_component_1 = rng.uniform(size=F.shape[0]) < pi
    y = np.empty(F.shape[0], dtype=np.float64)
    y[choose_component_1] = rng.normal(mu1[choose_component_1], sigma1)
    y[~choose_component_1] = rng.normal(mu2[~choose_component_1], sigma2)
    return y


def generate_factor_sample(
    n: int,
    loading: np.ndarray,
    sigma_u: float,
    rng: np.random.Generator,
) -> FactorSample:
    """Generate F, X, and Y for a fixed loading matrix.

    The factor model is X_i = Lambda F_i + U_i, represented in row-major form
    as X = F @ Lambda.T + U.
    """

    p, r = loading.shape
    F = rng.normal(size=(n, r))
    U = rng.normal(scale=sigma_u, size=(n, p))
    X = np.dot(F, loading.T) + U
    Y = sample_y_given_f(F, rng)
    return FactorSample(F=F.astype(np.float64), X=X.astype(np.float64), Y=Y.astype(np.float64))


def generate_train_test_data(
    n_train: int,
    n_test: int,
    p: int,
    r: int = 3,
    sigma_u: float = 1.0,
    seed: int = 0,
) -> TrainTestData:
    """Generate independent train/test samples sharing the same Lambda."""

    rng = np.random.default_rng(seed)
    loading = rng.normal(size=(p, r))
    train = generate_factor_sample(n_train, loading, sigma_u, rng)
    test = generate_factor_sample(n_test, loading, sigma_u, rng)
    return TrainTestData(train=train, test=test, loading=loading.astype(np.float64))


def make_y_grid(y_train: np.ndarray, y_test: np.ndarray, grid_size: int = 401) -> np.ndarray:
    """Create a broad y-grid for density recovery and TV integration."""

    y_all = np.concatenate([y_train, y_test])
    center = float(np.mean(y_train))
    scale = float(np.std(y_train, ddof=1))
    scale = max(scale, 1e-6)
    lower = min(float(np.min(y_all)), center - 6.0 * scale)
    upper = max(float(np.max(y_all)), center + 6.0 * scale)
    return np.linspace(lower, upper, grid_size, dtype=np.float64)
