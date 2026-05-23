"""Neural-network model and fitted CINDES estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn


class MLPClassifier(nn.Module):
    """MLP binary classifier for real-vs-reference CINDES training."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, depth: int = 3) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class Standardizer:
    """Column-wise standardization using training-set moments."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-6) -> "Standardizer":
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0, ddof=0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean.astype(np.float64), std=std.astype(np.float64))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


@dataclass(frozen=True)
class ReferenceDensity:
    """Gaussian reference density q used to generate fake responses."""

    mean: float
    std: float

    @classmethod
    def from_y_train(cls, y_train: np.ndarray) -> "ReferenceDensity":
        mean = float(np.mean(y_train))
        std = float(np.std(y_train, ddof=1))
        return cls(mean=mean, std=max(1.5 * std, 1e-6))

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(loc=self.mean, scale=self.std, size=size).astype(np.float64)

    def logpdf(self, y: np.ndarray) -> np.ndarray:
        z = (y - self.mean) / self.std
        return -0.5 * np.log(2.0 * np.pi) - np.log(self.std) - 0.5 * z**2

    def pdf(self, y: np.ndarray) -> np.ndarray:
        return np.exp(self.logpdf(y))


@dataclass
class CINDESDensityEstimator:
    """Fitted classifier plus density-recovery utilities.

    With balanced real/fake classes and fake samples drawn from q(y), the
    classifier logit estimates log f(y | z) / q(y). The recovered density is
    q(y) * exp(logit), normalized over the supplied y-grid.
    """

    model: MLPClassifier
    z_standardizer: Standardizer
    y_standardizer: Standardizer
    reference: ReferenceDensity
    device: torch.device
    batch_size: int = 8192

    def _standardized_pairs(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        z_scaled = self.z_standardizer.transform(np.asarray(z, dtype=np.float64))
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        y_scaled = self.y_standardizer.transform(y_arr)
        return np.hstack([z_scaled, y_scaled]).astype(np.float32)

    def predict_logits(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Predict logits for paired rows (z_i, y_i)."""

        x = self._standardized_pairs(z, y)
        outputs: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, x.shape[0], self.batch_size):
                xb = torch.from_numpy(x[start : start + self.batch_size]).to(self.device)
                outputs.append(self.model(xb).detach().cpu().numpy())
        return np.concatenate(outputs).astype(np.float64)

    def predict_logits_on_grid(
        self,
        z: np.ndarray,
        y_grid: np.ndarray,
        z_batch_size: int = 64,
    ) -> np.ndarray:
        """Predict logits for every z row crossed with all grid values."""

        z = np.asarray(z, dtype=np.float64)
        y_grid = np.asarray(y_grid, dtype=np.float64)
        n = z.shape[0]
        m = y_grid.shape[0]
        logits = np.empty((n, m), dtype=np.float64)

        for start in range(0, n, z_batch_size):
            end = min(start + z_batch_size, n)
            z_block = z[start:end]
            z_rep = np.repeat(z_block, m, axis=0)
            y_rep = np.tile(y_grid, end - start)
            block_logits = self.predict_logits(z_rep, y_rep).reshape(end - start, m)
            logits[start:end] = block_logits

        return logits


def iter_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield shuffled mini-batches."""

    indices = rng.permutation(x.shape[0])
    for start in range(0, x.shape[0], batch_size):
        idx = indices[start : start + batch_size]
        yield x[idx], y[idx]
