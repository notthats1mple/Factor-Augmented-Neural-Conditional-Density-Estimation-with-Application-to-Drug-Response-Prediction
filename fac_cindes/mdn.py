"""Mixture-density-network baseline for conditional response densities."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Iterable

import numpy as np
from scipy.special import ndtr
import torch
from torch import nn

from fac_cindes.models import Standardizer, iter_minibatches


@dataclass(frozen=True)
class MDNConfig:
    hidden_dim: int = 64
    depth: int = 3
    learning_rate: float = 1e-3
    epochs: int = 200
    patience: int = 25
    val_fraction: float = 0.2
    batch_size: int = 256
    min_delta: float = 1e-4
    device: str = "auto"
    components: tuple[int, ...] = (2, 3, 5)


@dataclass(frozen=True)
class MDNInfo:
    best_val_nll: float
    epochs_trained: int
    stopped_early: bool
    n_components: int


class MDNNetwork(nn.Module):
    """Gaussian mixture density network with scalar response output."""

    def __init__(self, input_dim: int, n_components: int, hidden_dim: int = 64, depth: int = 3) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if n_components < 1:
            raise ValueError("n_components must be positive")

        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dim, 3 * n_components)
        self.n_components = n_components

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.head(self.backbone(x))
        logits, means, raw_scales = torch.chunk(out, 3, dim=-1)
        scales = torch.nn.functional.softplus(raw_scales) + 1e-4
        return logits, means, scales


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _mixture_log_prob_standardized(
    y: torch.Tensor,
    logits: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    y = y.reshape(-1, 1)
    log_weights = torch.log_softmax(logits, dim=-1)
    z = (y - means) / scales
    log_component = -0.5 * np.log(2.0 * np.pi) - torch.log(scales) - 0.5 * z.pow(2)
    return torch.logsumexp(log_weights + log_component, dim=-1)


class MDNDensityEstimator:
    """Fitted MDN plus density and CDF-grid prediction helpers."""

    def __init__(
        self,
        model: MDNNetwork,
        z_standardizer: Standardizer,
        y_standardizer: Standardizer,
        device: torch.device,
        batch_size: int = 8192,
    ) -> None:
        self.model = model
        self.z_standardizer = z_standardizer
        self.y_standardizer = y_standardizer
        self.device = device
        self.batch_size = batch_size

    @property
    def y_scale(self) -> float:
        return float(np.asarray(self.y_standardizer.std).reshape(-1)[0])

    @property
    def y_mean(self) -> float:
        return float(np.asarray(self.y_standardizer.mean).reshape(-1)[0])

    def _predict_params(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z_scaled = self.z_standardizer.transform(np.asarray(z, dtype=np.float64)).astype(np.float32)
        weights_out: list[np.ndarray] = []
        means_out: list[np.ndarray] = []
        scales_out: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, z_scaled.shape[0], self.batch_size):
                xb = torch.from_numpy(z_scaled[start : start + self.batch_size]).to(self.device)
                logits, means, scales = self.model(xb)
                weights_out.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
                means_out.append(means.detach().cpu().numpy())
                scales_out.append(scales.detach().cpu().numpy())
        return (
            np.vstack(weights_out).astype(np.float64),
            np.vstack(means_out).astype(np.float64),
            np.vstack(scales_out).astype(np.float64),
        )

    def logpdf(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        z_scaled = self.z_standardizer.transform(np.asarray(z, dtype=np.float64)).astype(np.float32)
        y_scaled = self.y_standardizer.transform(np.asarray(y, dtype=np.float64).reshape(-1, 1)).astype(np.float32)

        outputs: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, z_scaled.shape[0], self.batch_size):
                xb = torch.from_numpy(z_scaled[start : start + self.batch_size]).to(self.device)
                yb = torch.from_numpy(y_scaled[start : start + self.batch_size, 0]).to(self.device)
                logits, means, scales = self.model(xb)
                logp = _mixture_log_prob_standardized(yb, logits, means, scales) - np.log(self.y_scale)
                outputs.append(logp.detach().cpu().numpy())
        return np.concatenate(outputs).astype(np.float64)

    def cdf(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        weights, means, scales = self._predict_params(z)
        y_std = self.y_standardizer.transform(np.asarray(y, dtype=np.float64).reshape(-1, 1))
        component_cdf = ndtr((y_std - means) / scales)
        return np.clip(np.sum(weights * component_cdf, axis=1), 0.0, 1.0)

    def interval(self, z: np.ndarray, alpha: float = 0.10, max_iter: int = 80) -> tuple[np.ndarray, np.ndarray]:
        weights, means, scales = self._predict_params(z)
        lower_prob = alpha / 2.0
        upper_prob = 1.0 - alpha / 2.0

        means_real = self.y_mean + self.y_scale * means
        scales_real = self.y_scale * scales
        lo = np.min(means_real - 12.0 * scales_real, axis=1)
        hi = np.max(means_real + 12.0 * scales_real, axis=1)

        def cdf_from_params(y_real: np.ndarray) -> np.ndarray:
            y_std = ((y_real - self.y_mean) / self.y_scale)[:, None]
            return np.clip(np.sum(weights * ndtr((y_std - means) / scales), axis=1), 0.0, 1.0)

        def quantile(target: float) -> np.ndarray:
            low = lo.copy()
            high = hi.copy()
            for _ in range(max_iter):
                mid = 0.5 * (low + high)
                cdf_mid = cdf_from_params(mid)
                low = np.where(cdf_mid < target, mid, low)
                high = np.where(cdf_mid >= target, mid, high)
            return 0.5 * (low + high)

        return quantile(lower_prob), quantile(upper_prob)

    def density_on_grid(self, z: np.ndarray, y_grid: np.ndarray, z_batch_size: int = 128) -> np.ndarray:
        z_scaled = self.z_standardizer.transform(np.asarray(z, dtype=np.float64)).astype(np.float32)
        y_grid_scaled = self.y_standardizer.transform(np.asarray(y_grid, dtype=np.float64).reshape(-1, 1))[:, 0]
        y_grid_t = torch.from_numpy(y_grid_scaled.astype(np.float32)).to(self.device)

        n = z_scaled.shape[0]
        m = y_grid_scaled.size
        density = np.empty((n, m), dtype=np.float64)
        self.model.eval()
        with torch.no_grad():
            for start in range(0, n, z_batch_size):
                end = min(start + z_batch_size, n)
                xb = torch.from_numpy(z_scaled[start:end]).to(self.device)
                logits, means, scales = self.model(xb)
                log_weights = torch.log_softmax(logits, dim=-1)[:, None, :]
                means = means[:, None, :]
                scales = scales[:, None, :]
                yb = y_grid_t[None, :, None]
                z_score = (yb - means) / scales
                log_component = -0.5 * np.log(2.0 * np.pi) - torch.log(scales) - 0.5 * z_score.pow(2)
                log_density = torch.logsumexp(log_weights + log_component, dim=-1) - np.log(self.y_scale)
                density[start:end] = torch.exp(log_density).detach().cpu().numpy()
        return density


def _fit_one_mdn(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int,
    config: MDNConfig,
    seed: int,
) -> tuple[MDNDensityEstimator, MDNInfo]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    z_standardizer = Standardizer.fit(Z_train)
    y_standardizer = Standardizer.fit(y_train.reshape(-1, 1))
    Z_scaled = z_standardizer.transform(Z_train).astype(np.float32)
    y_scaled = y_standardizer.transform(y_train.reshape(-1, 1))[:, 0].astype(np.float32)

    n = y_train.shape[0]
    val_size = max(1, int(round(config.val_fraction * n)))
    val_size = min(val_size, n - 1)
    perm = rng.permutation(n)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    x_train = Z_scaled[train_idx]
    y_train_scaled = y_scaled[train_idx]
    x_val = torch.from_numpy(Z_scaled[val_idx]).to(_resolve_device(config.device))
    y_val = torch.from_numpy(y_scaled[val_idx]).to(_resolve_device(config.device))

    device = _resolve_device(config.device)
    model = MDNNetwork(
        input_dim=Z_train.shape[1],
        n_components=n_components,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_nll = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    epochs_trained = 0
    stopped_early = False

    for epoch in range(1, config.epochs + 1):
        model.train()
        for xb_np, yb_np in iter_minibatches(x_train, y_train_scaled, config.batch_size, rng):
            xb = torch.from_numpy(xb_np).to(device)
            yb = torch.from_numpy(yb_np).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, means, scales = model(xb)
            loss = -torch.mean(_mixture_log_prob_standardized(yb, logits, means, scales))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, means, scales = model(x_val)
            val_nll = float(-torch.mean(_mixture_log_prob_standardized(y_val, logits, means, scales)).cpu().item())

        epochs_trained = epoch
        if val_nll < best_val_nll - config.min_delta:
            best_val_nll = val_nll
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            stopped_early = True
            break

    model.load_state_dict(best_state)
    estimator = MDNDensityEstimator(
        model=model,
        z_standardizer=z_standardizer,
        y_standardizer=y_standardizer,
        device=device,
    )
    info = MDNInfo(
        best_val_nll=best_val_nll + np.log(estimator.y_scale),
        epochs_trained=epochs_trained,
        stopped_early=stopped_early,
        n_components=n_components,
    )
    return estimator, info


def train_mdn_select_k(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    config: MDNConfig,
    seed: int = 0,
) -> tuple[MDNDensityEstimator, MDNInfo]:
    """Fit MDNs over candidate component counts and select by validation NLL."""

    Z_train = np.asarray(Z_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if Z_train.ndim != 2:
        raise ValueError("Z_train must be a 2D array.")
    if y_train.ndim != 1:
        raise ValueError("y_train must be a 1D array.")
    if Z_train.shape[0] != y_train.shape[0]:
        raise ValueError("Z_train and y_train must have the same number of rows.")

    best: tuple[MDNDensityEstimator, MDNInfo] | None = None
    for offset, n_components in enumerate(config.components):
        estimator, info = _fit_one_mdn(Z_train, y_train, int(n_components), config, seed + 1000 * offset)
        if best is None or info.best_val_nll < best[1].best_val_nll:
            best = (estimator, info)
    assert best is not None
    return best
