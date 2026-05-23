"""Training code for CINDES binary classifiers."""

from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np
import torch
from torch import nn

from fac_cindes.models import (
    CINDESDensityEstimator,
    MLPClassifier,
    ReferenceDensity,
    Standardizer,
    iter_minibatches,
)


@dataclass(frozen=True)
class TrainingConfig:
    hidden_dim: int = 64
    depth: int = 3
    learning_rate: float = 1e-3
    epochs: int = 200
    patience: int = 25
    val_fraction: float = 0.2
    batch_size: int = 256
    predict_batch_size: int = 8192
    min_delta: float = 1e-4
    device: str = "auto"


@dataclass(frozen=True)
class TrainingInfo:
    best_val_loss: float
    epochs_trained: int
    stopped_early: bool


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _make_classifier_dataset(
    Z: np.ndarray,
    Y: np.ndarray,
    reference: ReferenceDensity,
    z_standardizer: Standardizer,
    y_standardizer: Standardizer,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create balanced real/fake classifier data for CINDES."""

    n = Y.shape[0]
    y_fake = reference.sample(n, rng)

    z_all = np.vstack([Z, Z])
    y_all = np.concatenate([Y, y_fake]).reshape(-1, 1)
    labels = np.concatenate([np.ones(n), np.zeros(n)]).astype(np.float32)

    z_scaled = z_standardizer.transform(z_all)
    y_scaled = y_standardizer.transform(y_all)
    x_all = np.hstack([z_scaled, y_scaled]).astype(np.float32)
    return x_all, labels


def train_cindes(
    Z_train: np.ndarray,
    Y_train: np.ndarray,
    config: TrainingConfig,
    seed: int = 0,
    reference: ReferenceDensity | None = None,
) -> tuple[CINDESDensityEstimator, TrainingInfo]:
    """Fit a CINDES classifier for a selected conditioning variable Z."""

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    Z_train = np.asarray(Z_train, dtype=np.float64)
    Y_train = np.asarray(Y_train, dtype=np.float64)
    if Z_train.ndim != 2:
        raise ValueError("Z_train must be a 2D array.")
    if Y_train.ndim != 1:
        raise ValueError("Y_train must be a 1D array.")
    if Z_train.shape[0] != Y_train.shape[0]:
        raise ValueError("Z_train and Y_train must have the same number of rows.")

    reference = reference or ReferenceDensity.from_y_train(Y_train)
    z_standardizer = Standardizer.fit(Z_train)
    y_standardizer = Standardizer.fit(Y_train.reshape(-1, 1))

    n = Y_train.shape[0]
    val_size = max(1, int(round(config.val_fraction * n)))
    val_size = min(val_size, n - 1)
    perm = rng.permutation(n)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    x_train, labels_train = _make_classifier_dataset(
        Z_train[train_idx], Y_train[train_idx], reference, z_standardizer, y_standardizer, rng
    )
    x_val, labels_val = _make_classifier_dataset(
        Z_train[val_idx], Y_train[val_idx], reference, z_standardizer, y_standardizer, rng
    )

    device = _resolve_device(config.device)
    model = MLPClassifier(
        input_dim=Z_train.shape[1] + 1,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    x_val_t = torch.from_numpy(x_val).to(device)
    labels_val_t = torch.from_numpy(labels_val).to(device)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    epochs_trained = 0
    stopped_early = False

    for epoch in range(1, config.epochs + 1):
        model.train()
        for xb_np, yb_np in iter_minibatches(x_train, labels_train, config.batch_size, rng):
            xb = torch.from_numpy(xb_np).to(device)
            yb = torch.from_numpy(yb_np).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val_t), labels_val_t).detach().cpu().item())

        epochs_trained = epoch
        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            stopped_early = True
            break

    model.load_state_dict(best_state)
    estimator = CINDESDensityEstimator(
        model=model,
        z_standardizer=z_standardizer,
        y_standardizer=y_standardizer,
        reference=reference,
        device=device,
        batch_size=config.predict_batch_size,
    )
    info = TrainingInfo(
        best_val_loss=best_val_loss,
        epochs_trained=epochs_trained,
        stopped_early=stopped_early,
    )
    return estimator, info
