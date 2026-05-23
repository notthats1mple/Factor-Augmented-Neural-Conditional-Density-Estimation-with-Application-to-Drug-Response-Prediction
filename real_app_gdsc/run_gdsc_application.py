#!/usr/bin/env python3
"""Run real-data GDSC conditional density experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import time
from typing import Iterable
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from fac_cindes.mdn import MDNConfig, train_mdn_select_k
from fac_cindes.models import ReferenceDensity
from fac_cindes.train import TrainingConfig, train_cindes


APP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = APP_DIR / "data" / "processed"
RESULTS_DIR = APP_DIR / "results"
METHODS = [
    "plugin_pca_cindes",
    "raw_x_cindes",
    "gaussian_elasticnet_baseline",
    "pca_gaussian_baseline",
    "pca_mdn",
]


@dataclass(frozen=True)
class SplitData:
    train_idx: np.ndarray
    test_idx: np.ndarray
    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class TrainingOnlyPreprocess:
    selected_genes: np.ndarray
    X_train_std: np.ndarray
    X_test_std: np.ndarray
    y_grid: np.ndarray
    reference: ReferenceDensity


def append_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_key(analysis_id: str, seed: int, method: str, num_factors: int, top_genes: int, grid_size: int) -> tuple[str, int, str, int, int, int]:
    return (str(analysis_id), int(seed), str(method), int(num_factors), int(top_genes), int(grid_size))


def existing_run_keys(path: Path) -> tuple[set[tuple[str, int, str, int, int, int]], int]:
    if not path.exists():
        return set(), 0
    df = pd.read_csv(path)
    if df.empty:
        return set(), 0
    if "analysis_id" not in df.columns:
        df["analysis_id"] = df["drug_name"]
    if "grid_size" not in df.columns:
        return set(), int(df.duplicated(subset=["drug_name", "seed", "method", "num_factors", "top_genes"]).sum())
    keys = [
        run_key(row.analysis_id, row.seed, row.method, row.num_factors, row.top_genes, row.grid_size)
        for row in df.itertuples(index=False)
    ]
    return set(keys), len(keys) - len(set(keys))


def write_run_status_report(
    path: Path,
    output_path: Path,
    files: list[Path],
    methods: list[str],
    seeds: list[int],
    args: argparse.Namespace,
    duplicate_rows_before: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = []
    for file_path in files:
        meta = npz_metadata(file_path)
        for seed in seeds:
            for method in methods:
                expected.append(
                    run_key(str(meta["analysis_id"]), seed, method, args.num_factors, args.top_genes, args.grid_size)
                )
    expected_set = set(expected)
    completed_set, duplicate_rows_after = existing_run_keys(output_path)
    completed = expected_set & completed_set
    missing = expected_set - completed_set
    text = [
        "# GDSC Run Status",
        "",
        f"Output CSV: `{output_path}`",
        f"Eligible/prepared drug units: {len(files)}",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"Methods: {', '.join(methods)}",
        f"Expected rows: {len(expected_set)}",
        f"Completed expected rows: {len(completed)}",
        f"Missing rows: {len(missing)}",
        f"Duplicate rows before this run: {duplicate_rows_before}",
        f"Duplicate rows after this run: {duplicate_rows_after}",
        "",
    ]
    if missing:
        text.append("## Missing Keys")
        text.append("")
        for key in sorted(missing)[:200]:
            text.append(f"- {key}")
        if len(missing) > 200:
            text.append(f"- ... {len(missing) - 200} additional missing keys omitted")
    path.write_text("\n".join(text) + "\n")


def slugify(value: str) -> str:
    import re

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "drug"


def parse_comma_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def read_analysis_ids_file(path: str | None) -> list[str]:
    if path is None:
        return []
    id_path = Path(path)
    if not id_path.exists():
        raise FileNotFoundError(f"Analysis-id file not found: {id_path}")
    if id_path.suffix.lower() == ".csv":
        table = pd.read_csv(id_path)
        if table.empty:
            return []
        column = "analysis_id" if "analysis_id" in table.columns else table.columns[0]
        return dedupe_preserve_order(table[column].dropna().astype(str))
    values = []
    for line in id_path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        values.append(text)
    return dedupe_preserve_order(values)


def parse_methods(value: str | None) -> list[str]:
    if value is None or value.strip().lower() == "all":
        return METHODS.copy()
    methods = parse_comma_list(value)
    invalid = [method for method in methods if method not in METHODS]
    if invalid:
        raise ValueError(f"Unknown method(s): {', '.join(invalid)}. Valid methods: {', '.join(METHODS)}")
    return methods


def parse_int_tuple(value: str) -> tuple[int, ...]:
    out = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not out:
        raise ValueError("Expected at least one integer value.")
    return out


def logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    max_a = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(max_a + np.log(np.sum(np.exp(a - max_a), axis=axis, keepdims=True)), axis=axis)


def trapezoid_weights(y_grid: np.ndarray) -> np.ndarray:
    deltas = np.diff(y_grid)
    weights = np.empty_like(y_grid)
    weights[0] = 0.5 * deltas[0]
    weights[-1] = 0.5 * deltas[-1]
    if y_grid.size > 2:
        weights[1:-1] = 0.5 * (deltas[:-1] + deltas[1:])
    return weights


def make_y_grid(y_train: np.ndarray, grid_size: int) -> np.ndarray:
    center = float(np.mean(y_train))
    scale = max(float(np.std(y_train, ddof=1)), 1e-6)
    lower = center - 5.0 * scale
    upper = center + 5.0 * scale
    return np.linspace(lower, upper, grid_size, dtype=np.float64)


def density_to_cdf(density: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    increments = 0.5 * (density[:, 1:] + density[:, :-1]) * np.diff(y_grid)[None, :]
    cdf = np.hstack([np.zeros((density.shape[0], 1)), np.cumsum(increments, axis=1)])
    cdf /= np.maximum(cdf[:, [-1]], 1e-12)
    return np.clip(cdf, 0.0, 1.0)


def cdf_diagnostics(y_true: np.ndarray, y_grid: np.ndarray, density: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    cdf = density_to_cdf(density, y_grid)
    pit = np.array([np.interp(y_true[i], y_grid, cdf[i], left=0.0, right=1.0) for i in range(y_true.size)])
    pit_ks = float(stats.kstest(pit, "uniform").statistic)

    lower = np.array([np.interp(0.05, cdf[i], y_grid) for i in range(y_true.size)])
    upper = np.array([np.interp(0.95, cdf[i], y_grid) for i in range(y_true.size)])
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    width = float(np.mean(upper - lower))
    return pit, pit_ks, coverage, width


def evaluate_cindes(estimator, Z_test: np.ndarray, y_test: np.ndarray, y_grid: np.ndarray) -> tuple[float, np.ndarray, float, float, float, np.ndarray]:
    logits_grid = estimator.predict_logits_on_grid(Z_test, y_grid)
    log_q_grid = estimator.reference.logpdf(y_grid)[None, :]
    log_weights = np.log(trapezoid_weights(y_grid))[None, :]
    log_unnormalized = log_q_grid + logits_grid
    log_norm = logsumexp(log_unnormalized + log_weights, axis=1)

    logits_y = estimator.predict_logits(Z_test, y_test)
    log_density_y = estimator.reference.logpdf(y_test) + logits_y - log_norm
    nll = float(-np.mean(log_density_y))

    density = np.exp(log_unnormalized - log_norm[:, None])
    pit, pit_ks, coverage, width = cdf_diagnostics(y_test, y_grid, density)
    return nll, pit, pit_ks, coverage, width, density


def gaussian_diagnostics(y_true: np.ndarray, mean: np.ndarray, sigma: float) -> tuple[float, float, float, float]:
    sigma = max(float(sigma), 1e-6)
    z = (y_true - mean) / sigma
    nll = float(np.mean(0.5 * np.log(2.0 * np.pi) + np.log(sigma) + 0.5 * z**2))
    pit = stats.norm.cdf(z)
    pit_ks = float(stats.kstest(pit, "uniform").statistic)
    z90 = float(stats.norm.ppf(0.95))
    lower = mean - z90 * sigma
    upper = mean + z90 * sigma
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    width = float(np.mean(upper - lower))
    return nll, pit_ks, coverage, width


def select_top_genes(X_train: np.ndarray, top_genes: int) -> np.ndarray:
    variances = np.var(X_train, axis=0)
    top_k = min(top_genes, X_train.shape[1])
    return np.argsort(variances)[-top_k:]


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std


def make_train_test_split(X: np.ndarray, y: np.ndarray, train_fraction: float, seed: int) -> SplitData:
    train_idx, test_idx = train_test_split(
        np.arange(X.shape[0]),
        train_size=train_fraction,
        random_state=seed,
        shuffle=True,
    )
    if np.intersect1d(train_idx, test_idx).size:
        raise AssertionError("Train/test split contains overlapping rows.")
    if train_idx.size + test_idx.size != X.shape[0]:
        raise AssertionError("Train/test split does not partition all rows.")

    return SplitData(
        train_idx=train_idx,
        test_idx=test_idx,
        X_train_raw=X[train_idx],
        X_test_raw=X[test_idx],
        y_train=y[train_idx],
        y_test=y[test_idx],
    )


def fit_training_only_preprocess(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_train: np.ndarray,
    top_genes: int,
    grid_size: int,
) -> TrainingOnlyPreprocess:
    selected_genes = select_top_genes(X_train_raw, top_genes)
    X_train_top = X_train_raw[:, selected_genes]
    X_test_top = X_test_raw[:, selected_genes]
    X_train_std, X_test_std = standardize_train_test(X_train_top, X_test_top)
    y_grid = make_y_grid(y_train, grid_size)
    reference = ReferenceDensity.from_y_train(y_train)

    if selected_genes.size > X_train_raw.shape[1]:
        raise AssertionError("Selected more genes than exist in the training matrix.")
    if not np.all(np.diff(y_grid) > 0):
        raise AssertionError("Training-only response grid must be strictly increasing.")
    if not np.isfinite([reference.mean, reference.std]).all():
        raise AssertionError("Reference density parameters must be finite.")
    return TrainingOnlyPreprocess(
        selected_genes=selected_genes,
        X_train_std=X_train_std,
        X_test_std=X_test_std,
        y_grid=y_grid,
        reference=reference,
    )


def pca_scores(X_train: np.ndarray, X_test: np.ndarray, num_factors: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_components = min(num_factors, X_train.shape[0] - 1, X_train.shape[1])
    pca = PCA(n_components=n_components, random_state=seed, svd_solver="full")
    pca.fit(X_train)
    Z_train = np.dot(X_train - pca.mean_, pca.components_.T)
    Z_test = np.dot(X_test - pca.mean_, pca.components_.T)
    return Z_train, Z_test


def linear_predict(model, X: np.ndarray) -> np.ndarray:
    """Predict with linear model coefficients using np.dot to avoid matmul warnings."""

    return np.dot(X, model.coef_) + float(model.intercept_)


def npz_scalar(data, key: str, default: object = "") -> object:
    if key not in data:
        return default
    value = data[key]
    return value.item() if getattr(value, "shape", ()) == () else value


def save_plugin_examples(
    results_dir: Path,
    drug_slug: str,
    seed: int,
    y_test: np.ndarray,
    y_grid: np.ndarray,
    pit: np.ndarray,
    density: np.ndarray,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pit": pit}).to_csv(results_dir / f"pit_{drug_slug}_seed{seed}_plugin_pca_cindes.csv", index=False)
    weights = trapezoid_weights(y_grid)[None, :]
    mass = density * weights
    mass /= np.maximum(np.sum(mass, axis=1, keepdims=True), 1e-12)
    predicted_mean = np.sum(mass * y_grid[None, :], axis=1)
    cdf = density_to_cdf(density, y_grid)
    predicted_median = np.array([np.interp(0.5, cdf[i], y_grid) for i in range(density.shape[0])])
    np.savez_compressed(
        results_dir / f"density_examples_{drug_slug}_seed{seed}_plugin_pca_cindes.npz",
        y_grid=y_grid,
        density=density,
        y_observed=y_test,
        predicted_mean=predicted_mean,
        predicted_median=predicted_median,
    )


def split_diagnostics(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_grid: np.ndarray,
    test_y_outside_grid_count: int,
) -> dict[str, object]:
    return {
        "y_train_mean": float(np.mean(y_train)),
        "y_train_sd": float(np.std(y_train, ddof=1)),
        "y_test_mean": float(np.mean(y_test)),
        "y_test_sd": float(np.std(y_test, ddof=1)),
        "grid_min": float(y_grid[0]),
        "grid_max": float(y_grid[-1]),
        "test_y_outside_grid_count": int(test_y_outside_grid_count),
    }


def result_row(
    drug_name: str,
    seed: int,
    method: str,
    n_train: int,
    n_test: int,
    p_raw: int,
    top_genes: int,
    raw_x_top_genes: int,
    num_factors: int,
    test_nll: float,
    pit_ks: float,
    pi90_coverage: float,
    pi90_width: float,
    epochs_trained: float,
    best_val_loss: float,
    stopped_early: object,
    fit_eval_seconds: float,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    row = {
        "drug_name": drug_name,
        "seed": seed,
        "method": method,
        "n_train": n_train,
        "n_test": n_test,
        "p_raw": p_raw,
        "top_genes": top_genes,
        "raw_x_top_genes": raw_x_top_genes,
        "num_factors": num_factors,
        "test_nll": test_nll,
        "pit_ks": pit_ks,
        "pi90_coverage": pi90_coverage,
        "abs_coverage_error": abs(float(pi90_coverage) - 0.90),
        "pi90_width": pi90_width,
        "epochs_trained": epochs_trained,
        "best_val_loss": best_val_loss,
        "stopped_early": stopped_early,
        "cindes_epochs_trained": epochs_trained,
        "cindes_best_val_loss": best_val_loss,
        "cindes_stopped_early": stopped_early,
        "mdn_components": math.nan,
        "mdn_best_val_nll": math.nan,
        "mdn_grid_mass_mean": math.nan,
        "mdn_grid_mass_min": math.nan,
        "fit_eval_seconds": fit_eval_seconds,
    }
    row.update(diagnostics)
    return row


def run_one_dataset(path: Path, args: argparse.Namespace, output_path: Path) -> list[dict[str, object]]:
    with np.load(path, allow_pickle=True) as data:
        X = data["X"].astype(np.float64)
        y = data["y"].astype(np.float64)
        drug_name = str(npz_scalar(data, "drug_name"))
        analysis_id = str(npz_scalar(data, "analysis_id", drug_name))
        source = str(npz_scalar(data, "source", ""))
        drug_id = str(npz_scalar(data, "drug_id", ""))
        analysis_unit = str(npz_scalar(data, "analysis_unit", "drug_name"))
    drug_slug = slugify(analysis_id)

    rows: list[dict[str, object]] = []
    seeds = [args.base_seed + i for i in range(args.num_seeds)]
    for seed in seeds:
        split = make_train_test_split(X, y, args.train_fraction, seed)
        X_train_raw = split.X_train_raw
        X_test_raw = split.X_test_raw
        y_train = split.y_train
        y_test = split.y_test

        preprocess = fit_training_only_preprocess(
            X_train_raw,
            X_test_raw,
            y_train,
            top_genes=args.top_genes,
            grid_size=args.grid_size,
        )
        selected_genes = preprocess.selected_genes
        X_train_std = preprocess.X_train_std
        X_test_std = preprocess.X_test_std
        y_grid = preprocess.y_grid
        outside_grid = (y_test < y_grid[0]) | (y_test > y_grid[-1])
        outside_grid_count = int(np.sum(outside_grid))
        if outside_grid_count:
            print(
                f"warning: drug={drug_name} seed={seed} has {outside_grid_count} test Y values outside "
                f"the training-only y_grid [{y_grid[0]:.4f}, {y_grid[-1]:.4f}]"
            )
        diagnostics = split_diagnostics(y_train, y_test, y_grid, outside_grid_count)
        diagnostics.update(
            {
                "analysis_id": analysis_id,
                "source": source,
                "drug_id": drug_id,
                "analysis_unit": analysis_unit,
                "grid_size": int(args.grid_size),
            }
        )
        raw_x_top_genes_effective = min(int(args.raw_x_top_genes), X.shape[1])

        train_config = TrainingConfig(
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            device=args.device,
        )
        mdn_config = MDNConfig(
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            device=args.device,
            components=parse_int_tuple(args.mdn_components),
        )
        reference = preprocess.reference

        needs_pca = bool({"plugin_pca_cindes", "pca_gaussian_baseline", "pca_mdn"} & set(args.methods))
        if needs_pca:
            Z_train, Z_test = pca_scores(X_train_std, X_test_std, args.num_factors, seed)
        else:
            Z_train = Z_test = None

        # A. Plug-in PCA CINDES.
        if "plugin_pca_cindes" in args.methods and run_key(
            analysis_id, seed, "plugin_pca_cindes", args.num_factors, args.top_genes, args.grid_size
        ) not in args.completed_keys:
            started = time.perf_counter()
            assert Z_train is not None and Z_test is not None
            estimator, info = train_cindes(
                Z_train, y_train, config=train_config, seed=seed + 10_000, reference=reference
            )
            nll, pit, pit_ks, coverage, width, density = evaluate_cindes(estimator, Z_test, y_test, y_grid)
            elapsed = time.perf_counter() - started
            row = result_row(
                drug_name,
                seed,
                "plugin_pca_cindes",
                y_train.size,
                y_test.size,
                X.shape[1],
                selected_genes.size,
                raw_x_top_genes_effective,
                Z_train.shape[1],
                nll,
                pit_ks,
                coverage,
                width,
                info.epochs_trained,
                info.best_val_loss,
                info.stopped_early,
                elapsed,
                diagnostics,
            )
            append_csv(output_path, [row])
            rows.append(row)
            if args.save_density_examples and seed == seeds[0]:
                save_plugin_examples(output_path.parent, drug_slug, seed, y_test, y_grid, pit, density)
            print(f"done drug={drug_name} seed={seed} method=plugin_pca_cindes nll={nll:.4f}")

        # B. Raw-X CINDES.
        if "raw_x_cindes" in args.methods and run_key(
            analysis_id, seed, "raw_x_cindes", args.num_factors, args.top_genes, args.grid_size
        ) not in args.completed_keys:
            raw_selected_genes = select_top_genes(X_train_raw, args.raw_x_top_genes)
            X_train_raw_top = X_train_raw[:, raw_selected_genes]
            X_test_raw_top = X_test_raw[:, raw_selected_genes]
            X_train_raw_std, X_test_raw_std = standardize_train_test(X_train_raw_top, X_test_raw_top)
            started = time.perf_counter()
            estimator, info = train_cindes(
                X_train_raw_std,
                y_train,
                config=train_config,
                seed=seed + 20_000,
                reference=reference,
            )
            nll, _, pit_ks, coverage, width, _ = evaluate_cindes(estimator, X_test_raw_std, y_test, y_grid)
            elapsed = time.perf_counter() - started
            row = result_row(
                drug_name,
                seed,
                "raw_x_cindes",
                y_train.size,
                y_test.size,
                X.shape[1],
                selected_genes.size,
                raw_selected_genes.size,
                args.num_factors,
                nll,
                pit_ks,
                coverage,
                width,
                info.epochs_trained,
                info.best_val_loss,
                info.stopped_early,
                elapsed,
                diagnostics,
            )
            append_csv(output_path, [row])
            rows.append(row)
            print(f"done drug={drug_name} seed={seed} method=raw_x_cindes nll={nll:.4f}")

        # C. ElasticNet Gaussian baseline.
        if "gaussian_elasticnet_baseline" in args.methods and run_key(
            analysis_id, seed, "gaussian_elasticnet_baseline", args.num_factors, args.top_genes, args.grid_size
        ) not in args.completed_keys:
            started = time.perf_counter()
            elastic = ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.9],
                cv=min(5, y_train.size),
                random_state=seed,
                max_iter=20000,
                tol=1e-3,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                elastic.fit(X_train_std, y_train)
            train_mean = linear_predict(elastic, X_train_std)
            test_mean = linear_predict(elastic, X_test_std)
            sigma = float(np.std(y_train - train_mean, ddof=1))
            nll, pit_ks, coverage, width = gaussian_diagnostics(y_test, test_mean, sigma)
            elapsed = time.perf_counter() - started
            row = result_row(
                drug_name,
                seed,
                "gaussian_elasticnet_baseline",
                y_train.size,
                y_test.size,
                X.shape[1],
                selected_genes.size,
                raw_x_top_genes_effective,
                args.num_factors,
                nll,
                pit_ks,
                coverage,
                width,
                math.nan,
                math.nan,
                math.nan,
                elapsed,
                diagnostics,
            )
            append_csv(output_path, [row])
            rows.append(row)
            print(f"done drug={drug_name} seed={seed} method=gaussian_elasticnet_baseline nll={nll:.4f}")

        # D. PCA Gaussian baseline.
        if "pca_gaussian_baseline" in args.methods and run_key(
            analysis_id, seed, "pca_gaussian_baseline", args.num_factors, args.top_genes, args.grid_size
        ) not in args.completed_keys:
            started = time.perf_counter()
            assert Z_train is not None and Z_test is not None
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 13))
            ridge.fit(Z_train, y_train)
            train_mean = linear_predict(ridge, Z_train)
            test_mean = linear_predict(ridge, Z_test)
            sigma = float(np.std(y_train - train_mean, ddof=1))
            nll, pit_ks, coverage, width = gaussian_diagnostics(y_test, test_mean, sigma)
            elapsed = time.perf_counter() - started
            row = result_row(
                drug_name,
                seed,
                "pca_gaussian_baseline",
                y_train.size,
                y_test.size,
                X.shape[1],
                selected_genes.size,
                raw_x_top_genes_effective,
                Z_train.shape[1],
                nll,
                pit_ks,
                coverage,
                width,
                math.nan,
                math.nan,
                math.nan,
                elapsed,
                diagnostics,
            )
            append_csv(output_path, [row])
            rows.append(row)
            print(f"done drug={drug_name} seed={seed} method=pca_gaussian_baseline nll={nll:.4f}")

        # E. PCA mixture-density-network baseline.
        if "pca_mdn" in args.methods and run_key(
            analysis_id, seed, "pca_mdn", args.num_factors, args.top_genes, args.grid_size
        ) not in args.completed_keys:
            started = time.perf_counter()
            assert Z_train is not None and Z_test is not None
            estimator, info = train_mdn_select_k(Z_train, y_train, config=mdn_config, seed=seed + 30_000)
            log_density_y = estimator.logpdf(Z_test, y_test)
            nll = float(-np.mean(log_density_y))
            pit = estimator.cdf(Z_test, y_test)
            pit_ks = float(stats.kstest(pit, "uniform").statistic)
            lower, upper = estimator.interval(Z_test, alpha=0.10)
            coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
            width = float(np.mean(upper - lower))
            density = estimator.density_on_grid(Z_test, y_grid)
            grid_mass = np.sum(density * trapezoid_weights(y_grid)[None, :], axis=1)
            elapsed = time.perf_counter() - started
            row = result_row(
                drug_name,
                seed,
                "pca_mdn",
                y_train.size,
                y_test.size,
                X.shape[1],
                selected_genes.size,
                raw_x_top_genes_effective,
                Z_train.shape[1],
                nll,
                pit_ks,
                coverage,
                width,
                info.epochs_trained,
                info.best_val_nll,
                info.stopped_early,
                elapsed,
                diagnostics,
            )
            row["mdn_components"] = int(info.n_components)
            row["mdn_best_val_nll"] = float(info.best_val_nll)
            row["mdn_grid_mass_mean"] = float(np.mean(grid_mass))
            row["mdn_grid_mass_min"] = float(np.min(grid_mass))
            append_csv(output_path, [row])
            rows.append(row)
            print(
                f"done drug={drug_name} seed={seed} method=pca_mdn "
                f"k={info.n_components} nll={nll:.4f}"
            )

    return rows


def npz_drug_name(path: Path) -> str:
    with np.load(path, allow_pickle=True) as data:
        return str(npz_scalar(data, "drug_name"))


def npz_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        y = data["y"].astype(np.float64)
        return {
            "path": str(path),
            "drug_name": str(npz_scalar(data, "drug_name")),
            "analysis_id": str(npz_scalar(data, "analysis_id", npz_scalar(data, "drug_name"))),
            "source": str(npz_scalar(data, "source", "")),
            "drug_id": str(npz_scalar(data, "drug_id", "")),
            "analysis_unit": str(npz_scalar(data, "analysis_unit", "drug_name")),
            "n_matched": int(y.size),
            "response_sd": float(np.std(y, ddof=1)) if y.size > 1 else float("nan"),
        }


def filter_metadata_by_analysis_ids(metadata: pd.DataFrame, analysis_ids: list[str]) -> pd.DataFrame:
    if not analysis_ids:
        return metadata
    requested = set(analysis_ids)
    selected = metadata[metadata["analysis_id"].astype(str).isin(requested)].copy()
    found = set(selected["analysis_id"].astype(str))
    missing = sorted(requested - found)
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Requested analysis_id values were not found among eligible processed units: {preview}"
            + (f" ... plus {len(missing) - 10} more" if len(missing) > 10 else "")
        )
    order = {analysis_id: idx for idx, analysis_id in enumerate(analysis_ids)}
    selected["_analysis_id_order"] = selected["analysis_id"].map(order)
    return selected.sort_values("_analysis_id_order").drop(columns=["_analysis_id_order"]).reset_index(drop=True)


def processed_files(
    processed_dir: Path,
    num_drugs: int,
    drug_name: str | None,
    drug_names: list[str],
    all_eligible: bool,
    min_samples: int,
    min_response_sd: float,
    max_drugs: int | None,
    eligible_output: Path | None,
    analysis_ids: list[str],
) -> list[Path]:
    files = sorted(processed_dir.glob("gdsc_*.npz"))
    if drug_names:
        selected = []
        missing = []
        for requested in drug_names:
            matches = [npz_metadata(path) for path in files if npz_drug_name(path).lower() == requested.lower()]
            if not matches:
                missing.append(requested)
            else:
                matches = sorted(matches, key=lambda row: (row["analysis_unit"] == "source_drug", row["n_matched"]), reverse=True)
                if len(matches) > 1:
                    display = pd.DataFrame(matches)[["drug_name", "source", "drug_id", "analysis_id", "n_matched", "response_sd", "path"]]
                    print(
                        "\nwarning: requested drug name maps to multiple prepared units; "
                        "selecting the largest source-specific unit when available.\n"
                        f"{display.to_string(index=False)}\n",
                        flush=True,
                    )
                selected.append(Path(str(matches[0]["path"])))
        if missing:
            available = ", ".join(sorted({npz_drug_name(path).lower() for path in files})[:10])
            raise FileNotFoundError(
                "Requested processed drug file(s) not found by exact drug name: "
                f"{', '.join(missing)}. Run prepare_gdsc.py --drug-names first. "
                f"Example available processed drugs: {available}"
            )
        selected_meta = pd.DataFrame([npz_metadata(path) for path in selected])
        selected_meta = filter_metadata_by_analysis_ids(selected_meta, analysis_ids)
        return [Path(path) for path in selected_meta["path"]]
    if all_eligible:
        eligible = pd.DataFrame([npz_metadata(path) for path in files])
        if eligible.empty:
            raise FileNotFoundError("No processed GDSC .npz files found. Run prepare_gdsc.py first.")
        if (eligible["analysis_unit"] == "source_drug").any():
            eligible = eligible[eligible["analysis_unit"] == "source_drug"].copy()
        eligible["eligible"] = (eligible["n_matched"] >= min_samples) & (eligible["response_sd"] >= min_response_sd)
        eligible = eligible.sort_values(["eligible", "n_matched", "response_sd"], ascending=[False, False, False])
        if eligible_output is not None:
            eligible_output.parent.mkdir(parents=True, exist_ok=True)
            eligible.to_csv(eligible_output, index=False)
        selected = eligible[eligible["eligible"]].copy()
        selected = filter_metadata_by_analysis_ids(selected, analysis_ids)
        if max_drugs is not None:
            selected = selected.head(max_drugs)
        if selected.empty:
            raise ValueError(
                f"No processed drugs meet eligibility criteria: n_matched >= {min_samples}, "
                f"response_sd >= {min_response_sd}."
            )
        return [Path(path) for path in selected["path"]]
    if drug_name:
        requested = [slugify(item) for item in drug_name.split(",") if item.strip()]
        files = [path for path in files if any(req in path.stem for req in requested)]
    if not files:
        raise FileNotFoundError("No processed GDSC .npz files found. Run prepare_gdsc.py first.")
    selected_meta = pd.DataFrame([npz_metadata(path) for path in files])
    selected_meta = filter_metadata_by_analysis_ids(selected_meta, analysis_ids)
    selected_files = [Path(path) for path in selected_meta["path"]]
    return selected_files[:num_drugs]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR), help="Directory with prepared .npz files.")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="Output directory.")
    parser.add_argument("--output", default=None, help="Optional path for the per-split results CSV.")
    parser.add_argument("--drug-name", default=None, help="Optional legacy drug-name filter.")
    parser.add_argument(
        "--drug-names",
        default=None,
        help="Comma-separated exact drug names to run. When set, --num-drugs is ignored.",
    )
    parser.add_argument("--num-drugs", type=int, default=3)
    parser.add_argument("--all-eligible", action="store_true", help="Run all processed drugs meeting criteria.")
    parser.add_argument("--min-samples", type=int, default=120, help="Minimum matched rows for --all-eligible.")
    parser.add_argument("--min-response-sd", type=float, default=0.2, help="Minimum response SD for --all-eligible.")
    parser.add_argument("--max-drugs", type=int, default=None, help="Optional cap for --all-eligible.")
    parser.add_argument("--analysis-ids", default=None, help="Comma-separated exact analysis_id values to run.")
    parser.add_argument(
        "--analysis-ids-file",
        default=None,
        help="CSV or text file listing exact analysis_id values to run. CSV uses the analysis_id column when present.",
    )
    parser.add_argument(
        "--eligible-output",
        default=str(RESULTS_DIR / "gdsc_eligible_drugs.csv"),
        help="CSV path for processed-drug eligibility table.",
    )
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--top-genes", type=int, default=2000)
    parser.add_argument("--raw-x-top-genes", type=int, default=None)
    parser.add_argument("--num-factors", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grid-size", type=int, default=401)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--methods", default=",".join(METHODS), help="Comma-separated methods to run, or 'all'.")
    parser.add_argument("--mdn-components", default="2,3,5", help="Comma-separated MDN component counts to tune.")
    parser.add_argument(
        "--save-density-examples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save PIT and density-example files for the first seed of plug-in PCA CINDES.",
    )
    parser.add_argument("--skip-raw-x", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip completed drug/seed/method rows in the output CSV.")
    parser.add_argument(
        "--status-report",
        default=None,
        help="Optional Markdown run-status report path.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    import torch

    torch.set_num_threads(args.torch_num_threads)
    args.methods = parse_methods(args.methods)
    if args.skip_raw_x:
        args.methods = [method for method in args.methods if method != "raw_x_cindes"]
    if not args.methods:
        raise ValueError("No methods selected after applying --skip-raw-x.")
    if args.raw_x_top_genes is None:
        args.raw_x_top_genes = args.top_genes
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")
    analysis_ids = dedupe_preserve_order(parse_comma_list(args.analysis_ids) + read_analysis_ids_file(args.analysis_ids_file))

    results_dir = Path(args.results_dir)
    if args.output:
        output_path = Path(args.output)
    elif args.all_eligible:
        output_path = results_dir / "gdsc_all_drugs_metrics.csv"
    else:
        output_path = results_dir / "gdsc_application_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_path.exists():
        output_path.unlink()
    completed_keys, duplicate_rows_before = existing_run_keys(output_path) if args.resume else (set(), 0)
    args.completed_keys = completed_keys

    files = processed_files(
        Path(args.processed_dir),
        args.num_drugs,
        args.drug_name,
        parse_comma_list(args.drug_names),
        args.all_eligible,
        args.min_samples,
        args.min_response_sd,
        args.max_drugs,
        Path(args.eligible_output) if args.eligible_output else None,
        analysis_ids,
    )
    seeds = [args.base_seed + i for i in range(args.num_seeds)]
    expected_rows = len(files) * len(seeds) * len(args.methods)
    print(
        f"Run plan: drug_units={len(files)} seeds={len(seeds)} methods={len(args.methods)} "
        f"expected_rows={expected_rows} resume_completed={len(completed_keys)}",
        flush=True,
    )
    all_rows: list[dict[str, object]] = []
    for path in files:
        all_rows.extend(run_one_dataset(path, args, output_path))

    if all_rows:
        df = pd.DataFrame(all_rows)
        print("\nSummary by method:")
        print(df.groupby("method")[["test_nll", "pit_ks", "pi90_coverage", "pi90_width"]].mean().to_string())
        print(f"\nSaved results to {output_path}")
    report_path = (
        Path(args.status_report)
        if args.status_report
        else Path("reports") / ("gdsc_all_drugs_run_status.md" if args.all_eligible else "gdsc_run_status.md")
    )
    write_run_status_report(
        report_path,
        output_path,
        files,
        args.methods,
        seeds,
        args,
        duplicate_rows_before,
    )
    print(f"Saved run-status report to {report_path}")


if __name__ == "__main__":
    main()
