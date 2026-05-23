#!/usr/bin/env python3
"""Run factor-augmented CINDES simulations from the command line."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from sklearn.decomposition import PCA

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from fac_cindes.data import generate_train_test_data, make_y_grid
from fac_cindes.metrics import aligned_factor_mse_with_alignment, evaluate_nll_and_tv, fit_factor_alignment
from fac_cindes.models import ReferenceDensity
from fac_cindes.train import TrainingConfig, train_cindes


METHODS = ("oracle", "plugin_pca", "raw_x")


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown method(s): {', '.join(unknown)}. Valid methods are: {', '.join(METHODS)}."
        )
    if not methods:
        raise argparse.ArgumentTypeError("At least one method must be provided.")
    return methods


def seed_sequence(num_seeds: int, base_seed: int) -> list[int]:
    return [base_seed + i for i in range(num_seeds)]


def method_seed(base_seed: int, method: str) -> int:
    return base_seed + 10_000 * (METHODS.index(method) + 1)


def prepare_method_features(
    method: str,
    train_F: np.ndarray,
    test_F: np.ndarray,
    train_X: np.ndarray,
    test_X: np.ndarray,
    r: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return train/test Z and factor-MSE diagnostics for a CINDES method."""

    if method == "oracle":
        return train_F, test_F, 0.0, 0.0
    if method == "raw_x":
        return train_X, test_X, float("nan"), float("nan")
    if method == "plugin_pca":
        pca = PCA(n_components=r, random_state=seed, svd_solver="full")
        pca.fit(train_X)
        train_scores = np.dot(train_X - pca.mean_, pca.components_.T)
        test_scores = np.dot(test_X - pca.mean_, pca.components_.T)
        alignment = fit_factor_alignment(train_F, train_scores)
        factor_mse_train = aligned_factor_mse_with_alignment(train_F, train_scores, alignment)
        factor_mse_test = aligned_factor_mse_with_alignment(test_F, test_scores, alignment)
        return train_scores, test_scores, factor_mse_train, factor_mse_test
    raise ValueError(f"Unknown method: {method}")


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


def summarize(rows: list[dict[str, object]]) -> str:
    """Build a compact mean/standard-error summary table."""

    grouped: dict[tuple[int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["n"]), int(row["p"]), str(row["method"]))].append(row)

    lines = [
        "n     p     method       nll_mean  nll_se    tv_mean   tv_se     fmse_tr   fmse_te   reps",
        "-----------------------------------------------------------------------------------------",
    ]
    for key in sorted(grouped.keys()):
        n, p, method = key
        group = grouped[key]
        nll = np.array([float(row["nll"]) for row in group], dtype=np.float64)
        tv = np.array([float(row["tv"]) for row in group], dtype=np.float64)
        fmse_train = np.array([float(row["factor_mse_train"]) for row in group], dtype=np.float64)
        fmse_test = np.array([float(row["factor_mse_test"]) for row in group], dtype=np.float64)
        reps = len(group)
        nll_se = float(np.std(nll, ddof=1) / math.sqrt(reps)) if reps > 1 else float("nan")
        tv_se = float(np.std(tv, ddof=1) / math.sqrt(reps)) if reps > 1 else float("nan")
        fmse_train_mean = float(np.mean(fmse_train)) if np.isfinite(fmse_train).any() else float("nan")
        fmse_test_mean = float(np.mean(fmse_test)) if np.isfinite(fmse_test).any() else float("nan")
        lines.append(
            f"{n:<5d} {p:<5d} {method:<12s} "
            f"{np.mean(nll):<9.4f} {nll_se:<9.4f} {np.mean(tv):<9.4f} {tv_se:<9.4f} "
            f"{fmse_train_mean:<9.4f} {fmse_test_mean:<9.4f} {reps:<4d}"
        )
    return "\n".join(lines)


def run_one_configuration(
    n: int,
    p: int,
    r: int,
    seed: int,
    args: argparse.Namespace,
    train_config: TrainingConfig,
    output_path: Path,
) -> list[dict[str, object]]:
    data = generate_train_test_data(
        n_train=n,
        n_test=args.test_size,
        p=p,
        r=r,
        sigma_u=args.sigma_u,
        seed=seed,
    )
    y_grid = make_y_grid(data.train.Y, data.test.Y, grid_size=args.grid_size)
    reference = ReferenceDensity.from_y_train(data.train.Y)

    rows: list[dict[str, object]] = []
    for method in args.methods:
        started = time.perf_counter()
        Z_train, Z_test, factor_mse_train, factor_mse_test = prepare_method_features(
            method=method,
            train_F=data.train.F,
            test_F=data.test.F,
            train_X=data.train.X,
            test_X=data.test.X,
            r=r,
            seed=seed,
        )
        estimator, info = train_cindes(
            Z_train,
            data.train.Y,
            config=train_config,
            seed=method_seed(seed, method),
            reference=reference,
        )
        nll, tv = evaluate_nll_and_tv(
            estimator,
            Z_test,
            data.test.Y,
            data.test.F,
            y_grid,
            z_batch_size=args.z_batch_size,
        )
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "seed": seed,
                "n": n,
                "p": p,
                "r": r,
                "method": method,
                "nll": nll,
                "tv": tv,
                "factor_mse_train": factor_mse_train,
                "factor_mse_test": factor_mse_test,
                "epochs_trained": info.epochs_trained,
                "best_val_loss": info.best_val_loss,
                "stopped_early": info.stopped_early,
                "fit_eval_seconds": elapsed,
            }
        )
        append_csv(output_path, [rows[-1]])
        print(
            f"done seed={seed} n={n} p={p} method={method} "
            f"nll={nll:.4f} tv={tv:.4f} epochs={info.epochs_trained}",
            flush=True,
        )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", default="500,1000,2000", help="Comma-separated training sample sizes.")
    parser.add_argument("--p-values", default="100,500", help="Comma-separated covariate dimensions.")
    parser.add_argument("--r", type=int, default=3, help="Number of latent factors.")
    parser.add_argument("--num-seeds", type=int, default=20, help="Number of Monte Carlo seeds.")
    parser.add_argument("--base-seed", type=int, default=12345, help="First Monte Carlo seed.")
    parser.add_argument("--test-size", type=int, default=1000, help="Independent test-set size.")
    parser.add_argument("--sigma-u", type=float, default=1.0, help="Idiosyncratic noise standard deviation.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="MLP hidden width.")
    parser.add_argument("--depth", type=int, default=3, help="Number of hidden ReLU layers.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=25, help="Early-stopping patience.")
    parser.add_argument("--batch-size", type=int, default=256, help="Classifier training mini-batch size.")
    parser.add_argument("--grid-size", type=int, default=401, help="Number of y-grid points.")
    parser.add_argument("--z-batch-size", type=int, default=64, help="Number of test Z rows per grid-logit batch.")
    parser.add_argument("--device", default="auto", help="Torch device: cpu, mps, cuda, or auto.")
    parser.add_argument(
        "--methods",
        type=parse_methods,
        default=parse_methods(",".join(METHODS)),
        help="Comma-separated subset of methods: oracle, plugin_pca, raw_x.",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=1,
        help="Number of intra-op PyTorch CPU threads.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "results" / "simulation_results.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output CSV before running.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.torch_num_threads)
    n_values = parse_int_list(args.n_values)
    p_values = parse_int_list(args.p_values)
    seeds = seed_sequence(args.num_seeds, args.base_seed)
    output_path = Path(args.output)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    train_config = TrainingConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
    )

    all_rows: list[dict[str, object]] = []
    for n in n_values:
        for p in p_values:
            for seed in seeds:
                rows = run_one_configuration(
                    n=n,
                    p=p,
                    r=args.r,
                    seed=seed,
                    args=args,
                    train_config=train_config,
                    output_path=output_path,
                )
                all_rows.extend(rows)

    print("\nSummary:")
    print(summarize(all_rows))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
