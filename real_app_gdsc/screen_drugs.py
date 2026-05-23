#!/usr/bin/env python3
"""Screen GDSC drugs for expression-response signal before CINDES runs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

APP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = APP_DIR / "results"

if __package__ is None or __package__ == "":
    sys.path.append(str(APP_DIR))

from prepare_gdsc import load_expression, load_responses, require_files


def select_top_genes(X_train: np.ndarray, top_genes: int) -> np.ndarray:
    variances = np.var(X_train, axis=0)
    top_k = min(int(top_genes), X_train.shape[1])
    return np.argsort(variances)[-top_k:]


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std


def linear_predict(model: RidgeCV, X: np.ndarray) -> np.ndarray:
    return np.dot(X, model.coef_) + float(model.intercept_)


def parse_comma_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def stable_drug_seed(base_seed: int, drug_name: str) -> int:
    digest = hashlib.sha256(drug_name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int((int(base_seed) + offset) % (2**32 - 1))


def gaussian_nll(y_true: np.ndarray, mean: np.ndarray, sigma: float) -> float:
    sigma = max(float(sigma), 1e-6)
    z = (y_true - mean) / sigma
    return float(np.mean(0.5 * np.log(2.0 * np.pi) + np.log(sigma) + 0.5 * z**2))


def matched_drug_arrays(
    drug_name: str,
    responses: pd.DataFrame,
    X: np.ndarray,
    expression_ids: np.ndarray,
    id_to_row: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, int]:
    drug_response = responses[responses["drug_name"] == drug_name].copy()
    n_available = int(drug_response["cosmic_id"].nunique())
    drug_response = (
        drug_response.groupby("cosmic_id", as_index=False)
        .agg(response=("response", "mean"))
        .dropna(subset=["response"])
    )

    rows = []
    y_values = []
    for _, row in drug_response.iterrows():
        cosmic_id = str(row["cosmic_id"])
        if cosmic_id in id_to_row:
            rows.append(id_to_row[cosmic_id])
            y_values.append(float(row["response"]))

    if not rows:
        return np.empty((0, X.shape[1]), dtype=np.float64), np.empty(0, dtype=np.float64), n_available

    order = np.argsort(rows)
    row_array = np.asarray(rows, dtype=int)[order]
    y = np.asarray(y_values, dtype=np.float64)[order]
    return X[row_array].astype(np.float64), y, n_available


def cross_validated_pca_ridge(
    X: np.ndarray,
    y: np.ndarray,
    top_genes: int,
    num_factors: int,
    n_splits: int,
    seed: int,
) -> tuple[float, float, float]:
    if y.size < 3 or float(np.std(y, ddof=1)) < 1e-8:
        return float("nan"), float("nan"), float("nan")

    n_splits = min(int(n_splits), y.size)
    if n_splits < 2:
        return float("nan"), float("nan"), float("nan")

    predictions = np.empty_like(y, dtype=np.float64)
    null_nlls: list[float] = []
    ridge_nlls: list[float] = []
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, val_idx in splitter.split(X):
        X_train_raw = X[train_idx]
        X_val_raw = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        selected = select_top_genes(X_train_raw, top_genes)
        X_train_top = X_train_raw[:, selected]
        X_val_top = X_val_raw[:, selected]
        X_train_std, X_val_std = standardize_train_test(X_train_top, X_val_top)

        n_components = min(int(num_factors), X_train_std.shape[0] - 1, X_train_std.shape[1])
        if n_components < 1:
            val_pred = np.full(y_val.shape, float(np.mean(y_train)))
            train_pred = np.full(y_train.shape, float(np.mean(y_train)))
        else:
            pca = PCA(n_components=n_components, random_state=seed, svd_solver="randomized")
            pca.fit(X_train_std)
            Z_train = np.dot(X_train_std - pca.mean_, pca.components_.T)
            Z_val = np.dot(X_val_std - pca.mean_, pca.components_.T)
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 13))
            ridge.fit(Z_train, y_train)
            train_pred = linear_predict(ridge, Z_train)
            val_pred = linear_predict(ridge, Z_val)

        predictions[val_idx] = val_pred
        residual_sigma = float(np.std(y_train - train_pred, ddof=1))
        ridge_nlls.append(gaussian_nll(y_val, val_pred, residual_sigma))

        null_mean = np.full(y_val.shape, float(np.mean(y_train)))
        null_sigma = float(np.std(y_train, ddof=1))
        null_nlls.append(gaussian_nll(y_val, null_mean, null_sigma))

    sse = float(np.sum((y - predictions) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    cv_r2 = float(1.0 - sse / sst) if sst > 1e-12 else float("nan")
    return cv_r2, float(np.mean(null_nlls)), float(np.mean(ridge_nlls))


def select_screening_drugs(matched: pd.DataFrame, min_matched: int, requested_drug_names: list[str]) -> list[str]:
    drug_counts = (
        matched.groupby("drug_name")["cosmic_id"]
        .nunique()
        .sort_values(ascending=False)
        .rename("n_matched")
        .reset_index()
    )
    if requested_drug_names:
        selected = []
        missing = []
        for requested in requested_drug_names:
            exact = drug_counts[drug_counts["drug_name"].str.lower() == requested.lower()]
            if exact.empty:
                missing.append(requested)
            else:
                selected.append(str(exact.iloc[0]["drug_name"]))
        if missing:
            available = ", ".join(drug_counts["drug_name"].head(10).astype(str))
            raise ValueError(
                "Requested drug(s) not found by exact name among matched data: "
                f"{', '.join(missing)}. Example available drugs: {available}"
            )
        return selected

    drugs = drug_counts[drug_counts["n_matched"] >= min_matched]["drug_name"].astype(str).tolist()
    if not drugs:
        raise ValueError(f"No drugs have at least {min_matched} matched cell lines.")
    return drugs


def summarize_drug(
    drug_name: str,
    responses: pd.DataFrame,
    X: np.ndarray,
    expression_ids: np.ndarray,
    id_to_row: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, object]:
    started = time.perf_counter()
    X_drug, y, n_available = matched_drug_arrays(drug_name, responses, X, expression_ids, id_to_row)
    if y.size:
        q25, q75 = np.percentile(y, [25, 75])
        y_iqr = float(q75 - q25)
        y_sd = float(np.std(y, ddof=1)) if y.size > 1 else float("nan")
    else:
        y_iqr = float("nan")
        y_sd = float("nan")

    if y.size >= args.min_matched:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            cv_r2, null_nll, ridge_nll = cross_validated_pca_ridge(
                X_drug,
                y,
                top_genes=args.top_genes,
                num_factors=args.num_factors,
                n_splits=args.cv_folds,
                seed=args.seed,
            )
    else:
        cv_r2 = float("nan")
        null_nll = float("nan")
        ridge_nll = float("nan")

    return {
        "drug_name": drug_name,
        "n_matched": int(y.size),
        "y_mean": float(np.mean(y)) if y.size else float("nan"),
        "y_sd": y_sd,
        "y_iqr": y_iqr,
        "y_min": float(np.min(y)) if y.size else float("nan"),
        "y_max": float(np.max(y)) if y.size else float("nan"),
        "n_available_cell_lines": n_available,
        "cv_r2": cv_r2,
        "null_gaussian_nll": null_nll,
        "pca_ridge_gaussian_nll": ridge_nll,
        "top_genes": int(args.top_genes),
        "num_factors": int(args.num_factors),
        "cv_folds": int(args.cv_folds),
        "screen_seconds": time.perf_counter() - started,
    }


def summarize_shuffle_check(
    drug_name: str,
    responses: pd.DataFrame,
    X: np.ndarray,
    expression_ids: np.ndarray,
    id_to_row: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, object]:
    started = time.perf_counter()
    X_drug, y, n_available = matched_drug_arrays(drug_name, responses, X, expression_ids, id_to_row)
    if y.size:
        q25, q75 = np.percentile(y, [25, 75])
        y_iqr = float(q75 - q25)
        y_sd = float(np.std(y, ddof=1)) if y.size > 1 else float("nan")
    else:
        y_iqr = float("nan")
        y_sd = float("nan")

    if y.size >= args.min_matched:
        shuffle_seed = args.shuffle_seed if args.shuffle_seed is not None else args.seed
        rng = np.random.default_rng(stable_drug_seed(shuffle_seed, drug_name))
        y_shuffled = np.array(y, copy=True)
        rng.shuffle(y_shuffled)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            original_cv_r2, original_null_nll, original_ridge_nll = cross_validated_pca_ridge(
                X_drug,
                y,
                top_genes=args.top_genes,
                num_factors=args.num_factors,
                n_splits=args.cv_folds,
                seed=args.seed,
            )
            shuffled_cv_r2, shuffled_null_nll, shuffled_ridge_nll = cross_validated_pca_ridge(
                X_drug,
                y_shuffled,
                top_genes=args.top_genes,
                num_factors=args.num_factors,
                n_splits=args.cv_folds,
                seed=args.seed,
            )
    else:
        original_cv_r2 = float("nan")
        original_null_nll = float("nan")
        original_ridge_nll = float("nan")
        shuffled_cv_r2 = float("nan")
        shuffled_null_nll = float("nan")
        shuffled_ridge_nll = float("nan")

    return {
        "drug_name": drug_name,
        "n_matched": int(y.size),
        "n_available_cell_lines": n_available,
        "y_mean": float(np.mean(y)) if y.size else float("nan"),
        "y_sd": y_sd,
        "y_iqr": y_iqr,
        "original_cv_r2": original_cv_r2,
        "shuffled_cv_r2": shuffled_cv_r2,
        "r2_drop_original_minus_shuffled": original_cv_r2 - shuffled_cv_r2,
        "original_null_gaussian_nll": original_null_nll,
        "original_pca_ridge_gaussian_nll": original_ridge_nll,
        "shuffled_null_gaussian_nll": shuffled_null_nll,
        "shuffled_pca_ridge_gaussian_nll": shuffled_ridge_nll,
        "top_genes": int(args.top_genes),
        "num_factors": int(args.num_factors),
        "cv_folds": int(args.cv_folds),
        "shuffle_seed": int(args.shuffle_seed if args.shuffle_seed is not None else args.seed),
        "top_genes_training_fold_only": True,
        "standardization_training_fold_only": True,
        "pca_training_fold_only": True,
        "ridge_training_fold_only": True,
        "responses_aggregated_by_cosmic_id": True,
        "split_unit": "unique_cosmic_id_after_aggregation",
        "screen_seconds": time.perf_counter() - started,
    }


def rank_screening_table(df: pd.DataFrame, min_matched: int, min_y_sd: float, min_y_iqr: float) -> pd.DataFrame:
    out = df.copy()
    out["candidate_flag"] = (
        (out["n_matched"] >= min_matched)
        & (out["y_sd"] >= min_y_sd)
        & (out["y_iqr"] >= min_y_iqr)
        & (out["cv_r2"] > 0.0)
    )
    out["candidate_score"] = out["cv_r2"].where(out["candidate_flag"], -np.inf)
    out = out.sort_values(
        ["candidate_flag", "candidate_score", "cv_r2", "n_matched", "y_iqr"],
        ascending=[False, False, False, False, False],
    )
    out["screen_rank"] = np.arange(1, out.shape[0] + 1)
    return out.drop(columns=["candidate_score"]).reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", default="LN_IC50", choices=["LN_IC50", "AUC"], help="Preferred response column.")
    parser.add_argument("--top-genes", type=int, default=2000)
    parser.add_argument("--num-factors", type=int, default=10)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--drug-names", default=None, help="Comma-separated exact drug names to screen/check.")
    parser.add_argument(
        "--min-matched",
        type=int,
        default=500,
        help="Minimum matched cell lines required for PCA/Ridge screening.",
    )
    parser.add_argument("--min-y-sd", type=float, default=0.1)
    parser.add_argument("--min-y-iqr", type=float, default=0.1)
    parser.add_argument("--shuffle-y-check", action="store_true", help="Run original-vs-permuted-Y leakage check.")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="Seed for per-drug Y permutations.")
    parser.add_argument(
        "--shuffle-output",
        default=str(RESULTS_DIR / "drug_screening_shuffle_check.csv"),
        help="Output CSV path for --shuffle-y-check.",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "drug_screening_table.csv"),
        help="Output CSV path.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"openpyxl\.")
    require_files()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading GDSC expression and response data...", flush=True)
    X, expression_ids, _ = load_expression()
    expression_ids = expression_ids.astype(str)
    id_to_row = {cosmic_id: idx for idx, cosmic_id in enumerate(expression_ids)}
    responses = load_responses(args.response)

    matched = responses[responses["cosmic_id"].isin(set(expression_ids))].copy()
    drugs = select_screening_drugs(matched, args.min_matched, parse_comma_list(args.drug_names))

    if args.shuffle_y_check:
        shuffle_output_path = Path(args.shuffle_output)
        shuffle_output_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Running shuffle-Y leakage check for {len(drugs)} drugs using "
            f"top_genes={args.top_genes}, num_factors={args.num_factors}, cv_folds={args.cv_folds}.",
            flush=True,
        )
        rows = []
        for idx, drug_name in enumerate(drugs, start=1):
            row = summarize_shuffle_check(drug_name, responses, X, expression_ids, id_to_row, args)
            rows.append(row)
            print(
                f"[{idx:03d}/{len(drugs):03d}] {drug_name}: "
                f"n={row['n_matched']} original_r2={row['original_cv_r2']:.4f} "
                f"shuffled_r2={row['shuffled_cv_r2']:.4f}",
                flush=True,
            )

        table = pd.DataFrame(rows)
        table.to_csv(shuffle_output_path, index=False)
        display_cols = [
            "drug_name",
            "n_matched",
            "y_sd",
            "y_iqr",
            "original_cv_r2",
            "shuffled_cv_r2",
            "r2_drop_original_minus_shuffled",
            "original_pca_ridge_gaussian_nll",
            "shuffled_pca_ridge_gaussian_nll",
        ]
        print(f"\nSaved shuffle-Y check to {shuffle_output_path}", flush=True)
        print("\nShuffle-Y leakage check:", flush=True)
        print(table[display_cols].to_string(index=False), flush=True)
        print(
            "\nLeakage controls: top genes, standardization, PCA, and Ridge are fit within training folds; "
            "responses are aggregated by COSMIC_ID before splitting.",
            flush=True,
        )
        return

    print(
        f"Screening {len(drugs)} drugs with n_matched >= {args.min_matched} "
        f"using top_genes={args.top_genes}, num_factors={args.num_factors}, cv_folds={args.cv_folds}.",
        flush=True,
    )
    rows = []
    for idx, drug_name in enumerate(drugs, start=1):
        row = summarize_drug(drug_name, responses, X, expression_ids, id_to_row, args)
        rows.append(row)
        print(
            f"[{idx:03d}/{len(drugs):03d}] {drug_name}: "
            f"n={row['n_matched']} cv_r2={row['cv_r2']:.4f} "
            f"ridge_nll={row['pca_ridge_gaussian_nll']:.4f}",
            flush=True,
        )

    table = rank_screening_table(
        pd.DataFrame(rows),
        min_matched=args.min_matched,
        min_y_sd=args.min_y_sd,
        min_y_iqr=args.min_y_iqr,
    )
    table.to_csv(output_path, index=False)

    display_cols = [
        "screen_rank",
        "drug_name",
        "n_matched",
        "y_mean",
        "y_sd",
        "y_iqr",
        "cv_r2",
        "null_gaussian_nll",
        "pca_ridge_gaussian_nll",
        "candidate_flag",
    ]
    print(f"\nSaved screening table to {output_path}", flush=True)
    print("\nTop 20 candidate drugs:", flush=True)
    print(table[display_cols].head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
