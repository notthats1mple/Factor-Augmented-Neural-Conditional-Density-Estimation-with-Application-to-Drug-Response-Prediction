#!/usr/bin/env python3
"""Make tables and figures for the GDSC real-data application."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = APP_DIR / "results"
METHOD_ORDER = [
    "plugin_pca_cindes",
    "raw_x_cindes",
    "gaussian_elasticnet_baseline",
    "pca_gaussian_baseline",
    "pca_mdn",
]
METHOD_LABELS = {
    "plugin_pca_cindes": "Plug-in PCA CINDES",
    "raw_x_cindes": "Raw-X CINDES",
    "gaussian_elasticnet_baseline": "ElasticNet Gaussian",
    "pca_gaussian_baseline": "PCA/Ridge Gaussian",
    "pca_mdn": "PCA MDN",
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "drug"


def standard_error(values: pd.Series) -> float:
    count = values.count()
    if count <= 1:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(count))


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "abs_coverage_error" not in out.columns:
        out["abs_coverage_error"] = (out["pi90_coverage"] - 0.90).abs()

    def first_n_matched(values: pd.Series) -> float:
        idx = values.index[0]
        if {"n_train", "n_test"}.issubset(out.columns):
            return int(out.loc[idx, "n_train"] + out.loc[idx, "n_test"])
        if "n_matched" in out.columns and pd.notna(out.loc[idx, "n_matched"]):
            return int(out.loc[idx, "n_matched"])
        return float("nan")

    summary = (
        out.groupby(["drug_name", "method"])
        .agg(
            n_matched=("n_test", first_n_matched),
            test_nll_mean=("test_nll", "mean"),
            test_nll_se=("test_nll", standard_error),
            pit_ks_mean=("pit_ks", "mean"),
            pit_ks_se=("pit_ks", standard_error),
            pi90_coverage_mean=("pi90_coverage", "mean"),
            pi90_coverage_se=("pi90_coverage", standard_error),
            abs_coverage_error_mean=("abs_coverage_error", "mean"),
            abs_coverage_error_se=("abs_coverage_error", standard_error),
            pi90_width_mean=("pi90_width", "mean"),
            pi90_width_se=("pi90_width", standard_error),
            reps=("seed", "nunique"),
        )
        .reset_index()
    )
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    return summary.sort_values(["drug_name", "method"]).reset_index(drop=True)


def make_all_drug_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "abs_coverage_error" not in out.columns:
        out["abs_coverage_error"] = (out["pi90_coverage"] - 0.90).abs()
    unit_col = "analysis_id" if "analysis_id" in out.columns else "drug_name"
    summary = (
        out.groupby([unit_col, "drug_name", "method"], dropna=False)
        .agg(
            n_matched=("n_test", lambda values: int(out.loc[values.index[0], "n_train"] + out.loc[values.index[0], "n_test"])),
            mean_test_nll=("test_nll", "mean"),
            se_test_nll=("test_nll", standard_error),
            mean_pit_ks=("pit_ks", "mean"),
            se_pit_ks=("pit_ks", standard_error),
            mean_pi90_coverage=("pi90_coverage", "mean"),
            se_pi90_coverage=("pi90_coverage", standard_error),
            mean_abs_coverage_error=("abs_coverage_error", "mean"),
            se_abs_coverage_error=("abs_coverage_error", standard_error),
            mean_pi90_width=("pi90_width", "mean"),
            se_pi90_width=("pi90_width", standard_error),
            reps=("seed", "nunique"),
        )
        .reset_index()
    )
    if unit_col != "analysis_id":
        summary = summary.rename(columns={unit_col: "analysis_id"})
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    return summary.sort_values(["analysis_id", "method"]).reset_index(drop=True)


def make_delta_summary(summary: pd.DataFrame) -> pd.DataFrame:
    metric = summary.pivot(index="analysis_id", columns="method", values="mean_test_nll")
    coverage = summary.pivot(index="analysis_id", columns="method", values="mean_abs_coverage_error")
    names = summary.groupby("analysis_id")["drug_name"].first()
    out = pd.DataFrame({"analysis_id": metric.index, "drug_name": names.reindex(metric.index).to_numpy()})
    plugin_nll = metric.get("plugin_pca_cindes")
    plugin_cov = coverage.get("plugin_pca_cindes")
    for baseline, suffix in [
        ("gaussian_elasticnet_baseline", "elasticnet"),
        ("pca_gaussian_baseline", "pca_gaussian"),
        ("pca_mdn", "pca_mdn"),
    ]:
        if plugin_nll is not None and baseline in metric:
            out[f"delta_nll_plugin_vs_{suffix}"] = plugin_nll.to_numpy() - metric[baseline].to_numpy()
        else:
            out[f"delta_nll_plugin_vs_{suffix}"] = np.nan
        if plugin_cov is not None and baseline in coverage:
            out[f"delta_coverage_error_plugin_vs_{suffix}"] = plugin_cov.to_numpy() - coverage[baseline].to_numpy()
        else:
            out[f"delta_coverage_error_plugin_vs_{suffix}"] = np.nan
    return out


def cindes_pairwise_table(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baselines = [
        ("ElasticNet Gaussian", "elasticnet"),
        ("PCA/Ridge Gaussian", "pca_gaussian"),
        ("PCA MDN", "pca_mdn"),
    ]
    for label, suffix in baselines:
        nll_col = f"delta_nll_plugin_vs_{suffix}"
        cov_col = f"delta_coverage_error_plugin_vs_{suffix}"
        nll = delta[nll_col].dropna() if nll_col in delta else pd.Series(dtype=float)
        cov = delta[cov_col].dropna() if cov_col in delta else pd.Series(dtype=float)
        rows.append(
            {
                "baseline": label,
                "median_delta_nll": float(nll.median()) if not nll.empty else np.nan,
                "mean_delta_nll": float(nll.mean()) if not nll.empty else np.nan,
                "pct_drugs_cindes_better_nll": float(100.0 * (nll < 0).mean()) if not nll.empty else np.nan,
                "median_delta_abs_coverage_error": float(cov.median()) if not cov.empty else np.nan,
                "pct_drugs_cindes_better_coverage": float(100.0 * (cov < 0).mean()) if not cov.empty else np.nan,
                "n_drugs": int(nll.count()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_pairwise_deltas(delta: pd.DataFrame, n_boot: int = 2000, seed: int = 20260522) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    baselines = [
        ("elasticnet", "delta_nll_plugin_vs_elasticnet"),
        ("pca_gaussian", "delta_nll_plugin_vs_pca_gaussian"),
        ("pca_mdn", "delta_nll_plugin_vs_pca_mdn"),
    ]
    for baseline, col in baselines:
        if col not in delta:
            continue
        values = delta[col].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        boot_mean = np.empty(n_boot, dtype=float)
        boot_median = np.empty(n_boot, dtype=float)
        for idx in range(n_boot):
            sample = rng.choice(values, size=values.size, replace=True)
            boot_mean[idx] = float(np.mean(sample))
            boot_median[idx] = float(np.median(sample))
        rows.append(
            {
                "baseline": baseline,
                "n_analysis_units": int(values.size),
                "mean_delta_nll": float(np.mean(values)),
                "mean_delta_nll_ci_lower": float(np.quantile(boot_mean, 0.025)),
                "mean_delta_nll_ci_upper": float(np.quantile(boot_mean, 0.975)),
                "median_delta_nll": float(np.median(values)),
                "median_delta_nll_ci_lower": float(np.quantile(boot_median, 0.025)),
                "median_delta_nll_ci_upper": float(np.quantile(boot_median, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def all_drug_method_table(summary: pd.DataFrame) -> pd.DataFrame:
    metric = summary.pivot(index="analysis_id", columns="method", values="mean_test_nll")
    elastic = metric.get("gaussian_elasticnet_baseline")
    rows = []
    for method, group in summary.groupby("method", observed=True):
        method_text = str(method)
        nll = group.set_index("analysis_id")["mean_test_nll"]
        if elastic is not None:
            delta = nll.reindex(elastic.index) - elastic
            improves = delta < 0
            median_delta = float(delta.median(skipna=True))
            pct_improves = float(100.0 * improves.mean(skipna=True))
        else:
            median_delta = float("nan")
            pct_improves = float("nan")
        rows.append(
            {
                "method": METHOD_LABELS.get(method_text, method_text),
                "mean_nll": float(group["mean_test_nll"].mean()),
                "median_nll": float(group["mean_test_nll"].median()),
                "median_delta_nll_vs_elasticnet": median_delta,
                "pct_drugs_improved_vs_elasticnet": pct_improves,
                "median_pit_ks": float(group["mean_pit_ks"].median()),
                "median_pi90_coverage": float(group["mean_pi90_coverage"].median()),
                "median_abs_coverage_error": float(group["mean_abs_coverage_error"].median()),
                "median_pi90_width": float(group["mean_pi90_width"].median()),
                "n_drugs": int(group["analysis_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def save_all_drug_figures(summary: pd.DataFrame, delta: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    delta_cols = [
        "delta_nll_plugin_vs_elasticnet",
        "delta_nll_plugin_vs_pca_gaussian",
        "delta_nll_plugin_vs_pca_mdn",
    ]
    labels = ["vs ElasticNet", "vs PCA/Ridge", "vs PCA MDN"]
    values = [delta[col].dropna().to_numpy() for col in delta_cols if col in delta]
    used_labels = [label for col, label in zip(delta_cols, labels) if col in delta]
    if values:
        fig, ax = plt.subplots(figsize=(6.4, 4.3))
        ax.boxplot(values, tick_labels=used_labels, showmeans=True)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("Delta NLL: CINDES - baseline")
        ax.set_title("All Eligible Drugs: Pairwise NLL Differences")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / "gdsc_all_drugs_delta_nll_boxplot.pdf", bbox_inches="tight")
        plt.close(fig)

    if "delta_nll_plugin_vs_elasticnet" in delta:
        sorted_delta = delta.sort_values("delta_nll_plugin_vs_elasticnet")
        fig, ax = plt.subplots(figsize=(7.2, max(4.0, 0.13 * sorted_delta.shape[0])))
        y = np.arange(sorted_delta.shape[0])
        ax.barh(y, sorted_delta["delta_nll_plugin_vs_elasticnet"], color="#4C78A8")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks([])
        ax.set_xlabel("Delta NLL: CINDES - ElasticNet Gaussian")
        ax.set_title("All Eligible Drugs: Sorted CINDES NLL Difference")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / "gdsc_all_drugs_delta_nll_sorted.pdf", bbox_inches="tight")
        plt.close(fig)

    methods = [method for method in METHOD_ORDER if method in set(summary["method"].astype(str))]
    if methods:
        fig, ax = plt.subplots(figsize=(7.0, 4.3))
        values = [summary[summary["method"].astype(str) == method]["mean_abs_coverage_error"].dropna().to_numpy() for method in methods]
        ax.boxplot(values, tick_labels=[METHOD_LABELS.get(method, method) for method in methods], showmeans=True)
        ax.set_ylabel("Absolute 90% coverage error")
        ax.set_title("All Eligible Drugs: Coverage Error")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / "gdsc_all_drugs_coverage_error_boxplot.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.3))
        values = [summary[summary["method"].astype(str) == method]["mean_pit_ks"].dropna().to_numpy() for method in methods]
        ax.boxplot(values, tick_labels=[METHOD_LABELS.get(method, method) for method in methods], showmeans=True)
        ax.set_ylabel("PIT KS")
        ax.set_title("All Eligible Drugs: PIT Calibration")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / "gdsc_all_drugs_pitks_boxplot.pdf", bbox_inches="tight")
        plt.close(fig)


def make_paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["method_code"] = out["method"].astype(str)

    elastic_nll = (
        out[out["method_code"] == "gaussian_elasticnet_baseline"]
        .set_index("drug_name")["test_nll_mean"]
        .rename("elasticnet_nll")
    )
    pca_nll = (
        out[out["method_code"] == "pca_gaussian_baseline"]
        .set_index("drug_name")["test_nll_mean"]
        .rename("pca_gaussian_nll")
    )
    out = out.join(elastic_nll, on="drug_name").join(pca_nll, on="drug_name")
    out["abs_coverage_error"] = (out["pi90_coverage_mean"] - 0.90).abs()
    out["delta_nll_vs_elasticnet"] = out["test_nll_mean"] - out["elasticnet_nll"]
    out["delta_nll_vs_pca_gaussian"] = out["test_nll_mean"] - out["pca_gaussian_nll"]
    out["method"] = out["method_code"].map(METHOD_LABELS).fillna(out["method_code"])

    columns = [
        "drug_name",
        "method",
        "test_nll_mean",
        "test_nll_se",
        "delta_nll_vs_elasticnet",
        "delta_nll_vs_pca_gaussian",
        "pit_ks_mean",
        "pi90_coverage_mean",
        "abs_coverage_error",
        "pi90_width_mean",
        "reps",
    ]
    return out[columns].reset_index(drop=True)


def finite_yerr(series: pd.DataFrame, se_col: str) -> np.ndarray | None:
    yerr = series[se_col].to_numpy(dtype=float)
    if "reps" in series:
        yerr = np.where(series["reps"].to_numpy(dtype=float) > 1, yerr, np.nan)
    return yerr if np.isfinite(yerr).any() else None


def grouped_bar(
    summary: pd.DataFrame,
    value_col: str,
    se_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    target_line: float | None = None,
) -> None:
    drugs = list(summary["drug_name"].drop_duplicates())
    methods = [method for method in METHOD_ORDER if method in set(summary["method"].astype(str))]

    if len(drugs) == 1:
        series = summary.set_index("method").reindex(methods).reset_index()
        x = np.arange(len(methods))
        fig, ax = plt.subplots(figsize=(max(7.0, 1.55 * len(methods)), 4.8))
        ax.bar(
            x,
            series[value_col],
            yerr=finite_yerr(series, se_col),
            capsize=3,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(methods)],
        )
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(method, method) for method in methods], rotation=25, ha="right")
        ax.set_xlabel("Method")
    else:
        x = np.arange(len(drugs))
        width = 0.8 / max(len(methods), 1)
        fig, ax = plt.subplots(figsize=(max(7.5, 2.2 * len(drugs)), 4.8))

        for j, method in enumerate(methods):
            series = summary[summary["method"] == method].set_index("drug_name").reindex(drugs).reset_index()
            offset = (j - (len(methods) - 1) / 2) * width
            ax.bar(
                x + offset,
                series[value_col],
                width=width,
                yerr=finite_yerr(series, se_col),
                capsize=3,
                label=METHOD_LABELS.get(method, method),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(drugs, rotation=25, ha="right")
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")

    if target_line is not None:
        ax.axhline(target_line, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_pit_slug(path: Path) -> str | None:
    match = re.match(r"pit_(.+)_seed\d+_plugin_pca_cindes\.csv$", path.name)
    return match.group(1) if match else None


def density_to_cdf(density: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    increments = 0.5 * (density[:, 1:] + density[:, :-1]) * np.diff(y_grid)[None, :]
    cdf = np.hstack([np.zeros((density.shape[0], 1)), np.cumsum(increments, axis=1)])
    cdf /= np.maximum(cdf[:, [-1]], 1e-12)
    return np.clip(cdf, 0.0, 1.0)


def choose_representative_indices(scores: np.ndarray, probs: list[float]) -> list[int]:
    if scores.size <= len(probs):
        return list(np.argsort(scores))
    targets = np.quantile(scores, probs)
    selected: list[int] = []
    for target in targets:
        order = np.argsort(np.abs(scores - target))
        for idx in order:
            if int(idx) not in selected:
                selected.append(int(idx))
                break
    return selected


def pit_histogram(results_dir: Path, output_path: Path, representative_slug: str | None) -> None:
    if representative_slug:
        pit_files = sorted(results_dir.glob(f"pit_{representative_slug}_seed*_plugin_pca_cindes.csv"))
    else:
        pit_files = sorted(results_dir.glob("pit_*_plugin_pca_cindes.csv"))
    if not pit_files:
        print("No plugin PCA PIT files found; skipping PIT histogram.")
        return
    pit = pd.concat([pd.read_csv(path)["pit"].dropna() for path in pit_files], ignore_index=True).to_numpy()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.hist(pit, bins=10, range=(0, 1), density=True, alpha=0.75, edgecolor="white")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title("PIT Histogram: Plug-in PCA CINDES")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def density_examples(results_dir: Path, output_path: Path, representative_slug: str | None) -> None:
    if representative_slug:
        density_files = sorted(results_dir.glob(f"density_examples_{representative_slug}_seed*_plugin_pca_cindes.npz"))
    else:
        density_files = sorted(results_dir.glob("density_examples_*_plugin_pca_cindes.npz"))
    if not density_files:
        print("No plugin PCA density example files found; skipping density curves.")
        return
    with np.load(density_files[0]) as data:
        y_grid = data["y_grid"]
        density = data["density"]
        y_observed = data["y_observed"]
        if "predicted_median" in data:
            scores = data["predicted_median"]
        elif "predicted_mean" in data:
            scores = data["predicted_mean"]
        else:
            cdf = density_to_cdf(density, y_grid)
            scores = np.array([np.interp(0.5, cdf[i], y_grid) for i in range(density.shape[0])])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = ["Low predicted response", "Median predicted response", "High predicted response"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for plot_idx, idx in enumerate(choose_representative_indices(scores, [0.10, 0.50, 0.90])):
        color = colors[plot_idx % len(colors)]
        ax.plot(y_grid, density[idx], linewidth=2, color=color, label=labels[min(plot_idx, len(labels) - 1)])
        ax.axvline(y_observed[idx], linestyle="--", linewidth=1.2, color=color)
    ax.set_xlabel("Drug response")
    ax.set_ylabel("Estimated conditional density")
    ax.set_title("Example Conditional Density Curves")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(RESULTS_DIR / "gdsc_application_results.csv"),
        help="Path to per-split GDSC results CSV.",
    )
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="Output directory.")
    parser.add_argument("--figures-dir", default=str(APP_DIR / "figures"), help="Directory for all-drug PDF figures.")
    parser.add_argument("--tables-dir", default=str(APP_DIR.parent / "tables"), help="Directory for LaTeX tables.")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional prefix for summary and figure files, e.g. gdsc_real_v1.",
    )
    parser.add_argument(
        "--all-drug-outputs",
        action="store_true",
        help="Force all-eligible summary CSV, PDF figure, and LaTeX table outputs.",
    )
    parser.add_argument("--representative-drug", default=None, help="Optional exact drug name for PIT/density examples.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    summary = make_summary(df)
    if args.output_prefix:
        prefix = str(args.output_prefix).strip()
        summary_path = results_dir / f"{prefix}_summary.csv"
        paper_table_path = results_dir / f"{prefix}_paper_table.csv"
        nll_path = results_dir / f"{prefix}_test_nll_by_method.png"
        coverage_path = results_dir / f"{prefix}_pi90_coverage.png"
        width_path = results_dir / f"{prefix}_pi90_width_by_method.png"
        pit_path = results_dir / f"{prefix}_pit_hist_plugin_pca.png"
        density_path = results_dir / f"{prefix}_density_examples.png"
    else:
        summary_path = results_dir / "gdsc_application_summary.csv"
        paper_table_path = results_dir / "gdsc_application_paper_table.csv"
        nll_path = results_dir / "gdsc_figure_test_nll_by_method.png"
        coverage_path = results_dir / "gdsc_figure_pi90_coverage.png"
        width_path = results_dir / "gdsc_figure_pi90_width_by_method.png"
        pit_path = results_dir / "gdsc_figure_pit_hist_plugin_pca.png"
        density_path = results_dir / "gdsc_figure_density_examples.png"
    summary.to_csv(summary_path, index=False)
    paper_table = make_paper_table(summary)
    paper_table.to_csv(paper_table_path, index=False)
    should_make_all_drug = args.all_drug_outputs or "all_drugs" in input_path.stem or df["drug_name"].nunique() > 8
    if should_make_all_drug:
        all_drug_summary = make_all_drug_summary(df)
        delta_summary = make_delta_summary(all_drug_summary)
        method_table = all_drug_method_table(all_drug_summary)
        pairwise_table = cindes_pairwise_table(delta_summary)
        bootstrap_table = bootstrap_pairwise_deltas(delta_summary)
        all_summary_path = results_dir / "gdsc_all_drugs_summary.csv"
        delta_summary_path = results_dir / "gdsc_all_drugs_delta_summary.csv"
        pairwise_csv_path = results_dir / "gdsc_all_drugs_pairwise_delta_table.csv"
        bootstrap_path = results_dir / "gdsc_all_drugs_pairwise_bootstrap.csv"
        all_drug_summary.to_csv(all_summary_path, index=False)
        delta_summary.to_csv(delta_summary_path, index=False)
        pairwise_table.to_csv(pairwise_csv_path, index=False)
        bootstrap_table.to_csv(bootstrap_path, index=False)
        save_all_drug_figures(all_drug_summary, delta_summary, figures_dir)
        table_path = tables_dir / "table_gdsc_all_drugs_summary.tex"
        pairwise_table_path = tables_dir / "table_gdsc_cindes_pairwise_delta.tex"
        method_table.to_latex(table_path, index=False, float_format="%.4f", escape=True)
        pairwise_table.to_latex(pairwise_table_path, index=False, float_format="%.4f", escape=True)
        print(f"Saved {all_summary_path}")
        print(f"Saved {delta_summary_path}")
        print(f"Saved {pairwise_csv_path}")
        print(f"Saved {bootstrap_path}")
        print(f"Saved {table_path}")
        print(f"Saved {pairwise_table_path}")
    representative_drug = args.representative_drug
    if representative_drug is None and not df.empty:
        plugin_rows = df[df["method"] == "plugin_pca_cindes"]
        representative_drug = str(plugin_rows["drug_name"].iloc[0] if not plugin_rows.empty else df["drug_name"].iloc[0])
    representative_slug = slugify(representative_drug) if representative_drug else None

    unit_col = "analysis_id" if "analysis_id" in df.columns else "drug_name"
    n_analysis_units = int(df[unit_col].nunique())
    skip_legacy_figures = should_make_all_drug and n_analysis_units > 30
    if skip_legacy_figures:
        print(f"Skipping legacy per-drug PNG figures for {n_analysis_units} analysis units.")
    else:
        grouped_bar(
            summary,
            "test_nll_mean",
            "test_nll_se",
            "Test negative log-likelihood",
            "GDSC Drug Response: Test NLL",
            nll_path,
        )
        grouped_bar(
            summary,
            "pi90_coverage_mean",
            "pi90_coverage_se",
            "90% prediction interval coverage",
            "GDSC Drug Response: 90% Interval Coverage",
            coverage_path,
            target_line=0.90,
        )
        grouped_bar(
            summary,
            "pi90_width_mean",
            "pi90_width_se",
            "Mean 90% prediction interval width",
            "GDSC Drug Response: 90% Interval Width",
            width_path,
        )
        pit_histogram(results_dir, pit_path, representative_slug)
        density_examples(results_dir, density_path, representative_slug)

    print(f"Saved {summary_path}")
    print(f"Saved {paper_table_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nPaper table:")
    print(paper_table.to_string(index=False))


if __name__ == "__main__":
    main()
