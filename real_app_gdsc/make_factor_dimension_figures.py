#!/usr/bin/env python3
"""Summarize PCA factor-dimension sensitivity results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "plugin_pca_cindes",
    "pca_gaussian_baseline",
    "pca_mdn",
]
METHOD_LABELS = {
    "plugin_pca_cindes": "Plug-in PCA CINDES",
    "pca_gaussian_baseline": "PCA/Ridge Gaussian",
    "pca_mdn": "PCA MDN",
}
METHOD_COLORS = {
    "plugin_pca_cindes": "#4C78A8",
    "pca_gaussian_baseline": "#F58518",
    "pca_mdn": "#54A24B",
}


def standard_error(values: pd.Series) -> float:
    count = values.count()
    if count <= 1:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(count))


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "abs_coverage_error" not in out.columns:
        out["abs_coverage_error"] = (out["pi90_coverage"] - 0.90).abs()
    summary = (
        out.groupby(["num_factors", "method"], observed=True)
        .agg(
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
            n_rows=("test_nll", "count"),
            n_analysis_units=("analysis_id", "nunique"),
            reps=("seed", "nunique"),
        )
        .reset_index()
    )
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    return summary.sort_values(["method", "num_factors"]).reset_index(drop=True)


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    se_metric: str,
    ylabel: str,
    output: Path,
    target_line: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for method in METHOD_ORDER:
        group = summary[summary["method"].astype(str) == method].sort_values("num_factors")
        if group.empty:
            continue
        ax.errorbar(
            group["num_factors"],
            group[metric],
            yerr=group[se_metric],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
        )
    if target_line is not None:
        ax.axhline(target_line, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Number of PCA factors")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(summary["num_factors"].unique()))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GDSC Factor-Dimension Sensitivity",
        "",
        "Lower NLL is better. Coverage is compared with the nominal 0.90 target.",
        "",
    ]
    for method in METHOD_ORDER:
        group = summary[summary["method"].astype(str) == method].copy()
        if group.empty:
            continue
        label = METHOD_LABELS.get(method, method)
        best = group.loc[group["mean_test_nll"].idxmin()]
        r10 = group[group["num_factors"] == 10]
        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"- Best mean NLL at r={int(best['num_factors'])}: {best['mean_test_nll']:.4f}")
        if not r10.empty:
            r10_row = r10.iloc[0]
            delta = float(r10_row["mean_test_nll"] - best["mean_test_nll"])
            lines.append(f"- r=10 mean NLL: {r10_row['mean_test_nll']:.4f} (delta from best: {delta:.4f})")
            lines.append(f"- r=10 mean 90% coverage: {r10_row['mean_pi90_coverage']:.4f}")
        lines.append("")
    output.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Factor-dimension sensitivity CSV.")
    parser.add_argument("--results-dir", default=None, help="Directory for summary CSV. Defaults to input parent.")
    parser.add_argument("--figures-dir", default="real_app_gdsc/figures", help="Directory for PDF figures.")
    parser.add_argument("--tables-dir", default="tables", help="Directory for LaTeX table output.")
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional path for the summary CSV. Defaults to results-dir/gdsc_factor_dimension_sensitivity_summary.csv.",
    )
    parser.add_argument(
        "--report",
        default="reports/gdsc_factor_dimension_sensitivity_report.md",
        help="Markdown stability report path.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input)
    df = pd.read_csv(input_path)
    required = {"analysis_id", "seed", "method", "num_factors", "test_nll", "pit_ks", "pi90_coverage", "pi90_width"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    summary = make_summary(df)

    results_dir = Path(args.results_dir) if args.results_dir else input_path.parent
    summary_path = Path(args.summary_output) if args.summary_output else results_dir / "gdsc_factor_dimension_sensitivity_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    figures_dir = Path(args.figures_dir)
    plot_metric(
        summary,
        "mean_test_nll",
        "se_test_nll",
        "Mean held-out NLL",
        figures_dir / "gdsc_factor_dimension_nll.pdf",
    )
    plot_metric(
        summary,
        "mean_pi90_coverage",
        "se_pi90_coverage",
        "Mean 90% prediction interval coverage",
        figures_dir / "gdsc_factor_dimension_coverage.pdf",
        target_line=0.90,
    )

    table = summary.copy()
    table["method"] = table["method"].astype(str).map(lambda value: METHOD_LABELS.get(value, value))
    table = table[
        [
            "num_factors",
            "method",
            "mean_test_nll",
            "mean_pit_ks",
            "mean_pi90_coverage",
            "mean_abs_coverage_error",
            "mean_pi90_width",
            "n_analysis_units",
            "reps",
        ]
    ]
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / "table_factor_dimension_sensitivity.tex"
    table.to_latex(table_path, index=False, float_format="%.4f", escape=True)

    if args.report:
        write_report(summary, Path(args.report))

    print(f"Saved {summary_path}")
    print(f"Saved {figures_dir / 'gdsc_factor_dimension_nll.pdf'}")
    print(f"Saved {figures_dir / 'gdsc_factor_dimension_coverage.pdf'}")
    print(f"Saved {table_path}")
    if args.report:
        print(f"Saved {args.report}")


if __name__ == "__main__":
    main()
