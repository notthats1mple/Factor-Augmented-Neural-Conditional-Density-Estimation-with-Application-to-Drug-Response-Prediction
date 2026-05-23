#!/usr/bin/env python3
"""Create summary tables and figures from CINDES simulation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["oracle", "plugin_pca", "raw_x"]
METHOD_LABELS = {
    "oracle": "Oracle",
    "plugin_pca": "Plug-in PCA",
    "raw_x": "Raw-X",
}


def standard_error(values: pd.Series) -> float:
    """Sample standard error with NaN for a single replication."""

    count = values.count()
    if count <= 1:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(count))


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize NLL, TV, factor MSE, and replication counts."""

    summary = (
        df.groupby(["n", "p", "method"], sort=True)
        .agg(
            nll_mean=("nll", "mean"),
            nll_se=("nll", standard_error),
            tv_mean=("tv", "mean"),
            tv_se=("tv", standard_error),
            factor_mse_test_mean=("factor_mse_test", "mean"),
            factor_mse_test_se=("factor_mse_test", standard_error),
            reps=("seed", "nunique"),
        )
        .reset_index()
    )
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    return summary.sort_values(["n", "p", "method"]).reset_index(drop=True)


def figure_tv_by_p(summary: pd.DataFrame, output_path: Path) -> None:
    """Figure 1: mean TV versus p, one panel per n and one line per method."""

    n_values = sorted(summary["n"].unique())
    fig, axes = plt.subplots(1, len(n_values), figsize=(5.2 * len(n_values), 4.2), sharey=True)
    if len(n_values) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_values):
        panel = summary[summary["n"] == n]
        for method in METHOD_ORDER:
            series = panel[panel["method"] == method].sort_values("p")
            ax.errorbar(
                series["p"],
                series["tv_mean"],
                yerr=series["tv_se"],
                marker="o",
                linewidth=2,
                capsize=3,
                label=METHOD_LABELS[method],
            )
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Ambient dimension p")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(sorted(panel["p"].unique()))

    axes[0].set_ylabel("Integrated TV error")
    axes[-1].legend(frameon=False)
    fig.suptitle("Integrated TV Error by Ambient Dimension", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_factor_mse_by_p(summary: pd.DataFrame, output_path: Path) -> None:
    """Figure 2: plug-in PCA aligned factor MSE versus p, one line per n."""

    plugin = summary[summary["method"] == "plugin_pca"].copy()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for n in sorted(plugin["n"].unique()):
        series = plugin[plugin["n"] == n].sort_values("p")
        ax.errorbar(
            series["p"],
            series["factor_mse_test_mean"],
            yerr=series["factor_mse_test_se"],
            marker="o",
            linewidth=2,
            capsize=3,
            label=f"n = {n}",
        )

    ax.set_title("Aligned Factor MSE by Ambient Dimension")
    ax.set_xlabel("Ambient dimension p")
    ax.set_ylabel("Aligned factor MSE")
    ax.set_xticks(sorted(plugin["p"].unique()))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_tv_by_n(summary: pd.DataFrame, output_path: Path) -> None:
    """Figure 3: mean TV versus n, one panel per p and one line per method."""

    p_values = sorted(summary["p"].unique())
    fig, axes = plt.subplots(1, len(p_values), figsize=(5.2 * len(p_values), 4.2), sharey=True)
    if len(p_values) == 1:
        axes = [axes]

    for ax, p in zip(axes, p_values):
        panel = summary[summary["p"] == p]
        for method in METHOD_ORDER:
            series = panel[panel["method"] == method].sort_values("n")
            ax.errorbar(
                series["n"],
                series["tv_mean"],
                yerr=series["tv_se"],
                marker="o",
                linewidth=2,
                capsize=3,
                label=METHOD_LABELS[method],
            )
        ax.set_title(f"p = {p}")
        ax.set_xlabel("Training sample size n")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(sorted(panel["n"].unique()))

    axes[0].set_ylabel("Integrated TV error")
    axes[-1].legend(frameon=False)
    fig.suptitle("Integrated TV Error by Training Sample Size", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_overall_averages(df: pd.DataFrame) -> None:
    """Print overall average NLL and TV by method."""

    overall = (
        df.groupby("method", sort=False)
        .agg(mean_nll=("nll", "mean"), mean_tv=("tv", "mean"), reps=("seed", "count"))
        .reindex(METHOD_ORDER)
    )
    print("\nOverall average NLL and TV by method:")
    print(overall.to_string(float_format=lambda x: f"{x:.4f}"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_input = Path(__file__).resolve().parent / "results" / "pilot_v2.csv"
    parser.add_argument("--input", default=str(default_input), help="Simulation CSV path.")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for generated tables and figures. Defaults to the input CSV directory.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for generated artifacts. Defaults to the input CSV stem.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input)
    results_dir = Path(args.results_dir) if args.results_dir else input_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_prefix or input_path.stem

    df = pd.read_csv(input_path)
    summary = make_summary_table(df)

    summary_path = results_dir / f"{output_prefix}_summary.csv"
    figure1_path = results_dir / f"{output_prefix}_figure_tv_by_p.png"
    figure2_path = results_dir / f"{output_prefix}_figure_factor_mse_by_p.png"
    figure3_path = results_dir / f"{output_prefix}_figure_tv_by_n.png"

    summary.to_csv(summary_path, index=False)
    figure_tv_by_p(summary, figure1_path)
    figure_factor_mse_by_p(summary, figure2_path)
    figure_tv_by_n(summary, figure3_path)

    print(f"Read {input_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {figure1_path}")
    print(f"Saved {figure2_path}")
    print(f"Saved {figure3_path}")
    print_overall_averages(df)


if __name__ == "__main__":
    main()
