#!/usr/bin/env python3
"""Select source-specific GDSC units for PCA factor-dimension sensitivity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def sample_representative_units(candidates: pd.DataFrame, n_extra: int, seed: int) -> pd.DataFrame:
    if candidates.empty or n_extra <= 0:
        return candidates.iloc[0:0].copy()
    if candidates.shape[0] <= n_extra:
        return candidates.copy()

    rng = np.random.default_rng(seed)
    out = candidates.copy()
    out["_size_rank"] = out["n_matched"].rank(method="first")
    out["_size_bin"] = pd.qcut(out["_size_rank"], q=min(3, out.shape[0]), labels=False, duplicates="drop")
    out["_stratum"] = out["source"].astype(str) + "::" + out["_size_bin"].astype(str)

    groups = [(name, group.copy()) for name, group in out.groupby("_stratum", sort=True)]
    total = sum(group.shape[0] for _, group in groups)
    allocations = []
    used = 0
    for name, group in groups:
        raw = n_extra * group.shape[0] / total
        base = int(np.floor(raw))
        allocations.append([name, group, base, raw - base])
        used += base
    remaining = n_extra - used
    allocations.sort(key=lambda item: (item[3], item[1].shape[0]), reverse=True)
    for item in allocations:
        if remaining <= 0:
            break
        item[2] += 1
        remaining -= 1

    sampled = []
    for _, group, count, _ in allocations:
        count = min(count, group.shape[0])
        if count <= 0:
            continue
        positions = rng.choice(group.index.to_numpy(), size=count, replace=False)
        sampled.append(out.loc[positions])
    selected = pd.concat(sampled, axis=0) if sampled else out.iloc[0:0].copy()

    if selected.shape[0] < n_extra:
        remaining_pool = out.drop(index=selected.index)
        take = min(n_extra - selected.shape[0], remaining_pool.shape[0])
        if take > 0:
            positions = rng.choice(remaining_pool.index.to_numpy(), size=take, replace=False)
            selected = pd.concat([selected, out.loc[positions]], axis=0)

    return selected.drop(columns=["_size_rank", "_size_bin", "_stratum"], errors="ignore")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligible-input", required=True, help="CSV produced by prepare_gdsc.py --all-eligible.")
    parser.add_argument(
        "--case-drugs",
        default="Trametinib,Venetoclax,Vorinostat",
        help="Comma-separated drug names to always include when present.",
    )
    parser.add_argument("--n-extra", type=int, default=30, help="Number of additional representative units to sample.")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--output", required=True, help="Output CSV with selected analysis_id values.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    eligible = pd.read_csv(args.eligible_input)
    if "analysis_id" not in eligible.columns:
        raise ValueError("Eligible input must contain an analysis_id column.")
    if "eligible" in eligible.columns:
        eligible = eligible[eligible["eligible"].astype(bool)].copy()
    eligible = eligible.drop_duplicates("analysis_id").reset_index(drop=True)

    case_drugs = {drug.lower() for drug in parse_comma_list(args.case_drugs)}
    case_units = eligible[eligible["drug_name"].astype(str).str.lower().isin(case_drugs)].copy()
    case_units["selection_role"] = "case_study_drug"

    candidates = eligible[~eligible["analysis_id"].isin(case_units["analysis_id"])].copy()
    extra_units = sample_representative_units(candidates, args.n_extra, args.seed)
    extra_units["selection_role"] = "representative_subset"

    selected = pd.concat([case_units, extra_units], axis=0, ignore_index=True)
    selected["selection_seed"] = int(args.seed)
    selected["n_extra_requested"] = int(args.n_extra)
    columns = [
        "analysis_id",
        "drug_name",
        "source",
        "drug_id",
        "n_matched",
        "response_sd",
        "selection_role",
        "selection_seed",
        "n_extra_requested",
        "path",
    ]
    columns = [col for col in columns if col in selected.columns]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected[columns].to_csv(output, index=False)
    print(f"Selected {selected.shape[0]} analysis units: {case_units.shape[0]} case-study units plus {extra_units.shape[0]} representative units.")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
