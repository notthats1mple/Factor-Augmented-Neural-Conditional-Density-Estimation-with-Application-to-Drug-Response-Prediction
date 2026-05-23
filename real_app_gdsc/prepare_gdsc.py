#!/usr/bin/env python3
"""Prepare matched GDSC expression and drug-response datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
RAW_DIR = APP_DIR / "data" / "raw"
PROCESSED_DIR = APP_DIR / "data" / "processed"

EXPRESSION_FILE = RAW_DIR / "Cell_line_RMA_proc_basalExp.txt.zip"
GDSC1_FILE = RAW_DIR / "GDSC1_fitted_dose_response_27Oct23.xlsx"
GDSC2_FILE = RAW_DIR / "GDSC2_fitted_dose_response_27Oct23.xlsx"
CELL_LINES_FILE = RAW_DIR / "Cell_Lines_Details.xlsx"


def normalize_column(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(columns: Iterable[object], candidates: Iterable[str]) -> object | None:
    normalized = {normalize_column(col): col for col in columns}
    for candidate in candidates:
        key = normalize_column(candidate)
        if key in normalized:
            return normalized[key]
    for candidate in candidates:
        key = normalize_column(candidate)
        for norm, original in normalized.items():
            if key in norm:
                return original
    return None


def clean_cosmic_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.search(r"(\d{4,})", text)
    if not match:
        return None
    return match.group(1)


def clean_drug_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "drug"


def parse_drug_names(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def source_drug_analysis_id(source: object, drug_id: object, drug_name: object) -> str:
    source_text = str(source).strip()
    drug_name_text = str(drug_name).strip()
    drug_id_text = clean_drug_id(drug_id)
    if drug_id_text:
        return f"{source_text}::{drug_id_text}::{drug_name_text}"
    return f"{source_text}::{drug_name_text}"


def add_analysis_id(responses: pd.DataFrame, analysis_unit: str) -> pd.DataFrame:
    out = responses.copy()
    if analysis_unit == "drug_name":
        out["analysis_id"] = out["drug_name"].astype(str)
    elif analysis_unit == "source_drug":
        out["analysis_id"] = [
            source_drug_analysis_id(source, drug_id, drug_name)
            for source, drug_id, drug_name in zip(out["source"], out["drug_id"], out["drug_name"])
        ]
    else:
        raise ValueError(f"Unknown analysis_unit: {analysis_unit}")
    out["analysis_unit"] = analysis_unit
    return out


def require_files() -> None:
    missing = [path for path in [EXPRESSION_FILE, GDSC1_FILE, GDSC2_FILE, CELL_LINES_FILE] if not path.exists()]
    if missing:
        names = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required GDSC files:\n"
            f"{names}\nRun real_app_gdsc/download_gdsc.py or download them manually."
        )


def load_expression() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load basal expression as rows = cell lines and columns = genes."""

    expr = pd.read_csv(EXPRESSION_FILE, sep="\t", compression="zip", low_memory=False)
    gene_col = find_column(expr.columns, ["GENE_SYMBOLS", "gene_symbol", "gene"])
    title_col = find_column(expr.columns, ["GENE_title", "gene_title"])
    annotation_cols = [col for col in [gene_col, title_col] if col is not None]

    if gene_col is None:
        gene_names = np.array([f"gene_{i}" for i in range(expr.shape[0])], dtype=object)
    else:
        gene_names = expr[gene_col].astype(str).to_numpy()

    value_df = expr.drop(columns=annotation_cols)
    numeric_df = value_df.apply(pd.to_numeric, errors="coerce")
    numeric_cols = [col for col in numeric_df.columns if numeric_df[col].notna().any()]
    numeric_df = numeric_df[numeric_cols]

    cell_line_ids = np.array([clean_cosmic_id(col) or str(col) for col in numeric_df.columns], dtype=object)
    X = numeric_df.to_numpy(dtype=np.float64).T

    finite_gene = np.isfinite(X).all(axis=0)
    X = X[:, finite_gene]
    gene_names = gene_names[finite_gene]

    gene_var = np.var(X, axis=0)
    nonconstant = gene_var > 1e-12
    X = X[:, nonconstant]
    gene_names = gene_names[nonconstant]

    if pd.Series(cell_line_ids).duplicated().any():
        expr_by_id = pd.DataFrame(X, index=cell_line_ids)
        expr_by_id = expr_by_id.groupby(level=0, sort=False).mean()
        cell_line_ids = expr_by_id.index.astype(str).to_numpy()
        X = expr_by_id.to_numpy(dtype=np.float64)

    return X.astype(np.float32), cell_line_ids, gene_names


def load_response_file(path: Path, source: str, response_preference: str) -> pd.DataFrame:
    response = pd.read_excel(path)
    drug_col = find_column(response.columns, ["DRUG_NAME", "drug_name", "drug"])
    drug_id_col = find_column(response.columns, ["DRUG_ID", "drug_id", "drug id"])
    cosmic_col = find_column(response.columns, ["COSMIC_ID", "cosmic_id", "cosmic identifier"])
    cell_line_col = find_column(response.columns, ["CELL_LINE_NAME", "cell_line_name", "cell line name"])
    ln_ic50_col = find_column(response.columns, ["LN_IC50", "ln_ic50"])
    auc_col = find_column(response.columns, ["AUC", "auc"])

    if drug_col is None:
        raise ValueError(f"Could not find drug-name column in {path}")
    if cosmic_col is None and cell_line_col is None:
        raise ValueError(f"Could not find COSMIC_ID or cell-line-name column in {path}")

    response_col = None
    if response_preference.upper() == "LN_IC50" and ln_ic50_col is not None:
        response_col = ln_ic50_col
        response_name = "LN_IC50"
    elif response_preference.upper() == "AUC" and auc_col is not None:
        response_col = auc_col
        response_name = "AUC"
    elif ln_ic50_col is not None:
        response_col = ln_ic50_col
        response_name = "LN_IC50"
    elif auc_col is not None:
        response_col = auc_col
        response_name = "AUC"
    else:
        raise ValueError(f"Could not find LN_IC50 or AUC response column in {path}")

    out = pd.DataFrame(
        {
            "drug_name": response[drug_col].astype(str),
            "drug_id": response[drug_id_col].map(clean_drug_id) if drug_id_col is not None else None,
            "response": pd.to_numeric(response[response_col], errors="coerce"),
            "response_name": response_name,
            "source": source,
        }
    )
    if cosmic_col is not None:
        out["cosmic_id"] = response[cosmic_col].map(clean_cosmic_id)
    else:
        out["cosmic_id"] = None
    if cell_line_col is not None:
        out["cell_line_name"] = response[cell_line_col].astype(str).str.strip()
    else:
        out["cell_line_name"] = None
    return out.dropna(subset=["response"])


def load_cell_line_mapping() -> dict[str, str]:
    """Map normalized cell-line names to COSMIC IDs when needed."""

    details = pd.read_excel(CELL_LINES_FILE)
    cosmic_col = find_column(details.columns, ["COSMIC identifier", "COSMIC_ID", "cosmic id"])
    name_col = find_column(details.columns, ["Sample Name", "CELL_LINE_NAME", "cell line name"])
    if cosmic_col is None or name_col is None:
        return {}

    mapping: dict[str, str] = {}
    for _, row in details[[cosmic_col, name_col]].dropna().iterrows():
        cosmic_id = clean_cosmic_id(row[cosmic_col])
        if cosmic_id is not None:
            mapping[str(row[name_col]).strip().lower()] = cosmic_id
    return mapping


def load_responses(response_preference: str, analysis_unit: str = "source_drug") -> pd.DataFrame:
    gdsc1 = load_response_file(GDSC1_FILE, "GDSC1", response_preference)
    gdsc2 = load_response_file(GDSC2_FILE, "GDSC2", response_preference)
    responses = pd.concat([gdsc1, gdsc2], ignore_index=True)

    name_to_cosmic = load_cell_line_mapping()
    missing_cosmic = responses["cosmic_id"].isna()
    if missing_cosmic.any() and name_to_cosmic:
        responses.loc[missing_cosmic, "cosmic_id"] = responses.loc[missing_cosmic, "cell_line_name"].str.lower().map(
            name_to_cosmic
        )

    responses = responses.dropna(subset=["cosmic_id", "response"])
    return add_analysis_id(responses, analysis_unit)


def select_drugs(
    responses: pd.DataFrame,
    expression_ids: set[str],
    requested_drug_names: list[str],
    requested_drug: str | None,
    num_drugs: int,
    min_samples: int,
    min_response_sd: float = 0.0,
    all_eligible: bool = False,
    max_drugs: int | None = None,
    source_filter: str | None = None,
    drug_id_filter: str | None = None,
) -> list[dict[str, object]]:
    all_counts = summarize_drug_eligibility(
        responses=responses,
        expression_ids=expression_ids,
        min_samples=min_samples,
        min_response_sd=min_response_sd,
    )
    if all_counts.empty:
        raise ValueError("No drugs have matched expression-response data.")

    if requested_drug_names:
        selected_rows = []
        missing = []
        for requested in requested_drug_names:
            exact = all_counts[all_counts["drug_name"].str.lower() == requested.lower()].copy()
            if source_filter:
                exact = exact[exact["source"].astype(str).str.lower() == source_filter.lower()]
            if drug_id_filter:
                exact = exact[exact["drug_id"].astype(str).str.lower() == drug_id_filter.lower()]
            if exact.empty:
                missing.append(requested)
            else:
                if exact.shape[0] > 1:
                    display = exact[["drug_name", "source", "drug_id", "analysis_id", "n_matched", "response_sd"]]
                    print(
                        "\nwarning: requested drug name maps to multiple source/drug units; "
                        "selecting the largest n_matched unit. Use --source or --drug-id to disambiguate.\n"
                        f"{display.to_string(index=False)}\n",
                        flush=True,
                    )
                selected_rows.append(exact.iloc[0])
        if missing:
            available = ", ".join(all_counts["drug_name"].head(10).astype(str))
            raise ValueError(
                "Requested drug(s) not found by exact name among matched data: "
                f"{', '.join(missing)}. Example available drugs: {available}"
            )
        selected = pd.DataFrame(selected_rows)
        return selected.to_dict("records")

    counts = all_counts[all_counts["eligible"]].copy()
    if counts.empty:
        raise ValueError(
            f"No drugs meet eligibility criteria: n_matched >= {min_samples}, "
            f"response_sd >= {min_response_sd}."
        )

    if requested_drug:
        requested = [item.strip().lower() for item in requested_drug.split(",") if item.strip()]
        selected = counts[counts["drug_name"].str.lower().isin(requested)]
        if selected.empty:
            selected = counts[counts["drug_name"].str.lower().apply(lambda name: any(req in name for req in requested))]
        if source_filter:
            selected = selected[selected["source"].astype(str).str.lower() == source_filter.lower()]
        if drug_id_filter:
            selected = selected[selected["drug_id"].astype(str).str.lower() == drug_id_filter.lower()]
        if selected.empty:
            raise ValueError(f"Requested drug not found among matched data: {requested_drug}")
        counts = selected

    if all_eligible:
        selected = counts
        if max_drugs is not None:
            selected = selected.head(max_drugs)
    else:
        selected = counts.head(num_drugs)
    return selected.to_dict("records")


def summarize_drug_eligibility(
    responses: pd.DataFrame,
    expression_ids: set[str],
    min_samples: int,
    min_response_sd: float,
) -> pd.DataFrame:
    matched = responses[responses["cosmic_id"].isin(expression_ids)].copy()
    if matched.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["analysis_id", "drug_name", "source", "drug_id", "response_name", "analysis_unit"]
    for keys, group in matched.groupby(group_cols, dropna=False):
        analysis_id, drug_name, source, drug_id, response_name, analysis_unit = keys
        aggregated = (
            group.groupby("cosmic_id", as_index=False)
            .agg(response=("response", "mean"))
            .dropna(subset=["response"])
        )
        y = aggregated["response"].to_numpy(dtype=np.float64)
        if y.size:
            q25, q75 = np.percentile(y, [25, 75])
            y_iqr = float(q75 - q25)
        else:
            y_iqr = float("nan")
        y_sd = float(np.std(y, ddof=1)) if y.size > 1 else float("nan")
        rows.append(
            {
                "analysis_id": str(analysis_id),
                "analysis_unit": str(analysis_unit),
                "drug_name": str(drug_name),
                "source": str(source),
                "drug_id": "" if pd.isna(drug_id) else str(drug_id),
                "response_name": str(response_name),
                "n_matched": int(y.size),
                "response_sd": y_sd,
                "response_iqr": y_iqr,
                "response_min": float(np.min(y)) if y.size else float("nan"),
                "response_max": float(np.max(y)) if y.size else float("nan"),
            }
        )

    out = pd.DataFrame(rows)
    out["eligible"] = (out["n_matched"] >= min_samples) & (out["response_sd"] >= min_response_sd)
    out = out.sort_values(["eligible", "n_matched", "response_sd"], ascending=[False, False, False])
    return out.reset_index(drop=True)


def save_drug_dataset(
    selected_row: dict[str, object],
    responses: pd.DataFrame,
    X: np.ndarray,
    expression_ids: np.ndarray,
    gene_names: np.ndarray,
    response_name: str,
) -> Path:
    analysis_id = str(selected_row["analysis_id"])
    drug_name = str(selected_row["drug_name"])
    source = str(selected_row.get("source", ""))
    drug_id = str(selected_row.get("drug_id", ""))
    analysis_unit = str(selected_row.get("analysis_unit", "source_drug"))

    drug_response = responses[responses["analysis_id"] == analysis_id].copy()
    drug_response = (
        drug_response.groupby("cosmic_id", as_index=False)
        .agg(response=("response", "mean"), cell_line_name=("cell_line_name", "first"))
        .dropna(subset=["response"])
    )
    response_map = dict(zip(drug_response["cosmic_id"], drug_response["response"]))

    row_indices = [idx for idx, cosmic_id in enumerate(expression_ids) if cosmic_id in response_map]
    y = np.array([response_map[expression_ids[idx]] for idx in row_indices], dtype=np.float64)
    X_drug = X[row_indices]
    ids = expression_ids[row_indices]
    y_sd = float(np.std(y, ddof=1)) if y.size > 1 else float("nan")

    output_path = PROCESSED_DIR / f"gdsc_{slugify(analysis_id)}.npz"
    np.savez_compressed(
        output_path,
        X=X_drug.astype(np.float32),
        y=y,
        cell_line_ids=ids.astype(str),
        gene_names=gene_names.astype(str),
        drug_name=np.array(drug_name),
        source=np.array(source),
        drug_id=np.array(drug_id),
        analysis_id=np.array(analysis_id),
        analysis_unit=np.array(analysis_unit),
        response_name=np.array(response_name),
        n_matched=np.array(int(y.size)),
        response_sd=np.array(y_sd),
    )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drug-name", default=None, help="Optional legacy drug-name filter.")
    parser.add_argument(
        "--drug-names",
        default=None,
        help="Comma-separated exact drug names to prepare. When set, --num-drugs is ignored.",
    )
    parser.add_argument("--num-drugs", type=int, default=3, help="Number of high-coverage drugs to prepare.")
    parser.add_argument("--response", default="LN_IC50", choices=["LN_IC50", "AUC"], help="Preferred response column.")
    parser.add_argument(
        "--analysis-unit",
        default="source_drug",
        choices=["drug_name", "source_drug"],
        help="Drug analysis unit. Use source_drug for publication runs.",
    )
    parser.add_argument("--source", default=None, help="Optional GDSC source filter, e.g. GDSC1 or GDSC2.")
    parser.add_argument("--drug-id", default=None, help="Optional GDSC drug ID filter for disambiguation.")
    parser.add_argument("--min-samples", type=int, default=50, help="Minimum matched cell lines per drug.")
    parser.add_argument("--min-response-sd", type=float, default=0.0, help="Minimum matched response SD per drug.")
    parser.add_argument("--all-eligible", action="store_true", help="Prepare all drugs meeting eligibility criteria.")
    parser.add_argument("--max-drugs", type=int, default=None, help="Optional cap for --all-eligible.")
    parser.add_argument(
        "--eligible-output",
        default=str(PROCESSED_DIR.parent.parent / "results" / "gdsc_eligible_drugs.csv"),
        help="CSV path for the eligible-drug table.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    require_files()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X, expression_ids, gene_names = load_expression()
    responses = load_responses(args.response, analysis_unit=args.analysis_unit)
    eligibility = summarize_drug_eligibility(
        responses=responses,
        expression_ids=set(expression_ids.astype(str)),
        min_samples=args.min_samples,
        min_response_sd=args.min_response_sd,
    )
    eligible_output = Path(args.eligible_output)
    eligible_output.parent.mkdir(parents=True, exist_ok=True)
    eligibility.to_csv(eligible_output, index=False)
    selected = select_drugs(
        responses=responses,
        expression_ids=set(expression_ids.astype(str)),
        requested_drug_names=parse_drug_names(args.drug_names),
        requested_drug=args.drug_name,
        num_drugs=args.num_drugs,
        min_samples=args.min_samples,
        min_response_sd=args.min_response_sd,
        all_eligible=args.all_eligible,
        max_drugs=args.max_drugs,
        source_filter=args.source,
        drug_id_filter=args.drug_id,
    )

    response_name = responses["response_name"].mode().iloc[0]
    print(f"Expression matrix: {X.shape[0]} cell lines x {X.shape[1]} genes after zero-variance filtering")
    print(f"Response column: {response_name}")
    print(f"Analysis unit: {args.analysis_unit}")
    print(f"Saved eligible-drug table to {eligible_output}")
    print("Selected drugs:")
    metadata_rows = []
    for row in selected:
        output_path = save_drug_dataset(row, responses, X, expression_ids, gene_names, response_name)
        with np.load(output_path, allow_pickle=True) as data:
            metadata_rows.append(
                {
                    "drug_name": str(data["drug_name"]),
                    "source": str(data["source"]),
                    "drug_id": str(data["drug_id"]),
                    "analysis_id": str(data["analysis_id"]),
                    "analysis_unit": str(data["analysis_unit"]),
                    "response_name": str(data["response_name"]),
                    "n_matched": int(data["n_matched"]),
                    "response_sd": float(data["response_sd"]),
                    "path": str(output_path),
                }
            )
            print(
                f"  {data['analysis_id']}: matched={int(data['n_matched'])}, "
                f"saved_rows={data['X'].shape[0]}, genes={data['X'].shape[1]}, file={output_path}"
            )
    metadata_path = eligible_output.parent / "gdsc_prepared_metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    print(f"Saved prepared metadata to {metadata_path}")


if __name__ == "__main__":
    main()
