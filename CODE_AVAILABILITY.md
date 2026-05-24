# Code and Data Availability

This repository contains the code and curated computational outputs supporting
the manuscript "Factor-Augmented Neural Conditional Density Estimation with
Application to Drug Response Prediction".

## Included

- Simulation code for oracle, plug-in PCA, and raw-X CINDES experiments.
- GDSC download and preprocessing scripts.
- Source-specific GDSC all-eligible benchmark scripts.
- PCA-space mixture density network baseline code.
- Leakage audit and run-completeness reports.
- Manuscript-ready CSV results, PDF figures, and LaTeX tables.

## Not Included

The repository does not commit raw GDSC spreadsheets, raw gene-expression
matrices, or prepared per-drug `.npz` matrices. These are either public external
data files or large intermediate artifacts. They can be downloaded and recreated
using the scripts in `real_app_gdsc/`.

## Main Reproducibility Entry Points

- `README.md`: end-to-end commands.
- `real_app_gdsc/README.md`: GDSC application details.
- `reports/leakage_audit.md`: leakage controls.
- `reports/gdsc_all_drugs_status.md`: all-eligible benchmark completeness.
- `reports/gdsc_factor_dimension_sensitivity_report.md`: factor-dimension
  stability summary.

## Notes for Review

The committed CSV files are sufficient to regenerate the manuscript GDSC tables
and figures without rerunning the full benchmark. Re-running the full
all-eligible benchmark requires downloading GDSC data and can take substantial
time depending on hardware.
