# Factor-Augmented CINDES

This repository contains simulation and real-data code for factor-augmented
classification-induced neural density estimation (CINDES).

The code accompanies the manuscript "Factor-Augmented Neural Conditional
Density Estimation with Application to Drug Response Prediction". The empirical
application uses source-specific GDSC drug-screen units and high-dimensional
basal gene expression to estimate full conditional distributions of continuous
drug response.

## Repository Contents

- `fac_cindes/`: simulation data-generating mechanisms, CINDES models, training
  utilities, and simulation figure/table scripts.
- `real_app_gdsc/`: GDSC download, preprocessing, leakage checks, all-drug
  benchmark, MDN baseline, and sensitivity-analysis scripts.
- `real_app_gdsc/results/`: curated CSV outputs used for the manuscript tables
  and figures. Raw GDSC data and prepared `.npz` matrices are not committed.
- `real_app_gdsc/figures/`: PDF figures generated from the curated result CSVs.
- `tables/`: LaTeX-ready tables.
- `reports/`: leakage audit, benchmark completeness reports, and sensitivity
  summaries.

## Environment

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The commands below assume they are run from the repository root.

## Simulation

Smoke run:

```bash
.venv/bin/python -m fac_cindes.run_simulation \
  --n-values 200 \
  --p-values 100 \
  --num-seeds 1 \
  --epochs 5 \
  --patience 2 \
  --output fac_cindes/results/smoke_simulation.csv \
  --overwrite
```

Main simulation:

```bash
.venv/bin/python -m fac_cindes.run_simulation \
  --n-values 500,1000,2000 \
  --p-values 100,500,1000 \
  --num-seeds 10 \
  --output fac_cindes/results/final_v1.csv \
  --overwrite
```

Simulation tables and figures:

```bash
.venv/bin/python fac_cindes/make_tables_figures.py \
  --input fac_cindes/results/final_v1.csv
```

## GDSC Data

Download the raw GDSC files:

```bash
.venv/bin/python real_app_gdsc/download_gdsc.py
```

Prepare selected case-study drugs:

```bash
.venv/bin/python real_app_gdsc/prepare_gdsc.py \
  --analysis-unit source_drug \
  --drug-names "Trametinib,Venetoclax,Vorinostat"
```

Run the selected-drug real-data application without raw-X CINDES:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python real_app_gdsc/run_gdsc_application.py \
  --drug-names "Trametinib,Venetoclax,Vorinostat" \
  --num-seeds 5 \
  --top-genes 2000 \
  --num-factors 10 \
  --epochs 150 \
  --patience 20 \
  --grid-size 401 \
  --skip-raw-x \
  --output real_app_gdsc/results/gdsc_real_v1.csv \
  --overwrite
```

Run the same selected-drug analysis with the PCA-space MDN baseline included:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python real_app_gdsc/run_gdsc_application.py \
  --drug-names "Trametinib,Venetoclax,Vorinostat" \
  --num-seeds 5 \
  --top-genes 2000 \
  --num-factors 10 \
  --epochs 150 \
  --patience 20 \
  --grid-size 401 \
  --methods plugin_pca_cindes,gaussian_elasticnet_baseline,pca_gaussian_baseline,pca_mdn \
  --mdn-components 2,3,5 \
  --output real_app_gdsc/results/gdsc_real_v1_with_mdn.csv \
  --overwrite
```

Prepare all eligible drugs with pre-specified eligibility criteria. This can
create many per-drug `.npz` files because each prepared drug stores its matched
expression matrix; check available disk space before running without
`--max-drugs`.

```bash
.venv/bin/python real_app_gdsc/prepare_gdsc.py \
  --analysis-unit source_drug \
  --all-eligible \
  --min-samples 120 \
  --min-response-sd 0.2 \
  --eligible-output real_app_gdsc/results/gdsc_eligible_drugs.csv
```

Run an all-eligible benchmark from already prepared `.npz` files:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python real_app_gdsc/run_gdsc_application.py \
  --all-eligible \
  --min-samples 120 \
  --min-response-sd 0.2 \
  --num-seeds 5 \
  --top-genes 2000 \
  --num-factors 10 \
  --epochs 150 \
  --patience 20 \
  --grid-size 401 \
  --methods plugin_pca_cindes,gaussian_elasticnet_baseline,pca_gaussian_baseline,pca_mdn \
  --output real_app_gdsc/results/gdsc_all_drugs_metrics.csv \
  --resume
```

Use `--overwrite` only for deliberate clean reruns. For long all-eligible
benchmarks, prefer `--resume`; completed keys are
`(analysis_id, seed, method, num_factors, top_genes, grid_size)`.

Generate GDSC tables and figures:

```bash
.venv/bin/python real_app_gdsc/make_gdsc_figures.py \
  --input real_app_gdsc/results/gdsc_all_drugs_metrics.csv \
  --results-dir real_app_gdsc/results \
  --figures-dir real_app_gdsc/figures/all_drugs \
  --tables-dir tables \
  --output-prefix gdsc_all_drugs \
  --all-drug-outputs
```

Run the factor-dimension sensitivity analysis on the pre-specified subset:

```bash
.venv/bin/python real_app_gdsc/select_factor_sensitivity_units.py \
  --eligible-input real_app_gdsc/results/gdsc_eligible_drugs.csv \
  --case-drugs Trametinib,Venetoclax,Vorinostat \
  --n-extra 30 \
  --seed 20260524 \
  --output real_app_gdsc/results/gdsc_factor_dimension_units.csv

for r in 3 5 10 15 20; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python real_app_gdsc/run_gdsc_application.py \
    --all-eligible \
    --analysis-ids-file real_app_gdsc/results/gdsc_factor_dimension_units.csv \
    --min-samples 120 \
    --min-response-sd 0.2 \
    --num-seeds 5 \
    --top-genes 2000 \
    --num-factors "$r" \
    --epochs 150 \
    --patience 20 \
    --grid-size 401 \
    --methods plugin_pca_cindes,pca_gaussian_baseline,pca_mdn \
    --mdn-components 2,3,5 \
    --output real_app_gdsc/results/gdsc_factor_dimension_sensitivity.csv \
    --status-report "reports/gdsc_factor_dimension_sensitivity_r${r}_status.md" \
    --no-save-density-examples \
    --resume
done

.venv/bin/python real_app_gdsc/make_factor_dimension_figures.py \
  --input real_app_gdsc/results/gdsc_factor_dimension_sensitivity.csv \
  --results-dir real_app_gdsc/results \
  --figures-dir real_app_gdsc/figures \
  --tables-dir tables
```

## Leakage Checks

Drug-screening response-shuffle check:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python real_app_gdsc/screen_drugs.py \
  --drug-names "Trametinib,Venetoclax,Vorinostat" \
  --shuffle-y-check \
  --shuffle-output real_app_gdsc/results/drug_screening_shuffle_check.csv
```

The formal leakage audit is documented in `reports/leakage_audit.md`.

## Output Directories

- `fac_cindes/results/`: simulation CSVs
- `fac_cindes/figures/`: simulation figures
- `real_app_gdsc/results/`: GDSC CSVs and selected diagnostic artifacts
- `real_app_gdsc/figures/`: GDSC manuscript figures for extended analyses
- `tables/`: LaTeX-ready tables
- `reports/`: audit and reproducibility reports

Generated data, figures, and large raw GDSC files should not be committed unless
explicitly needed for manuscript archival.

The committed GDSC result CSVs are small enough for GitHub and correspond to the
manuscript analyses. They do not contain raw gene-expression matrices or raw
dose-response spreadsheets.

## Clean Code Archive

On macOS, set `COPYFILE_DISABLE=1` while packaging so tar does not emit
AppleDouble `._*` metadata files:

```bash
COPYFILE_DISABLE=1 tar -czf cindes_code_clean.tar.gz \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.DS_Store' \
  --exclude='._*' \
  --exclude='*/._*' \
  --exclude='fac_cindes/results' \
  --exclude='fac_cindes/figures' \
  --exclude='fac_cindes/tables' \
  --exclude='real_app_gdsc/data' \
  --exclude='real_app_gdsc/results' \
  --exclude='real_app_gdsc/figures' \
  --exclude='results' \
  --exclude='figures' \
  --exclude='tables' \
  --exclude='*.zip' \
  --exclude='*.tar.gz' \
  README.md requirements.txt .gitignore fac_cindes real_app_gdsc reports
```

Confirm archive hygiene before packaging:

```bash
find . -name '._*' -o -name '.DS_Store'
```
