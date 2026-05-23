# GDSC Real-Data Application

This folder adds a real biomedical application for factor-augmented CINDES.

The scientific task is conditional density estimation for cancer drug response:

```text
X = high-dimensional basal gene expression profile of a cancer cell line
Y = continuous drug response, preferably LN_IC50
```

The goal is to estimate the full conditional distribution `p(Y | X)`, not only a conditional mean. This is a pharmacogenomic precision-oncology setting: given a transcriptomic profile, estimate the distribution of drug sensitivity.

This application is not intended to prove clinical utility. It illustrates distributional prediction of cancer drug response in a high-dimensional biomedical setting.

## Data

The scripts use the Genomics of Drug Sensitivity in Cancer (GDSC) release 8.5 files:

- `GDSC1_fitted_dose_response_27Oct23.xlsx`
- `GDSC2_fitted_dose_response_27Oct23.xlsx`
- `Cell_Lines_Details.xlsx`
- `Cell_line_RMA_proc_basalExp.txt.zip`

Run:

```bash
.venv/bin/python real_app_gdsc/download_gdsc.py
```

The original GDSC expression URL may return HTTP 410 on some systems. The downloader tries the original GDSC URL first and then a public Argonne CANDLE/IMPROVE mirror for the same expression zip. If all direct URLs fail, download the missing files manually from the GDSC bulk download page and place them in:

```text
real_app_gdsc/data/raw/
```

## Prepare Matched Drug Datasets

Before preparing final datasets, screen drugs for expression-response signal:

```bash
.venv/bin/python real_app_gdsc/screen_drugs.py \
  --top-genes 2000 \
  --num-factors 10 \
  --output real_app_gdsc/results/drug_screening_table.csv
```

The screening table ranks drugs by matched sample size, response variation, and quick 5-fold PCA/Ridge CV signal. Drug screening is recommended before the final application; the highest-coverage drug is not necessarily the best scientific example.

Prepare one or more drug-specific datasets:

```bash
.venv/bin/python real_app_gdsc/prepare_gdsc.py --num-drugs 3
```

Or prepare explicit drugs selected from screening:

```bash
.venv/bin/python real_app_gdsc/prepare_gdsc.py \
  --drug-names "DrugA,DrugB,DrugC"
```

The script:

- loads expression with rows as cell lines and columns as genes,
- removes annotation and zero-variance genes,
- loads GDSC1/GDSC2 fitted dose response files,
- uses `LN_IC50` when available,
- merges expression and response by `COSMIC_ID`,
- preserves `source`, `DRUG_ID`, and `DRUG_NAME`,
- defines source-specific publication analysis units as
  `source::drug_id::drug_name`,
- saves matched `.npz` files in `real_app_gdsc/data/processed/`.

`LN_IC50` is a continuous drug-response measure; lower `LN_IC50` generally indicates higher sensitivity.

## Methods

The application compares:

- `plugin_pca_cindes`: top variable genes -> training-set standardization -> PCA factors -> CINDES.
- `raw_x_cindes`: top variable genes -> training-set standardization -> CINDES directly on gene expression.
- `gaussian_elasticnet_baseline`: ElasticNet mean model plus Gaussian residual density.
- `pca_gaussian_baseline`: PCA factors plus Ridge Gaussian residual density.
- `pca_mdn`: PCA factors plus a mixture density network baseline, with analytic
  Gaussian-mixture CDFs for PIT and prediction intervals.

Real data does not have a known true conditional density, so integrated TV against truth cannot be computed. The evaluation metrics are:

- held-out negative log-likelihood, where lower is better,
- PIT Kolmogorov-Smirnov statistic against Uniform(0,1), where lower is better,
- 90% prediction interval coverage, which should be close to 0.90,
- mean 90% interval width, which must be considered together with coverage.

## Smoke Test

The smoke test is only pipeline validation, not a scientific result. It uses one drug, one seed, few epochs, a small gene subset, and no raw-X CINDES.

```bash
.venv/bin/python real_app_gdsc/download_gdsc.py

.venv/bin/python real_app_gdsc/prepare_gdsc.py \
  --num-drugs 1

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python real_app_gdsc/run_gdsc_application.py \
  --num-drugs 1 \
  --num-seeds 1 \
  --top-genes 500 \
  --num-factors 5 \
  --epochs 5 \
  --patience 2 \
  --grid-size 101 \
  --skip-raw-x \
  --overwrite
```

Then create tables and figures:

```bash
.venv/bin/python real_app_gdsc/make_gdsc_figures.py
```

## Selected-Drug Application

After screening and preparing selected drugs, run plug-in PCA CINDES and Gaussian baselines first:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python real_app_gdsc/run_gdsc_application.py \
  --drug-names "DrugA,DrugB,DrugC" \
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

Create figures from that result file with:

```bash
.venv/bin/python real_app_gdsc/make_gdsc_figures.py \
  --input real_app_gdsc/results/gdsc_real_v1.csv
```

Then consider a smaller raw-X pilot for one selected drug:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python real_app_gdsc/run_gdsc_application.py \
  --drug-names "DrugA" \
  --num-seeds 3 \
  --top-genes 2000 \
  --raw-x-top-genes 500 \
  --num-factors 10 \
  --epochs 150 \
  --patience 20 \
  --grid-size 401 \
  --methods plugin_pca_cindes,raw_x_cindes,gaussian_elasticnet_baseline,pca_gaussian_baseline \
  --output real_app_gdsc/results/gdsc_rawx_pilot.csv \
  --overwrite
```

## All-Eligible Benchmark

The manuscript benchmark uses source-specific GDSC drug-screen units satisfying
minimum sample-size and response-variation criteria:

```bash
.venv/bin/python real_app_gdsc/prepare_gdsc.py \
  --analysis-unit source_drug \
  --all-eligible \
  --min-samples 120 \
  --min-response-sd 0.2 \
  --eligible-output real_app_gdsc/results/gdsc_eligible_drugs.csv
```

Run the benchmark with resumable writes:

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
  --mdn-components 2,3,5 \
  --output real_app_gdsc/results/gdsc_all_drugs_metrics.csv \
  --status-report reports/gdsc_all_drugs_status.md \
  --resume
```

Create all-drug summary figures and LaTeX tables:

```bash
.venv/bin/python real_app_gdsc/make_gdsc_figures.py \
  --input real_app_gdsc/results/gdsc_all_drugs_metrics.csv \
  --results-dir real_app_gdsc/results \
  --figures-dir real_app_gdsc/figures/all_drugs \
  --tables-dir tables \
  --output-prefix gdsc_all_drugs \
  --all-drug-outputs
```

## Factor-Dimension Sensitivity

The factor-dimension sensitivity analysis uses the three case-study drugs plus
30 fixed representative source-specific units:

```bash
.venv/bin/python real_app_gdsc/select_factor_sensitivity_units.py \
  --eligible-input real_app_gdsc/results/gdsc_eligible_drugs.csv \
  --case-drugs Trametinib,Venetoclax,Vorinostat \
  --n-extra 30 \
  --seed 20260524 \
  --output real_app_gdsc/results/gdsc_factor_dimension_units.csv
```

Then run `run_gdsc_application.py` with `--analysis-ids-file
real_app_gdsc/results/gdsc_factor_dimension_units.csv` for each value of
`--num-factors` in `{3, 5, 10, 15, 20}`, and summarize with:

```bash
.venv/bin/python real_app_gdsc/make_factor_dimension_figures.py \
  --input real_app_gdsc/results/gdsc_factor_dimension_sensitivity.csv \
  --results-dir real_app_gdsc/results \
  --figures-dir real_app_gdsc/figures \
  --tables-dir tables
```
