# GDSC Leakage Audit

Date: 2026-05-22

## Scope

This audit inspected the GDSC data-preparation, drug-screening, and real-data
application pipeline:

- `real_app_gdsc/prepare_gdsc.py`
- `real_app_gdsc/screen_drugs.py`
- `real_app_gdsc/run_gdsc_application.py`
- `fac_cindes/train.py`
- `fac_cindes/models.py`

The audit focuses on whether held-out test responses or held-out test
covariates are used during training, tuning, feature selection, response-grid
construction, reference-density construction, or model selection.

## Findings

No test-set leakage was found in the current selected-drug GDSC pipeline.

### Data Preparation

`prepare_gdsc.py` loads expression rows by `COSMIC_ID` and preserves the GDSC
screen source and drug identifier when available. Publication-oriented runs use
the source-specific drug-screen unit
`analysis_id = source::drug_id::drug_name` when a drug ID is available and
`analysis_id = source::drug_name` otherwise. This avoids silently averaging
GDSC1 and GDSC2 measurements that share a drug name but may correspond to
different screen contexts.

Within each `analysis_id`, duplicated drug-response measurements are aggregated
by `COSMIC_ID` before saving one row per matched cell line. This aggregation
happens before train/test splitting, but it does not mix train and test rows
because each `COSMIC_ID` appears once in the prepared `.npz` file and the later
split is performed over these unique rows.

Checked functions:

- `load_expression`
- `load_responses`
- `save_drug_dataset`
- `source_drug_analysis_id`
- `add_analysis_id`

Conclusion: no train/test leakage was found in preparation. The saved modeling
unit is a unique matched cell line for a source-specific drug-screen unit.

### Drug Screening

`screen_drugs.py` screens drugs using a 5-fold PCA/Ridge pipeline. Within each
fold:

- top variable genes are selected from `X_train_raw` only;
- standardization moments are computed from the training fold only;
- PCA is fit on the standardized training fold only;
- Ridge is fit on the PCA training scores only;
- validation rows are transformed using the training-fitted gene subset,
  standardization, and PCA objects.

Checked functions:

- `cross_validated_pca_ridge`
- `select_top_genes`
- `standardize_train_test`
- `matched_drug_arrays`

Additional sanity check already run:

| Drug | Original CV R2 | Shuffled CV R2 |
| --- | ---: | ---: |
| Trametinib | 0.4574 | -0.0144 |
| Venetoclax | 0.4298 | -0.0079 |
| Vorinostat | 0.4407 | 0.0033 |

Conclusion: the screening signal is not consistent with an obvious response
leakage artifact.

### Real-Data Train/Test Evaluation

`run_gdsc_application.py` performs the train/test split before all
training-fitted operations. After patching this audit, the split and
training-only preprocessing are explicit helper steps:

- `make_train_test_split`
- `fit_training_only_preprocess`

For each seed and drug:

- train/test row indices are asserted to be disjoint;
- top variable genes are selected from `X_train_raw` only;
- expression standardization is fit on training rows only;
- PCA is fit on training rows only;
- the CINDES response grid is constructed from `y_train` only;
- the CINDES reference density is estimated from `y_train` only;
- ElasticNetCV is fit on training rows only;
- RidgeCV is fit on training PCA scores only;
- test rows are used only for final metric evaluation.

The output diagnostic field `test_y_outside_grid_count` records whether any
held-out response falls outside the training-based response grid. In the
`gdsc_real_v1.csv` run, the total count was zero across all drugs, seeds, and
methods.

### CINDES Internal Validation

`fac_cindes/train.py` creates an internal validation split from the training
rows passed to `train_cindes`. It uses this internal validation subset for
early stopping only. The reference pseudo-responses are sampled from a
reference density estimated from the external training response vector.

Checked functions/classes:

- `train_cindes`
- `TrainingConfig`
- `ReferenceDensity.from_y_train`
- `Standardizer.fit`

Conclusion: CINDES early stopping does not use held-out test rows.

## Patches Applied

`real_app_gdsc/run_gdsc_application.py` was patched to make leakage controls
explicit:

1. Added `SplitData` and `TrainingOnlyPreprocess` helper dataclasses.
2. Added `make_train_test_split`, including assertions that train/test row
   indices are disjoint and partition the dataset.
3. Added `fit_training_only_preprocess`, which constructs selected genes,
   standardized matrices, response grid, and reference density from training
   data only.
4. Rewired `run_one_dataset` to use these helpers.

These changes do not alter the intended estimator; they make the existing
training-only sequence more auditable.

## Final Conclusion

For the current selected-drug GDSC analysis and the planned all-eligible
extension, no test response is used in training, tuning, gene filtering,
standardization, PCA fitting, reference density construction, response-grid
construction, or model selection. Test data enter only at final metric
evaluation.

The remaining empirical limitation is not leakage; it is scope. The current
manuscript still relies on selected drugs. The next planned extension is a
source-specific all-eligible-drug benchmark with the PCA-space MDN baseline.
