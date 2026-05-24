# Repository Deposit Guide

This repository is prepared as a reproducibility package for the manuscript
"Factor-Augmented Neural Conditional Density Estimation with Application to Drug
Response Prediction".

Statistics in Medicine expects supporting code, simulations, and data files used
to support the manuscript results to be archived or uploaded as Data Files in
the submission system. This repository can be used directly as the code
repository, or it can be packaged into a clean archive for Zenodo, OSF, or the
journal submission portal.

## Recommended Deposit Contents

Use a clean archive containing:

```text
code/
results/
figures/
tables/
reports/
README.md
CODE_AVAILABILITY.md
CITATION.cff
LICENSE
requirements.txt
```

The archive should include committed computational outputs needed to reproduce
the manuscript tables and figures, but should not include raw GDSC spreadsheets,
raw gene-expression matrices, prepared per-drug `.npz` matrices, virtual
environments, cache directories, or local machine metadata.

## Suggested Repository Platforms

- Zenodo: best for DOI minting.
- OSF: good for project-level documentation.
- GitHub + Zenodo: best if code will be actively maintained.

## Suggested Data and Code Availability Text

Replace the manuscript placeholder with repository-specific text like:

> The GDSC data are publicly available from the Genomics of Drug Sensitivity in
> Cancer resource. Code and processed analysis outputs used to reproduce the
> simulations, GDSC benchmark, factor-dimension sensitivity analysis,
> supplemental simulation, CINDES-versus-MDN heterogeneity analysis, and
> grid-tail sensitivity diagnostic are archived at [Repository name],
> DOI/URL: [insert persistent identifier].

## Build a Clean Archive

From the repository root, run:

```bash
bash scripts/create_submission_archive.sh
```

This creates:

```text
cindes_statsmed_submission_code_results.tar.gz
```

The script stages only tracked manuscript-support files and uses
`COPYFILE_DISABLE=1` on macOS so the archive does not include AppleDouble
`._*` files.

## Archive Hygiene Check

Before uploading an archive, verify that it does not contain local macOS
metadata:

```bash
tar -tzf cindes_statsmed_submission_code_results.tar.gz | grep -E '(^|/)(\._|\.DS_Store)' || true
```

The command should print nothing.
