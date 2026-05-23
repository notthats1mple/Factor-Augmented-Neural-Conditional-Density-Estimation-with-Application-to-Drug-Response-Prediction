# Factor-Augmented CINDES Simulation

This project simulates three conditional density estimators:

1. `oracle`: CINDES using the true latent factors `F`.
2. `plugin_pca`: CINDES using PCA-estimated latent factors from high-dimensional `X`.
3. `raw_x`: CINDES using the original high-dimensional covariates `X`.

The binary classifier is trained to distinguish real pairs `(Z_i, Y_i)` from fake pairs `(Z_i, Y_tilde_i)`, where `Y_tilde_i` is drawn from the Gaussian reference density

```text
q = Normal(mean(Y_train), 1.5 * std(Y_train)).
```

With balanced real/fake classes, the classifier logit estimates `log f(y | z) / q(y)`. The recovered density is

```text
f_hat(y | z) = q(y) * exp(logit(z, y)),
```

followed by trapezoidal-rule normalization over the evaluation `y` grid. The reported TV metric uses the CINDES-paper convention `int |p_hat - p_true| dy`.

## Install

Create an environment with NumPy, PyTorch, scikit-learn, pandas, and matplotlib:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Full Simulation

From the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 fac_cindes/run_simulation.py --overwrite
```

Defaults:

- `n in {500, 1000, 2000}` as training sample sizes.
- `p in {100, 500}`.
- `r = 3`.
- `20` Monte Carlo seeds.
- `1000` independent test observations per configuration.
- MLP: hidden dimension `64`, depth `3`, ReLU activations.
- Adam learning rate `1e-3`.
- Maximum epochs `200` with early stopping.
- Device defaults to `auto`, which uses CUDA when available, Apple Silicon MPS when available, otherwise CPU.
- Methods default to `oracle,plugin_pca,raw_x`; use `--methods oracle` or another comma-separated subset for debugging.
- PyTorch CPU thread count defaults to `--torch-num-threads 1`.

Results are saved to:

```text
fac_cindes/results/simulation_results.csv
```

The script also prints a summary table with mean and standard error for test NLL and integrated TV distance grouped by method, `n`, and `p`.

## Quick Smoke Test

Use a smaller run while developing:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python fac_cindes/run_simulation.py \
  --n-values 50 \
  --p-values 20 \
  --num-seeds 1 \
  --test-size 50 \
  --epochs 3 \
  --patience 2 \
  --grid-size 101 \
  --methods oracle,plugin_pca,raw_x \
  --overwrite
```

## Data-Generating Process

The latent factor model is:

```text
F_i ~ N(0, I_r)
Lambda_jk ~ N(0, 1)
U_i ~ N(0, sigma_u^2 I_p)
X_i = Lambda F_i + U_i
```

The conditional response distribution is a two-component Gaussian mixture:

```text
pi(F) = sigmoid(F_1)
mu1(F) = F_1 + 0.5 * F_2^2
mu2(F) = -F_1 + 0.5 * sin(F_2)
sigma1 = 0.5
sigma2 = 0.7
Y | F ~ pi(F) * N(mu1(F), sigma1^2)
        + (1 - pi(F)) * N(mu2(F), sigma2^2)
```

## Output Columns

- `seed`, `n`, `p`, `r`
- `method`
- `nll`: average test negative log-likelihood
- `tv`: average `int |p_hat - p_true| dy` over the test points
- `factor_mse_train`: least-squares aligned factor MSE on the training set
- `factor_mse_test`: least-squares aligned factor MSE on the test set
- `epochs_trained`
- `best_val_loss`
- `stopped_early`
- `fit_eval_seconds`
