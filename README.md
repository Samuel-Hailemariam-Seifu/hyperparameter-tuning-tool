# Hyperparameter tuning tool

Small CLI to run **grid search**, **randomized search**, or **Optuna** on tabular CSV data (or built-in sklearn toy datasets).

## Install

```bash
pip install -e .
```

## Web UI (for a quick, plain-language tour)

```bash
pip install -e .
hpt-ui
```

Opens a browser page that explains what tuning does and lets you pick a demo dataset or upload a CSV.

## CLI usage

```bash
# Built-in breast cancer dataset, random forest, 5-fold CV, randomized search
hpt-tune --dataset breast_cancer --model random_forest --method random --n-iter 24

# Your CSV (last column or use --target)
hpt-tune --data ./train.csv --target label --model gradient_boosting --method optuna --n-trials 40
```

Options:

- `--method`: `grid` | `random` | `optuna`
- `--model`: `random_forest` | `gradient_boosting` | `logistic_regression` | `svc_rbf`
- `--dataset`: `iris` | `wine` | `breast_cancer` (used when `--data` is omitted)
- `--metric`: `accuracy` | `f1_macro` | `roc_auc` (binary/multiclass where applicable)
- `--output`: write JSON summary path

## License

MIT
