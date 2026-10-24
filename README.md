# Concrete strength regression

- `symbolic_regression.py`: symbolic regression on ten original features, without PCA.
- `linear_regression.py`: linear regression on the same features and split.
- `analysis_common.py`: shared data loading, statistics, and plotting.
- `data.xlsx`: the dataset. `models/`: two saved equations and their preprocessing.
- `tests/`: reference-result and portability checks.

## Run

Use Python 3.9. From the project folder:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python symbolic_regression.py
python linear_regression.py
```

On Windows, create the environment with `py -3.9 -m venv .venv` and use
`.venv\Scripts\python.exe` instead of activating it.

Defaults evaluate the saved models and write to `results/<model>/`. Each run writes:

- `metrics.csv`: MSE, RMSE, MAE, R², Pearson r and p-value.
- `predictions.csv`: train/test predictions, row indices, and residuals.
- `sensitivity.csv`: mean absolute analytical derivatives in scaled and original units.
- `predictions.svg`, `main_effects.svg`, `sensitivity.svg`.
- `equation.txt`: equation expressed in original input units.

Main-effect curves vary one feature while holding the others at training medians.
Sensitivity is local derivative sensitivity, not a causal or global importance measure.

## Train

```bash
python linear_regression.py --train
python -m pip install -r requirements-sr.txt
python symbolic_regression.py --train
```

SR uses seed 42, serial execution, Julia 1.10.10, and SymbolicRegression.jl 1.11.3.
Julia is installed on first training; saved-model evaluation needs no Julia.
On macOS, serial training disables Julia signal handling to avoid an Apple Python conflict.
PySR prints warnings about these intentional single-thread settings.

Training fits scaling only on training rows and keeps test data out of equation selection.
Both models use an 85/15 split with seed 42. Training also writes `model.json`;
SR writes `candidate_equations.csv`. Temporary search files are deleted automatically.

Options: `--data`, `--model`, `--output`; SR also accepts `--iterations` (default 800)
and `--maxsize` (default 120). To evaluate a newly trained SR model:

```bash
python symbolic_regression.py --model results/symbolic_regression/model.json
```

## Reproduce and move

Copy the source files, requirements files, dataset, `models/`, and `tests/`.
Recreate `.venv` on the destination; do not copy installed dependencies or caches.
Default paths are relative to the scripts, so execution works from any directory.

Saved-model replay preserves the reference results (test R²: SR 0.81389149,
LR 0.55898703). Across devices, small floating-point differences are possible.
The historical SR model used full-dataset scaling and test-based equation selection;
its saved metrics are historical reproduction, not an unbiased holdout comparison.
New SR training corrects those choices and will produce a different equation.

```bash
python -m unittest discover -s tests -v
```

Tests check reference metrics, repeatable outputs after relocation, LR refitting,
analytical derivatives, and rejection of changed data.
