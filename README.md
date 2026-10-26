# Concrete strength regression

- `symbolic_regression.py`: symbolic regression on the ten original features, without PCA.
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
python symbolic_regression.py --legacy
python linear_regression.py
```

On Windows, create the environment with `py -3.9 -m venv .venv` and use
`.venv\Scripts\python.exe` instead of activating it.

`linear_regression.py` evaluates its saved model by default. `symbolic_regression.py`
trains a new model with the revised selection strategy by default. Each run writes:

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
python symbolic_regression.py
```

SR uses Julia 1.10.10 and SymbolicRegression.jl 1.11.3. Training uses
shared-memory multithreading by default, capped at eight threads to avoid overwhelming a
laptop. Use `--workers` to choose a different thread count or `--workers 1` for deterministic
serial execution with seed 42. Parallel searches are stochastic because PySR only guarantees
seeded, bit-for-bit deterministic behavior in serial mode.
The maximum expression size is warmed up over the first quarter of the search to spend less
time evaluating very large expressions before simpler candidates have evolved.
Julia is installed on first training; saved-model evaluation needs no Julia.

Revised SR training follows the original workflow's 85/15 random row split with seed 42,
but fits min-max scaling on the training rows only. Test observations do not participate in
preprocessing, search, physical auditing, or equation selection. Training writes
`model.json`, `candidate_equations.csv`, and the final reports under
`results/symbolic_regression/`.

PySR fits the ten original variables directly. Every eligible equation must contain all ten
features. The search permits addition, subtraction, multiplication, and squares, while
operator constraints and an exact SymPy audit restrict accepted equations to polynomial
degree two, single-variable squares, and at most five pairwise interaction terms. Nested
higher-order interactions are rejected.

Physical checks are restricted to interpolation: observed training rows and one-at-a-time
sweeps between each feature's training minimum and maximum. Equations must remain finite and
nonnegative and preserve a nonincreasing water/binder response in that domain. There is no
extrapolation buffer or upper prediction ceiling. Among eligible equations, selection uses
the highest PySR accuracy-complexity score, with lower complexity and training loss as
tie-breakers. Test metrics are calculated only after the equation has been selected.

The PySR hall of fame is retained under `results/symbolic_regression/search_run/`, and the
complete audit is written before selection. If no equation passes, the program exits with a
concise message while preserving the search and audit.

Options: `--data`, `--model`, `--output`; SR also accepts `--iterations` (default 800),
`--maxsize` (default 50), and `--workers`. The search uses 32-bit evaluation and at least two
populations per worker. PySR displays its progress bar and evolving equation table during
training. To evaluate a newly trained SR model without starting another search:

```bash
python symbolic_regression.py --model results/symbolic_regression/model.json
```

The published historical model and its outputs are retained under `models/legacy/` and
`results/legacy/`. Reproduce them without starting PySR using:

```bash
python symbolic_regression.py --legacy
```

## Reproduce and move

Copy the source files, requirements files, dataset, `models/`, and `tests/`.
Recreate `.venv` on the destination; do not copy installed dependencies or caches.
Default paths are relative to the scripts, so execution works from any directory.

Saved-model replay preserves the reference results (test R²: SR 0.81389149,
LR 0.55898703). Across devices, small floating-point differences are possible.
The legacy SR model used full-dataset scaling and test-based equation selection; its saved
metrics are historical reproduction, not an unbiased holdout comparison.
New SR training corrects those choices and will produce a different equation.

```bash
python -m unittest discover -s tests -v
```

Tests check reference metrics, repeatable outputs after relocation, LR refitting,
analytical derivatives, changed-data rejection, and the SR candidate-selection gates.
