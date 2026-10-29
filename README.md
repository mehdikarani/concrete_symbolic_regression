# Concrete strength regression

## Abstract

This project investigates interpretable prediction of the compressive strength
of concrete containing iron tailings using linear, additive quadratic polynomial,
and symbolic regression. Models relate strength to oxide composition, water–binder
ratio, fineness modulus, and mixture quantities using a reproducible 85/15
training/test row split and feature scaling fitted on training data. Symbolic
regression searches for explicit equations with PySR. Selection uses training
rows only and requires polynomial equations of degree at most nine containing
at least seven input features. Under the current command-line policy, candidates
within 15% of the lowest eligible study-weighted training MSE form the primary
pool; the equation with the highest PySR score is selected, with ties resolved
by lower training MSE and then lower complexity. Robustness analysis compares
up to six equations spanning distinct complexities within 25% of the lowest
eligible training MSE, including the primary equation. Equations are frozen
before test performance is reported. Analytical sensitivities and
composition-constrained main-effect curves describe model behavior and its
robustness across the selected equations.

## Project structure

```text
concrete_symbolic_regression/
├── data.xlsx                   # Concrete mixture and strength dataset
├── data_preprocessing.py       # Data loading, splitting, scaling, and study weights
├── linear_regression.py        # Linear baseline
├── polynomial_regression.py    # Additive quadratic baseline
├── symbolic_regression.py      # PySR equation search
├── sr_equation_selection.py    # Equation audit, selection, and evaluation
├── robustness_analysis.py      # Main-effect and sensitivity robustness
├── sr_plotting.py              # Selection and robustness figures
├── analysis_common.py          # Shared model fitting, metrics, and reporting
├── requirements.txt            # Python dependency pins
├── models/                     # Saved baseline models
├── results/                    # Generated models, tables, and plots
│   ├── linear_regression/
│   ├── polynomial_regression/
│   └── symbolic_regression/
│       ├── search/             # Timestamped search runs and Pareto candidates
│       └── selection/          # Frozen equations and evaluation reports
│           └── <run>/robustness_analysis/
│               └── sensitivity/ # Sensitivity tables and figures
└── Manuscript/                 # Manuscript and reviewer-response materials
```
