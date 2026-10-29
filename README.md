# Concrete strength regression

## Abstract

This project investigates interpretable prediction of the compressive strength
of concrete containing iron tailings using linear, additive quadratic polynomial,
and symbolic regression. Models relate strength to oxide composition, water–binder
ratio, fineness modulus, and mixture quantities using a reproducible 85/15
training/test row split and feature scaling fitted on training data. Symbolic
regression searches for explicit equations with PySR, then selects equations
using training performance, feature coverage, and complexity before reporting
test performance. Analytical sensitivities and composition-constrained
main-effect curves describe model behavior and its robustness across selected
equations.

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
