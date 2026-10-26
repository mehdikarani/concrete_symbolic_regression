"""Train and evaluate symbolic regression on the ten original features."""
import os
import warnings

import numpy as np
import pandas as pd
import sympy as sp
from sklearn.preprocessing import MinMaxScaler

from analysis_common import (FEATURES, ROOT, arguments, equation, load_model, make_model,
                             read_data, report, save_model, split_indices)

REQUIRED_FEATURES = tuple(FEATURES)
EXPECTED_DIRECTIONS = {'w_c': -1}
AUDIT_POINTS_PER_FEATURE = 41
MAX_DIRECTION_VIOLATION_FRACTION = 0.05
MAX_INTERACTIONS = 5
DEFAULT_POPULATIONS = 31


class CandidateSelectionError(ValueError):
    """Raised after a complete candidate audit has been saved."""


def parallel_settings(workers):
    """Build PySR parallel settings from one user-facing worker count."""
    if workers < 1:
        raise ValueError('workers must be at least 1.')
    parallel = workers > 1
    return {'parallelism': 'multithreading' if parallel else 'serial',
            'procs': None,
            'populations': max(DEFAULT_POPULATIONS, 2 * workers),
            'deterministic': not parallel}


def interpolation_points(Z):
    """Return observed and one-at-a-time points inside the training range."""
    baseline = np.median(Z, axis=0)
    points = [Z]
    for column in range(Z.shape[1]):
        low, high = Z[:, column].min(), Z[:, column].max()
        if low == high:
            continue
        sweep = np.tile(baseline, (AUDIT_POINTS_PER_FEATURE, 1))
        sweep[:, column] = np.linspace(low, high, AUDIT_POINTS_PER_FEATURE)
        points.append(sweep)
    return np.vstack(points)


def structural_audit(expression, symbols):
    """Allow constants, linear terms, squares, and sparse pairwise interactions."""
    failures = []
    interaction_count = degree = 0
    try:
        polynomial = sp.Poly(sp.expand(expression), *symbols)
        degree = polynomial.total_degree()
        for monomial, _ in polynomial.terms():
            variables = sum(power > 0 for power in monomial)
            if sum(monomial) > 2 or variables > 2:
                failures.append('higher_order_polynomial')
                break
            if variables == 2:
                interaction_count += 1
    except sp.PolynomialError:
        failures.append('nonpolynomial_term')
    if degree > 2:
        failures.append('polynomial_degree_above_two')
    if interaction_count > MAX_INTERACTIONS:
        failures.append('too_many_interactions')
    return failures, degree, interaction_count


def audit_candidate(expression, audit_points):
    """Check structure and physical behavior only inside the training domain."""
    expression, symbols, predict = equation(expression)
    expression = sp.simplify(expression)
    symbol_by_name = dict(zip(FEATURES, symbols))
    missing = [feature for feature in REQUIRED_FEATURES
               if symbol_by_name[feature] not in expression.free_symbols]
    failures = []
    if missing:
        failures.append('missing:' + '|'.join(missing))
    structure = structural_audit(expression, symbols)
    failures.extend(structure[0])

    try:
        predictions = predict(audit_points)
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return {'eligible': False, 'rejection_reason': 'nonfinite_prediction',
                'minimum_prediction': np.nan, 'maximum_prediction': np.nan,
                'polynomial_degree': structure[1],
                'interaction_count': structure[2],
                'max_direction_violation': np.nan}
    if predictions.min() < 0:
        failures.append('negative_prediction')

    max_violation = 0.0
    for feature, direction in EXPECTED_DIRECTIONS.items():
        try:
            derivative = sp.lambdify(
                symbols, sp.diff(expression, symbol_by_name[feature]), 'numpy')(*audit_points.T)
            derivative = np.broadcast_to(derivative, len(audit_points)).astype(float)
        except (TypeError, ValueError, FloatingPointError, OverflowError):
            failures.append(f'nonfinite_derivative:{feature}')
            continue
        if not np.isfinite(derivative).all():
            failures.append(f'nonfinite_derivative:{feature}')
            continue
        violation = np.mean(direction * derivative < -1e-10)
        max_violation = max(max_violation, float(violation))
        if violation > MAX_DIRECTION_VIOLATION_FRACTION:
            failures.append(f'wrong_direction:{feature}')

    return {'eligible': not failures,
            'rejection_reason': ';'.join(dict.fromkeys(failures)),
            'minimum_prediction': float(predictions.min()),
            'maximum_prediction': float(predictions.max()),
            'polynomial_degree': structure[1],
            'interaction_count': structure[2],
            'max_direction_violation': max_violation}


def audit_candidates(equations, Z_train):
    """Audit all equations without using validation or test observations."""
    audited = equations.copy().reset_index(drop=True)
    if 'sympy_format' not in audited:
        raise RuntimeError('This PySR expression type does not support the required SymPy export.')
    points = interpolation_points(Z_train)
    expressions = [sp.sympify(value) for value in audited['sympy_format']]
    diagnostics = [audit_candidate(value, points) for value in expressions]
    audited = pd.concat([audited, pd.DataFrame(diagnostics)], axis=1)
    audited['sympy_equation'] = [str(value) for value in expressions]
    audited['selected'] = False
    return audited


def choose_candidate(audited):
    """Choose the highest PySR score among eligible equations."""
    eligible = audited[audited['eligible']].copy()
    if eligible.empty:
        common = dict(list(audited['rejection_reason'].value_counts().items())[:5])
        raise CandidateSelectionError(
            f'No candidate passed the interpolation rules. Most common rejections: {common}')
    eligible['_score'] = pd.to_numeric(eligible['score'], errors='coerce').fillna(-np.inf)
    selected_index = eligible.sort_values(
        ['_score', 'complexity', 'loss'], ascending=[False, True, True]).index[0]
    audited.loc[selected_index, 'selected'] = True
    return sp.simplify(audited.loc[selected_index, 'sympy_format']), audited


def select_candidate(equations, Z_train):
    """Audit and select without consulting validation or test data."""
    return choose_candidate(audit_candidates(equations, Z_train))


def train(args, X, y):
    minimum_size = 2 * len(REQUIRED_FEATURES) - 1
    if args.iterations < 1 or args.maxsize < minimum_size:
        raise ValueError(
            f'iterations must be positive and maxsize must be at least {minimum_size}.')
    runtime = parallel_settings(args.workers)
    os.environ['PYTHON_JULIACALL_THREADS'] = str(args.workers)
    warnings.filterwarnings(
        'ignore', message='PYTHON_JULIACALL_THREADS environment variable.*')
    # Let PySR provision compatible Julia and SymbolicRegression.jl versions.
    from pysr import PySRRegressor

    train_indices, test_indices = split_indices(X)
    scaler = MinMaxScaler().fit(X[train_indices])
    Z_train = scaler.transform(X[train_indices])
    args.output.mkdir(parents=True, exist_ok=True)
    model = PySRRegressor(
        niterations=args.iterations, maxsize=args.maxsize,
        binary_operators=['+', '-', '*'], unary_operators=['square'],
        constraints={'*': (5, 5), 'square': 1},
        nested_constraints={'*': {'*': 1}, 'square': {'square': 0, '*': 0}},
        complexity_of_operators={'*': 2, 'square': 2},
        random_state=42 if runtime['deterministic'] else None,
        warmup_maxsize_by=.25, precision=32, model_selection='best',
        progress=True, verbosity=1,
        output_directory=str(args.output / 'search_run'), **runtime)
    model.fit(Z_train, y[train_indices], variable_names=FEATURES)

    candidates = audit_candidates(model.equations_, Z_train)
    columns = ['complexity', 'loss', 'score', 'equation', 'sympy_equation',
               'eligible', 'rejection_reason', 'minimum_prediction',
               'maximum_prediction', 'polynomial_degree', 'interaction_count',
               'max_direction_violation', 'selected']
    audit_path = args.output / 'candidate_equations.csv'
    candidates[columns].to_csv(audit_path, index=False)
    try:
        expression, candidates = choose_candidate(candidates)
    except CandidateSelectionError as error:
        raise CandidateSelectionError(f'{error} Full audit: {audit_path.resolve()}') from None
    candidates[columns].to_csv(audit_path, index=False)

    return make_model(args.data, expression, scaler, train_indices, test_indices,
                      {'seed': 42 if runtime['deterministic'] else None,
                       'test_size': 0.15, 'split': 'random_rows',
                       'scaling': 'training_only',
                       'selection': 'highest_pysr_score_with_required_features',
                       'required_features': list(REQUIRED_FEATURES),
                       'expected_directions': EXPECTED_DIRECTIONS,
                       'interpolation_audit': 'observed_and_one_at_a_time_training_range',
                       'maximum_interactions': MAX_INTERACTIONS,
                       'maximum_polynomial_degree': 2,
                       'parallelism': runtime['parallelism'],
                       'workers': args.workers,
                       'populations': runtime['populations'],
                       'deterministic': runtime['deterministic'],
                       'precision': 32,
                       'warmup_maxsize_by': .25,
                       'iterations': args.iterations, 'maxsize': args.maxsize})


def main():
    args = arguments('symbolic_regression')
    X, y = read_data(args.data)
    if args.legacy:
        if args.train or args.model:
            raise ValueError('--legacy cannot be combined with --train or --model.')
        model = load_model(ROOT / 'models' / 'legacy' / 'symbolic_regression.json', args.data)
    elif args.model and not args.train:
        model = load_model(args.model, args.data)
    else:
        try:
            model = train(args, X, y)
        except CandidateSelectionError as error:
            raise SystemExit(f'Symbolic-regression selection stopped: {error}') from None
    report(model, X, y, args.output)
    if not args.legacy and (args.train or args.model is None):
        save_model(args.output / 'model.json', model)


if __name__ == '__main__':
    main()
