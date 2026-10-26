"""Run symbolic regression on original features, without PCA."""
import os
import sys
from tempfile import TemporaryDirectory

from sklearn.preprocessing import MinMaxScaler

from analysis_common import (FEATURES, arguments, load_model, make_model, read_data,
                             report, save_model, split_indices)


def train(args, X, y):
    if args.iterations < 1 or args.maxsize < 7:
        raise ValueError('iterations must be positive and maxsize must be at least 7.')
    # Serial execution is required for reproducible PySR searches.
    os.environ['PYTHON_JULIACALL_THREADS'] = '1'
    os.environ.setdefault('JULIA_NUM_PRECOMPILE_TASKS', '1')
    if sys.platform == 'darwin':
        os.environ.setdefault('PYTHON_JULIACALL_HANDLE_SIGNALS', 'no')
    # Let PySR provision compatible Julia and SymbolicRegression.jl versions.
    from pysr import PySRRegressor

    train_indices, test_indices = split_indices(X)
    scaler = MinMaxScaler().fit(X[train_indices])
    with TemporaryDirectory(prefix='concrete-sr-') as search:
        model = PySRRegressor(
            niterations=args.iterations, maxsize=args.maxsize,
            binary_operators=['+', '-', '*'], unary_operators=['square'],
            nested_constraints={'*': {'*': 2}, 'square': {'square': 0}},
            random_state=42, deterministic=True, parallelism='serial', precision=64,
            model_selection='best', progress=False, output_directory=search)
        model.fit(scaler.transform(X[train_indices]), y[train_indices], variable_names=FEATURES)
        # Selection uses training loss and complexity, never test scores.
        expression = model.get_best()['sympy_format']
        args.output.mkdir(parents=True, exist_ok=True)
        model.equations_[['complexity', 'loss', 'equation']].to_csv(
            args.output / 'candidate_equations.csv', index=False)
    return make_model(args.data, expression, scaler, train_indices, test_indices,
                      {'seed': 42, 'test_size': 0.15, 'scaling': 'training_only',
                       'iterations': args.iterations, 'maxsize': args.maxsize})


def main():
    args = arguments('symbolic_regression')
    X, y = read_data(args.data)
    model = train(args, X, y) if args.train else load_model(args.model, args.data)
    report(model, X, y, args.output)
    if args.train:
        save_model(args.output / 'model.json', model)


if __name__ == '__main__':
    main()
