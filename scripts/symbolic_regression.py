"""Search training data with PySR and save Pareto equations for sr_equation_selection.py."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from data_preprocessing import FEATURES, ROOT, SPLIT_SEED, prepare_data


def run(args):
    julia_env = Path(sys.prefix) / 'julia_env'
    if not julia_env.exists() and os.environ.get('CONDA_PREFIX'):
        julia_env = Path(os.environ['CONDA_PREFIX']) / 'julia_env'
    os.environ.setdefault('PYTHON_JULIAPKG_PROJECT', str(julia_env))
    os.environ.setdefault('PYTHON_JULIACALL_THREADS', 'auto')
    from pysr import PySRRegressor

    data = prepare_data(args.data)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output or ROOT / 'results' / 'symbolic_regression' / 'search' / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)

    serial = args.workers == 1
    model = PySRRegressor(
        niterations=args.iterations,
        timeout_in_seconds=args.seconds,
        maxsize=args.maxsize,
        binary_operators=['+', '-', '*'],
        unary_operators=['square'],
        nested_constraints={'*': {'*': 2}, 'square': {'square': 0}},
        random_state=args.seed,
        deterministic=serial,
        parallelism='serial' if serial else 'multithreading',
        procs=None,
        progress=True,
        warm_start=False,
        output_directory=str(output / 'pysr'),
    )
    model.fit(data.Z[data.train_indices], data.y[data.train_indices],
              variable_names=FEATURES)
    equations = model.equations_
    required = {'complexity', 'loss', 'score', 'sympy_format'}
    if equations is None or equations.empty or not required.issubset(equations.columns):
        raise RuntimeError('PySR did not return a usable Pareto front.')
    equations.to_csv(output / 'pareto_candidates.csv', index=False)

    context = {
        'status': 'search_complete',
        'source_format': 'training_only_pysr',
        'data_path': str(data.path),
        'train_indices': data.train_indices.tolist(),
        'test_indices': data.test_indices.tolist(),
        'target_mode': 'original_mpa',
        'feature_scaling': 'training_only_minmax',
        'settings': {
            'iterations': args.iterations, 'seconds': args.seconds,
            'workers': args.workers, 'seed': args.seed,
            'split_seed': SPLIT_SEED, 'maxsize': args.maxsize,
            'operators': ['+', '-', '*', 'square'],
            'min_features': args.min_features,
            'max_degree': args.max_degree,
        },
    }
    (output / 'run.json').write_text(json.dumps(context, indent=2) + '\n', encoding='utf-8')
    print(f'Saved {len(equations)} Pareto equations to {output}')
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data.xlsx')
    parser.add_argument('--output', type=Path, help='New directory for this search run')
    parser.add_argument('--iterations', type=int, default=800)
    parser.add_argument('--seconds', type=int)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--maxsize', type=int, default=120)
    parser.add_argument('--min-features', type=int, default=7,
                        help='Selection-stage eligibility threshold saved with the run')
    parser.add_argument('--max-degree', type=int, default=9,
                        help='Selection-stage polynomial degree limit saved with the run')
    return parser.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
