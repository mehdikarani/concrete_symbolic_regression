"""Check reference results, analytical derivatives, and portable execution."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_common import equation, load_model, read_data
from linear_regression import train


class Workflows(unittest.TestCase):
    def test_reference_results_and_portability(self):
        expected = {'symbolic_regression': [0.8386176165625329, 0.813891490985118],
                    'linear_regression': [0.5588468707095975, 0.5589870278958655]}
        with tempfile.TemporaryDirectory() as folder:
            moved = Path(folder) / 'project with spaces'
            moved.mkdir()
            for name in ('analysis_common.py', 'linear_regression.py', 'symbolic_regression.py', 'data.xlsx'):
                shutil.copy2(ROOT / name, moved)
            shutil.copytree(ROOT / 'models', moved / 'models')
            for name, scores in expected.items():
                command = [sys.executable, str(moved / f'{name}.py')]
                subprocess.run(command, cwd=folder, check=True, capture_output=True)
                output = moved / 'results' / name
                np.testing.assert_allclose(pd.read_csv(output / 'metrics.csv').r2, scores, atol=1e-10, rtol=0)
                first = {p.name: p.read_bytes() for p in output.iterdir()}
                subprocess.run(command, cwd=folder, check=True, capture_output=True)
                self.assertEqual(first, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_lr_refit_and_derivatives(self):
        X, y = read_data(ROOT / 'data.xlsx')
        fitted = train(ROOT / 'data.xlsx', X, y)
        frozen = load_model(ROOT / 'models/linear_regression.json', ROOT / 'data.xlsx')
        _, _, fitted_predict = equation(fitted['expression'])
        _, _, frozen_predict = equation(frozen['expression'])
        Z = X * fitted['scale'] + fitted['offset']
        np.testing.assert_allclose(fitted_predict(Z), frozen_predict(Z), rtol=1e-12, atol=1e-10)
        for name in ('symbolic_regression', 'linear_regression'):
            model = load_model(ROOT / 'models' / f'{name}.json', ROOT / 'data.xlsx')
            expression, symbols, predict = equation(model['expression'])
            Z = X[:5] * model['scale'] + model['offset']
            for j, symbol in enumerate(symbols):
                delta = np.zeros_like(Z)
                delta[:, j] = 1e-5
                numerical = (predict(Z + delta) - predict(Z - delta)) / 2e-5
                analytic = sp.lambdify(symbols, sp.diff(expression, symbol), 'numpy')(*Z.T)
                np.testing.assert_allclose(numerical, analytic, rtol=1e-5, atol=1e-7)

    def test_changed_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / 'data.xlsx'
            data.write_bytes((ROOT / 'data.xlsx').read_bytes() + b'changed')
            with self.assertRaises(ValueError):
                load_model(ROOT / 'models/linear_regression.json', data)


if __name__ == '__main__':
    unittest.main()
