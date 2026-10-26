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
from analysis_common import FEATURES, equation, load_model, read_data, split_indices
from linear_regression import train
from symbolic_regression import (REQUIRED_FEATURES, audit_candidate, parallel_settings,
                                 select_candidate)


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
                if name == 'symbolic_regression':
                    command.append('--legacy')
                subprocess.run(command, cwd=folder, check=True, capture_output=True)
                output = (moved / 'results' / 'legacy' / name
                          if name == 'symbolic_regression' else moved / 'results' / name)
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
            model_path = (ROOT / 'models' / 'legacy' / f'{name}.json'
                          if name == 'symbolic_regression' else ROOT / 'models' / f'{name}.json')
            model = load_model(model_path, ROOT / 'data.xlsx')
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

    def test_sr_row_split(self):
        X, _ = read_data(ROOT / 'data.xlsx')
        train_indices, test_indices = split_indices(X)
        self.assertEqual(len(train_indices), 130)
        self.assertEqual(len(test_indices), 23)
        self.assertTrue(set(train_indices).isdisjoint(test_indices))

    def test_sr_candidate_selection_uses_prespecified_training_rules(self):
        all_features = ' + '.join(REQUIRED_FEATURES)
        stable = sp.sympify(f'20 + ({all_features}) - 2*w_c')
        simpler = sp.sympify(f'18 + ({all_features}) - 2*w_c')
        missing = sp.sympify('20 - 2*w_c')
        negative = sp.sympify(f'-100 + ({all_features}) - 2*w_c')
        equations = pd.DataFrame([
            dict(complexity=10, loss=1.0, score=.5, equation='stable', sympy_format=stable),
            dict(complexity=25, loss=1.2, score=.1, equation='simpler', sympy_format=simpler),
            dict(complexity=5, loss=.5, score=10.0, equation='missing', sympy_format=missing),
            dict(complexity=30, loss=.5, score=9.0, equation='negative', sympy_format=negative),
        ])
        selected, audited = select_candidate(
            equations, np.full((12, len(FEATURES)), .5))
        self.assertEqual(selected, stable)
        self.assertEqual(audited.selected.sum(), 1)
        self.assertIn('missing:', audited.loc[audited.equation == 'missing',
                                               'rejection_reason'].item())
        self.assertIn('negative_prediction', audited.loc[audited.equation == 'negative',
                                                         'rejection_reason'].item())

    def test_sr_structural_rules(self):
        base = sp.sympify('20 + (' + ' + '.join(REQUIRED_FEATURES) + ') - 2*w_c')
        points = np.full((12, len(FEATURES)), .5)
        quartic = base + sp.sympify('Al2O3**2 * Fineness_modulus**2')
        pairwise = base + sp.sympify('Al2O3 * Fineness_modulus + Fe2O3**2')
        self.assertIn('higher_order_polynomial',
                      audit_candidate(quartic, points)['rejection_reason'])
        self.assertTrue(audit_candidate(pairwise, points)['eligible'])

    def test_sr_candidate_selection_rejects_wrong_water_ratio_direction(self):
        expression = sp.sympify('20 + ' + ' + '.join(REQUIRED_FEATURES))
        equations = pd.DataFrame([
            dict(complexity=30, loss=1.0, score=.2, equation='wrong direction',
                 sympy_format=expression),
        ])
        with self.assertRaisesRegex(ValueError, 'wrong_direction:w_c'):
            select_candidate(equations, np.full((12, len(FEATURES)), .5))

    def test_sr_parallel_settings(self):
        parallel = parallel_settings(4)
        self.assertEqual(parallel['parallelism'], 'multithreading')
        self.assertIsNone(parallel['procs'])
        self.assertFalse(parallel['deterministic'])
        serial = parallel_settings(1)
        self.assertEqual(serial['parallelism'], 'serial')
        self.assertIsNone(serial['procs'])
        self.assertTrue(serial['deterministic'])


if __name__ == '__main__':
    unittest.main()
