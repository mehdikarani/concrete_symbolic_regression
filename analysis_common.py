"""Shared data preparation, statistics, and plots for both regression models."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
TARGET = 'Compressive_strength_Mpa'
FEATURES = ['SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'w_c', 'Fineness_modulus',
            'Portland_cement_kg_m3', 'Fine_aggregate_kg_m3', 'Iron_tailing_kg_m3',
            'Coarse_aggregates_kg_m3']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.hashsalt': 'concrete'})


def arguments(name):
    parser = argparse.ArgumentParser(description='Original features, no PCA. Replay saved results by default.')
    parser.add_argument('--data', type=Path, default=ROOT / 'data.xlsx')
    parser.add_argument('--model', type=Path, default=ROOT / 'models' / f'{name}.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / name)
    parser.add_argument('--train', action='store_true')
    if name == 'symbolic_regression':
        parser.add_argument('--iterations', type=int, default=800)
        parser.add_argument('--maxsize', type=int, default=120)
    return parser.parse_args()


def read_data(path):
    data = pd.read_excel(path)[FEATURES + [TARGET]].apply(pd.to_numeric)
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError('Data must contain finite numeric values without missing entries.')
    return data[FEATURES].to_numpy(), data[TARGET].to_numpy()


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_indices(X):
    return train_test_split(np.arange(len(X)), test_size=0.15, random_state=42)


def save_model(path, model):
    path.write_text(json.dumps(model, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def make_model(data, expression, scaler, train, test, settings):
    return {'features': FEATURES, 'data_sha256': checksum(data), 'expression': str(expression),
            'scale': scaler.scale_.tolist(), 'offset': scaler.min_.tolist(),
            'train_indices': train.tolist(), 'test_indices': test.tolist(), 'settings': settings}


def load_model(path, data):
    model = json.loads(path.read_text(encoding='utf-8'))
    if model['features'] != FEATURES or model['data_sha256'] != checksum(data):
        raise ValueError('Saved model requires the original dataset and feature order. Use --train for new data.')
    return model


def equation(expression):
    symbols = sp.symbols(' '.join(FEATURES))
    expression = sp.sympify(expression, locals=dict(zip(FEATURES, symbols)))
    function = sp.lambdify(symbols, expression, 'numpy')

    def predict(values):
        result = np.broadcast_to(function(*values.T), (len(values),)).astype(float)
        if not np.isfinite(result).all():
            raise ValueError('Equation produced nonfinite predictions.')
        return result

    return expression, symbols, predict


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, metadata={'Date': None})
    plt.close(fig)


def prediction_report(y, Z, model, predict, output):
    metrics, predictions = [], []
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for row, split in enumerate(('train', 'test')):
        indices = model[f'{split}_indices']
        actual, predicted = y[indices], predict(Z[indices])
        r, p = pearsonr(actual, predicted) if np.ptp(actual) and np.ptp(predicted) else (np.nan, np.nan)
        mse = mean_squared_error(actual, predicted)
        scores = dict(split=split, n=len(indices), mse=mse, rmse=np.sqrt(mse),
                      mae=mean_absolute_error(actual, predicted), r2=r2_score(actual, predicted),
                      pearson_r=r, pearson_p=p)
        metrics.append(scores)
        predictions.append(pd.DataFrame(dict(split=split, row_index=indices, y_true=actual,
                                              y_pred=predicted, residual=actual - predicted)))
        ax, residual_ax = axes[row]
        ax.scatter(actual, predicted, s=20, alpha=.7)
        limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
        ax.plot(limits, limits, 'r--')
        ax.set(title=f"{split.title()}: R² = {scores['r2']:.4f}", xlabel='Measured strength (MPa)',
               ylabel='Predicted strength (MPa)')
        residual_ax.scatter(predicted, actual - predicted, s=20, alpha=.7)
        residual_ax.axhline(0, color='red', linestyle='--')
        residual_ax.set(title=split.title(), xlabel='Predicted strength (MPa)', ylabel='Residual (MPa)')
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(output / 'metrics.csv', index=False, float_format='%.17g')
    pd.concat(predictions).to_csv(output / 'predictions.csv', index=False, float_format='%.17g')
    save_figure(fig, output / 'predictions.svg')
    print(metrics.to_string(index=False))


def main_effects(X, model, predict, output):
    train = X[model['train_indices']]
    baseline = np.median(train, axis=0)
    fig, axes = plt.subplots(5, 2, figsize=(12, 16))
    for j, (feature, ax) in enumerate(zip(FEATURES, axes.flat)):
        low, high = train[:, j].min(), train[:, j].max()
        grid = np.linspace(low + .05 * (high - low), low + .95 * (high - low), 200)
        values = np.tile(baseline, (len(grid), 1))
        values[:, j] = grid
        ax.plot(grid, predict(values * model['scale'] + model['offset']))
        ax.set(xlabel=feature, ylabel='Predicted strength (MPa)')
    save_figure(fig, output / 'main_effects.svg')


def sensitivity_report(Z, model, expression, symbols, output):
    gradients = np.column_stack([
        np.broadcast_to(sp.lambdify(symbols, sp.diff(expression, symbol), 'numpy')(*Z.T), len(Z))
        for symbol in symbols])
    rows = []
    for split in ('train', 'test'):
        values = gradients[model[f'{split}_indices']]
        for j, feature in enumerate(FEATURES):
            rows.append(dict(split=split, feature=feature, mean_abs_scaled=np.abs(values[:, j]).mean(),
                             std_abs_scaled=np.abs(values[:, j]).std(ddof=1),
                             mean_abs_original=np.abs(values[:, j] * model['scale'][j]).mean()))
    pd.DataFrame(rows).to_csv(output / 'sensitivity.csv', index=False, float_format='%.17g')
    importance = np.abs(gradients[model['train_indices']]).mean(axis=0)
    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([FEATURES[j] for j in order], importance[order])
    ax.set_xlabel('Train mean |dy/d(scaled feature)|')
    save_figure(fig, output / 'sensitivity.svg')


def write_equation(model, expression, symbols, output):
    original = expression.subs(
        {symbol: symbol * model['scale'][j] + model['offset'][j] for j, symbol in enumerate(symbols)},
        simultaneous=True)
    (output / 'equation.txt').write_text(
        f'{TARGET} = {original}\n\nInputs are in original dataset units.\n', encoding='utf-8')


def report(model, X, y, output):
    output.mkdir(parents=True, exist_ok=True)
    if model['settings'].get('warning'):
        print(model['settings']['warning'])
    expression, symbols, predict = equation(model['expression'])
    Z = X * model['scale'] + model['offset']
    prediction_report(y, Z, model, predict, output)
    main_effects(X, model, predict, output)
    sensitivity_report(Z, model, expression, symbols, output)
    write_equation(model, expression, symbols, output)
    print(f'Results: {output.resolve()}')
