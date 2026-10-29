"""Shared model representation, baseline fitting, metrics, and reporting."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import FEATURES, ROOT, TARGET, SPLIT_SEED, TRAIN_RATIO, prepare_data

SYMBOLS = sp.symbols(' '.join(FEATURES))
SYMBOL_MAP = dict(zip(FEATURES, SYMBOLS))

OXIDE_FEATURES = ('SiO2', 'Al2O3', 'Fe2O3', 'CaO')
OXIDE_INDICES = [FEATURES.index(f) for f in OXIDE_FEATURES]

DISPLAY_NAMES = {
    'SiO2': 'SiO$_2$',
    'Al2O3': 'Al$_2$O$_3$',
    'Fe2O3': 'Fe$_2$O$_3$',
    'CaO': 'CaO',
    'w_c': 'w/b',
    'Fineness_modulus': 'Fineness modulus',
    'Portland_cement_kg_m3': 'Portland cement (kg/m3)',
    'Fine_aggregate_kg_m3': 'Fine aggregate (kg/m3)',
    'Iron_tailing_kg_m3': 'Iron tailings (kg/m3)',
    'Coarse_aggregates_kg_m3': 'Coarse aggregate (kg/m3)',
}
DISPLAY_NAMES_SCALED = {k: v.replace(' (kg/m3)', '') for k, v in DISPLAY_NAMES.items()}

plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'svg.hashsalt': 'concrete',
})


def make_model(data, expression, settings):
    return {
        'features': FEATURES,
        'expression': str(expression),
        'scale': data.scaler.scale_.tolist(),
        'offset': data.scaler.min_.tolist(),
        'train_indices': data.train_indices.tolist(),
        'test_indices': data.test_indices.tolist(),
        'settings': {
            'split': 'random_rows',
            'train_ratio': TRAIN_RATIO,
            'split_seed': SPLIT_SEED,
            'scaling': 'training_only',
            **settings,
        },
    }


def save_model(path, model):
    Path(path).write_text(json.dumps(model, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fit_polynomial_model(data, degree=1):
    """Additive powers only; no interaction terms. Equal row weights."""
    powers = range(1, degree + 1)
    Z_train = np.hstack([data.Z[data.train_indices] ** p for p in powers])
    fit = LinearRegression().fit(Z_train, data.y[data.train_indices])

    term_names = [feat if p == 1 else f'{feat}**{p}' for p in powers for feat in FEATURES]
    terms = ''.join(f' + ({float(coef)!r})*{term}' for coef, term in zip(fit.coef_, term_names))
    expression = repr(float(fit.intercept_)) + terms

    settings = {
        'model_type': 'linear' if degree == 1 else 'additive_polynomial',
        'weighting': 'equal_row',
        'degree': degree,
    }
    return make_model(data, expression, settings)


def parse_equation(expression):
    sym_expr = sp.sympify(expression, locals=SYMBOL_MAP)
    fn = sp.lambdify(SYMBOLS, sym_expr, 'numpy')

    def predict(values):
        values = np.asarray(values, dtype=float)
        result = np.broadcast_to(fn(*values.T), (len(values),)).astype(float)
        if not np.isfinite(result).all():
            raise ValueError('Equation produced non-finite predictions.')
        return result

    return sym_expr, predict


def parse_polynomial(expression, max_degree=4):
    expanded = sp.expand(sp.sympify(expression, locals=SYMBOL_MAP))
    if not expanded.free_symbols.issubset(set(SYMBOLS)):
        raise ValueError('Unexpected input symbol.')
    poly = sp.Poly(expanded, *SYMBOLS)
    if poly.total_degree() > max_degree or not all(c.is_finite for c in poly.coeffs()):
        raise ValueError('Non-finite coefficients or excessive polynomial degree.')
    _, predict = parse_equation(expanded)
    return expanded, poly, predict


def evaluate_gradients(expression, values):
    gradients = np.column_stack([
        np.broadcast_to(sp.lambdify(SYMBOLS, sp.diff(expression, symbol), 'numpy')(*values.T), len(values))
        for symbol in SYMBOLS
    ]).astype(float)
    if not np.isfinite(gradients).all():
        raise ValueError('Non-finite sensitivity values.')
    return gradients


def weighted_quantiles(values, weights, quantiles):
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    positions = (np.cumsum(weights) - weights / 2) / weights.sum()
    return np.interp(quantiles, positions, values)


def to_original_units(expression, scale, offset):
    substitutions = {
        sym: sym * scale[j] + offset[j]
        for j, sym in enumerate(SYMBOLS)
    }
    return expression.subs(substitutions, simultaneous=True)


def save_figure(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', metadata={'Date': None})
    plt.close(fig)


def split_metrics(split, actual, predicted):
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    r, p = pearsonr(actual, predicted) if np.ptp(actual) and np.ptp(predicted) else (np.nan, np.nan)
    return {
        'split': split,
        'n': len(actual),
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mean_absolute_error(actual, predicted),
        'r2': r2,
        'r_value': np.sqrt(r2) if r2 >= 0 else np.nan,
        'pearson': r,
        'pearson_p': p,
    }


def report_predictions(y, Z, model, predict, output):
    colors = {'train': None, 'test': 'orange'}
    metrics = []

    for split in ('train', 'test'):
        indices = model[f'{split}_indices']
        actual, predicted = y[indices], predict(Z[indices])
        scores = split_metrics(split, actual, predicted)
        metrics.append(scores)
        pd.DataFrame({'y_true': actual, 'y_pred': predicted}).to_csv(
            output / f'{split}_predictions.csv', index=False
        )

        fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.scatter(actual, predicted, alpha=0.6, edgecolors='k', linewidth=0.5, s=100,
                   color=colors[split], label=split.title())
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        ax.set_xlabel('Measured Compressive Strength (MPa)', fontsize=20, labelpad=10)
        ax.set_ylabel('Predicted Compressive Strength (MPa)', fontsize=20, labelpad=10)
        ax.tick_params(axis='both', labelsize=18)
        r_label = f"R = {scores['r_value']:.4f}" if np.isfinite(scores['r_value']) else 'R = N/A'
        ax.text(0.05, 0.95, r_label, transform=ax.transAxes,
                fontsize=20, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend(fontsize=18, loc='lower right')
        save_figure(fig, output / f'{split}_true_vs_predicted.svg')

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output / 'equation_metrics.csv', index=False)
    print(metrics_df.to_string(index=False))


def main_effect_grid(X, train_indices, j):
    """Build evaluation grid for feature j, keeping oxide percentages summing to 100%."""
    baseline = np.median(X[train_indices], axis=0).copy()
    baseline[OXIDE_INDICES] *= 100.0 / baseline[OXIDE_INDICES].sum()

    low, high = float(X[:, j].min()), float(X[:, j].max())
    grid = np.linspace(low + 0.05 * (high - low), low + 0.95 * (high - low), 200)
    values = np.tile(baseline, (len(grid), 1))
    values[:, j] = grid

    if j in OXIDE_INDICES:
        others = [idx for idx in OXIDE_INDICES if idx != j]
        weights = baseline[others] / baseline[others].sum()
        values[:, others] = np.outer(100.0 - grid, weights)
    return grid, values


def report_main_effects(X, model, predict, output):
    main_effect_dir = output / 'main_effect'
    main_effect_dir.mkdir(parents=True, exist_ok=True)

    for j, feat in enumerate(FEATURES):
        grid, values = main_effect_grid(X, model['train_indices'], j)
        y_grid = predict(values * model['scale'] + model['offset'])

        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        ax.plot(grid, y_grid, color='#2E86AB', linewidth=3.5, label='Main Effect', zorder=3)
        ax.set_xlabel(DISPLAY_NAMES.get(feat, feat), fontsize=22, labelpad=10)
        ax.set_ylabel('Compressive strength (MPa)', fontsize=22, labelpad=10)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray', zorder=1)
        ax.set_axisbelow(True)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_linewidth(1.2)
            ax.spines[spine].set_color('#333333')
        ax.tick_params(axis='both', which='major', labelsize=18, length=6, width=1.2, colors='#333333')
        fig.tight_layout()
        save_figure(fig, main_effect_dir / f'main_effect_{j + 1:02d}_{feat}.svg')


def write_derivative_formulas(model, expression, grad_formulas, path):
    original = to_original_units(expression, model['scale'], model['offset'])
    sep = '=' * 80
    lines = [
        sep, 'ANALYTICAL PARTIAL DERIVATIVE FORMULAS', sep, '',
        'Scaled Equation (inputs in [0, 1]):', f'y = {expression}', '',
        'Original Equation (inputs in original dataset units):', f'{TARGET} = {original}', '',
        sep, '',
    ]
    for feat, formula in grad_formulas.items():
        lines.extend([
            f'Partial derivative with respect to {feat}:',
            '-' * 80,
            f'∂y/∂{feat} = ',
            sp.pretty(formula, use_unicode=True),
            '', sep, '',
        ])
    path.write_text('\n'.join(lines), encoding='utf-8')


def plot_pareto(summary, sensitivity_dir):
    features_sorted = summary['Feature'].values
    labels = [DISPLAY_NAMES_SCALED.get(f, f) for f in features_sorted]
    importances = summary['Train_Mean_Abs_Gradient'].values
    total = importances.sum()
    cum_pct = (np.cumsum(importances) / total * 100) if total else np.zeros_like(importances)

    fig, ax1 = plt.subplots(figsize=(14, 8), facecolor='white')
    x_pos = np.arange(len(features_sorted))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features_sorted)))
    bars = ax1.bar(x_pos, importances, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    ax1.set_ylim([0, max(105, float(importances.max()) * 1.15)])
    ax1.set_ylabel('Mean Absolute Sensitivity (Scaled)', fontsize=22, labelpad=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=18)
    ax1.tick_params(axis='y', labelsize=18)

    ax2 = ax1.twinx()
    ax2.plot(x_pos, cum_pct, color='red', marker='o', linewidth=2.5,
             markersize=8, label='Cumulative %', zorder=10)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=22, labelpad=10, color='red')
    ax2.tick_params(axis='y', labelcolor='red', labelsize=18)
    ax2.set_ylim([0, 105])

    for bar, val in zip(bars, importances):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{val:.2f}',
                 ha='center', va='bottom', fontsize=12)
    for x, y_val in zip(x_pos, cum_pct):
        ax2.text(x, y_val + 2, f'{y_val:.1f}%', ha='center', va='bottom',
                 fontsize=12, color='darkred')

    ax1.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    fig.tight_layout()
    save_figure(fig, sensitivity_dir / 'pareto_chart.svg')


def plot_gradient_heatmap(gradients_train, gradients_test, sensitivity_dir):
    labels = [DISPLAY_NAMES_SCALED.get(f, f) for f in FEATURES]
    gradients_all = pd.concat([gradients_train, gradients_test], axis=0, ignore_index=True)
    sample_indices = np.linspace(0, len(gradients_all) - 1, min(150, len(gradients_all)), dtype=int)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
    vmax = float(np.abs(gradients_all.values).max()) or 1.0
    im = ax.imshow(gradients_all.iloc[sample_indices].T, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(labels, fontsize=18)
    ax.set_xlabel('Sample Index', fontsize=18, labelpad=10)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=18)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Gradient Value', fontsize=18, labelpad=15)
    cbar.ax.tick_params(labelsize=16)
    fig.tight_layout()
    save_figure(fig, sensitivity_dir / 'gradient_heatmap.svg')


def plot_gradient_boxplot(summary, gradients_train, sensitivity_dir):
    ordered = summary['Feature'].values
    data = [np.abs(gradients_train[f].values) for f in ordered]
    labels = [DISPLAY_NAMES_SCALED.get(f, f) for f in ordered]

    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.6)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Absolute Local Sensitivity |∂y/∂x|', fontsize=22, labelpad=10)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, sensitivity_dir / 'gradient_boxplot.svg')


def report_sensitivity(Z, model, expression, output):
    sensitivity_dir = output / 'sensitivity_analysis'
    sensitivity_dir.mkdir(parents=True, exist_ok=True)

    grad_formulas = {feat: sp.diff(expression, sym) for feat, sym in zip(FEATURES, SYMBOLS)}
    gradients = evaluate_gradients(expression, Z)

    gradients_train = pd.DataFrame(gradients[model['train_indices']], columns=FEATURES)
    gradients_test = pd.DataFrame(gradients[model['test_indices']], columns=FEATURES)
    gradients_train.to_csv(sensitivity_dir / 'gradients_train.csv', index=False)
    gradients_test.to_csv(sensitivity_dir / 'gradients_test.csv', index=False)

    summary = pd.DataFrame({
        'Feature': FEATURES,
        'Train_Mean_Abs_Gradient': np.abs(gradients_train).mean(axis=0).values,
        'Train_Std_Abs_Gradient': np.abs(gradients_train).std(axis=0).values,
        'Test_Mean_Abs_Gradient': np.abs(gradients_test).mean(axis=0).values,
        'Test_Std_Abs_Gradient': np.abs(gradients_test).std(axis=0).values,
    }).sort_values('Train_Mean_Abs_Gradient', ascending=False)
    summary['Rank'] = range(1, len(summary) + 1)
    summary.to_csv(sensitivity_dir / 'sensitivity_summary.csv', index=False)

    write_derivative_formulas(
        model, expression, grad_formulas,
        sensitivity_dir / 'analytical_derivative_formulas.txt'
    )
    plot_pareto(summary, sensitivity_dir)
    plot_gradient_heatmap(gradients_train, gradients_test, sensitivity_dir)
    plot_gradient_boxplot(summary, gradients_train, sensitivity_dir)

    print('\nSensitivity Analysis Results:')
    print(summary.to_string(index=False))


def report_scaling_factors(model, output):
    """Write min-max scaling factors: z = (x - min) / (max - min) = scale * x + offset."""
    scale = np.asarray(model['scale'], dtype=float)
    offset = np.asarray(model['offset'], dtype=float)
    x_min = np.where(offset == 0, 0.0, -offset / scale)
    x_max = (1.0 - offset) / scale
    pd.DataFrame({
        'Feature': FEATURES,
        'Min': x_min,
        'Max': x_max,
        'Range': x_max - x_min,
        'Scale': scale,
        'Offset': offset,
    }).to_csv(output / 'scaling_factors.csv', index=False)


def write_equation(model, expression, output):
    original = to_original_units(expression, model['scale'], model['offset'])
    text = (
        f'Scaled equation (inputs scaled to [0, 1], see scaling_factors.csv):\n'
        f'{TARGET} = {expression}\n\n'
        f'Original-units equation (inputs in original dataset units):\n'
        f'{TARGET} = {original}\n'
    )
    (output / 'equation.txt').write_text(text, encoding='utf-8')


def report(model, X, y, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expression, predict = parse_equation(model['expression'])
    Z = X * model['scale'] + model['offset']
    write_equation(model, expression, output)
    report_scaling_factors(model, output)
    report_predictions(y, Z, model, predict, output)
    report_main_effects(X, model, predict, output)
    report_sensitivity(Z, model, expression, output)
    print(f'\nAll results saved to: {output.resolve()}')


def run_baseline(data_path, model_path, output_dir, degree):
    data = prepare_data(data_path)
    model = fit_polynomial_model(data, degree=degree)
    report(model, data.X, data.y, output_dir)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    save_model(model_path, model)
    save_model(Path(output_dir) / 'model.json', model)
