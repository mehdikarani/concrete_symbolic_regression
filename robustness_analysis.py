"""Main-effect and sensitivity robustness across frozen equations."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

from analysis_common import (
    OXIDE_INDICES, SYMBOL_MAP, evaluate_gradients, main_effect_grid, parse_polynomial,
)
from data_preprocessing import FEATURES, study_weights
from sr_plotting import plot_main_effect_robustness, plot_sensitivity_robustness


def analyse_main_effect_robustness(candidates, selected, data, output, max_degree, make_plots=True):
    curves = []
    for rank, idx in enumerate(selected, 1):
        row = candidates.loc[idx]
        _, _, predict = parse_polynomial(row.expression, max_degree=max_degree)
        for j, feature in enumerate(FEATURES):
            grid, values = main_effect_grid(data.X[data.train_indices],
                                            np.arange(len(data.train_indices)), j)
            predictions = predict(data.scaler.transform(values))
            for x, y in zip(grid, predictions):
                curves.append({'rank': rank, 'feature': feature, 'feature_value': x,
                               'predicted_strength': y})
    curves = pd.DataFrame(curves)
    curves.to_csv(output / 'main_effect_curves.csv', index=False)

    rows = []
    for feature in FEATURES:
        primary_curve = curves[(curves['rank'] == 1) & (curves.feature == feature)].predicted_strength.to_numpy()
        for rank, idx in enumerate(selected, 1):
            curve = curves[(curves['rank'] == rank) & (curves.feature == feature)].predicted_strength.to_numpy()
            delta = float(curve[-1] - curve[0])
            rows.append({'feature': feature, 'rank': rank,
                         'complexity': int(candidates.loc[idx, 'complexity']),
                         'included': SYMBOL_MAP[feature] in sp.sympify(candidates.loc[idx, 'expression'], locals=SYMBOL_MAP).free_symbols,
                         'low_to_high_change_mpa': delta,
                         'effect_range_mpa': float(np.ptp(curve)),
                         'curve_rmse_from_primary_mpa': float(np.sqrt(np.mean((curve - primary_curve) ** 2)))})
    details = pd.DataFrame(rows)
    details.to_csv(output / 'feature_effects_by_equation.csv', index=False)
    summary = []
    for feature, group in details.groupby('feature', sort=False):
        changes = group.low_to_high_change_mpa.to_numpy()
        nonzero = changes[np.abs(changes) > 1e-8]
        summary.append({'feature': feature, 'equations': len(group),
                        'inclusion_fraction': float(group.included.mean()),
                        'median_effect_range_mpa': float(group.effect_range_mpa.median()),
                        'median_low_to_high_change_mpa': float(np.median(changes)),
                        'direction_agreement_fraction': float(max(np.mean(nonzero > 0), np.mean(nonzero < 0))) if len(nonzero) else np.nan,
                        'median_curve_rmse_from_primary_mpa': float(group.curve_rmse_from_primary_mpa.median())})
    pd.DataFrame(summary).to_csv(output / 'feature_robustness_summary.csv', index=False)
    if make_plots:
        plot_main_effect_robustness(curves, candidates, selected, output)


def composition_gradients(expression, X, scaler):
    """Differentiate along proportional redistribution of the other oxides, per scaled input unit."""
    values = X.copy()
    totals = values[:, OXIDE_INDICES].sum(axis=1)
    if np.any(values[:, OXIDE_INDICES] < 0) or np.any(totals <= 0):
        raise ValueError('Oxide totals must be positive.')
    values[:, OXIDE_INDICES] *= 100 / totals[:, None]
    gradients = evaluate_gradients(expression, scaler.transform(values))
    physical = gradients * scaler.scale_
    result = gradients.copy()
    for j in OXIDE_INDICES:
        others = [k for k in OXIDE_INDICES if k != j]
        denominator = values[:, others].sum(axis=1)
        if np.any(denominator <= 0):
            raise ValueError('Oxide redistribution requires a positive remainder.')
        redistribution = values[:, others] / denominator[:, None]
        result[:, j] = (physical[:, j] - (physical[:, others] * redistribution).sum(axis=1)) / scaler.scale_[j]
    return result


def analyse_sensitivity_robustness(candidates, selected, data, output, make_plots=True, zero_tolerance=1e-6):
    """Compare frozen equations on training rows with equal total weight per study."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    training_indices = data.train_indices
    oxide_values = data.X[training_indices][:, OXIDE_INDICES]
    composition_valid = ((oxide_values >= 0).all(axis=1)
                         & (oxide_values.sum(axis=1) > 0)
                         & ((oxide_values.sum(axis=1)[:, None] - oxide_values) > 0).all(axis=1))
    equations = [sp.sympify(candidates.loc[idx, 'expression'], locals=SYMBOL_MAP) for idx in selected]
    metadata = {
        'split': 'training', 'weighting': 'equal_study',
        'gradient_units': 'MPa per unit of training-minmax-scaled feature',
        'zero_tolerance': zero_tolerance,
        'composition_aware': 'Normalize the four modeled oxides to a 100% subcomposition; increase one oxide and proportionally reduce the others. Non-oxide derivatives evaluated at the same normalized compositions.',
        'sign_agreement': 'Compare alternatives with primary only where both exceed zero_tolerance in magnitude; report comparison coverage separately.',
        'rank_ties': 'minimum rank',
        'distribution_boxes': 'Study-weighted 25th/50th/75th percentiles; whiskers at 5th/95th percentiles; no outlier points.',
        'training_rows': len(training_indices),
        'composition_aware_rows': int(composition_valid.sum()),
        'composition_excluded_rows': training_indices[~composition_valid].tolist(),
        'composition_exclusion_reason': 'Negative oxide, zero total, or zero remainder for redistribution.',
        'included': 'Direct occurrence in equation; a composition derivative can be nonzero through redistribution even when the feature is absent.',
        'scope': 'Selected-equation robustness; not bootstrap uncertainty or causal effects.',
    }
    (output / 'methodology.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for mode in ('partial', 'composition_aware'):
        indices = training_indices if mode == 'partial' else training_indices[composition_valid]
        if not len(indices):
            raise ValueError('No training rows support composition-aware sensitivity.')
        X, Z = data.X[indices], data.Z[indices]
        weights = study_weights(data.groups[indices])
        gradients = np.stack([
            evaluate_gradients(eq, Z) if mode == 'partial' else composition_gradients(eq, X, data.scaler)
            for eq in equations
        ])
        importance = np.average(np.abs(gradients), axis=1, weights=weights)
        totals = importance.sum(axis=1, keepdims=True)
        relative = np.divide(100 * importance, totals, out=np.zeros_like(importance), where=totals > 0)
        ranks = pd.DataFrame(importance).rank(axis=1, ascending=False, method='min').to_numpy()
        signs = np.where(np.abs(gradients) <= zero_tolerance, 0, np.sign(gradients))
        details, local, summary = [], [], []
        for r, idx in enumerate(selected):
            for j, feature in enumerate(FEATURES):
                comparable = (signs[0, :, j] != 0) & (signs[r, :, j] != 0)
                agreement = np.average(signs[0, comparable, j] == signs[r, comparable, j], weights=weights[comparable]) if comparable.any() else np.nan
                details.append({'equation_rank': r + 1, 'complexity': int(candidates.loc[idx, 'complexity']),
                    'feature': feature, 'included': SYMBOL_MAP[feature] in equations[r].free_symbols,
                    'mean_absolute_gradient': importance[r, j], 'importance_pct': relative[r, j], 'importance_rank': ranks[r, j],
                    'positive_fraction': np.average(signs[r, :, j] > 0, weights=weights),
                    'negative_fraction': np.average(signs[r, :, j] < 0, weights=weights),
                    'near_zero_fraction': np.average(signs[r, :, j] == 0, weights=weights),
                    'sign_agreement_with_primary': agreement,
                    'sign_comparison_coverage': np.average(comparable, weights=weights)})
                local.extend({'equation_rank': r + 1, 'row': int(row), 'study': data.groups[row],
                    'weight': float(weight), 'feature': feature, 'gradient': float(value)}
                    for row, weight, value in zip(indices, weights, gradients[r, :, j]))
        details = pd.DataFrame(details)
        for feature, group in details.groupby('feature', sort=False):
            alternatives = group[group.equation_rank > 1]
            summary.append({'feature': feature, 'inclusion_fraction': group.included.mean(),
                'importance_pct_min': group.importance_pct.min(), 'importance_pct_max': group.importance_pct.max(),
                'absolute_gradient_min': group.mean_absolute_gradient.min(), 'absolute_gradient_max': group.mean_absolute_gradient.max(),
                'rank_min': group.importance_rank.min(), 'rank_max': group.importance_rank.max(),
                'alternative_sign_agreement_mean': alternatives.sign_agreement_with_primary.mean(),
                'alternative_sign_comparison_coverage_mean': alternatives.sign_comparison_coverage.mean()})
        details.to_csv(output / f'{mode}_by_equation.csv', index=False)
        pd.DataFrame(local).to_csv(output / f'{mode}_local_gradients.csv', index=False)
        pd.DataFrame(summary).to_csv(output / f'{mode}_summary.csv', index=False)
        if make_plots:
            plot_sensitivity_robustness(gradients, weights, relative, ranks, candidates, selected, output, mode)

