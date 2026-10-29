"""Step 2: Audit and select saved SR equations on training data and generate reports."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from analysis_common import (
    SYMBOL_MAP, fit_polynomial_model, make_model, parse_equation, parse_polynomial,
    report, save_model, write_equation,
)
from data_preprocessing import FEATURES, ROOT, STUDY_COL, TARGET, make_prepared_data, prepare_data
from robustness_analysis import analyse_main_effect_robustness, analyse_sensitivity_robustness
from sr_plotting import plot_selection


def compute_scores(actual, predicted, groups):
    errors = (np.asarray(actual) - predicted) ** 2
    per_study_mse = [errors[groups == study].mean() for study in np.unique(groups)]
    return {
        'rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
        'r2': float(r2_score(actual, predicted)),
        'study_rmse': float(np.sqrt(np.mean(per_study_mse))),
    }


def audit_candidates(equations, data, min_features, max_degree):
    Z_train = data.Z[data.train_indices]
    y_train = data.y[data.train_indices]
    train_groups = data.groups[data.train_indices]
    y_center, y_scale = data.target_center, data.target_scale

    rows = []
    for _, candidate in equations.iterrows():
        complexity = int(candidate['complexity'])
        raw_text = str(candidate['sympy_format'])
        row = {
            'complexity': complexity,
            'search_loss': float(candidate['loss']),
            'search_score': float(candidate['score']) if 'score' in candidate and pd.notna(candidate['score']) else np.nan,
            'search_expression': raw_text,
            'eligible': False,
        }
        try:
            raw_expr = sp.sympify(raw_text, locals=SYMBOL_MAP)
            unscaled = sp.Float(y_center) + sp.Float(y_scale) * raw_expr
            expr, poly, predict = parse_polynomial(unscaled, max_degree=max_degree)
            predicted = predict(Z_train)
            train_scores = compute_scores(y_train, predicted, train_groups)
            mse = float(np.average((predicted - y_train) ** 2, weights=data.weights))
            n_features = len(expr.free_symbols)
            norm_mse = mse / (y_scale ** 2)
            n_terms = len(poly.monoms())
            eligible = n_features >= min_features
            row.update(
                expression=str(expr),
                n_features=n_features,
                train_mse=mse,
                train_rmse=train_scores['rmse'],
                train_r2=train_scores['r2'],
                normalized_train_mse=norm_mse,
                n_terms=n_terms,
                eligible=eligible,
                reason='ok' if eligible else 'insufficient_features',
            )
        except (ValueError, TypeError, sp.PolynomialError, OverflowError) as error:
            row['reason'] = str(error)
        rows.append(row)

    return pd.DataFrame(rows).drop_duplicates('search_expression').reset_index(drop=True)


def weighted_primary_scores(pool, score_weight):
    """Normalize training MSE and PySR score to [0, 1]; smaller total is better."""
    mse = pool.train_mse.astype(float)
    mse_span = mse.max() - mse.min()
    mse_cost = (mse - mse.min()) / mse_span if mse_span else mse * 0
    if score_weight == 0:
        return mse_cost
    score = pool.search_score.astype(float)
    if not np.isfinite(score).all():
        raise ValueError('Weighted selection requires finite PySR scores in the primary pool.')
    score_span = score.max() - score.min()
    score_cost = (score.max() - score) / score_span if score_span else score * 0
    return (1 - score_weight) * mse_cost + score_weight * score_cost


def select_diverse(candidates, *, selection_policy='weighted_score', score_weight=0.5,
                   primary_margin=0.05, robustness_margin=0.10, max_selected=6):
    eligible = candidates[candidates.eligible]
    if eligible.empty:
        return []

    best_mse = eligible.train_mse.min()
    pool = eligible[eligible.train_mse <= best_mse * (1 + robustness_margin)].copy()
    primary_pool = eligible[eligible.train_mse <= best_mse * (1 + primary_margin)].copy()
    if selection_policy == 'lowest_error':
        best_idx = eligible.sort_values(['train_mse', 'complexity']).index[0]
    elif selection_policy == 'simplest_near_best':
        best_idx = primary_pool.sort_values(['complexity', 'train_mse']).index[0]
    elif selection_policy == 'weighted_score':
        costs = weighted_primary_scores(primary_pool, score_weight)
        best_idx = primary_pool.assign(weighted_cost=costs).sort_values(
            ['weighted_cost', 'train_mse', 'complexity']).index[0]
    else:
        raise ValueError(f'Unknown selection policy: {selection_policy}')
    selected = [best_idx]
    complexities = sorted(pool.complexity.unique())
    if len(complexities) == 1:
        return selected
    targets = [complexities[0], complexities[-1]]
    targets.extend(np.linspace(complexities[0], complexities[-1], max_selected)[1:-1])
    for target in targets:
        if len(selected) >= max_selected:
            break
        used = {pool.loc[idx, 'complexity'] for idx in selected}
        choices = pool[~pool.complexity.isin(used)]
        if choices.empty:
            break
        idx = min(choices.index, key=lambda i: (
            abs(float(pool.loc[i, 'complexity']) - target),
            float(pool.loc[i, 'train_mse']), int(i)))
        selected.append(idx)

    return selected


def load_search(run_directory, data_path=None):
    run_directory = Path(run_directory).resolve()
    legacy_path = run_directory / 'pysr_equations.csv'
    if legacy_path.is_file() and not (run_directory / 'run.json').is_file():
        # Legacy equations require scaling fitted on all rows.
        frame = pd.read_excel(data_path or ROOT / 'data.xlsx')
        X = frame[FEATURES].to_numpy(float)
        y = frame[TARGET].to_numpy(float)
        groups = frame[STUDY_COL].to_numpy() if STUDY_COL in frame else np.arange(len(frame))
        train_idx, test_idx = train_test_split(np.arange(len(frame)), test_size=0.15,
                                               random_state=42)
        data = make_prepared_data(Path(data_path or ROOT / 'data.xlsx'), X, y, groups,
                                  train_idx, test_idx)
        data.scaler = MinMaxScaler().fit(X)
        data.Z = data.scaler.transform(X)
        data.weights = np.ones(len(train_idx))
        data.target_center, data.target_scale = 0.0, 1.0
        equations = pd.read_csv(legacy_path)
        required_cols = {'complexity', 'loss', 'sympy_format'}
        if not required_cols.issubset(equations.columns):
            raise ValueError(f'{legacy_path} lacks {sorted(required_cols - set(equations.columns))}')
        return {'min_features': 7, 'max_degree': 9, 'parsimony': None,
                'source_format': 'legacy_pysr_equations',
                'legacy_global_scaling': True}, data, equations[list(required_cols) + (['score'] if 'score' in equations else [])].copy()
    context = json.loads((run_directory / 'run.json').read_text(encoding='utf-8'))

    arguments = context.get('arguments', {})
    settings = context.get('settings', {})
    resolved_data_path = data_path or context.get('data_path') or arguments.get('data') or (ROOT / 'data.xlsx')
    data = prepare_data(resolved_data_path)
    if (context.get('train_indices') != data.train_indices.tolist()
            or context.get('test_indices') != data.test_indices.tolist()):
        raise ValueError('Search run uses a different or unknown data split. Rerun symbolic_regression.py.')
    if context.get('target_mode') == 'original_mpa':
        data.target_center, data.target_scale = 0.0, 1.0

    candidates_path = run_directory / 'pareto_candidates.csv'
    if not candidates_path.is_file():
        raise FileNotFoundError(candidates_path)
    equations = pd.read_csv(candidates_path)
    required_cols = {'complexity', 'loss', 'sympy_format'}
    if not required_cols.issubset(equations.columns):
        raise ValueError(f'pareto_candidates.csv must contain columns: {sorted(required_cols)}')

    config = {
        'min_features': context.get('min_features') or settings.get('min_features') or arguments.get('min_features', 7),
        'max_degree': context.get('max_degree') or settings.get('max_degree') or arguments.get('max_degree', 4),
        'parsimony': context.get('parsimony') or settings.get('parsimony', 0.0032),
        'source_format': context.get('source_format', 'saved_search'),
    }
    return config, data, equations[['complexity', 'loss', 'sympy_format'] + (['score'] if 'score' in equations else [])].copy()


def run_selection(run_directory, *, data_path=None, output=None, make_plots=True,
                  selection_policy='weighted_score', score_weight=0.5, primary_margin=0.05,
                  robustness_margin=0.10):
    if not 0 <= primary_margin <= robustness_margin:
        raise ValueError('Margins must satisfy 0 <= primary_margin <= robustness_margin.')
    if not 0 <= score_weight <= 1:
        raise ValueError('score_weight must be between 0 and 1.')
    if selection_policy not in ('simplest_near_best', 'lowest_error', 'weighted_score'):
        raise ValueError(f'Unknown selection policy: {selection_policy}')
    config, data, equations = load_search(run_directory, data_path)
    min_features = config['min_features']
    max_degree = config['max_degree']
    parsimony = config['parsimony']
    candidates = audit_candidates(equations, data, min_features, max_degree)
    selected = select_diverse(candidates, selection_policy=selection_policy,
                              score_weight=score_weight,
                              primary_margin=primary_margin,
                              robustness_margin=robustness_margin)
    eligible = candidates[candidates.eligible]
    if not eligible.empty:
        best_mse = float(eligible.train_mse.min())
        candidates['training_mse_above_best_pct'] = (
            (candidates.train_mse / best_mse - 1) * 100 if best_mse > 0
            else np.where(candidates.train_mse == 0, 0.0, np.inf)
        )

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if output:
        output = Path(output).resolve()
    else:
        root = ROOT / 'results' / 'symbolic_regression' / 'selection'
        output = root / f'{Path(run_directory).name}-{stamp}'
    output.mkdir(parents=True, exist_ok=False)

    candidates['selected_rank'] = np.nan
    for rank, idx in enumerate(selected, 1):
        candidates.loc[idx, 'selected_rank'] = rank
    candidates.to_csv(output / 'candidates.csv', index=False)

    selection_info = {
        'search_directory': str(Path(run_directory).resolve()),
        'rule': selection_policy,
        'score_weight': score_weight,
        'score_normalization': 'minmax_within_primary_pool; higher_PySR_score_is_better',
        'primary_margin': primary_margin,
        'robustness_margin': robustness_margin,
        'robustness_policy': 'even_complexity_grid_with_endpoints',
        'min_features': min_features,
        'max_degree': max_degree,
        'parsimony': parsimony,
        'source_format': config['source_format'],
        'legacy_global_scaling': config.get('legacy_global_scaling', False),
        'test_used_for_selection': False,
        'status': 'selected' if selected else 'no_eligible_equation',
    }
    (output / 'selection.json').write_text(json.dumps(selection_info, indent=2) + '\n', encoding='utf-8')
    if not selected:
        raise RuntimeError(f'No >= {min_features}-feature candidate found. Audit saved to: {output}')

    best = candidates.loc[selected[0]]
    model = make_model(data, best.expression, {'model_type': 'symbolic_regression', **selection_info})
    save_model(output / 'model.json', model)

    best_expr, _, sr_predict = parse_polynomial(best.expression, max_degree=max_degree)
    write_equation(model, best_expr, output)

    lr_model = fit_polynomial_model(data, degree=1)
    pr_model = fit_polynomial_model(data, degree=2)
    _, lr_predict = parse_equation(lr_model['expression'])
    _, pr_predict = parse_equation(pr_model['expression'])

    predictors = {'SR': sr_predict, 'LR': lr_predict, 'PR': pr_predict}
    for rank, idx in enumerate(selected[1:], 2):
        _, _, candidate_predict = parse_polynomial(candidates.loc[idx, 'expression'], max_degree=max_degree)
        predictors[f'Rank_{rank:02d}'] = candidate_predict

    comparison, predictions = [], []
    for split_name, indices in [('train', data.train_indices), ('test', data.test_indices)]:
        for label, predictor in predictors.items():
            values = predictor(data.Z[indices])
            comparison.append({
                'model': label,
                'split': split_name,
                'n': len(indices),
                **compute_scores(data.y[indices], values, data.groups[indices]),
            })
            predictions.extend(
                {
                    'model': label,
                    'row': int(i),
                    'study': data.groups[i],
                    'split': split_name,
                    'actual': data.y[i],
                    'predicted': val,
                }
                for i, val in zip(indices, values)
            )

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(output / 'comparison.csv', index=False)
    pd.DataFrame(predictions).to_csv(output / 'predictions.csv', index=False)
    robustness = output / 'robustness_analysis'
    robustness.mkdir(parents=True, exist_ok=True)
    candidates.loc[selected].to_csv(robustness / 'robustness_equations.csv', index=False)
    analyse_main_effect_robustness(candidates, selected, data, robustness, max_degree,
                                   make_plots=make_plots)

    analyse_sensitivity_robustness(candidates, selected, data, robustness / 'sensitivity',
                                   make_plots=make_plots)

    if make_plots:
        plot_selection(candidates, selected, comparison_df, output)
        report(model, data.X, data.y, output)

    print(comparison_df[comparison_df.model.isin(['SR', 'LR', 'PR'])].to_string(index=False))
    print(f'Selected {len(selected)} frozen equations; output: {output}')
    return output


def resolve_run_directory(run_directory=None, *, search_root=ROOT / 'results' / 'symbolic_regression' / 'search'):
    if run_directory is not None:
        directory = Path(run_directory).resolve()
        if not ((directory / 'run.json').is_file() and (directory / 'pareto_candidates.csv').is_file()) and not (directory / 'pysr_equations.csv').is_file():
            raise FileNotFoundError(
                f'{directory} must contain run.json and pareto_candidates.csv, or pysr_equations.csv.'
            )
        return directory

    completed = []
    for path in Path(search_root).glob('*/run.json'):
        if not (path.parent / 'pareto_candidates.csv').is_file():
            continue
        try:
            context = json.loads(path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(context, dict) and context.get('status') == 'search_complete':
            completed.append(path.parent)

    if not completed:
        legacy = Path(search_root).parent / 'pysr_equations.csv'
        if legacy.is_file():
            return legacy.parent.resolve()
        raise FileNotFoundError(
            f'No completed SR search found under {Path(search_root).resolve()}. '
            'Run symbolic_regression.py first, or pass a saved run directory.'
        )
    return max(completed, key=lambda d: d.name).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', nargs='?', type=Path,
                        help='Saved search directory; defaults to the latest completed SR run')
    parser.add_argument('--data', type=Path, help='Override dataset path')
    parser.add_argument('--output', type=Path, help='Custom output directory (must not exist)')
    parser.add_argument('--no-plots', action='store_true', help='Write tables and models without plots')
    parser.add_argument('--selection-policy', choices=('weighted_score', 'simplest_near_best', 'lowest_error'),
                        default='weighted_score', help='Primary equation rule (training data only)')
    parser.add_argument('--score-weight', type=float, default=1,
                        help='Weight of normalized PySR score in weighted_score; 0=MSE only, 1=score only')
    parser.add_argument('--primary-margin', type=float, default=0.15,
                        help='Fractional training MSE tolerance for primary pool (default: 0.15)')
    parser.add_argument('--robustness-margin', type=float, default=0.25,
                        help='Fractional training MSE tolerance for robustness set (default: 0.25)')
    args = parser.parse_args()

    try:
        directory = resolve_run_directory(args.run_directory)
    except FileNotFoundError as error:
        parser.error(str(error))

    print(f'Using SR search: {directory}')
    run_selection(directory, data_path=args.data, output=args.output, make_plots=not args.no_plots,
                  selection_policy=args.selection_policy, score_weight=args.score_weight,
                  primary_margin=args.primary_margin,
                  robustness_margin=args.robustness_margin)


if __name__ == '__main__':
    main()
