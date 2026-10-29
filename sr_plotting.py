"""Publication figures for symbolic regression selection and robustness."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis_common import DISPLAY_NAMES, DISPLAY_NAMES_SCALED, weighted_quantiles
from data_preprocessing import FEATURES

EQUATION_COLORS = ('#222222', '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00')
EQUATION_STYLES = ('-', '--', '-.', ':', (0, (5, 2)), (0, (3, 1, 1, 1)))


def style_axis(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='out', labelsize=14)
    ax.grid(axis='y', color='#dddddd', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)


def export_plot(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(path.with_suffix('.' + extension), dpi=600,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)


def equation_label(rank, complexity):
    name = 'Primary' if rank == 1 else f'Alternative {rank - 1}'
    return f'{name} (C = {int(complexity)})'


def plot_main_effect_robustness(curves, candidates, selected, output):
    fig, axes = plt.subplots(5, 2, figsize=(11, 16), layout='constrained')
    fig.get_layout_engine().set(wspace=0.10, rect=(0, 0.095, 1, 0.905))
    for panel, (ax, feature) in enumerate(zip(axes.flat, FEATURES)):
        subset = curves[curves.feature == feature]
        for rank, curve in subset.groupby('rank', sort=True):
            rank = int(rank)
            complexity = candidates.loc[selected[rank - 1], 'complexity']
            ax.plot(curve.feature_value, curve.predicted_strength,
                    color=EQUATION_COLORS[(rank - 1) % len(EQUATION_COLORS)],
                    linestyle=EQUATION_STYLES[(rank - 1) % len(EQUATION_STYLES)],
                    linewidth=1.8 if rank == 1 else 1.1,
                    zorder=10 if rank == 1 else 2,
                    label=equation_label(rank, complexity))
        label = DISPLAY_NAMES.get(feature, feature).replace('kg/m3', 'kg/m³')
        if feature in ('SiO2', 'Al2O3', 'Fe2O3', 'CaO'):
            label += ' (%)'
        ax.set_xlabel(label)
        ax.set_ylabel('Compressive strength (MPa)' if panel % 2 == 0 else '')
        ax.set_title(f'({chr(97 + panel)})', loc='left', fontsize=15)
        ax.margins(x=0.02)
        style_axis(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.005), ncol=3,
               frameon=True, fontsize=15, edgecolor='#777777', facecolor='white',
               framealpha=1, borderpad=0.8, borderaxespad=0.5,
               labelspacing=0.6, columnspacing=1.5)
    export_plot(fig, Path(output) / 'main_effect_robustness')


def plot_sensitivity_robustness(gradients, weights, importance, ranks, candidates, selected, output, mode):
    order = np.argsort(-importance[0], kind='stable')
    labels = [DISPLAY_NAMES_SCALED[FEATURES[j]] for j in order]
    equation_labels = [equation_label(r, candidates.loc[idx, 'complexity']).replace(' (', '\n(')
                       for r, idx in enumerate(selected, 1)]
    fig, ax = plt.subplots(figsize=(11, 8), layout='constrained')
    im = ax.imshow(importance[:, order].T, cmap='cividis', aspect='auto', vmin=0)
    ax.set_xticks(range(len(selected)), equation_labels, fontsize=13)
    ax.set_yticks(range(len(order)), labels, fontsize=15)
    for j in range(len(order)):
        for r in range(len(selected)):
            value = importance[r, order[j]]
            ax.text(r, j, f'{value:.1f}', ha='center', va='center', fontsize=13,
                    color='white' if im.norm(value) < 0.5 else 'black')
    colorbar = fig.colorbar(im, ax=ax, pad=0.02)
    colorbar.set_label('Relative sensitivity (%)', fontsize=16)
    colorbar.ax.tick_params(labelsize=14)
    export_plot(fig, output / f'{mode}_importance_heatmap')

    fig, ax = plt.subplots(figsize=(11, 7), layout='constrained')
    y = np.arange(len(order))
    ax.hlines(y, ranks[:, order].min(axis=0), ranks[:, order].max(axis=0), color='#bbbbbb', linewidth=3)
    for r in range(len(selected)):
        ax.scatter(ranks[r, order], y + (r - (len(selected) - 1) / 2) * 0.08,
                   color=EQUATION_COLORS[r % len(EQUATION_COLORS)], marker='*' if r == 0 else 'o',
                   s=140 if r == 0 else 35, label=equation_labels[r].replace('\n', ' '), zorder=5 if r == 0 else 3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xticks(range(1, len(FEATURES) + 1))
    ax.set_xlabel('Sensitivity rank (1 = most influential)')
    style_axis(ax)
    ax.legend(loc='upper right', ncol=2, frameon=True, fontsize=14,
              facecolor='white', edgecolor='#777777', framealpha=1,
              borderpad=0.7, labelspacing=0.5, columnspacing=1.2)
    export_plot(fig, output / f'{mode}_rank_stability')

    fig, axes = plt.subplots(5, 2, figsize=(11, 16), layout='constrained')
    fig.get_layout_engine().set(wspace=0.10, rect=(0, 0.095, 1, 0.905))
    for j, (ax, feature) in enumerate(zip(axes.flat, FEATURES)):
        stats = []
        for r in range(len(selected)):
            low, q1, median, q3, high = weighted_quantiles(gradients[r, :, j], weights, [0.05, 0.25, 0.5, 0.75, 0.95])
            stats.append(dict(whislo=low, q1=q1, med=median, q3=q3, whishi=high, fliers=[]))
        boxes = ax.bxp(stats, patch_artist=True, showfliers=False, widths=0.55)
        for r, box in enumerate(boxes['boxes']):
            box.set_facecolor(EQUATION_COLORS[r % len(EQUATION_COLORS)])
            box.set_alpha(0.65)
        for median in boxes['medians']:
            median.set_color('#222222')
            median.set_linewidth(1.4)
        ax.axhline(0, color='#777777', linestyle='--', linewidth=0.8)
        ax.set_xticks(range(1, len(selected) + 1), ['P'] + [f'A{r}' for r in range(1, len(selected))])
        ax.set_title(f'({chr(97 + j)}) {DISPLAY_NAMES_SCALED[feature]}', loc='left', fontsize=15)
        ax.set_ylabel('Signed sensitivity' if j % 2 == 0 else '')
        style_axis(ax)
    handles = [plt.Line2D([], [], color=EQUATION_COLORS[r % len(EQUATION_COLORS)], linewidth=4,
                         label=equation_labels[r].replace('\n', ' ')) for r in range(len(selected))]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.005),
               ncol=3, frameon=True, fontsize=15, edgecolor='#777777',
               facecolor='white', framealpha=1, borderpad=0.8,
               borderaxespad=0.5, labelspacing=0.6, columnspacing=1.5)
    export_plot(fig, output / f'{mode}_signed_sensitivity')


def plot_selection(candidates, selected, comparison, output):
    output = Path(output)
    robustness = output / 'robustness_analysis'
    robustness.mkdir(parents=True, exist_ok=True)
    valid = candidates[candidates.expression.notna()].sort_values('complexity')
    chosen = candidates.loc[selected]
    best = candidates.loc[selected[0]]

    fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
    ax.scatter(valid.complexity, np.sqrt(valid.train_mse), s=40,
               color='#aaaaaa', label='Other candidates', zorder=2)
    eligible = valid[valid.eligible]
    ax.scatter(eligible.complexity, np.sqrt(eligible.train_mse), s=45,
               color='#0072B2', label='Eligible candidates', zorder=3)
    ax.scatter(chosen.complexity, np.sqrt(chosen.train_mse), s=155,
               facecolors='none', edgecolors='#009E73', linewidths=1.4,
               label='Robustness set', zorder=4)
    ax.scatter([best.complexity], [np.sqrt(best.train_mse)], marker='*', s=285,
               color='#D55E00', edgecolors='white', linewidths=0.5,
               label='Selected primary', zorder=5)
    ax.set(xlabel='Equation complexity', ylabel='RMSE')
    style_axis(ax)
    ax.tick_params(axis='both', labelsize=18, length=6, width=1.2)
    ax.xaxis.label.set_size(22)
    ax.yaxis.label.set_size(22)
    ax.legend(frameon=False, fontsize=18, loc='upper right')
    export_plot(fig, output / 'pareto_front')

    alternatives = comparison[(comparison.model == 'SR') | comparison.model.str.startswith('Rank_')]
    pivot = alternatives.pivot(index='model', columns='split', values='study_rmse')
    order = ['SR'] + [f'Rank_{rank:02d}' for rank in range(2, len(selected) + 1)]
    pivot = pivot.reindex(order)
    x = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
    ax.bar(x - 0.18, pivot['train'], width=0.34, color='#0072B2', label='Training')
    ax.bar(x + 0.18, pivot['test'], width=0.34, color='#E69F00', label='Test', hatch='//')
    ax.set_xticks(x, [equation_label(rank, candidates.loc[idx, 'complexity']).replace(' (', '\n(')
                         for rank, idx in enumerate(selected, 1)])
    ax.set_ylabel('RMSE')
    ax.set_ylim(0, float(pivot[['train', 'test']].max().max()) * 1.2)
    style_axis(ax)
    ax.tick_params(axis='both', labelsize=18, length=6, width=1.2)
    ax.xaxis.label.set_size(22)
    ax.yaxis.label.set_size(22)
    ax.tick_params(axis='x', labelsize=16)
    ax.legend(frameon=False, fontsize=18, ncol=2)
    export_plot(fig, robustness / 'robustness_performance')

