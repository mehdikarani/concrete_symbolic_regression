"""Shared data loading, reproducible train/test split, and training-only scaling."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent
FEATURES = [
    'SiO2',
    'Al2O3',
    'Fe2O3',
    'CaO',
    'w_c',
    'Fineness_modulus',
    'Portland_cement_kg_m3',
    'Fine_aggregate_kg_m3',
    'Iron_tailing_kg_m3',
    'Coarse_aggregates_kg_m3',
]
TARGET = 'Compressive_strength_Mpa'
STUDY_COL = 'Studies'
TRAIN_RATIO = 0.85
SPLIT_SEED = 42


def study_weights(groups):
    """Assign each study equal total weight, normalized to a mean weight of 1."""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.mean()


@dataclass
class PreparedData:
    path: Path
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    scaler: MinMaxScaler
    Z: np.ndarray
    weights: np.ndarray
    target_center: float
    target_scale: float

    @property
    def standardized_target(self):
        return (self.y[self.train_indices] - self.target_center) / self.target_scale


def make_prepared_data(path, X, y, groups, train_indices, test_indices):
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    if len(train_indices) < 2 or len(test_indices) < 2:
        raise ValueError('Both training and test partitions must contain at least 2 rows.')
    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError('Training and test partitions overlap.')

    scaler = MinMaxScaler().fit(X[train_indices])
    Z = scaler.transform(X)
    weights = study_weights(groups[train_indices])
    y_train = y[train_indices]
    target_center = float(np.average(y_train, weights=weights))
    target_scale = float(np.sqrt(np.average((y_train - target_center) ** 2, weights=weights)))
    if target_scale <= 0:
        raise ValueError('Training target must have non-zero variance.')

    return PreparedData(
        path=path,
        X=X,
        y=y,
        groups=groups,
        train_indices=train_indices,
        test_indices=test_indices,
        scaler=scaler,
        Z=Z,
        weights=weights,
        target_center=target_center,
        target_scale=target_scale,
    )


def prepare_data(path=ROOT / 'data.xlsx'):
    path = Path(path).resolve()
    frame = pd.read_excel(path)
    numeric = frame[FEATURES + [TARGET]].apply(pd.to_numeric, errors='raise').to_numpy(float)
    if not np.isfinite(numeric).all() or frame[STUDY_COL].isna().any():
        raise ValueError('Features and target must be finite, and every row needs a study ID.')

    X, y = numeric[:, :-1], numeric[:, -1]
    groups = frame[STUDY_COL].to_numpy()
    indices = np.random.default_rng(SPLIT_SEED).permutation(len(X))
    train_size = int(len(X) * TRAIN_RATIO)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return make_prepared_data(
        path, X, y, groups, train_indices, test_indices
    )
