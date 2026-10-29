"""Fit the linear regression baseline."""
from analysis_common import ROOT, fit_polynomial_model, run_baseline
from data_preprocessing import prepare_data

DATA_PATH = ROOT / 'data.xlsx'
MODEL_PATH = ROOT / 'models' / 'linear_regression.json'
OUTPUT_DIR = ROOT / 'results' / 'linear_regression'


def train(data_path=DATA_PATH, *, prepared=None):
    data = prepared if prepared is not None else prepare_data(data_path)
    return fit_polynomial_model(data, degree=1)


def main():
    run_baseline(DATA_PATH, MODEL_PATH, OUTPUT_DIR, degree=1)


if __name__ == '__main__':
    main()
