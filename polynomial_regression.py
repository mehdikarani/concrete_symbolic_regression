"""Fit the additive quadratic polynomial baseline."""
from analysis_common import ROOT, fit_polynomial_model, run_baseline
from data_preprocessing import prepare_data

DATA_PATH = ROOT / 'data.xlsx'
MODEL_PATH = ROOT / 'models' / 'polynomial_regression.json'
OUTPUT_DIR = ROOT / 'results' / 'polynomial_regression'
DEGREE = 2


def train(data_path=DATA_PATH, degree=DEGREE, *, prepared=None):
    data = prepared if prepared is not None else prepare_data(data_path)
    return fit_polynomial_model(data, degree=degree)


def main():
    run_baseline(DATA_PATH, MODEL_PATH, OUTPUT_DIR, degree=DEGREE)


if __name__ == '__main__':
    main()
