"""Run linear regression and its statistics and plots."""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from analysis_common import (FEATURES, arguments, load_model, make_model, read_data,
                             report, save_model, split_indices)


def train(data, X, y):
    train_indices, test_indices = split_indices(X)
    scaler = MinMaxScaler().fit(X[train_indices])
    model = LinearRegression().fit(scaler.transform(X[train_indices]), y[train_indices])
    expression = repr(float(model.intercept_)) + ''.join(
        f' + ({float(coefficient)!r})*{feature}' for coefficient, feature in zip(model.coef_, FEATURES))
    return make_model(data, expression, scaler, train_indices, test_indices,
                      {'seed': 42, 'test_size': 0.15, 'scaling': 'training_only'})


def main():
    args = arguments('linear_regression')
    X, y = read_data(args.data)
    model = train(args.data, X, y) if args.train else load_model(args.model, args.data)
    report(model, X, y, args.output)
    if args.train:
        save_model(args.output / 'model.json', model)


if __name__ == '__main__':
    main()
