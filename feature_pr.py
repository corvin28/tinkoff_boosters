import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold


#counters
def make_counters(X, y, folding=True, n_folds=10):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    new_X = np.zeros((n_samples, n_features), dtype=float)
    if folding:
        kf = KFold(n=n_samples, n_folds=n_folds, shuffle=True)
    else:
        kf = [(np.arange(n_samples), np.arange(n_samples))]
    for train_ind, test_ind in kf:
        # train_ind - на основе чего строим счетчики
        for i in range(0, n_features):
            uniq_values, counts = np.unique(X[train_ind, i], return_counts=True)
            for j in range(0, uniq_values.shape[0]):
                ind = test_ind[np.where(X[test_ind, i] == uniq_values[j])[0]]
                # new_X[ind, i] = counts[j] / train_ind.shape[0]  # учитываем размер train
                successes = np.sum(y[train_ind[np.where(X[train_ind, i] == uniq_values[j])[0]]])
                # new_X[ind, n_features + i] = successes / train_ind.shape[0]
                # new_X[ind, 2 * n_features + i] = (successes + 1) / (counts[j] + 2)
                new_X[ind, i] = (successes + 1) / (counts[j] + 2)
    return new_X


def make_counters_test(X_test, X_train, y_train):
    n_samples = X_test.shape[0]
    n_features = X_test.shape[1]
    new_X = np.zeros((n_samples, n_features), dtype=float)
    for i in range(0, n_features):
        uniq_values, counts = np.unique(X_train[:, i], return_counts=True)
        for j in range(0, uniq_values.shape[0]):
            ind = np.where(X_test[:, i] == uniq_values[j])[0]
            # new_X[ind, i] = counts[j] / X_train.shape[0]
            successes = np.sum(y_train[np.where(X_train[:, i] == uniq_values[j])[0]])
            # new_X[ind, n_features + i] = successes / X_train.shape[0]
            # new_X[ind, 2 * n_features + i] = (successes + 1) / (counts[j] + 2)
            new_X[ind, i] = (successes + 1) / (counts[j] + 2)
    return new_X


def make_pairs(d, col_names):
    n_features = len(col_names)
    new_col_names = []
    for i in range(n_features - 1):
        for j in range(i + 1, n_features):
            d[col_names[i] + '__' + col_names[j]] = d[col_names[i]].astype(str) + '#' + d[col_names[j]].astype(str)
            new_col_names.append(col_names[i] + '__' + col_names[j])
    return d, new_col_names


#rare values compression
def compress_vals(dtr, dte, col_name, threshold, value):
    a = dtr[col_name].value_counts()[dtr[col_name].value_counts() < threshold].index.values
    for val in a:
        dtr.loc[dtr[col_name] == val, col_name] = value
        dte.loc[dte[col_name] == val, col_name] = value
    return None


def replace_val(d, col_name, olds, new):
    for old in olds:
        d.loc[d[col_name] == old, col_name] = new
    return None
