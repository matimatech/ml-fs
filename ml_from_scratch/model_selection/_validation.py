import copy

import numpy as np

from ..metrics import __all__
from ..model_selection import KFold


def cross_val_score(estimator, X, y, cv=5, scoring="mean_squared_error"):
    """
    Evaluate a score by cross-validation

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data

    X : array-like of shape (n_samples, n_features)
        The data to fit

    y : array-like of shape (n_samples,)
        Target variabel to try to predict

    scoring : str, default = "mean_squared_error"
        A scoring function

    cv : int, default=5
        The k of k-fold cross validation
    """
    # Extract data
    X = np.array(X).copy()
    y = np.array(y).copy()

    # split data
    kf = KFold(n_splits=cv)

    scoring = __all__[scoring]
    score_train_list = []
    score_test_list = []

    for i, (ind_train, ind_test) in enumerate(kf.split(X)):
        # extract data
        X_train = X[ind_train]
        y_train = y[ind_train]

        X_test = X[ind_test]
        y_test = y[ind_test]

        # create and fit model
        mdl = copy.deepcopy(estimator)
        mdl.fit(X=X_train, y=y_train)

        # predict
        y_pred_train = mdl.predict(X_train)
        y_pred_test = mdl.predict(X_test)

        # calculate error
        score_train = scoring(y_train, y_pred_train)
        score_test = scoring(y_test, y_pred_test)

        # append
        score_train_list.append(score_train)
        score_test_list.append(score_test)

    return score_train_list, score_test_list
