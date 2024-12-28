import numpy as np


class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # prepare data
        X = np.array(X).copy()
        y = np.array(y).copy()

        # Extract size
        n_samples, _ = X.shape

        # panggil normal equation
        # buat design matrix
        if self.fit_intercept:
            A = np.column_stack((np.ones(n_samples), X))
        else:
            A = X

        # cari model parameter terbaik
        beta = np.linalg.inv(A.T @ A) @ A.T @ y

        # extract model parameter
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta[1:]

    def predict(self, X):
        X = np.array(X)
        y = np.dot(X, self.coef_) + self.intercept_
        return y
