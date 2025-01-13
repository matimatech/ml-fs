import numpy as np

from ._base import LinearRegression


class Ridge(LinearRegression):
    def __init__(self, fit_intercept=True, alpha=1.0, max_iter=1000, tol=1e-4):
        super().__init__(fit_intercept=fit_intercept)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.array(X).copy()
        y = np.array(y).copy()

        n_samples, n_features = X.shape

        # Initialize the design matrix, A
        if self.fit_intercept:
            alpha_I = self.alpha * np.identity(n_features + 1)
            alpha_I[-1, -1] = 0.0
            A = np.column_stack((X, np.ones(n_samples)))
        else:
            alpha_I = self.alpha * np.identity(n_features)
            A = X

        print(A.shape, (self.alpha * np.identity(n_features)).shape, A.T.shape, y.shape)
        # cari model paramater terbaik
        beta = np.linalg.inv(A.T @ A + alpha_I) @ A.T @ y

        if self.fit_intercept:
            self.intercept_ = beta[-1]
            self.coef_ = beta[:n_features]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta[:]
            
