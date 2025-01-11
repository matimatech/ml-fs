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


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=10000, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def predict_proba(self, X):
        logits = np.dot(X, self.coef_) + self.intercept_
        proba = 1.0 / (1 + np.exp(-logits))

        return proba

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype("int")

    def fit(self, X, y):
        X = np.array(X).copy()
        y = np.array(y).copy()

        self.n_samples, self.n_features = X.shape

        self.coef_ = np.zeros(self.n_features)
        self.intercept_ = 0

        # tuning parameter
        for _ in range(self.max_iter):
            # make a new pred
            y_pred = self.predict_proba(X)

            # calculate the gradient
            grad_coef_ = -(y - y_pred).dot(X) / self.n_samples
            grad_intercept_ = (
                -(y - y_pred).dot(np.ones(self.n_samples)) / self.n_samples
            )

            # update parameter
            self.coef_ -= self.learning_rate * grad_coef_
            self.intercept_ -= self.learning_rate * grad_intercept_

            # break the iteration
            grad_stack = np.hstack((grad_coef_, grad_intercept_))
            if all(np.abs(grad_stack) < self.tol):
                break
