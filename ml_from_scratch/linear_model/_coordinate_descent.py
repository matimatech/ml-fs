import numpy as np

from ._base import LinearRegression


def _soft_thresholding(rho_j, z_j, lamda):
    if rho_j < -lamda:
        theta_j = rho_j + lamda
    elif (-lamda <= rho_j) and (rho_j <= lamda):
        theta_j = 0
    else:
        theta_j = rho_j - lamda

    return theta_j


class Lasso(LinearRegression):
    def __init__(self, fit_intercept=True, alpha=1.0, max_iter=1000, tol=1e-4):
        super().__init__(fit_intercept=fit_intercept)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        # prepare data
        X = np.array(X).copy()
        y = np.array(y).copy()

        # extract s8ze
        n_sample, n_feature = X.shape

        # design matrix

        if self.fit_intercept:
            A = np.column_stack((X, np.ones(n_sample)))
            n_feature += 1

        else:
            A = X

        # initialize theta
        theta = np.zeros(n_feature)
        print(n_feature)

        for iter in range(self.max_iter):
            for j in range(n_feature):
                x_j = A[:, j]
                x_k = np.delete(A, j, axis=1)
                theta_k = np.delete(theta, j)

                res_j = y - np.dot(x_k, theta_k)

                rho_j = np.dot(x_j, res_j)

                z_j = np.dot(x_j, x_j)

                print("theta j lama", theta[j])

                # compute new theta j
                if self.fit_intercept:
                    if j == (n_feature - 1):
                        theta[j] = rho_j
                    else:
                        theta[j] = _soft_thresholding(rho_j, z_j, n_sample * self.alpha)
                else:
                    theta[j] = _soft_thresholding(rho_j, z_j, n_sample * self.alpha)

                theta[j] /= z_j

                print(theta[j])

            # if self.fit_intercept:
            #     self.intercept_ =
