import numpy as np

from ._base import LinearRegression

def _soft_thresholding(rho_j, z_j, lamda):
    """
    Soft threshold function used for normalized data and Lasso regression
    """
    if rho_j < -lamda:
        theta_j = rho_j + lamda
    elif (-lamda <= rho_j) and (rho_j <= lamda):
        theta_j = 0
    else:
        theta_j = rho_j - lamda

    return theta_j

def _compute_cost_function(X, y, theta, lamda, fit_intercept):
    """
    Function to compute the cost/objective function
        (1/(2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
        err_rss                              + err_l1
    """
    
    n_samples, n_features = X.shape

    pred = np.dot(X, theta)
    err_rss = np.dot(y - pred, y - pred) / (2 * n_samples)

    if fit_intercept:
        err_l1 = lamda * np.sum(np.abs(theta[: n_features - 1]))
    else:
        err_l1 = lamda * np.sum(np.abs(theta))

    cost = err_rss + err_l1

    return cost


class Lasso(LinearRegression):
    """
    Linear model trained with L1 regularizer

    The optimization objective for Lasso is:
        (1/(2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    The algorithm used to fit hte model is coordinate descent

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1 penalty.
        Controlling the regularization strength.
        `alpha` must be a non negative float i.e. [0, inf)

        When `alpha=0`, the objective is equivalent to
        ordinary least squares.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
        If set to False, no intercept will be used in calculations
        i.e. data is expected to be centered (manually)

    max_iter : int, default=1000
        The maximum number of iterations

    tol : float, default=1e-4
        The tolerance for the optimization.


    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coef. for the linear regression problem.

    intercept_ : float
        Independent term in the linear model.
        Set to 0.0 if `fit_intercept = False`

    Examples
    --------
    >>> import numpy as np
    >>> from yara.linear_model import Lasso
    >>> X = np.array([[0,0], [1,1], [2,2]])
    >>> y = np.array([0, 1, 2])
    >>> clf = Lasso(alpha = 0.1)
    >>> clf.fit(X, y)
    >>> clf.coef_
    array([0.85 0.  ])
    >>> clf.intercept_
    0.1499999999999998
    >>> clf.predict([[1.0,1.5], [2.0,-1.0]])
    array([1.  , 1.85])
    """

    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        super().__init__(fit_intercept=fit_intercept)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        """
        Fit the model with cyclic coordinate descent
        See: https://xavierbourretsicotte.github.io/lasso_derivation.html

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Prepare data
        X = np.array(X).copy()
        y = np.array(y).copy()

        # Extract size
        n_samples, n_features = X.shape

        # Initialize the design matrix, A
        if self.fit_intercept:
            A = np.column_stack((X, np.ones(n_samples)))
            n_features += 1  # add 1 to accommodate the intercept
        else:
            A = X

        # Initialize theta
        theta = np.zeros(n_features)

        # Start the coordinate descent
        for iter in range(self.max_iter):
            for j in range(n_features):
                # Extract needed data
                X_j = A[:, j]
                X_k = np.delete(A, j, axis=1)
                theta_k = np.delete(theta, j)

                # Calculate rho_j
                res_j = y - np.dot(X_k, theta_k)
                rho_j = np.dot(X_j, res_j)

                # Compute z_j
                z_j = np.dot(X_j, X_j)

                # Compute new theta_j from soft thresholding
                if self.fit_intercept:
                    if j == (n_features - 1):
                        theta[j] = rho_j
                    else:
                        theta[j] = _soft_thresholding(
                            rho_j, z_j, n_samples * self.alpha
                        )
                else:
                    theta[j] = _soft_thresholding(rho_j, z_j, n_samples * self.alpha)

                theta[j] /= z_j
            print(theta[j])

            # Stopping criterion
            cost_current = _compute_cost_function(
                A, y, theta, self.alpha, self.fit_intercept
            )
            if cost_current < self.tol:
                break

        # Extract parameters
        if self.fit_intercept:
            self.coef_ = theta[: n_features - 1]
            self.intercept_ = theta[-1]
        else:
            self.coef_ = theta
            self.intercept_ = 0.0
