import numpy as np

class SVC:
    """
    Support Vector Classifier
    Menggunakan SMO Algorithm
    Ref: https://cs229.stanford.edu/materials/smo.pdf

    Parameters
    ----------
    C: float, default=1.0
        Regularization parameter

    tol : float, default=1e-5
        Tolerance for stopping criteria

    max_passes : int, default=10
        Maximum # of times to iterate over a's without changing

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Coefficient of the features in the decision function

    intercept_ : ndarray of shape (1,)

    Examples
    --------
    >>> import numpy as np
    >>> from ml_from_scratch.svm import SVC
    >>> X = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> y = np.array([-1, 1, 1, 1])
    >>> clf = SVC(C=10)
    >>> clf.fit(X, y)
    >>> clf.coef_
    array([2. 2.])
    >>> clf.intercept_
    -1.0
    """

    def __init__(self, C=1.0, tol=1e-5, max_passes=10):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes

    def _initialize_parameters(self, X):
        """Initialize parameters"""
        n_samples, n_features = X.shape

        self.alpha = np.zeros(n_samples)
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

    def _get_random_sample(self, idx, n):
        """Get random sample where j!=idx"""
        # Initialize counter
        j = np.random.choice(n)
        while j == idx:
            j = np.random.choice(n)

        return j

    def _compute_boundary(self, y_i, y_j, a_i, a_j):
        """
        Obtain the lower & upper boundary for a_j
        Eq. (10) & (11)
        """
        if y_i != y_j:
            L = max(0, a_j - a_i)
            H = min(self.C, self.C + (a_j - a_i))
        else:
            L = max(0, a_i + a_j - self.C)
            H = min(self.C, a_i + a_j)

        return L, H

    def _compute_coef(self, X, y):
        """w = sigma(alpha * y * x)"""
        self.coef_ = np.dot(self.alpha * y, X)

    def _calculate_F(self, X_star, X, y):
        """Solve Eq.(2)"""
        return np.dot(self.alpha * y, np.dot(X, X_star.T)) + self.intercept_

    def _calculate_E(self, X_star, y_star, X, y):
        """Solve Eq.(13)"""
        f = self._calculate_F(X_star, X, y)
        return f - y_star

    def fit(self, X, y):
        """
        Fit the model according to the given training data

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectori, where `n_samples` is the number of samples and
            `n_features` is the number of features

        y : array-like of shape (n_samples,)
            Target vector relative to X

        Returns
        -------
        self
            Fitted estimator
        """
        # Change input format
        X = np.array(X)
        y = np.array(y)
        n_samples, _ = X.shape

        # Initialize variables
        self._initialize_parameters(X)
        passes = 0

        # Start tuning the parameters
        while passes < self.max_passes:
            num_changed_alphas = 0

            # Iterate over samples
            for i in range(n_samples):
                # Extract data
                X_i, y_i, a_i = X[i, :], y[i], self.alpha[i]

                # Calculate E_i
                E_i = self._calculate_E(X_i, y_i, X, y)

                # Check condition
                cond_1 = (y_i * E_i < -self.tol) and (a_i < self.C)
                cond_2 = (y_i * E_i > self.tol) and (a_i > 0)
                if cond_1 or cond_2:
                    # Select j randomly
                    j = self._get_random_sample(i, n_samples)

                    # Extract data
                    X_j, y_j, a_j = X[j, :], y[j], self.alpha[j]

                    # Calculate E_j
                    E_j = self._calculate_E(X_j, y_j, X, y)

                    # Save old a's
                    a_i_old, a_j_old = a_i, a_j

                    # Compute L and H
                    L, H = self._compute_boundary(y_i, y_j, a_i_old, a_j_old)
                    if L == H:
                        continue

                    # Compute eta Eq (14)
                    eta = 2 * np.dot(X_i, X_j) - np.dot(X_i, X_i) - np.dot(X_j, X_j)
                    if eta >= 0:
                        continue

                    # Clip value for a_j
                    # Get a_j_unclipped, Eq.(12)
                    a_j_unclipped = a_j_old - (y_j * (E_i - E_j)) / eta

                    # Get a_j_clipped, Eq.(15)
                    if a_j_unclipped > H:
                        a_j_new = H
                    elif (a_j_unclipped >= L) and (a_j_unclipped <= H):
                        a_j_new = a_j_unclipped
                    else:
                        a_j_new = L

                    if np.abs(a_j_new - a_j_old) < self.tol:
                        continue

                    # Get the a_i, Eq.(16)
                    a_i_new = a_i_old + (y_i * y_j) * (a_j_old - a_j_new)

                    # Compute b_1 and b_2
                    b_old = self.intercept_

                    # compute b_1, Eq.(17)
                    b_1 = (
                        b_old
                        - E_i
                        - (y_i * (a_i_new - a_i_old)) * np.dot(X_i, X_i)
                        - (y_j * (a_j_new - a_j_old)) * np.dot(X_i, X_j)
                    )

                    # compute b_2, Eq.(18)
                    b_2 = (
                        b_old
                        - E_j
                        - (y_i * (a_i_new - a_i_old)) * np.dot(X_i, X_j)
                        - (y_j * (a_j_new - a_j_old)) * np.dot(X_j, X_j)
                    )

                    # compute b, Eq. (19)
                    if (a_i > 0) & (a_i < self.C):
                        b_new = b_1
                    elif (a_j > 0) & (a_j < self.C):
                        b_new = b_2
                    else:
                        b_new = 0.5 * (b_1 + b_2)

                    # Update variables
                    self.alpha[i], self.alpha[j] = a_i_new, a_j_new
                    self.intercept_ = b_new
                    self._compute_coef(X, y)

                    # Update counter
                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

    def predict(self, X):
        """
        Prediksi Kelas

        Parameters
        ----------
        X : Array berukuran (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features

        Returns
        -------
        T : array-like of shape (n_samples,)
            Returns the probability of the sample for each class in the model,
        """
        return np.sign(np.dot(X, self.coef_) + self.intercept_).astype("int")
