import numpy as np

from ._base import NearestNeighbor, _get_weights


class KNeighborsRegressor(NearestNeighbor):
    def __init__(self, n_neighbors=5, weights="uniform", p=2):
        super().__init__(n_neighbors=n_neighbors, p=p)
        self.weights = weights

    def predict(self, X):
        # konversi input ke numpy array
        X = np.array(X)

        # cari tau weights
        if self.weights == "uniform":
            neigh_ind = self._kneighbors(X, return_distance=False)
            neigh_dist = None

        else:
            neigh_ind, neigh_dist = self._kneighbors(X)

        # Tentukan bobot melakukan prediksi
        weights = _get_weights(neigh_dist, self.weights)

        # predict
        if self.weights == "uniform":
            y_pred = np.mean(self._y[neigh_ind], axis=1)
        else:
            num = np.sum(self._y[neigh_ind] * weights, axis=1)
            denom = np.sum(weights, axis=1)
            y_pred = num / denom

        return y_pred
