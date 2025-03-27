import numpy as np


def _get_weights(dist, weights):
    if weights == "uniform":
        weights_arr = None
    else:
        weights_arr = 1.0 / (dist**2 + 1e-6)
    return weights_arr


class NearestNeighbor:
    def __init__(self, n_neighbors=5, p=2):
        self.n_neighbors = n_neighbors
        self.p = p

    def fit(self, X, y):
        self._X = np.array(X)
        self._y = np.array(y)

    def _compute_distance(self, x1, x2):
        abs_diff = np.abs(x1 - x2)
        abs_diff_p = np.power(abs_diff, self.p)
        sum_ = np.sum(abs_diff_p)
        dist = np.power(sum_, 1.0 / self.p)

        return dist

    def _kneighbors(self, X, return_distance=True):
        # hitung ukuran distance
        n_queries = X.shape[0]
        n_samples = self._X.shape[0]
        list_dist = np.empty((n_queries, n_samples))

        # looping untuk menentukan jarak per data point
        for i in range(n_queries):
            X_i = X[i]

            # looping ke seluruh sample data
            for j in range(n_samples):
                # cari sample data ke-j
                X_j = self._X[j]

                # ukur jarak antara x_i dan X_j
                dist_ij = self._compute_distance(x1=X_i, x2=X_j)

                list_dist[i, j] = dist_ij

        neigh_ind = np.argsort(list_dist, axis=1)[:, : self.n_neighbors]
        if return_distance:
            neigh_dist = np.sort(list_dist, axis=1)[:, : self.n_neighbors]
            return neigh_dist, neigh_ind

        else:
            return neigh_ind
