import numpy as np

class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=42):  # reproduce random
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(self, X):
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)

        n_splits = self.n_splits
        n_fold = n_samples // n_splits
        fold_sizes = np.ones(n_splits, dtype=int) * n_fold
        fold_sizes[: n_samples % n_splits] += 1
        print(fold_sizes)

        current = 0
        for fold_size in fold_sizes:
            start = current
            stop = current + fold_size

            yield indices[start:stop]

            current = stop

    def split(self, X):
        n_samples = len(X)
        indices = np.arange(n_samples)
        # print(indices)

        for test_index in self._iter_test_indices(X):
            print(f"TEST INDEX {test_index}")
            # print(test_index)
            train_index = np.array([ind for ind in indices if ind not in test_index])

            yield (train_index, test_index)
