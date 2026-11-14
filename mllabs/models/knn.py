"""K ближайших соседей на numpy: полный перебор с евклидовым расстоянием."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class _KNNBase(BaseEstimator):
    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", chunk_size: int = 1024):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.chunk_size = chunk_size

    def _store(self, X, y):
        # «Обучение» KNN — просто запоминание выборки.
        self.X_ = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y, dtype=np.float64)
        self.sq_norms_ = (self.X_ ** 2).sum(axis=1)
        return self

    def _neighbor_average(self, X):
        """Для каждого объекта — среднее (взвешенное) y по k ближайшим соседям."""
        X = np.asarray(X, dtype=np.float64)
        k = self.n_neighbors
        out = np.empty(len(X))
        # Считаем по частям, чтобы матрица расстояний помещалась в память.
        for start in range(0, len(X), self.chunk_size):
            batch = X[start:start + self.chunk_size]
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2ab — без явного цикла по парам.
            d2 = (batch ** 2).sum(axis=1)[:, None] + self.sq_norms_[None, :] - 2 * batch @ self.X_.T
            np.maximum(d2, 0, out=d2)
            idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
            neigh_y = self.y_[idx]
            if self.weights == "distance":
                dist = np.sqrt(np.take_along_axis(d2, idx, axis=1))
                w = 1.0 / (dist + 1e-9)
                out[start:start + len(batch)] = (w * neigh_y).sum(axis=1) / w.sum(axis=1)
            else:
                out[start:start + len(batch)] = neigh_y.mean(axis=1)
        return out


class KNNClassifierScratch(ClassifierMixin, _KNNBase):
    """Бинарный классификатор: вероятность класса 1 — доля соседей этого класса."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self._store(X, (np.asarray(y) == self.classes_[1]).astype(float))

    def predict_proba(self, X):
        p = self._neighbor_average(X)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


class KNNRegressorScratch(RegressorMixin, _KNNBase):
    def fit(self, X, y):
        return self._store(X, y)

    def predict(self, X):
        return self._neighbor_average(X)
