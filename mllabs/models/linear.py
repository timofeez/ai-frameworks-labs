"""Линейная и логистическая регрессия на numpy."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


def _add_intercept(X):
    X = np.asarray(X, dtype=np.float64)
    return np.column_stack([np.ones(len(X)), X])


class LinearRegressionScratch(RegressorMixin, BaseEstimator):
    """МНК с необязательной L2-регуляризацией (гребневая регрессия).

    alpha = 0 — обычная линейная регрессия через псевдообратную матрицу
    (устойчиво к линейно зависимым one-hot столбцам).
    Свободный член не штрафуется.
    """

    def __init__(self, alpha: float = 0.0):
        self.alpha = alpha

    def fit(self, X, y):
        A = _add_intercept(X)
        y = np.asarray(y, dtype=np.float64)
        if self.alpha == 0:
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
        else:
            penalty = self.alpha * np.eye(A.shape[1])
            penalty[0, 0] = 0
            w = np.linalg.solve(A.T @ A + penalty, A.T @ y)
        self.intercept_, self.coef_ = w[0], w[1:]
        return self

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_


class LogisticRegressionScratch(ClassifierMixin, BaseEstimator):
    """Логистическая регрессия, обучаемая методом Ньютона (IRLS).

    Минимизирует C * sum(w_i * logloss_i) + 0.5 * ||w||^2 — та же постановка, что в sklearn.
    class_weight='balanced' умножает потери объектов на веса классов.
    """

    def __init__(self, C: float = 1.0, class_weight=None, max_iter: int = 50, tol: float = 1e-6):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y = (np.asarray(y) == self.classes_[1]).astype(float)
        A = _add_intercept(X)
        sw = np.ones(len(y))
        if self.class_weight == "balanced":
            pos = y.mean()
            sw = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))

        reg = np.eye(A.shape[1]) / self.C
        reg[0, 0] = 0
        w = np.zeros(A.shape[1])
        for self.n_iter_ in range(1, self.max_iter + 1):
            p = 1 / (1 + np.exp(-np.clip(A @ w, -35, 35)))
            grad = A.T @ (sw * (p - y)) + reg @ w
            hess = (A * (sw * p * (1 - p))[:, None]).T @ A + reg
            step = np.linalg.solve(hess + 1e-10 * np.eye(len(w)), grad)
            w -= step
            if np.abs(step).max() < self.tol:
                break
        self.intercept_, self.coef_ = w[0], w[1:]
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-np.clip(self.decision_function(X), -35, 35)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
