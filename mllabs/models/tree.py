"""Решающее дерево (CART) на numpy.

Чтобы дерево обучалось быстро, признаки заранее разбиваются на корзины (до 64
значений, как в LightGBM / HistGradientBoosting). Тогда лучший порог в узле
ищется по гистограммам за один проход, а не перебором всех значений.

Критерий разбиения — уменьшение взвешенной суммы квадратов отклонений (SSE).
Для бинарной классификации с y ∈ {0, 1} это ровно критерий Джини:
дисперсия p(1 - p) пропорциональна индексу Джини 2p(1 - p).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class Binner:
    """Переводит числовые признаки в номера корзин (uint8)."""

    def __init__(self, max_bins: int = 64):
        self.max_bins = max_bins

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.edges_ = []
        for col in X.T:
            uniq = np.unique(col)
            if len(uniq) <= self.max_bins:
                edges = (uniq[:-1] + uniq[1:]) / 2      # граница между соседними значениями
            else:
                qs = np.linspace(0, 1, self.max_bins + 1)[1:-1]
                edges = np.unique(np.quantile(col, qs))  # квантили — корзины равной наполненности
            self.edges_.append(edges)
        self.n_bins_ = max(len(e) for e in self.edges_) + 1
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape, dtype=np.uint8)
        for j, edges in enumerate(self.edges_):
            # Корзина b означает: x меньше edges[b] и не меньше edges[b - 1].
            out[:, j] = np.searchsorted(edges, X[:, j], side="right")
        return out


def resolve_max_features(max_features, n_features: int) -> int:
    if max_features is None:
        return n_features
    if max_features == "sqrt":
        return max(1, int(np.sqrt(n_features)))
    if isinstance(max_features, float):
        return max(1, int(max_features * n_features))
    return min(n_features, int(max_features))


class TreeArrays:
    """Обученное дерево в виде плоских массивов: так прогноз векторизуется."""

    def __init__(self, feature, threshold, left, right, value, gain_by_feature):
        self.feature, self.threshold = feature, threshold
        self.left, self.right, self.value = left, right, value
        self.gain_by_feature = gain_by_feature

    @property
    def n_leaves(self):
        return int((self.left == -1).sum())

    @property
    def depth(self):
        depth = np.zeros(len(self.left), dtype=int)
        for node in range(len(self.left)):     # дети всегда создаются после родителя
            if self.left[node] != -1:
                depth[self.left[node]] = depth[self.right[node]] = depth[node] + 1
        return int(depth.max())

    def apply(self, Xb):
        """Номер листа для каждого объекта: все объекты спускаются по дереву одновременно."""
        node = np.zeros(len(Xb), dtype=np.int64)
        rows = np.arange(len(Xb))
        active = self.left[node] != -1
        while active.any():
            r, nd = rows[active], node[active]
            go_left = Xb[r, self.feature[nd]] <= self.threshold[nd]
            node[r] = np.where(go_left, self.left[nd], self.right[nd])
            active = self.left[node] != -1
        return node

    def predict(self, Xb):
        return self.value[self.apply(Xb)]


def build_tree(Xb, y, w, n_bins, max_depth=None, min_samples_split=2, min_samples_leaf=1,
               max_features=None, rng=None) -> TreeArrays:
    """Жадно строит дерево в глубину. w — веса объектов (0 — объект не участвует)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    n, d = Xb.shape
    max_depth = np.inf if max_depth is None else max_depth
    k_features = resolve_max_features(max_features, d)

    feature, threshold, left, right, value = [], [], [], [], []
    gain_by_feature = np.zeros(d)

    def new_node(idx):
        wi = w[idx]
        for lst, v in ((feature, -1), (threshold, 0), (left, -1), (right, -1)):
            lst.append(v)
        value.append(float((wi * y[idx]).sum() / wi.sum()))
        return len(value) - 1

    root = np.flatnonzero(w > 0)
    stack = [(root, 0, new_node(root))]
    while stack:
        idx, depth, node = stack.pop()
        if depth >= max_depth or len(idx) < max(min_samples_split, 2 * min_samples_leaf):
            continue

        feats = (np.arange(d) if k_features == d
                 else np.sort(rng.choice(d, k_features, replace=False)))
        f = len(feats)
        sub = Xb[np.ix_(idx, feats)].astype(np.int64)
        flat = (sub + np.arange(f) * n_bins).ravel()
        wi, yi = w[idx], y[idx]
        size = f * n_bins

        def hist(weights):
            return np.bincount(flat, weights=np.repeat(weights, f), minlength=size).reshape(f, n_bins)

        # Гистограммы по корзинам: число объектов, сумма весов, сумма wy и wy².
        cnt = np.bincount(flat, minlength=size).reshape(f, n_bins)
        W, S, Q = hist(wi), hist(wi * yi), hist(wi * yi * yi)
        Wl, Sl, Ql, Cl = (a.cumsum(axis=1)[:, :-1] for a in (W, S, Q, cnt))
        Wt, St, Qt, Ct = W.sum(1)[:, None], S.sum(1)[:, None], Q.sum(1)[:, None], cnt.sum(1)[:, None]
        Wr, Sr, Qr, Cr = Wt - Wl, St - Sl, Qt - Ql, Ct - Cl

        with np.errstate(divide="ignore", invalid="ignore"):
            sse_children = (Ql - Sl ** 2 / Wl) + (Qr - Sr ** 2 / Wr)
        parent_sse = float(Qt[0, 0] - St[0, 0] ** 2 / Wt[0, 0])
        gain = parent_sse - sse_children
        valid = (Cl >= min_samples_leaf) & (Cr >= min_samples_leaf) & (Wl > 0) & (Wr > 0)
        gain = np.where(valid, gain, -np.inf)

        best = int(np.argmax(gain))
        best_gain = gain.flat[best]
        if not np.isfinite(best_gain) or best_gain <= 1e-12 * max(parent_sse, 1.0):
            continue

        fj, b = divmod(best, n_bins - 1)
        feat = int(feats[fj])
        mask = Xb[idx, feat] <= b
        li, ri = idx[mask], idx[~mask]
        feature[node], threshold[node] = feat, b
        gain_by_feature[feat] += best_gain
        left[node], right[node] = new_node(li), new_node(ri)
        stack.append((ri, depth + 1, right[node]))
        stack.append((li, depth + 1, left[node]))

    return TreeArrays(np.array(feature), np.array(threshold, dtype=np.int64),
                      np.array(left), np.array(right), np.array(value), gain_by_feature)


def class_weights(y, class_weight):
    if class_weight != "balanced":
        return np.ones(len(y))
    pos = y.mean()
    return np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))


class _TreeEstimator(BaseEstimator):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, max_bins=64, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_bins = max_bins
        self.random_state = random_state

    def _fit(self, X, y, w):
        self.binner_ = Binner(self.max_bins).fit(X)
        Xb = self.binner_.transform(X)
        self.tree_ = build_tree(Xb, y, w, self.binner_.n_bins_, self.max_depth,
                                self.min_samples_split, self.min_samples_leaf,
                                self.max_features, np.random.default_rng(self.random_state))
        total = self.tree_.gain_by_feature.sum()
        self.feature_importances_ = self.tree_.gain_by_feature / (total if total > 0 else 1)
        return self

    def _raw_predict(self, X):
        return self.tree_.predict(self.binner_.transform(X))


class DecisionTreeRegressorScratch(RegressorMixin, _TreeEstimator):
    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=np.float64)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        return self._fit(X, y, w)

    def predict(self, X):
        return self._raw_predict(X)


class DecisionTreeClassifierScratch(ClassifierMixin, _TreeEstimator):
    """Бинарная классификация: лист хранит долю класса 1 (с учётом весов)."""

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, max_bins=64, class_weight=None, random_state=None):
        super().__init__(max_depth, min_samples_split, min_samples_leaf, max_features,
                         max_bins, random_state)
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        y = (np.asarray(y) == self.classes_[1]).astype(float)
        w = class_weights(y, self.class_weight)
        if sample_weight is not None:
            w = w * np.asarray(sample_weight, float)
        return self._fit(X, y, w)

    def predict_proba(self, X):
        p = self._raw_predict(X)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
