"""Готовые препроцессоры, чтобы все лабораторные сравнивались на одинаковых данных."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import data as D


def onehot():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def _columns(numeric, categorical, scale):
    parts = []
    if numeric:
        parts.append(("num", StandardScaler() if scale else "passthrough", list(numeric)))
    if categorical:
        parts.append(("cat", onehot(), list(categorical)))
    return ColumnTransformer(parts)


def bank_preprocessor(rich: bool = False, scale: bool = False):
    """Бейзлайн: исходные числа как есть + one-hot категорий.
    rich=True добавляет признаки из data.add_bank_features вместо скошенных исходников."""
    numeric = D.BANK_RICH_NUMERIC if rich else D.BANK_NUMERIC
    steps = [FunctionTransformer(D.add_bank_features)] if rich else []
    return make_pipeline(*steps, _columns(numeric, D.BANK_CATEGORICAL, scale))


def metro_preprocessor(numeric=None, categorical=None, rich: bool = True, scale: bool = False):
    """rich=False — признаки бейзлайна (metro_basic_features), иначе metro_rich_features."""
    if not rich:
        numeric = numeric or D.METRO_BASIC_NUMERIC
        categorical = D.METRO_BASIC_CATEGORICAL if categorical is None else categorical
    fe = FunctionTransformer(D.metro_rich_features if rich else D.metro_basic_features)
    return make_pipeline(fe, _columns(numeric, categorical, scale))
