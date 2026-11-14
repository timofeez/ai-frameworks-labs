"""Загрузка, разбиение и признаки для двух датасетов.

Bank Marketing (UCI 222) — классификация: согласится ли клиент открыть депозит.
Metro Interstate Traffic Volume (UCI 492) — регрессия: почасовой трафик на шоссе I-94.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 42

# ---------------------------------------------------------------- Bank Marketing

BANK_NUMERIC = ["age", "balance", "day", "campaign", "pdays", "previous"]
BANK_CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
                    "contact", "month", "poutcome"]
# Признаки после add_bank_features: логарифмы вместо сильно скошенных исходников.
BANK_RICH_NUMERIC = ["age", "day", "balance_slog", "campaign_log", "previous_log",
                     "pdays_log", "was_contacted"]


def load_bank() -> pd.DataFrame:
    """Читает bank-full.csv и убирает duration.

    duration (длительность звонка) известна только после разговора и почти
    определяет ответ, поэтому её использование — утечка целевой переменной.
    """
    df = pd.read_csv(DATA_DIR / "bank-full.csv", sep=";")
    df = df.drop(columns="duration")
    df["y"] = (df["y"] == "yes").astype(int)
    return df


def bank_split(test_size: float = 0.2):
    """Стратифицированное разбиение 80/20 с фиксированным seed."""
    df = load_bank()
    X, y = df.drop(columns="y"), df["y"]
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=RANDOM_STATE)


def bank_cv(n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def add_bank_features(X: pd.DataFrame) -> pd.DataFrame:
    """Новые признаки для банковского датасета (гипотезы улучшенного бейзлайна)."""
    X = X.copy()
    # pdays = -1 означает «клиенту раньше не звонили»: это отдельная категория, а не число.
    X["was_contacted"] = (X["pdays"] != -1).astype(int)
    X["pdays_log"] = np.log1p(X["pdays"].clip(lower=0))
    # Баланс бывает отрицательным, поэтому знаковый логарифм.
    X["balance_slog"] = np.sign(X["balance"]) * np.log1p(X["balance"].abs())
    X["campaign_log"] = np.log1p(X["campaign"])
    X["previous_log"] = np.log1p(X["previous"])
    return X


# ------------------------------------------------------------------------ Metro

METRO_BASIC_NUMERIC = ["temp", "rain_1h", "snow_1h", "clouds_all", "is_holiday",
                       "hour", "dayofweek", "month"]
METRO_BASIC_CATEGORICAL = ["weather_main"]

METRO_CYCLIC = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
METRO_WEATHER = ["temp_c", "rain_log", "snow_1h", "clouds_all"]


def load_metro() -> pd.DataFrame:
    """Читает данные, сортирует по времени и убирает дубли часов.

    В исходнике один час повторяется, если погода описана несколькими метками
    (например, Rain и Mist). Трафик в дублях одинаковый, оставляем первую строку.
    """
    df = pd.read_csv(DATA_DIR / "metro_interstate_traffic_volume.csv",
                     parse_dates=["date_time"])
    df = df.sort_values("date_time").drop_duplicates("date_time", keep="first")
    return df.reset_index(drop=True)


def metro_split(test_size: float = 0.2):
    """Хронологическое разбиение: последние 20% по времени — тест.

    Случайное разбиение дало бы утечку: соседние часы почти одинаковы,
    и модель «подсматривала» бы будущее.
    """
    df = load_metro()
    cut = int(len(df) * (1 - test_size))
    X, y = df.drop(columns="traffic_volume"), df["traffic_volume"]
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def metro_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """Кросс-валидация, где валидация всегда идёт после обучения по времени."""
    return TimeSeriesSplit(n_splits=n_splits)


def metro_basic_features(X: pd.DataFrame) -> pd.DataFrame:
    """Минимальные признаки бейзлайна: сырые погодные столбцы и части даты числами."""
    dt = X["date_time"]
    return pd.DataFrame({
        "temp": X["temp"],
        "rain_1h": X["rain_1h"],
        "snow_1h": X["snow_1h"],
        "clouds_all": X["clouds_all"],
        "is_holiday": X["holiday"].notna().astype(int),
        "hour": dt.dt.hour,
        "dayofweek": dt.dt.dayofweek,
        "month": dt.dt.month,
        "weather_main": X["weather_main"],
    }, index=X.index)


def metro_rich_features(X: pd.DataFrame) -> pd.DataFrame:
    """Признаки улучшенного бейзлайна (каждый — отдельная гипотеза в ноутбуках)."""
    dt = X["date_time"]
    out = metro_basic_features(X)

    # Очистка: 0 K — ошибка датчика, 9831 мм дождя за час — тоже.
    temp = X["temp"].where(X["temp"] > 200)
    out["temp_c"] = temp.interpolate(limit_direction="both") - 273.15
    out["rain_log"] = np.log1p(X["rain_1h"].clip(upper=60))

    # Праздник в исходнике отмечен только в 00:00 — распространяем на весь день.
    holiday_days = set(dt[X["holiday"].notna()].dt.normalize())
    out["is_holiday"] = dt.dt.normalize().isin(holiday_days).astype(int)

    # Тип дня: в праздники трафик похож на воскресный.
    dow = dt.dt.dayofweek
    daytype = np.where(out["is_holiday"] == 1, "holiday",
               np.where(dow == 5, "saturday", np.where(dow == 6, "sunday", "workday")))
    out["daytype"] = daytype
    out["is_weekend"] = (daytype != "workday").astype(int)
    # Час внутри типа дня: утренний пик есть только в будни.
    out["hour_daytype"] = pd.Series(daytype, index=X.index) + "_" + out["hour"].astype(str)

    # Циклическое кодирование: 23 часа и 0 часов должны быть рядом.
    for name, value, period in [("hour", out["hour"], 24), ("dow", dow, 7),
                                ("month", out["month"], 12)]:
        out[f"{name}_sin"] = np.sin(2 * np.pi * value / period)
        out[f"{name}_cos"] = np.cos(2 * np.pi * value / period)

    # Редкие погодные метки объединяем.
    rare = ~X["weather_main"].isin(["Clouds", "Clear", "Mist", "Rain", "Snow",
                                    "Drizzle", "Haze", "Thunderstorm", "Fog"])
    out["weather_main"] = X["weather_main"].where(~rare, "Other")
    return out
