"""Метрики, кросс-валидация гипотез и сохранение результатов для итогового сравнения."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_predict, cross_validate

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

STAGES = {
    "baseline": "Бейзлайн sklearn",
    "improved": "Улучшенный sklearn",
    "scratch": "Своя реализация",
    "scratch_improved": "Своя реализация + улучшения",
}


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """ROC-AUC и PR-AUC не зависят от порога, F1/precision/recall — при пороге threshold."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, pred),
        "threshold": float(threshold),
    }


def regression_metrics(y_true, pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "R2": r2_score(y_true, pred),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Порог, максимизирующий F1. Подбирается только на out-of-fold прогнозах train."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def oof_threshold(model, X, y, cv) -> float:
    """Подбирает порог по out-of-fold вероятностям, не трогая тест."""
    proba = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba",
                              n_jobs=-1)[:, 1]
    return best_f1_threshold(y, proba)


def compare_cv(variants: dict, X, y, cv, scoring: dict, n_jobs: int = -1) -> pd.DataFrame:
    """Проверка гипотез: одна и та же CV для каждого варианта модели.

    variants: {название гипотезы: estimator или (estimator, X_variant)}.
    """
    rows = []
    for name, spec in variants.items():
        est, X_used = spec if isinstance(spec, tuple) else (spec, X)
        res = cross_validate(clone(est), X_used, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
        row = {"Гипотеза": name}
        for key in scoring:
            vals = res[f"test_{key}"]
            if key.startswith("neg_") or key in ("MAE", "RMSE"):
                vals = -vals
            row[key] = vals.mean()
            row[f"{key} std"] = vals.std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("Гипотеза")


class Timer:
    """Контекстный менеджер для замера времени обучения."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


class ResultStore:
    """Копит метрики лабораторной и сохраняет их в results/<name>.json."""

    def __init__(self, name: str, algorithm: str):
        self.name, self.algorithm, self.records = name, algorithm, []

    def add(self, task: str, stage: str, model: str, metrics: dict, seconds=None):
        rec = {"algorithm": self.algorithm, "task": task, "stage": stage,
               "model": model, **{k: float(v) for k, v in metrics.items()}}
        if seconds is not None:
            rec["fit_seconds"] = float(seconds)
        self.records = [r for r in self.records
                        if not (r["task"] == task and r["stage"] == stage)] + [rec]
        return rec

    def table(self, task: str) -> pd.DataFrame:
        df = pd.DataFrame([r for r in self.records if r["task"] == task])
        df["Этап"] = df["stage"].map(STAGES)
        cols = (["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "threshold"]
                if task == "classification" else ["MAE", "RMSE", "R2"])
        cols = [c for c in cols + ["fit_seconds"] if c in df]
        return df.set_index("Этап")[["model"] + cols].round(4)

    def save(self):
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"{self.name}.json"
        path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path


def load_all_results() -> pd.DataFrame:
    frames = [pd.DataFrame(json.loads(p.read_text(encoding="utf-8")))
              for p in sorted(RESULTS_DIR.glob("lab*.json"))]
    return pd.concat(frames, ignore_index=True)
