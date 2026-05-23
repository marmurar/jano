from __future__ import annotations

import numpy as np
import pandas as pd


def build_frame(size: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=size, freq="D"),
            "feature": range(size),
            "target": range(100, 100 + size),
        }
    )


def write_csv_frame(tmp_path, frame: pd.DataFrame, name: str = "frame.csv") -> str:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return str(path)


class SimpleLinearRegressor:
    def fit(self, X: pd.DataFrame, y: pd.Series):
        matrix = X.to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(matrix)), matrix])
        self.coef_, *_ = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        matrix = X.to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(matrix)), matrix])
        return design @ self.coef_


class MeanRegressor:
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.mean_ = float(np.asarray(y).mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.repeat(self.mean_, len(X))


class RunningMeanPartialFitRegressor:
    def __init__(self) -> None:
        self.total_ = 0.0
        self.count_ = 0

    def partial_fit(self, X: pd.DataFrame, y: pd.Series):
        values = np.asarray(y, dtype=float)
        self.total_ += float(values.sum())
        self.count_ += int(len(values))
        self.mean_ = self.total_ / self.count_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.repeat(self.mean_, len(X))
