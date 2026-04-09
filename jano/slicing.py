from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TimeIndexer:
    """Index helper that keeps temporal ordering and positional mappings."""

    frame: pd.DataFrame
    time_col: str

    def __post_init__(self) -> None:
        if self.time_col not in self.frame.columns:
            raise ValueError(f"time_col '{self.time_col}' was not found in the dataset")

        ordered = self.frame.copy()
        ordered["__jano_position__"] = np.arange(len(self.frame))
        ordered[self.time_col] = pd.to_datetime(ordered[self.time_col])
        self.ordered = ordered.sort_values(self.time_col, kind="mergesort").reset_index(drop=True)
        self.timestamps = self.ordered[self.time_col]

    @property
    def original_index(self) -> np.ndarray:
        return self.ordered["__jano_position__"].to_numpy()

    @property
    def min_time(self) -> pd.Timestamp:
        return self.timestamps.iloc[0]

    @property
    def max_time(self) -> pd.Timestamp:
        return self.timestamps.iloc[-1]

    def slice_positional(self, start: int, end: int) -> np.ndarray:
        return self.original_index[start:end]

    def slice_between(self, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        mask = (self.timestamps >= start) & (self.timestamps < end)
        return self.original_index[mask.to_numpy()]
