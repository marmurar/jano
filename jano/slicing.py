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

        timestamps = pd.to_datetime(self.frame[self.time_col])
        timestamp_array = timestamps.to_numpy(dtype="datetime64[ns]")
        order = np.argsort(timestamp_array, kind="mergesort")

        self.timestamp_array = timestamp_array[order]
        self.position_array = np.arange(len(self.frame), dtype=np.int64)[order]
        self.total_rows = len(self.position_array)

    @property
    def original_index(self) -> np.ndarray:
        return self.position_array

    @property
    def min_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.timestamp_array[0])

    @property
    def max_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.timestamp_array[-1])

    def slice_positional(self, start: int, end: int) -> np.ndarray:
        return self.original_index[start:end]

    def timestamp_at(self, position: int) -> pd.Timestamp:
        return pd.Timestamp(self.timestamp_array[position])

    def bounds_between(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, int]:
        start_value = np.datetime64(start.to_datetime64())
        end_value = np.datetime64(end.to_datetime64())
        left = int(np.searchsorted(self.timestamp_array, start_value, side="left"))
        right = int(np.searchsorted(self.timestamp_array, end_value, side="left"))
        return left, right

    def slice_between(self, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        left, right = self.bounds_between(start, end)
        return self.original_index[left:right]
