from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .engines import PartitionEngine, missing_columns
from .types import TemporalSemanticsSpec


@dataclass
class TimeIndexer:
    """Index helper that keeps temporal ordering and positional mappings."""

    engine: PartitionEngine
    semantics: TemporalSemanticsSpec

    def __post_init__(self) -> None:
        self._sorted_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        required_columns = set(self._referenced_columns())
        missing = missing_columns(required_columns, self.engine.columns)
        if missing:
            raise ValueError(
                "Temporal semantics reference columns that were not found in the dataset: "
                + ", ".join(str(column) for column in missing)
            )

        order_col = self.semantics.effective_order_col
        ordered_timestamps = pd.to_datetime(self.engine.column_values(order_col)).to_numpy(
            dtype="datetime64[ns]"
        )
        order = np.argsort(ordered_timestamps, kind="mergesort")

        self.position_array = np.arange(self.engine.total_rows, dtype=np.int64)[order]
        self.total_rows = len(self.position_array)

        for column in required_columns:
            timestamps = pd.to_datetime(self.engine.column_values(column)).to_numpy(
                dtype="datetime64[ns]"
            )
            column_order = np.argsort(timestamps, kind="mergesort")
            self._sorted_arrays[column] = (
                timestamps[column_order],
                np.arange(self.engine.total_rows, dtype=np.int64)[column_order],
            )

        self.timeline_array = self._sorted_arrays[self.semantics.timeline_col][0]

    @property
    def original_index(self) -> np.ndarray:
        return self.position_array

    @property
    def min_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.timeline_array[0])

    @property
    def max_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.timeline_array[-1])

    def slice_positional(self, start: int, end: int) -> np.ndarray:
        return self.original_index[start:end]

    def timestamp_at(self, position: int) -> pd.Timestamp:
        return pd.Timestamp(self.timeline_array[position])

    def bounds_between(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, int]:
        return self.bounds_between_for_column(self.semantics.timeline_col, start, end)

    def bounds_between_for_column(
        self, column: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[int, int]:
        timestamp_array, _ = self._sorted_arrays[column]
        start_value = np.datetime64(start.to_datetime64())
        end_value = np.datetime64(end.to_datetime64())
        left = int(np.searchsorted(timestamp_array, start_value, side="left"))
        right = int(np.searchsorted(timestamp_array, end_value, side="left"))
        return left, right

    def slice_between(self, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        left, right = self.bounds_between(start, end)
        return self.original_index[left:right]

    def slice_between_for_segment(
        self, segment_name: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> np.ndarray:
        column = self.semantics.column_for_segment(segment_name)
        timestamp_array, positions = self._sorted_arrays[column]
        start_value = np.datetime64(start.to_datetime64())
        end_value = np.datetime64(end.to_datetime64())
        left = int(np.searchsorted(timestamp_array, start_value, side="left"))
        right = int(np.searchsorted(timestamp_array, end_value, side="left"))
        return positions[left:right]

    def _referenced_columns(self) -> Iterable[str]:
        yield self.semantics.timeline_col
        yield self.semantics.effective_order_col
        for column in self.semantics.segment_time_cols.values():
            yield column
