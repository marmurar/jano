from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import pandas as pd

SizeValue = Union[str, int, float, pd.Timedelta]


@dataclass(frozen=True)
class SizeSpec:
    """Normalized specification for segment sizes."""

    value: Union[pd.Timedelta, int, float]
    kind: str

    @classmethod
    def from_value(cls, value: SizeValue) -> "SizeSpec":
        if isinstance(value, pd.Timedelta):
            return cls(value=value, kind="duration")
        if isinstance(value, str):
            return cls(value=pd.to_timedelta(value), kind="duration")
        if isinstance(value, bool):
            raise TypeError("Size values cannot be booleans.")
        if isinstance(value, int):
            if value <= 0:
                raise ValueError("Integer size values must be greater than zero.")
            return cls(value=value, kind="rows")
        if isinstance(value, float):
            if not 0 < value < 1:
                raise ValueError("Fractional size values must be between 0 and 1.")
            return cls(value=value, kind="fraction")
        raise TypeError(
            "Size values must be a pandas Timedelta, duration string, int or float."
        )


@dataclass(frozen=True)
class TemporalPartitionSpec:
    """High-level description of a temporal partition layout."""

    layout: str
    train_size: SizeValue
    test_size: SizeValue | None = None
    validation_size: SizeValue | None = None
    gap_before_validation: SizeValue | None = None
    gap_before_test: SizeValue | None = None


@dataclass(frozen=True)
class SegmentBoundaries:
    """Closed-open boundaries for a named temporal segment."""

    start: pd.Timestamp
    end: pd.Timestamp
