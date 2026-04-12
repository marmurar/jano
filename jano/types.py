from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence, Union

import pandas as pd

SizeValue = Union[str, int, float, pd.Timedelta]
ColumnRef = Union[str, int]


@dataclass(frozen=True)
class SizeSpec:
    """Normalized specification for segment sizes.

    Attributes:
        value: Parsed size value as ``Timedelta``, integer row count or fraction.
        kind: Unit family for the value: ``duration``, ``rows`` or ``fraction``.
    """

    value: Union[pd.Timedelta, int, float]
    kind: str

    @classmethod
    def from_value(cls, value: SizeValue) -> "SizeSpec":
        """Normalize a raw size value into a typed ``SizeSpec``."""
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
    """High-level description of a temporal partition layout.

    Attributes:
        layout: Either ``train_test`` or ``train_val_test``.
        train_size: Size of the train segment.
        test_size: Size of the test segment when present.
        validation_size: Size of the validation segment when present.
        gap_before_validation: Optional gap inserted before validation.
        gap_before_test: Optional gap inserted before test.
    """

    layout: str
    train_size: SizeValue
    test_size: SizeValue | None = None
    validation_size: SizeValue | None = None
    gap_before_train: SizeValue | None = None
    gap_before_validation: SizeValue | None = None
    gap_before_test: SizeValue | None = None
    gap_after_test: SizeValue | None = None


@dataclass(frozen=True)
class TemporalSemanticsSpec:
    """Temporal semantics for ordering, reporting and segment eligibility.

    Attributes:
        timeline_col: Column used to anchor the global simulation timeline and reports.
        order_col: Optional column used to sort the dataset internally. Defaults to
            ``timeline_col``.
        segment_time_cols: Optional per-segment timestamp mapping. Use this when a
            segment should be sliced by a different temporal column than the global
            timeline. For example, train can be filtered by ``arrived_at`` while test
            stays anchored on ``departured_at``.
    """

    timeline_col: ColumnRef
    order_col: ColumnRef | None = None
    segment_time_cols: Mapping[str, ColumnRef] = field(default_factory=dict)

    @property
    def effective_order_col(self) -> ColumnRef:
        """Return the ordering column used by the engine."""
        return self.order_col or self.timeline_col

    def column_for_segment(self, name: str) -> ColumnRef:
        """Return the timestamp column used to assign rows to ``name``."""
        return self.segment_time_cols.get(name, self.timeline_col)


@dataclass(frozen=True)
class FeatureLookbackSpec:
    """Lookback requirements for feature groups within the same fold.

    Attributes:
        default_lookback: Optional fallback lookback applied to features that do not
            belong to an explicit group.
        group_lookbacks: Mapping from feature-group name to the temporal lookback
            needed to build that group.
        feature_groups: Mapping from group name to the feature columns that belong to it.

    All lookbacks must use duration-based sizes.
    """

    default_lookback: SizeValue | None = None
    group_lookbacks: Mapping[str, SizeValue] = field(default_factory=dict)
    feature_groups: Mapping[str, Sequence[ColumnRef]] = field(default_factory=dict)

    def normalized_group_lookbacks(self) -> dict[str, SizeSpec]:
        """Return validated duration lookbacks for each explicit feature group."""
        normalized: dict[str, SizeSpec] = {}
        for name, value in self.group_lookbacks.items():
            spec = SizeSpec.from_value(value)
            if spec.kind != "duration":
                raise ValueError("Feature lookbacks must use duration-based sizes")
            normalized[name] = spec
        return normalized

    def normalized_default_lookback(self) -> SizeSpec | None:
        """Return the validated duration lookback for ungrouped features."""
        if self.default_lookback is None:
            return None
        spec = SizeSpec.from_value(self.default_lookback)
        if spec.kind != "duration":
            raise ValueError("Feature lookbacks must use duration-based sizes")
        return spec


@dataclass(frozen=True)
class SegmentBoundaries:
    """Closed-open boundaries for a named temporal segment."""

    start: pd.Timestamp
    end: pd.Timestamp
