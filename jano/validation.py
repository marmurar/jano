from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .types import SizeSpec, TemporalPartitionSpec, TemporalSemanticsSpec


@dataclass(frozen=True)
class ValidatedPartitionSpec:
    """Partition specification after normalization."""

    layout: str
    segments: Dict[str, SizeSpec]
    gaps: Dict[str, SizeSpec]
    tail_gap: SizeSpec | None
    size_kind: str
    calendar_frequency: str | None = None


def validate_strategy(strategy: str) -> str:
    """Validate and normalize a split strategy name."""
    valid = {"single", "rolling", "expanding"}
    if strategy not in valid:
        raise ValueError(f"strategy must be one of {sorted(valid)}")
    return strategy


def validate_partition_spec(partition: TemporalPartitionSpec) -> ValidatedPartitionSpec:
    """Validate a high-level partition spec and normalize its sizes."""
    if partition.layout not in {"train_test", "train_val_test"}:
        raise ValueError("layout must be 'train_test' or 'train_val_test'")

    segments = {"train": SizeSpec.from_value(partition.train_size)}
    gaps = {}

    if partition.gap_before_train is not None:
        gaps["train"] = SizeSpec.from_value(partition.gap_before_train)

    if partition.layout == "train_test":
        if partition.test_size is None:
            raise ValueError("test_size is required for the 'train_test' layout")
        segments["test"] = SizeSpec.from_value(partition.test_size)
        if partition.gap_before_test is not None:
            gaps["test"] = SizeSpec.from_value(partition.gap_before_test)
    else:
        if partition.validation_size is None or partition.test_size is None:
            raise ValueError(
                "validation_size and test_size are required for the 'train_val_test' layout"
            )
        segments["validation"] = SizeSpec.from_value(partition.validation_size)
        segments["test"] = SizeSpec.from_value(partition.test_size)
        if partition.gap_before_validation is not None:
            gaps["validation"] = SizeSpec.from_value(partition.gap_before_validation)
        if partition.gap_before_test is not None:
            gaps["test"] = SizeSpec.from_value(partition.gap_before_test)

    kinds = {spec.kind for spec in segments.values()}
    kinds.update(spec.kind for spec in gaps.values())
    if partition.gap_after_test is not None:
        kinds.add(SizeSpec.from_value(partition.gap_after_test).kind)

    if len(kinds) > 1:
        raise ValueError(
            "All partition sizes and gaps must use the same unit family: duration, rows or fraction."
        )

    size_kind = kinds.pop()
    tail_gap = None
    if partition.gap_after_test is not None:
        tail_gap = SizeSpec.from_value(partition.gap_after_test)

    calendar_frequency = partition.calendar_frequency
    if calendar_frequency is not None:
        if size_kind != "duration":
            raise ValueError("calendar_frequency can only be used with duration-based partitions")
        try:
            pd.Timestamp("2024-01-01").floor(calendar_frequency)
        except ValueError as exc:
            raise ValueError("calendar_frequency must be a valid fixed pandas frequency") from exc

    return ValidatedPartitionSpec(
        layout=partition.layout,
        segments=segments,
        gaps=gaps,
        tail_gap=tail_gap,
        size_kind=size_kind,
        calendar_frequency=calendar_frequency,
    )


def validate_temporal_semantics(semantics: TemporalSemanticsSpec) -> TemporalSemanticsSpec:
    """Validate a temporal semantics configuration."""
    if semantics.timeline_col is None:
        raise ValueError("timeline_col must not be None")
    if semantics.effective_order_col is None:
        raise ValueError("order_col must resolve to a non-null column reference")

    for name, column in semantics.segment_time_cols.items():
        if not name:
            raise ValueError("segment_time_cols keys must be non-empty strings")
        if column is None:
            raise ValueError("segment_time_cols values must be non-null column references")

    return semantics
