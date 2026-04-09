from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .types import SizeSpec, TemporalPartitionSpec


@dataclass(frozen=True)
class ValidatedPartitionSpec:
    """Partition specification after normalization."""

    layout: str
    segments: Dict[str, SizeSpec]
    gaps: Dict[str, SizeSpec]
    size_kind: str


def validate_strategy(strategy: str) -> str:
    valid = {"single", "rolling", "expanding"}
    if strategy not in valid:
        raise ValueError(f"strategy must be one of {sorted(valid)}")
    return strategy


def validate_partition_spec(partition: TemporalPartitionSpec) -> ValidatedPartitionSpec:
    if partition.layout not in {"train_test", "train_val_test"}:
        raise ValueError("layout must be 'train_test' or 'train_val_test'")

    segments = {"train": SizeSpec.from_value(partition.train_size)}
    gaps = {}

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

    if len(kinds) > 1:
        raise ValueError(
            "All partition sizes and gaps must use the same unit family: duration, rows or fraction."
        )

    size_kind = kinds.pop()
    return ValidatedPartitionSpec(
        layout=partition.layout,
        segments=segments,
        gaps=gaps,
        size_kind=size_kind,
    )
