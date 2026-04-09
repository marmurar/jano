"""Public package interface for Jano."""

from .splitters import TemporalBacktestSplitter
from .splits import TimeSplit
from .types import SegmentBoundaries, SizeSpec, TemporalPartitionSpec

__all__ = [
    "SegmentBoundaries",
    "SizeSpec",
    "TemporalBacktestSplitter",
    "TemporalPartitionSpec",
    "TimeSplit",
]
