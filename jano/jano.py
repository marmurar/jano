"""Backward-compatible import surface for legacy users."""

from .splitters import TemporalBacktestSplitter
from .reporting import SimulationSummary
from .splits import TimeSplit
from .types import SegmentBoundaries, SizeSpec, TemporalPartitionSpec

__all__ = [
    "SegmentBoundaries",
    "SimulationSummary",
    "SizeSpec",
    "TemporalBacktestSplitter",
    "TemporalPartitionSpec",
    "TimeSplit",
]
