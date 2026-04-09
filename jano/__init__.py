"""Public package interface for Jano."""

from .reporting import SimulationSummary
from .splitters import TemporalBacktestSplitter
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
