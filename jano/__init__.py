"""Public package interface for Jano."""

from .reporting import SimulationChartData, SimulationSummary
from .splitters import TemporalBacktestSplitter
from .splits import TimeSplit
from .types import SegmentBoundaries, SizeSpec, TemporalPartitionSpec

__all__ = [
    "SegmentBoundaries",
    "SimulationChartData",
    "SimulationSummary",
    "SizeSpec",
    "TemporalBacktestSplitter",
    "TemporalPartitionSpec",
    "TimeSplit",
]
