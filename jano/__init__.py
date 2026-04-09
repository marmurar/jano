"""Public package interface for Jano."""

from .reporting import SimulationChartData, SimulationSummary
from .simulation import SimulationResult, TemporalSimulation
from .splitters import TemporalBacktestSplitter
from .splits import TimeSplit
from .types import SegmentBoundaries, SizeSpec, TemporalPartitionSpec

__all__ = [
    "SegmentBoundaries",
    "SimulationChartData",
    "SimulationResult",
    "SimulationSummary",
    "SizeSpec",
    "TemporalSimulation",
    "TemporalBacktestSplitter",
    "TemporalPartitionSpec",
    "TimeSplit",
]
