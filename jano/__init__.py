"""Public package interface for Jano."""

from ._version import __version__
from .planning import PartitionPlan, PlannedFold, SimulationPlan
from .policies import (
    PerformanceDecayPolicy,
    PerformanceDecayResult,
    TrainGrowthPolicy,
    TrainGrowthResult,
)
from .reporting import SimulationChartData, SimulationSummary
from .simulation import SimulationResult, TemporalSimulation
from .splitters import TemporalBacktestSplitter
from .splits import TimeSplit
from .types import (
    FeatureLookbackSpec,
    SegmentBoundaries,
    SizeSpec,
    TemporalPartitionSpec,
    TemporalSemanticsSpec,
)

__all__ = [
    "FeatureLookbackSpec",
    "PartitionPlan",
    "PerformanceDecayPolicy",
    "PerformanceDecayResult",
    "PlannedFold",
    "SegmentBoundaries",
    "SimulationChartData",
    "SimulationPlan",
    "SimulationResult",
    "SimulationSummary",
    "SizeSpec",
    "TemporalSimulation",
    "TemporalBacktestSplitter",
    "TemporalPartitionSpec",
    "TemporalSemanticsSpec",
    "TrainGrowthPolicy",
    "TrainGrowthResult",
    "TimeSplit",
    "__version__",
]
