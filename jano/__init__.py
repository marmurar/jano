"""Public package interface for Jano."""

from ._version import __version__
from .engines import PartitionEngineMetadata
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
from .workflows import (
    DriftMonitoringPolicy,
    RollingTrainHistoryPolicy,
    RollingTrainHistoryResult,
    TrainHistoryPolicy,
    WalkForwardPolicy,
)

__all__ = [
    "FeatureLookbackSpec",
    "DriftMonitoringPolicy",
    "PartitionPlan",
    "PartitionEngineMetadata",
    "PerformanceDecayPolicy",
    "PerformanceDecayResult",
    "PlannedFold",
    "RollingTrainHistoryPolicy",
    "RollingTrainHistoryResult",
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
    "TrainHistoryPolicy",
    "TrainGrowthPolicy",
    "TrainGrowthResult",
    "TimeSplit",
    "WalkForwardPolicy",
    "__version__",
]
