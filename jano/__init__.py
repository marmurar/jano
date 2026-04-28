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
from .runner import (
    AlwaysRetrain,
    DriftBasedRetrain,
    NeverRetrain,
    PeriodicRetrain,
    RetrainContext,
    RetrainPolicy,
    WalkForwardRunResult,
    WalkForwardRunner,
)
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
    "AlwaysRetrain",
    "DriftBasedRetrain",
    "FeatureLookbackSpec",
    "DriftMonitoringPolicy",
    "NeverRetrain",
    "PartitionPlan",
    "PartitionEngineMetadata",
    "PerformanceDecayPolicy",
    "PerformanceDecayResult",
    "PeriodicRetrain",
    "PlannedFold",
    "RetrainContext",
    "RetrainPolicy",
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
    "WalkForwardRunResult",
    "WalkForwardRunner",
    "__version__",
]
