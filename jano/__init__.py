"""Public package interface for Jano."""

from ._version import __version__
from .engines import PartitionEngineMetadata
from .evaluation import (
    ClassificationProfile,
    EvaluationProfile,
    OrdinalClassificationProfile,
    RankingProfile,
    RegressionProfile,
    ResolvedEvaluationProfile,
)
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
    FunctionRetrainPolicy,
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
    "ClassificationProfile",
    "DriftBasedRetrain",
    "EvaluationProfile",
    "FeatureLookbackSpec",
    "DriftMonitoringPolicy",
    "FunctionRetrainPolicy",
    "NeverRetrain",
    "OrdinalClassificationProfile",
    "PartitionPlan",
    "PartitionEngineMetadata",
    "PerformanceDecayPolicy",
    "PerformanceDecayResult",
    "PeriodicRetrain",
    "PlannedFold",
    "RankingProfile",
    "RegressionProfile",
    "RetrainContext",
    "RetrainPolicy",
    "ResolvedEvaluationProfile",
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
