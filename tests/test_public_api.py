from __future__ import annotations

import builtins
import runpy
import sys
import types
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import polars as pl

import jano.engines as engines_module
import jano.io as io_module
import jano.mcp_server as mcp_server_module
import jano.mcp_tools as mcp_tools_module
import jano.planning as planning_module
import jano.policies as policies_module
import jano.runner as runner_module
import jano.validation as validation_module
from conftest import build_frame, write_csv_frame, SimpleLinearRegressor, MeanRegressor
from jano import (
    AlwaysRetrain,
    DriftBasedRetrain,
    DriftMonitoringPolicy,
    FeatureLookbackSpec,
    NeverRetrain,
    PartitionPlan,
    PeriodicRetrain,
    PerformanceDecayPolicy,
    PlannedFold,
    RetrainContext,
    RetrainPolicy,
    RollingTrainHistoryPolicy,
    RollingTrainHistoryResult,
    SimulationPlan,
    SystemEvaluationResult,
    SystemRunResult,
    SystemUpdateResult,
    TemporalBacktestSplitter,
    TemporalPartitionSpec,
    TemporalSystemRunner,
    TemporalSemanticsSpec,
    TemporalSimulation,
    TrainHistoryPolicy,
    TrainGrowthPolicy,
    UpdateableSystem,
    WalkForwardRunResult,
    WalkForwardRunner,
    WalkForwardPolicy,
    __version__,
)
from jano.describe import SimulationSummary as LegacySimulationSummary
from jano.engines import PartitionEngine, detect_backend, missing_columns
from jano.jano import TemporalBacktestSplitter as LegacyTemporalBacktestSplitter
from jano.mcp_server import build_server
from jano.mcp_tools import load_dataset_frame, plan_walk_forward, preview_dataset, run_walk_forward
from jano.policies import PerformanceDecayResult, TrainGrowthResult
from jano.reporting import SimulationChartData, SimulationSummary
from jano.simulation import SimulationResult
from jano.splits import TimeSplit
from jano.types import SegmentBoundaries, SizeSpec

def test_public_version_matches_installed_distribution_metadata() -> None:
    assert __version__ == "0.4.1"
    assert version("jano") == __version__

def test_legacy_import_surface_matches_public_splitter() -> None:
    assert LegacyTemporalBacktestSplitter is TemporalBacktestSplitter

def test_legacy_describe_import_surface_matches_public_summary() -> None:
    assert LegacySimulationSummary is SimulationSummary


def test_system_runner_surface_is_exported() -> None:
    assert TemporalSystemRunner.__name__ == "TemporalSystemRunner"
    assert SystemRunResult.__name__ == "SystemRunResult"
    assert SystemUpdateResult.__name__ == "SystemUpdateResult"
    assert SystemEvaluationResult.__name__ == "SystemEvaluationResult"
    assert UpdateableSystem.__name__ == "UpdateableSystem"
