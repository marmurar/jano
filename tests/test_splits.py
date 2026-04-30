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
    TemporalBacktestSplitter,
    TemporalPartitionSpec,
    TemporalSemanticsSpec,
    TemporalSimulation,
    TrainHistoryPolicy,
    TrainGrowthPolicy,
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

def test_slice_xy_returns_named_segments() -> None:
    frame = build_frame(size=10)
    features = frame[["timestamp", "feature"]]
    target = frame["target"]

    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="4D",
            test_size="2D",
        ),
        step="1D",
        strategy="single",
    )

    split = next(splitter.iter_splits(frame))
    sliced = split.slice_xy(features, target)

    assert list(sliced.keys()) == ["X_train", "y_train", "X_test", "y_test"]
    assert len(sliced["X_train"]) == 4
    assert len(sliced["y_test"]) == 2

def test_summary_exposes_segment_metadata() -> None:
    frame = build_frame(size=10)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size=5,
            test_size=3,
        ),
        step=1,
        strategy="single",
    )

    split = next(splitter.iter_splits(frame))
    summary = split.summary()

    assert summary["fold"] == 0
    assert summary["strategy"] == "single"
    assert summary["segments"]["train"]["rows"] == 5
    assert summary["segments"]["test"]["rows"] == 3

def test_split_slice_returns_named_frames() -> None:
    frame = build_frame(size=10)
    split = TimeSplit(
        fold=0,
        segments={"train": pd.Index([0, 1, 2]).to_numpy(), "test": pd.Index([3, 4]).to_numpy()},
        boundaries={
            "train": SegmentBoundaries(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04")),
            "test": SegmentBoundaries(pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-06")),
        },
    )

    sliced = split.slice(frame)

    assert list(sliced.keys()) == ["train", "test"]
    assert len(sliced["train"]) == 3
    assert len(sliced["test"]) == 2

def test_time_split_can_compute_feature_history_bounds_for_multiple_groups() -> None:
    split = TimeSplit(
        fold=0,
        segments={"train": np.array([10, 11, 12]), "test": np.array([13, 14])},
        boundaries={
            "train": SegmentBoundaries(pd.Timestamp("2025-03-10"), pd.Timestamp("2025-03-15")),
            "test": SegmentBoundaries(pd.Timestamp("2025-03-15"), pd.Timestamp("2025-03-17")),
        },
    )

    lookbacks = FeatureLookbackSpec(
        default_lookback="15D",
        group_lookbacks={"lag_features": "65D"},
        feature_groups={"lag_features": ["lag_30", "lag_60"]},
    )

    bounds = split.feature_history_bounds(lookbacks)

    assert bounds["lag_features"].start == pd.Timestamp("2025-01-04")
    assert bounds["lag_features"].end == pd.Timestamp("2025-03-10")
    assert bounds["__default__"].start == pd.Timestamp("2025-02-23")

def test_time_split_can_slice_feature_history_per_group() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=90, freq="D"),
            "feature_recent": np.arange(90),
            "lag_60": np.arange(90) * 2,
        }
    )
    split = TimeSplit(
        fold=0,
        segments={"train": np.array([70, 71, 72]), "test": np.array([73, 74])},
        boundaries={
            "train": SegmentBoundaries(pd.Timestamp("2025-03-12"), pd.Timestamp("2025-03-15")),
            "test": SegmentBoundaries(pd.Timestamp("2025-03-15"), pd.Timestamp("2025-03-17")),
        },
    )
    lookbacks = FeatureLookbackSpec(
        default_lookback="15D",
        group_lookbacks={"lag_features": "65D"},
        feature_groups={"lag_features": ["lag_60"]},
    )

    history = split.slice_feature_history(frame, lookbacks, time_col="timestamp")

    assert history["__default__"]["timestamp"].min() == pd.Timestamp("2025-02-25")
    assert history["__default__"]["timestamp"].max() == pd.Timestamp("2025-03-11")
    assert history["lag_features"]["timestamp"].min() == pd.Timestamp("2025-01-06")
    assert history["lag_features"]["timestamp"].max() == pd.Timestamp("2025-03-11")

def test_feature_lookback_spec_rejects_non_duration_sizes() -> None:
    split = TimeSplit(
        fold=0,
        segments={"train": np.array([0]), "test": np.array([1])},
        boundaries={
            "train": SegmentBoundaries(pd.Timestamp("2025-03-10"), pd.Timestamp("2025-03-11")),
            "test": SegmentBoundaries(pd.Timestamp("2025-03-11"), pd.Timestamp("2025-03-12")),
        },
    )

    with pytest.raises(ValueError, match="duration-based sizes"):
        split.feature_history_bounds(
            FeatureLookbackSpec(group_lookbacks={"lag_features": 10}),
        )

def test_feature_lookback_spec_rejects_non_duration_default() -> None:
    spec = FeatureLookbackSpec(default_lookback=5)

    with pytest.raises(ValueError, match="duration-based sizes"):
        spec.normalized_default_lookback()
