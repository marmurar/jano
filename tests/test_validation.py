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

def test_calendar_frequency_requires_duration_partitions() -> None:
    with pytest.raises(ValueError, match="duration-based"):
        TemporalBacktestSplitter(
            time_col="timestamp",
            partition=TemporalPartitionSpec(
                layout="train_test",
                train_size=5,
                test_size=2,
                calendar_frequency="D",
            ),
            step=1,
            strategy="single",
        )

def test_expanding_train_val_test_layout() -> None:
    frame = build_frame(size=12)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_val_test",
            train_size=4,
            validation_size=2,
            test_size=2,
        ),
        step=2,
        strategy="expanding",
    )

    splits = list(splitter.iter_splits(frame))

    assert len(splits) == 3
    first = splits[0]
    second = splits[1]

    assert list(first.segments.keys()) == ["train", "validation", "test"]
    assert len(first.segments["train"]) == 4
    assert len(first.segments["validation"]) == 2
    assert len(first.segments["test"]) == 2
    assert len(second.segments["train"]) == 6
    assert len(second.segments["validation"]) == 2

def test_invalid_layout_requires_validation_and_test_sizes() -> None:
    with pytest.raises(ValueError, match="validation_size and test_size"):
        TemporalBacktestSplitter(
            time_col="timestamp",
            partition=TemporalPartitionSpec(
                layout="train_val_test",
                train_size=0.6,
                test_size=0.2,
            ),
            step=0.2,
            strategy="single",
        )

def test_mixed_unit_families_are_rejected() -> None:
    with pytest.raises(ValueError, match="same unit family"):
        TemporalBacktestSplitter(
            time_col="timestamp",
            partition=TemporalPartitionSpec(
                layout="train_test",
                train_size="7D",
                test_size=0.3,
            ),
            step="1D",
            strategy="single",
        )

@pytest.mark.parametrize("value", [True, 0, 1.2, "not-a-duration"])
def test_invalid_size_values_are_rejected(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        SizeSpec.from_value(value)

def test_validation_accepts_train_val_test_with_both_gaps() -> None:
    simulation = TemporalSimulation(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_val_test",
            train_size="5D",
            validation_size="2D",
            test_size="2D",
            gap_before_validation="1D",
            gap_before_test="1D",
        ),
        step="1D",
        strategy="rolling",
    )

    assert simulation.partition.gaps["validation"].kind == "duration"
    assert simulation.partition.gaps["test"].kind == "duration"

def test_validation_rejects_missing_test_size_for_train_test() -> None:
    with pytest.raises(ValueError, match="test_size is required"):
        TemporalSimulation(
            time_col="timestamp",
            partition=TemporalPartitionSpec(
                layout="train_test",
                train_size=5,
            ),
            step=1,
        )

def test_slicing_validation_types_and_policy_helpers_cover_error_branches(monkeypatch) -> None:
    frame = build_frame(size=8)
    semantics = TemporalSemanticsSpec(timeline_col="timestamp", segment_time_cols={"train": "timestamp"})
    indexer = runner_module.TemporalSimulation(
        "timestamp",
        TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
        2,
        strategy="rolling",
    ).as_splitter().plan(frame)._engine
    time_indexer = planning_module.TimeIndexer(engine=indexer, semantics=semantics)
    assert time_indexer.bounds_between(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")) == (0, 2)
    assert time_indexer.bounds_between_for_column("timestamp", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")) == (0, 2)
    assert time_indexer.slice_between(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")).tolist() == [0, 1]

    with pytest.raises(ValueError, match="layout must be"):
        runner_module.TemporalBacktestSplitter(
            time_col="timestamp",
            partition=TemporalPartitionSpec(layout="invalid", train_size=4, test_size=2),
            step=2,
            strategy="rolling",
        )
    with pytest.raises(ValueError, match="calendar_frequency must be a valid fixed pandas frequency"):
        TemporalBacktestSplitter(
            time_col="timestamp",
            partition=TemporalPartitionSpec(
                layout="train_test",
                train_size="4D",
                test_size="2D",
                calendar_frequency="ME",
            ),
            step="1D",
            strategy="rolling",
        )
    with pytest.raises(ValueError, match="timeline_col must not be None"):
        TemporalBacktestSplitter(
            time_col=TemporalSemanticsSpec(timeline_col=None),  # type: ignore[arg-type]
            partition=TemporalPartitionSpec(layout="train_test", train_size="4D", test_size="2D"),
            step="1D",
            strategy="rolling",
        )
    with pytest.raises(ValueError, match="segment_time_cols keys must be non-empty strings"):
        validation_module.validate_temporal_semantics(
            TemporalSemanticsSpec(timeline_col="timestamp", segment_time_cols={"": "timestamp"})
        )
    with pytest.raises(ValueError, match="segment_time_cols values must be non-null"):
        validation_module.validate_temporal_semantics(
            TemporalSemanticsSpec(timeline_col="timestamp", segment_time_cols={"train": None})  # type: ignore[arg-type]
        )

    assert policies_module._resolve_column(frame, 1) == "feature"
    assert policies_module._resolve_columns(frame, [0, "feature"]) == ["timestamp", "feature"]
    with pytest.raises(TypeError, match="mapping"):
        policies_module._normalize_metric_mapping("unknown")
    with pytest.raises(ValueError, match="must not be empty"):
        policies_module._normalize_metric_mapping({})
    with pytest.raises(TypeError, match="mapping"):
        policies_module._normalize_metric_mapping(["rmse", "weird"])
    with pytest.raises(TypeError, match="mapping"):
        policies_module._normalize_metric_mapping([])
    with pytest.raises(TypeError, match="must be callable"):
        policies_module._normalize_metric_mapping({"custom": "not-callable"})
    assert policies_module._normalize_metric_mapping(None) == ({}, {})
    assert policies_module._normalize_metric_mapping({"custom": lambda y, p: 0.0})[1]["custom"] == "min"

    with pytest.raises(ValueError, match="X must contain at least one row"):
        policies_module._prepare_supervised_frame(
            frame.iloc[0:0],
            time_col="timestamp",
            target_col="target",
            feature_cols=["feature"],
        )
    with pytest.raises(ValueError, match="target_col 'missing'"):
        policies_module._prepare_supervised_frame(
            frame,
            time_col="timestamp",
            target_col="missing",
            feature_cols=["feature"],
        )
    with pytest.raises(ValueError, match="empty feature set"):
        policies_module._prepare_supervised_frame(
            frame[["timestamp", "target"]],
            time_col="timestamp",
            target_col="target",
            feature_cols=None,
        )

    no_default = FeatureLookbackSpec(group_lookbacks={"lags": "3D"})
    assert no_default.normalized_default_lookback() is None
    with pytest.raises(ValueError, match="Feature lookbacks must use duration-based sizes"):
        FeatureLookbackSpec(default_lookback=2).normalized_default_lookback()

    train_growth = TrainGrowthResult(
        pd.DataFrame({"variant": [0, 1], "train_rows": [5, 10], "rmse": [1.0, 0.5], "accuracy": [0.7, 0.8]}),
        {"rmse": "min", "accuracy": "max"},
    )
    assert train_growth.find_optimal_train_size(metric="accuracy")["train_rows"] == 10
    with pytest.raises(ValueError, match="not present"):
        train_growth.find_optimal_train_size(metric="mae")
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        train_growth.find_optimal_train_size(tolerance=-0.1)

    decay = PerformanceDecayResult(
        pd.DataFrame({"window": [0, 1], "rmse": [0.0, 0.2], "accuracy": [0.0, -0.2]}),
        {"rmse": "min", "accuracy": "max"},
    )
    assert decay.find_drift_onset(metric="rmse", threshold=0.1, baseline="first") is not None
    assert decay.find_drift_onset(metric="accuracy", threshold=0.1, baseline="first") is not None
    assert decay.find_drift_onset(metric="rmse", threshold=1.0, baseline=0.0, relative=False) is None
    with pytest.raises(ValueError, match="not present"):
        decay.find_drift_onset(metric="mae")
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        decay.find_drift_onset(threshold=-0.1)
    assert decay.find_drift_onset(
        metric="accuracy",
        threshold=0.1,
        baseline=0.0,
        relative=False,
    ) is not None

    with pytest.raises(ValueError, match="train_sizes must not be empty"):
        RollingTrainHistoryPolicy(
            "timestamp",
            partition=TemporalPartitionSpec(layout="train_test", train_size="5D", test_size="1D"),
            step="1D",
            train_sizes=[],
        )

def test_remaining_validation_and_workflow_branches(monkeypatch) -> None:
    with pytest.raises(ValueError, match="order_col must resolve to a non-null column reference"):
        validation_module.validate_temporal_semantics(
            types.SimpleNamespace(
                timeline_col="timestamp",
                effective_order_col=None,
                segment_time_cols={},
            )
        )

    empty_partition = planning_module.PartitionPlan(
        frame=build_frame(),
        temporal_semantics=TemporalSemanticsSpec(timeline_col="timestamp"),
        strategy="rolling",
        size_kind="rows",
        folds=[],
    )

    policy = RollingTrainHistoryPolicy(
        "timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size="5D", test_size="1D"),
        step="1D",
        train_sizes=["1D"],
    )
    monkeypatch.setattr(policy._walk_forward, "plan", lambda X, title=None: SimulationPlan(empty_partition, "Empty"))
    with pytest.raises(ValueError, match="did not produce any valid outer iterations"):
        policy.evaluate(
            build_frame(size=10),
            model=SimpleLinearRegressor(),
            target_col="target",
            feature_cols=["feature"],
        )
