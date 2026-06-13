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

def test_splitter_plan_exposes_iterations_boundaries_and_counts() -> None:
    frame = build_frame(size=12)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="3D",
            test_size="2D",
        ),
        step="1D",
        strategy="rolling",
    )

    plan = splitter.plan(frame)
    plan_frame = plan.to_frame()

    assert isinstance(plan, PartitionPlan)
    assert plan.total_folds == 7
    assert plan_frame["iteration"].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert plan_frame["train_rows"].tolist() == [3, 3, 3, 3, 3, 3, 3]
    assert plan_frame["test_rows"].tolist() == [2, 2, 2, 2, 2, 2, 2]
    assert pd.Timestamp(plan_frame.loc[0, "train_start"]) == pd.Timestamp("2024-01-01")

def test_partition_plan_can_select_iterations_and_materialize() -> None:
    frame = build_frame(size=12)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size="3D", test_size="2D"),
        step="1D",
        strategy="rolling",
    )

    plan = splitter.plan(frame).select_iterations([1, 3])
    splits = plan.materialize()

    assert plan.to_frame()["iteration"].tolist() == [1, 3]
    assert [split.fold for split in splits] == [1, 3]

def test_partition_plan_can_select_from_iteration() -> None:
    frame = build_frame(size=12)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size="3D", test_size="2D"),
        step="1D",
        strategy="rolling",
    )

    plan = splitter.plan(frame).select_from_iteration(2)

    assert plan.to_frame()["iteration"].tolist() == [2, 3, 4, 5, 6]

def test_partition_plan_can_exclude_windows_overlapping_train() -> None:
    frame = build_frame(size=12)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size="3D", test_size="2D"),
        step="1D",
        strategy="rolling",
    )

    filtered = splitter.plan(frame).exclude_windows(
        train=[("2024-01-02", "2024-01-04")],
    )

    assert filtered.to_frame()["iteration"].tolist() == [3, 4, 5, 6]

def test_partition_plan_exclude_windows_validates_ranges() -> None:
    frame = build_frame(size=12)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size="3D", test_size="2D"),
        step="1D",
        strategy="rolling",
    )

    with pytest.raises(ValueError, match="end greater than start"):
        splitter.plan(frame).exclude_windows(train=[("2024-01-05", "2024-01-05")])

def test_simulation_plan_can_be_filtered_before_materialization() -> None:
    frame = build_frame(size=20)
    simulation = TemporalSimulation(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size="4D", test_size="2D"),
        step="1D",
        strategy="rolling",
        max_folds=5,
    )

    plan = simulation.plan(frame, title="Planned simulation")
    trimmed = plan.select_from_iteration(2)
    result = trimmed.materialize()

    assert isinstance(plan, SimulationPlan)
    assert plan.to_frame()["iteration"].tolist() == [0, 1, 2, 3, 4]
    assert trimmed.to_frame()["iteration"].tolist() == [2, 3, 4]
    assert result.total_folds == 3
    assert result.summary.title == "Planned simulation"

def test_partition_plan_iter_splits_rejects_empty_plan() -> None:
    frame = build_frame(size=8)
    empty_plan = PartitionPlan(
        frame=frame,
        temporal_semantics=TemporalSemanticsSpec(timeline_col="timestamp"),
        strategy="rolling",
        size_kind="duration",
        folds=[],
    )

    with pytest.raises(ValueError, match="does not contain any folds"):
        empty_plan.materialize()

def test_planned_fold_properties_are_exposed() -> None:
    fold = PlannedFold(
        iteration=3,
        boundaries={
            "train": SegmentBoundaries(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")),
            "test": SegmentBoundaries(pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")),
        },
        counts={"train": 2, "test": 1},
        metadata={"is_partial": True},
    )

    assert fold.fold == 3
    assert fold.is_partial is True
    assert fold.to_dict()["test_rows"] == 1

def test_planning_simulation_and_splitter_cover_remaining_helpers(monkeypatch) -> None:
    frame = build_frame(size=12)
    semantics = TemporalSemanticsSpec(
        timeline_col="timestamp",
        order_col="timestamp",
        segment_time_cols={"train": "timestamp", "test": "timestamp"},
    )
    splitter = TemporalBacktestSplitter(
        time_col=semantics,
        partition=TemporalPartitionSpec(layout="train_test", train_size="4D", test_size="2D"),
        step="2D",
        strategy="rolling",
    )
    plan = splitter.plan(frame)
    assert plan.engine_metadata.engine in {"pandas", "numpy", "polars"}
    assert list(plan.iter_splits())
    assert plan.select_iterations([0]).total_folds == 1
    assert plan.select_from_iteration(1).total_folds == max(plan.total_folds - 1, 0)
    assert plan.select_until_iteration(0).total_folds == 1
    assert plan.exclude_windows(train=[("2024-01-01", "2024-01-02")]).total_folds <= plan.total_folds
    with pytest.raises(ValueError, match="end greater than start"):
        plan.exclude_windows(train=[("2024-01-02", "2024-01-01")])

    sim_plan = SimulationPlan(plan, "Coverage")
    materialized = sim_plan.materialize()
    assert sim_plan.describe().title == "Coverage"
    assert materialized.to_dict()["engine"]["engine"] == materialized.engine_metadata.engine
    assert list(materialized.iter_splits())

    with pytest.raises(ValueError, match="greater than zero"):
        TemporalSimulation("timestamp", TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2), 1, max_folds=0)

    simulation = TemporalSimulation(
        0,
        TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
        2,
        strategy="rolling",
        max_folds=2,
    )
    numpy_frame = frame[["timestamp", "feature", "target"]].to_numpy()
    assert simulation.plan(numpy_frame).total_folds == 2
    assert simulation.run(numpy_frame).total_folds == 2
    assert simulation._timeline_column_name(frame[["timestamp", "feature", "target"]]) == "timestamp"
    assert simulation.select_input(frame).equals(frame)
    assert simulation._select_input(frame).equals(frame)

    with pytest.raises(ValueError, match="did not produce any valid folds"):
        TemporalSimulation(
            "timestamp",
            TemporalPartitionSpec(layout="train_test", train_size=20, test_size=10),
            1,
            strategy="single",
        ).plan(frame)

    with pytest.raises(ValueError, match="engine must be one of"):
        TemporalBacktestSplitter(
            time_col="timestamp",
            partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
            step=2,
            strategy="rolling",
            engine="spark",
        )
    with pytest.raises(ValueError, match="Per-segment temporal semantics currently require duration-based"):
        TemporalBacktestSplitter(
            time_col=TemporalSemanticsSpec(
                timeline_col="timestamp",
                segment_time_cols={"train": "feature"},
            ),
            partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
            step=2,
            strategy="rolling",
        )
    with pytest.raises(ValueError, match="output must be one of"):
        splitter.describe_simulation(frame, output="json")
    assert splitter.describe_simulation(frame, output="chart_data").to_dict()
    assert TemporalBacktestSplitter._is_valid_segments({"train": np.array([0])}) is True
    assert TemporalBacktestSplitter._is_valid_segments({"train": np.array([])}) is False
    assert TemporalBacktestSplitter._is_valid_count_map({"train": 1}) is True
    assert TemporalBacktestSplitter._is_valid_count_map({"train": 0}) is False
    with pytest.raises(ValueError, match="resolved to zero rows"):
        TemporalBacktestSplitter._resolve_position_size(SizeSpec.from_value(0.01), 1)

    empty_plan = planning_module.PartitionPlan(
        frame=frame,
        temporal_semantics=TemporalSemanticsSpec(timeline_col="timestamp"),
        strategy="rolling",
        size_kind="rows",
        folds=[],
    )
    with pytest.raises(ValueError, match="does not contain any folds"):
        empty_plan.materialize()

    monkeypatch.setattr(planning_module.PartitionPlan, "materialize", lambda self: [])
    monkeypatch.setattr(planning_module, "build_simulation_summary", lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not be called")))
