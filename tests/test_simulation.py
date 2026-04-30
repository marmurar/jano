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

def test_describe_simulation_returns_summary_with_html(tmp_path) -> None:
    frame = build_frame(size=10)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="4D",
            test_size="2D",
        ),
        step="1D",
        strategy="rolling",
    )

    summary = splitter.describe_simulation(frame)

    assert isinstance(summary, SimulationSummary)
    assert summary.total_folds == 4
    assert summary.segment_order == ["train", "test"]
    assert isinstance(summary.chart_data, SimulationChartData)
    assert summary.chart_data.segment_stats["train"]["avg_rows"] == 4.0
    assert "Fold 0" in summary.html
    assert "Temporal partition simulation overview" in summary.html
    assert "Segment profile" in summary.html

    output_path = tmp_path / "simulation.html"
    written_path = summary.write_html(output_path)
    assert written_path == output_path
    assert output_path.exists()
    assert "Jano simulation summary" in output_path.read_text(encoding="utf-8")

def test_describe_simulation_can_write_html_directly(tmp_path) -> None:
    frame = build_frame(size=10)
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

    output_path = tmp_path / "report.html"
    summary = splitter.describe_simulation(
        frame,
        output_path=output_path,
        title="Walk-forward report",
    )

    assert output_path.exists()
    assert summary.title == "Walk-forward report"
    assert "validation" in summary.to_dict()["segment_order"]
    assert not summary.to_frame().empty

def test_describe_simulation_rejects_empty_fold_configuration() -> None:
    frame = build_frame(size=4)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="4D",
            test_size="4D",
        ),
        step="1D",
        strategy="single",
    )

    with pytest.raises(ValueError, match="did not produce any valid folds"):
        splitter.describe_simulation(frame)

def test_describe_simulation_can_return_html_or_chart_data() -> None:
    frame = build_frame(size=10)
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

    html = splitter.describe_simulation(frame, output="html", title="HTML only")
    chart_data = splitter.describe_simulation(frame, output="chart_data")

    assert isinstance(html, str)
    assert "HTML only" in html
    assert isinstance(chart_data, SimulationChartData)
    assert chart_data.segment_order == ["train", "validation", "test"]
    assert chart_data.segment_colors["validation"]
    assert chart_data.folds[0]["segments"]["train"]["offset_pct"] == 0.0

def test_describe_simulation_rejects_invalid_output_mode() -> None:
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

    with pytest.raises(ValueError, match="output must be one of"):
        splitter.describe_simulation(frame, output="svg")

def test_temporal_simulation_runs_without_manual_iteration(tmp_path) -> None:
    frame = build_frame(size=10)
    simulation = TemporalSimulation(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="4D",
            test_size="2D",
        ),
        step="1D",
        strategy="rolling",
    )

    output_path = tmp_path / "simulation.html"
    result = simulation.run(frame, output_path=output_path, title="Production month")

    assert isinstance(result, SimulationResult)
    assert result.total_folds == 4
    assert isinstance(result.summary, SimulationSummary)
    assert isinstance(result.chart_data, SimulationChartData)
    assert "Production month" in result.html
    assert output_path.exists()
    assert len(list(result.iter_splits())) == 4
    assert not result.to_frame().empty
    assert result.to_dict()["total_folds"] == 4

def test_temporal_simulation_exposes_low_level_splitter() -> None:
    simulation = TemporalSimulation(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size=5,
            test_size=3,
        ),
        step=1,
        strategy="single",
    )

    splitter = simulation.as_splitter()

    assert isinstance(splitter, TemporalBacktestSplitter)
    assert simulation.time_col == "timestamp"

def test_temporal_simulation_can_start_from_specific_date_and_limit_folds() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-08-20", periods=80, freq="D"),
            "feature": range(80),
            "target": range(200, 280),
        }
    )
    simulation = TemporalSimulation(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="15D",
            test_size="4D",
        ),
        step="1D",
        strategy="rolling",
        start_at="2025-09-01",
        max_folds=15,
    )

    result = simulation.run(frame, title="From explicit start")

    assert result.total_folds == 15
    assert result.frame["timestamp"].min() == pd.Timestamp("2025-09-01")
    assert result.splits[0].boundaries["train"].start == pd.Timestamp("2025-09-01")
    assert result.splits[-1].fold == 14

def test_temporal_simulation_rejects_empty_window() -> None:
    frame = build_frame(size=10)
    simulation = TemporalSimulation(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size=5,
            test_size=3,
        ),
        step=1,
        strategy="single",
        start_at="2026-01-01",
    )

    with pytest.raises(ValueError, match="does not contain any rows"):
        simulation.run(frame)

def test_temporal_simulation_can_constrain_with_end_at() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-09-01", periods=20, freq="D"),
            "feature": range(20),
            "target": range(300, 320),
        }
    )
    simulation = TemporalSimulation(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="5D",
            test_size="2D",
        ),
        step="1D",
        strategy="rolling",
        end_at="2025-09-12",
    )

    result = simulation.run(frame)

    assert result.frame["timestamp"].max() == pd.Timestamp("2025-09-12")

def test_temporal_simulation_partition_property_returns_validated_partition() -> None:
    simulation = TemporalSimulation(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="5D",
            test_size="2D",
        ),
        step="1D",
        strategy="rolling",
    )

    assert simulation.partition.layout == "train_test"
    assert simulation.partition.size_kind == "duration"

def test_temporal_simulation_rejects_invalid_max_folds() -> None:
    with pytest.raises(ValueError, match="max_folds must be greater than zero"):
        TemporalSimulation(
            time_col="timestamp",
            partition=TemporalPartitionSpec(
                layout="train_test",
                train_size=5,
                test_size=3,
            ),
            step=1,
            strategy="single",
            max_folds=0,
        )

def test_walk_forward_policy_wraps_temporal_simulation() -> None:
    frame = build_frame(size=20)
    policy = WalkForwardPolicy(
        "timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size="4D", test_size="2D"),
        step="1D",
        strategy="rolling",
        max_folds=3,
    )

    plan = policy.plan(frame, title="Walk-forward")
    result = policy.run(frame, title="Walk-forward")

    assert isinstance(plan, SimulationPlan)
    assert isinstance(result, SimulationResult)
    assert plan.total_folds == 3
    assert result.total_folds == 3
    assert isinstance(policy.as_splitter(), TemporalBacktestSplitter)
