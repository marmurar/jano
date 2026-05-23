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
from conftest import build_frame, write_csv_frame, SimpleLinearRegressor, MeanRegressor, mae, rmse
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

def test_train_growth_policy_finds_smallest_train_with_best_rmse() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=50, freq="D"),
            "feature": np.arange(50, dtype=float),
        }
    )
    frame["target"] = (2.0 * frame["feature"]) + 3.0

    policy = TrainGrowthPolicy(
        "timestamp",
        cutoff="2025-02-10",
        train_sizes=["5D", "10D", "15D"],
        test_size="5D",
    )

    result = policy.evaluate(
        frame,
        model=SimpleLinearRegressor(),
        target_col="target",
        metrics={"rmse": rmse},
    )
    best = result.find_optimal_train_size(metric="rmse", tolerance=0.0)

    assert result.to_frame()["train_size"].tolist() == ["5 days 00:00:00", "10 days 00:00:00", "15 days 00:00:00"]
    assert best["train_size"] == "5 days 00:00:00"
    assert best["rmse"] == pytest.approx(0.0)

def test_train_growth_policy_convenience_method_returns_best_variant() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=40, freq="D"),
            "feature": np.arange(40, dtype=float),
        }
    )
    frame["target"] = frame["feature"] + 1.0

    policy = TrainGrowthPolicy(
        "timestamp",
        cutoff="2025-01-30",
        train_sizes=["4D", "8D", "12D"],
        test_size="4D",
    )

    evaluated = policy.evaluate(
        frame,
        model=SimpleLinearRegressor(),
        target_col="target",
        metrics={"mae": mae},
    )
    best = policy.find_optimal_train_size(
        frame,
        model=SimpleLinearRegressor(),
        target_col="target",
        metric="mae",
        metrics={"mae": mae},
    )

    assert best == evaluated.find_optimal_train_size(metric="mae")
    assert best["train_size"] == "12 days 00:00:00"

def test_performance_decay_policy_detects_first_problem_window() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=70, freq="D"),
            "feature": np.arange(70, dtype=float),
        }
    )
    frame["target"] = frame["feature"] + 2.0
    frame.loc[frame["timestamp"] >= pd.Timestamp("2025-02-15"), "target"] += 50.0

    policy = PerformanceDecayPolicy(
        "timestamp",
        cutoff="2025-02-01",
        train_size="20D",
        test_size="5D",
        step="5D",
        max_windows=5,
    )

    result = policy.evaluate(
        frame,
        model=SimpleLinearRegressor(),
        target_col="target",
        metrics={"rmse": rmse},
    )
    onset = result.find_drift_onset(metric="rmse", threshold=10.0, relative=False)

    assert result.to_frame()["window"].tolist() == [0, 1, 2, 3, 4]
    assert onset is not None
    assert onset["window"] == 2
    assert onset["rmse"] > 10.0

def test_performance_decay_policy_convenience_method_uses_first_window_baseline() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=60, freq="D"),
            "feature": np.arange(60, dtype=float),
        }
    )
    frame["target"] = frame["feature"] + 5.0
    frame.loc[frame["timestamp"] >= pd.Timestamp("2025-02-10"), "target"] += 30.0

    policy = PerformanceDecayPolicy(
        "timestamp",
        cutoff="2025-01-25",
        train_size="15D",
        test_size="5D",
        step="5D",
        max_windows=4,
    )

    onset = policy.find_drift_onset(
        frame,
        model=SimpleLinearRegressor(),
        target_col="target",
        metric="mae",
        metrics={"mae": mae},
        threshold=5.0,
        relative=False,
    )

    assert onset is not None
    assert onset["window"] == 3

def test_train_growth_result_validates_metric_name_and_tolerance() -> None:
    result = TrainGrowthResult(
        records=pd.DataFrame(
            {
                "variant": [0, 1],
                "train_rows": [10, 20],
                "rmse": [1.0, 0.8],
            }
        ),
        metric_directions={"rmse": "min"},
    )

    with pytest.raises(ValueError, match="not present"):
        result.find_optimal_train_size(metric="mae")

    with pytest.raises(ValueError, match="greater than or equal to zero"):
        result.find_optimal_train_size(tolerance=-0.1)

def test_train_growth_result_supports_max_direction_metrics() -> None:
    result = TrainGrowthResult(
        records=pd.DataFrame(
            {
                "variant": [0, 1, 2],
                "train_rows": [10, 20, 30],
                "accuracy": [0.80, 0.90, 0.86],
            }
        ),
        metric_directions={"accuracy": "max"},
    )

    best = result.find_optimal_train_size(metric="accuracy", tolerance=0.05)

    assert best["train_rows"] == 20
    assert best["accuracy"] == pytest.approx(0.90)

def test_performance_decay_result_validates_metric_and_threshold() -> None:
    result = PerformanceDecayResult(
        records=pd.DataFrame({"window": [0, 1], "rmse": [1.0, 1.2]}),
        metric_directions={"rmse": "min"},
    )

    with pytest.raises(ValueError, match="not present"):
        result.find_drift_onset(metric="mae")

    with pytest.raises(ValueError, match="greater than or equal to zero"):
        result.find_drift_onset(threshold=-0.1)

def test_performance_decay_result_supports_best_baseline_and_absolute_mode() -> None:
    result = PerformanceDecayResult(
        records=pd.DataFrame({"window": [0, 1, 2], "rmse": [1.4, 1.0, 1.25]}),
        metric_directions={"rmse": "min"},
    )

    onset = result.find_drift_onset(
        metric="rmse",
        threshold=0.2,
        baseline="best",
        relative=False,
    )

    assert onset is not None
    assert onset["window"] == 0

def test_performance_decay_result_supports_accuracy_and_zero_baseline_branch() -> None:
    result = PerformanceDecayResult(
        records=pd.DataFrame({"window": [0, 1, 2], "accuracy": [0.0, 0.0, -0.2]}),
        metric_directions={"accuracy": "max"},
    )

    onset = result.find_drift_onset(metric="accuracy", threshold=0.1, baseline="first")

    assert onset is not None
    assert onset["window"] == 2

def test_performance_decay_result_returns_none_when_no_degradation_detected() -> None:
    result = PerformanceDecayResult(
        records=pd.DataFrame({"window": [0, 1, 2], "rmse": [1.0, 1.02, 1.03]}),
        metric_directions={"rmse": "min"},
    )

    assert result.find_drift_onset(metric="rmse", threshold=0.1) is None

def test_train_growth_policy_rejects_invalid_metric_inputs() -> None:
    frame = build_frame(size=20)
    policy = TrainGrowthPolicy(
        "timestamp",
        cutoff="2024-01-15",
        train_sizes=["4D"],
        test_size="2D",
    )

    with pytest.raises(TypeError, match="mapping"):
        policy.evaluate(frame, model=SimpleLinearRegressor(), target_col="target", metrics="bogus")

    with pytest.raises(ValueError, match="metrics must not be empty"):
        policy.evaluate(frame, model=SimpleLinearRegressor(), target_col="target", metrics={})

    with pytest.raises(TypeError, match="mapping"):
        policy.evaluate(frame, model=SimpleLinearRegressor(), target_col="target", metrics=[])

def test_train_growth_policy_supports_custom_metric_mapping() -> None:
    frame = build_frame(size=25)
    policy = TrainGrowthPolicy(
        "timestamp",
        cutoff="2024-01-20",
        train_sizes=["5D", "8D"],
        test_size="2D",
    )

    result = policy.evaluate(
        frame,
        model=SimpleLinearRegressor(),
        target_col="target",
        metrics={"signed_error": lambda y_true, y_pred: float(np.mean(y_pred - y_true))},
    )

    assert "signed_error" in result.to_frame().columns

def test_train_growth_policy_infers_feature_columns_from_semantics() -> None:
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2025-01-01", periods=30, freq="D"),
            "available_at": pd.date_range("2025-01-02", periods=30, freq="D"),
            "feature": np.arange(30, dtype=float),
            "target": np.arange(30, dtype=float) * 2.0,
        }
    )
    policy = TrainGrowthPolicy(
        TemporalSemanticsSpec(
            timeline_col="event_time",
            segment_time_cols={"train": "available_at", "test": "event_time"},
        ),
        cutoff="2025-01-20",
        train_sizes=["5D"],
        test_size="3D",
    )

    result = policy.evaluate(frame, model=SimpleLinearRegressor(), target_col="target")

    assert result.to_frame()["train_rows"].iloc[0] > 0

def test_train_growth_policy_rejects_invalid_target_and_empty_feature_set() -> None:
    frame = build_frame(size=20)
    policy = TrainGrowthPolicy(
        "timestamp",
        cutoff="2024-01-15",
        train_sizes=["4D"],
        test_size="2D",
    )

    with pytest.raises(ValueError, match="target_col"):
        policy.evaluate(frame, model=SimpleLinearRegressor(), target_col="missing")

    with pytest.raises(ValueError, match="empty feature set"):
        policy.evaluate(
            frame,
            model=SimpleLinearRegressor(),
            target_col="target",
            feature_cols=[],
        )

def test_train_growth_policy_rejects_empty_train_sizes_and_non_duration_sizes() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        TrainGrowthPolicy("timestamp", cutoff="2024-01-10", train_sizes=[], test_size="2D")

    with pytest.raises(ValueError, match="duration-based size"):
        TrainGrowthPolicy("timestamp", cutoff="2024-01-10", train_sizes=[5], test_size="2D")

def test_train_growth_policy_rejects_empty_test_window_and_train_windows() -> None:
    frame = build_frame(size=3)

    policy = TrainGrowthPolicy(
        "timestamp",
        cutoff="2024-02-01",
        train_sizes=["2D"],
        test_size="2D",
    )
    with pytest.raises(ValueError, match="test window did not produce any rows"):
        policy.evaluate(frame, model=SimpleLinearRegressor(), target_col="target")

    policy = TrainGrowthPolicy(
        "timestamp",
        cutoff="2024-01-01",
        train_sizes=["2D"],
        test_size="1D",
    )
    with pytest.raises(ValueError, match="No valid train windows"):
        policy.evaluate(frame, model=SimpleLinearRegressor(), target_col="target")

def test_performance_decay_policy_rejects_invalid_sizes_and_windows() -> None:
    with pytest.raises(ValueError, match="duration-based size"):
        PerformanceDecayPolicy(
            "timestamp",
            cutoff="2024-01-10",
            train_size=5,
            test_size="2D",
            step="1D",
        )

    with pytest.raises(ValueError, match="greater than zero"):
        PerformanceDecayPolicy(
            "timestamp",
            cutoff="2024-01-10",
            train_size="5D",
            test_size="2D",
            step="1D",
            max_windows=0,
        )

def test_performance_decay_policy_rejects_empty_train_or_test_paths() -> None:
    frame = build_frame(size=3)

    policy = PerformanceDecayPolicy(
        "timestamp",
        cutoff="2024-02-01",
        train_size="2D",
        test_size="1D",
        step="1D",
    )
    with pytest.raises(ValueError, match="train window did not produce any rows"):
        policy.evaluate(frame, model=SimpleLinearRegressor(), target_col="target")

    policy = PerformanceDecayPolicy(
        "timestamp",
        cutoff="2024-01-03",
        train_size="2D",
        test_size="1D",
        step="1D",
    )
    with pytest.raises(ValueError, match="No valid test windows"):
        policy.evaluate(frame.iloc[:2], model=SimpleLinearRegressor(), target_col="target")

def test_performance_decay_policy_honors_max_windows_and_custom_metrics() -> None:
    frame = build_frame(size=30)
    policy = PerformanceDecayPolicy(
        "timestamp",
        cutoff="2024-01-15",
        train_size="5D",
        test_size="2D",
        step="1D",
        max_windows=2,
    )

    result = policy.evaluate(
        frame,
        model=SimpleLinearRegressor(),
        target_col="target",
        metrics={"signed_error": lambda y_true, y_pred: float(np.mean(y_pred - y_true))},
    )

    assert result.to_frame()["window"].tolist() == [0, 1]
    assert "signed_error" in result.to_frame().columns

def test_train_history_policy_wraps_train_growth_policy() -> None:
    frame = build_frame(size=30)
    policy = TrainHistoryPolicy(
        "timestamp",
        cutoff="2024-01-20",
        train_sizes=["5D", "10D", "15D"],
        test_size="3D",
    )

    result = policy.evaluate(
        frame,
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
        metrics={"rmse": rmse},
    )

    best = policy.find_optimal_train_size(
        frame,
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
        metrics={"rmse": rmse},
        metric="rmse",
        tolerance=0.0,
    )

    assert isinstance(result, TrainGrowthResult)
    assert result.to_frame()["train_size"].tolist() == ["5 days 00:00:00", "10 days 00:00:00", "15 days 00:00:00"]
    assert "train_rows" in best

def test_drift_monitoring_policy_wraps_performance_decay_policy() -> None:
    frame = build_frame(size=40)
    policy = DriftMonitoringPolicy(
        "timestamp",
        cutoff="2024-01-20",
        train_size="10D",
        test_size="2D",
        step="1D",
        max_windows=4,
    )

    result = policy.evaluate(
        frame,
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
        metrics={"rmse": rmse},
    )
    onset = policy.find_drift_onset(
        frame,
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
        metrics={"rmse": rmse},
        metric="rmse",
        threshold=0.0,
        baseline="first",
        relative=False,
    )

    assert isinstance(result, PerformanceDecayResult)
    assert len(result.to_frame()) == 4
    assert onset is not None

def test_rolling_train_history_policy_optimizes_inner_train_size_per_iteration() -> None:
    frame = build_frame(size=60)
    policy = RollingTrainHistoryPolicy(
        "timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size="15D", test_size="1D"),
        step="1D",
        strategy="rolling",
        max_folds=5,
        train_sizes=["5D", "10D", "15D"],
    )

    plan = policy.plan(frame)
    result = policy.evaluate(
        frame,
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
        metrics={"rmse": rmse},
        metric="rmse",
        tolerance=0.0,
    )

    summary = result.summary()

    assert isinstance(plan, SimulationPlan)
    assert isinstance(result, RollingTrainHistoryResult)
    assert result.to_frame()["iteration"].tolist() == [0, 1, 2, 3, 4]
    assert "optimal_train_size" in result.to_frame().columns
    assert summary["iterations"] == 5
    assert summary["metric"] == "rmse"
