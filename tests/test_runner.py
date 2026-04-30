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

def test_walk_forward_runner_retrains_every_fold_by_default() -> None:
    frame = build_frame(size=10)
    policy = WalkForwardPolicy(
        "timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
        step=2,
        strategy="rolling",
    )

    result = WalkForwardRunner(
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
        metrics="rmse",
    ).run(policy, frame)

    assert isinstance(result, WalkForwardRunResult)
    assert result.retrain_policy == "AlwaysRetrain"
    assert result.to_frame()["retrained"].tolist() == [True, True, True]
    assert len(result.predictions_frame()) == 6
    assert result.metric_names == ["rmse"]

def test_walk_forward_run_result_exposes_plot_ready_execution_data() -> None:
    frame = build_frame(size=10)
    policy = WalkForwardPolicy(
        "timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
        step=2,
        strategy="rolling",
    )

    result = WalkForwardRunner(
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
        retrain="periodic",
        retrain_interval=2,
        metrics=["mae", "rmse"],
    ).run(policy, frame)

    fold_summary = result.fold_summary()
    trajectory = result.metric_trajectory()
    retrain_events = result.retrain_events()
    report = result.report_data(include_predictions=True)

    assert list(fold_summary.columns) == [
        "fold",
        "retrained",
        "last_retrain_fold",
        "train_rows",
        "test_rows",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
    ]
    assert trajectory["metric"].tolist() == ["mae", "mae", "mae", "rmse", "rmse", "rmse"]
    assert set(trajectory["direction"]) == {"min"}
    assert retrain_events["fold"].tolist() == [0, 2]
    assert report["summary"]["metrics"] == ["mae", "rmse"]
    assert report["summary"]["retrain_events"] == 2
    assert report["retraining"]["policy"] == "PeriodicRetrain"
    assert report["folds"][0]["train_start"] == "2024-01-01T00:00:00"
    assert len(report["predictions"]) == len(result.predictions_frame())
    assert result.to_dict(include_predictions=True)["predictions"][0]["prediction"] == report["predictions"][0]["prediction"]

def test_walk_forward_run_result_data_helpers_handle_minimal_and_max_metric_payloads() -> None:
    result = WalkForwardRunResult(
        records=pd.DataFrame(
            {
                "fold": [0, 1],
                "retrained": [True, False],
                "last_retrain_fold": [0, 0],
                "train_rows": [4, 4],
                "test_rows": [2, 2],
                "train_start": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
                "train_end": [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-06")],
                "test_start": [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-06")],
                "test_end": [pd.Timestamp("2024-01-07"), pd.Timestamp("2024-01-08")],
                "accuracy": [np.float64(0.6), np.float64(0.8)],
            }
        ),
        predictions=pd.DataFrame(
            {
                "fold": [0],
                "row_index": [np.int64(4)],
                "retrained": [True],
                "y_true": [np.int64(1)],
                "prediction": [np.int64(1)],
                "latency": [pd.Timedelta(days=1)],
            }
        ),
        metric_directions={"accuracy": "max"},
        retrain_policy="ManualPolicy",
    )

    summary = result.summary()
    report = result.report_data()
    report_with_predictions = result.report_data(include_predictions=True)

    assert summary["accuracy_best"] == 0.8
    assert summary["accuracy_best_fold"] == 1
    assert "predictions" not in report
    assert report_with_predictions["predictions"][0]["row_index"] == 4
    assert report_with_predictions["predictions"][0]["latency"] == "1 days 00:00:00"

    no_metric = WalkForwardRunResult(
        records=result.fold_summary(),
        predictions=pd.DataFrame(),
        metric_directions={},
        retrain_policy="ManualPolicy",
    )
    assert no_metric.metric_names == []
    assert list(no_metric.metric_trajectory().columns) == [
        "fold",
        "metric",
        "value",
        "direction",
        "retrained",
    ]
    assert no_metric.report_data()["metrics"] == []

def test_walk_forward_runner_can_keep_same_model_without_retraining() -> None:
    frame = build_frame(size=10)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
        step=2,
        strategy="rolling",
    )

    result = WalkForwardRunner(
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
        retrain=False,
        metrics="rmse",
    ).run(splitter, frame)

    assert result.retrain_policy == "NeverRetrain"
    assert result.to_frame()["retrained"].tolist() == [True, False, False]
    assert result.summary()["retrain_events"] == 1

def test_walk_forward_runner_supports_periodic_retraining() -> None:
    frame = build_frame(size=10)
    policy = WalkForwardPolicy(
        "timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
        step=2,
        strategy="rolling",
    )

    result = WalkForwardRunner(
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
        retrain="periodic",
        retrain_interval=2,
        metrics="rmse",
    ).run(policy, frame)

    assert result.retrain_policy == "PeriodicRetrain"
    assert result.to_frame()["retrained"].tolist() == [True, False, True]

def test_walk_forward_runner_accepts_separate_y_input() -> None:
    frame = build_frame(size=10)
    policy = WalkForwardPolicy(
        "timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
        step=2,
        strategy="rolling",
    )

    result = WalkForwardRunner(
        model=SimpleLinearRegressor(),
        feature_cols=["feature"],
        metrics="rmse",
    ).run(policy, frame[["timestamp", "feature"]], frame["target"])

    assert "rmse" in result.to_frame().columns
    assert len(result.predictions_frame()) == 6

def test_walk_forward_runner_respects_policy_max_folds() -> None:
    frame = build_frame(size=12)
    policy = WalkForwardPolicy(
        "timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
        step=1,
        strategy="rolling",
        max_folds=2,
    )

    result = WalkForwardRunner(
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
        retrain="always",
        metrics="rmse",
    ).run(policy, frame)

    assert len(result.to_frame()) == 2

def test_walk_forward_runner_can_retrain_on_observed_drift() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
            "feature": np.arange(10),
            "target": np.arange(10),
        }
    )
    policy = WalkForwardPolicy(
        "timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
        step=2,
        strategy="rolling",
    )

    result = WalkForwardRunner(
        model=MeanRegressor(),
        target_col="target",
        feature_cols=["feature"],
        retrain_policy=DriftBasedRetrain(metric="mae", threshold=0.5, baseline="last_retrain"),
        metrics="mae",
    ).run(policy, frame)

    assert result.retrain_policy == "DriftBasedRetrain"
    assert result.to_frame()["retrained"].tolist() == [True, False, True]

def test_retrain_policy_base_interface_raises_not_implemented() -> None:
    context = RetrainContext(
        fold=0,
        split=TimeSplit(
            fold=0,
            segments={"train": np.array([0]), "test": np.array([1])},
            boundaries={
                "train": SegmentBoundaries(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")),
                "test": SegmentBoundaries(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")),
            },
        ),
        history=pd.DataFrame(),
        metric_directions={},
        last_retrain_fold=None,
    )

    with pytest.raises(NotImplementedError):
        RetrainPolicy().should_retrain(context)

def test_retrain_policies_cover_validation_and_baseline_branches() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        PeriodicRetrain(0)
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        DriftBasedRetrain(threshold=-0.1)
    with pytest.raises(ValueError, match="baseline must be one of"):
        DriftBasedRetrain(baseline="unknown")

    split = TimeSplit(
        fold=2,
        segments={"train": np.array([0, 1]), "test": np.array([2])},
        boundaries={
            "train": SegmentBoundaries(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")),
            "test": SegmentBoundaries(pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")),
        },
    )
    history = pd.DataFrame(
        {
            "fold": [0, 1],
            "retrained": [True, False],
            "rmse": [0.0, 0.2],
            "accuracy": [0.0, 0.3],
        }
    )

    empty_context = RetrainContext(0, split, pd.DataFrame(), {"rmse": "min"}, None)
    assert PeriodicRetrain(2).should_retrain(empty_context) is True
    assert DriftBasedRetrain(metric="rmse").should_retrain(empty_context) is True

    with pytest.raises(ValueError, match="not present in runner history"):
        DriftBasedRetrain(metric="mae").should_retrain(
            RetrainContext(1, split, history, {"rmse": "min"}, 0)
        )

    assert DriftBasedRetrain(metric="rmse", threshold=0.1, baseline="first").should_retrain(
        RetrainContext(2, split, history, {"rmse": "min"}, 0)
    )
    assert DriftBasedRetrain(
        metric="rmse", threshold=0.1, baseline="best", relative=False
    ).should_retrain(RetrainContext(2, split, history, {"rmse": "min"}, 0))
    assert not DriftBasedRetrain(
        metric="accuracy", threshold=0.1, baseline="previous_fold"
    ).should_retrain(
        RetrainContext(
            2,
            split,
            pd.DataFrame({"fold": [0, 1], "retrained": [True, False], "accuracy": [0.0, -0.3]}),
            {"accuracy": "max"},
            0,
        )
    )
    assert DriftBasedRetrain(
        metric="accuracy", threshold=0.1, baseline="best", relative=False
    ).should_retrain(
        RetrainContext(
            2,
            split,
            pd.DataFrame({"fold": [0, 1], "retrained": [True, False], "accuracy": [0.8, 0.6]}),
            {"accuracy": "max"},
            0,
        )
    )
    assert DriftBasedRetrain(metric="rmse", baseline="last_retrain")._baseline_value(
        RetrainContext(
            2,
            split,
            pd.DataFrame({"fold": [0, 1], "retrained": [False, False], "rmse": [0.5, 0.6]}),
            {"rmse": "min"},
            1,
        )
    ) == 0.5

def test_walk_forward_runner_covers_error_paths_and_workflow_variants(tmp_path) -> None:
    frame = build_frame(size=10)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
        step=2,
        strategy="rolling",
    )
    simulation = TemporalSimulation(
        "timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
        step=2,
        strategy="rolling",
        max_folds=2,
    )

    class SimulationWrapper:
        def __init__(self, simulation):
            self.simulation = simulation

    class SplitterWrapper:
        def __init__(self, splitter):
            self._splitter = splitter

        def as_splitter(self):
            return self._splitter

    result = WalkForwardRunner(
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
        metrics="rmse",
    ).run(simulation, frame)
    assert len(result.to_frame()) == 2

    wrapped_result = WalkForwardRunner(
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
        prediction_column="y_hat",
    ).run(SimulationWrapper(simulation), frame)
    assert "y_hat" in wrapped_result.predictions_frame().columns

    splitter_result = WalkForwardRunner(
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
    ).run(SplitterWrapper(splitter), frame)
    assert splitter_result.summary()["folds"] == 3

    with pytest.raises(ValueError, match="same number of rows"):
        WalkForwardRunner(model=SimpleLinearRegressor()).run(simulation, frame, frame["target"][:-1])
    with pytest.raises(ValueError, match="target_col is required"):
        WalkForwardRunner(model=SimpleLinearRegressor()).run(simulation, frame)
    with pytest.raises(ValueError, match="retrain_interval cannot be used together"):
        WalkForwardRunner(
            model=SimpleLinearRegressor(),
            target_col="target",
            retrain_interval=2,
            retrain_policy=AlwaysRetrain(),
        )
    with pytest.raises(ValueError, match="retrain must be True, False"):
        WalkForwardRunner(model=SimpleLinearRegressor(), target_col="target", retrain="sometimes")
    with pytest.raises(ValueError, match="retrain_interval is required"):
        WalkForwardRunner(model=SimpleLinearRegressor(), target_col="target", retrain="periodic")
    with pytest.raises(TypeError, match="workflow must be"):
        WalkForwardRunner(model=SimpleLinearRegressor(), target_col="target").run(object(), frame)

    class MissingTrainTestWorkflow:
        def as_splitter(self):
            class _Splitter:
                temporal_semantics = TemporalSemanticsSpec(timeline_col="timestamp")

                def iter_splits(self, X):
                    yield TimeSplit(
                        fold=0,
                        segments={"validation": np.array([0, 1]), "test": np.array([2, 3])},
                        boundaries={
                            "validation": SegmentBoundaries(
                                pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")
                            ),
                            "test": SegmentBoundaries(
                                pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-05")
                            ),
                        },
                    )

            return _Splitter()

    with pytest.raises(ValueError, match="requires folds with 'train' and 'test'"):
        WalkForwardRunner(
            model=SimpleLinearRegressor(),
            target_col="target",
            feature_cols=["feature"],
        ).run(MissingTrainTestWorkflow(), frame)

    empty_simulation = TemporalSimulation(
        "timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
        step=2,
        strategy="rolling",
        start_at="2030-01-01",
    )
    with pytest.raises(ValueError, match="does not contain any rows"):
        empty_simulation.run(frame)

def test_remaining_planning_runner_simulation_and_splitter_branches(monkeypatch) -> None:
    frame = build_frame(size=12)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size="4D", test_size="2D"),
        step="2D",
        strategy="rolling",
    )
    plan = splitter.plan(frame)
    sim_plan = SimulationPlan(plan, "Coverage")
    assert sim_plan.select_iterations([0]).total_folds == 1
    assert sim_plan.select_from_iteration(1).total_folds == max(sim_plan.total_folds - 1, 0)
    assert sim_plan.select_until_iteration(0).total_folds == 1
    assert sim_plan.exclude_windows(validation=[("2024-01-01", "2024-01-02")]).total_folds == sim_plan.total_folds

    assert policies_module._normalize_metric_mapping(["rmse"])[1]["rmse"] == "min"

    no_default_split = TimeSplit(
        fold=0,
        segments={"train": np.array([0, 1])},
        boundaries={"train": SegmentBoundaries(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"))},
    )
    with pytest.raises(ValueError, match="Unknown segment"):
        no_default_split.feature_history_bounds(FeatureLookbackSpec(group_lookbacks={"lags": "1D"}), segment_name="missing")
    assert "__default__" not in no_default_split.feature_history_bounds(
        FeatureLookbackSpec(group_lookbacks={"lags": "1D"}),
    )

    assert DriftBasedRetrain(metric="rmse", threshold=0.1)._baseline_value(
        RetrainContext(
            1,
            no_default_split,
            pd.DataFrame({"fold": [0], "retrained": [True], "rmse": [0.5]}),
            {"rmse": "min"},
            None,
        )
    ) == 0.5
    assert DriftBasedRetrain(
        metric="accuracy",
        threshold=0.1,
        baseline="best",
    ).should_retrain(
        RetrainContext(
            2,
            no_default_split,
            pd.DataFrame({"fold": [0, 1], "retrained": [True, False], "accuracy": [0.0, -0.3]}),
            {"accuracy": "max"},
            0,
        )
    )

    no_fold_splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=50, test_size=50),
        step=1,
        strategy="single",
    )
    assert list(no_fold_splitter.split(frame)) == []

    with pytest.raises(ValueError, match="did not produce any valid folds"):
        WalkForwardRunner(
            model=SimpleLinearRegressor(),
            target_col="target",
            feature_cols=["feature"],
        ).run(no_fold_splitter, frame)

    with pytest.raises(ValueError, match="did not produce any valid folds"):
        TemporalSimulation(
            "timestamp",
            TemporalPartitionSpec(layout="train_test", train_size=50, test_size=50),
            1,
            strategy="single",
        ).run(frame)
    with pytest.raises(ValueError, match="did not produce any valid folds"):
        TemporalSimulation(
            "timestamp",
            TemporalPartitionSpec(layout="train_test", train_size=50, test_size=50),
            1,
            strategy="single",
        ).plan(frame)

    assert isinstance(
        WalkForwardRunner(model=SimpleLinearRegressor(), target_col="target", retrain=True, retrain_interval=2).retrain_policy,
        PeriodicRetrain,
    )
    assert isinstance(
        WalkForwardRunner(model=SimpleLinearRegressor(), target_col="target", retrain="never").retrain_policy,
        NeverRetrain,
    )

    class WrappedSimulation:
        def __init__(self):
            self.simulation = TemporalSimulation(
                "timestamp",
                TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
                2,
                strategy="rolling",
                max_folds=1,
            )

    wrapped = WalkForwardRunner(
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
    ).run(WrappedSimulation(), frame)
    assert len(wrapped.to_frame()) == 1

    with pytest.raises(ValueError, match="step must use the same unit family"):
        TemporalBacktestSplitter(
            time_col="timestamp",
            partition=TemporalPartitionSpec(layout="train_test", train_size="4D", test_size="2D"),
            step=1,
            strategy="rolling",
        )

    expanding = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size="1D", test_size="10D"),
        step="10D",
        strategy="expanding",
        allow_partial=False,
    )
    assert list(expanding.iter_splits(frame))

    no_duration_fold = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="1D",
            test_size="20D",
            gap_before_train="2D",
        ),
        step="1D",
        strategy="expanding",
        allow_partial=False,
    )
    assert list(no_duration_fold.iter_splits(frame)) == []

    positional = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=1, test_size=20),
        step=1,
        strategy="single",
        allow_partial=False,
    )
    assert list(positional.iter_splits(frame)) == []
