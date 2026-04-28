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


def build_frame(size: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=size, freq="D"),
            "feature": range(size),
            "target": range(100, 100 + size),
        }
    )


def write_csv_frame(tmp_path, frame: pd.DataFrame, name: str = "frame.csv") -> str:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return str(path)


class SimpleLinearRegressor:
    def fit(self, X: pd.DataFrame, y: pd.Series):
        matrix = X.to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(matrix)), matrix])
        self.coef_, *_ = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        matrix = X.to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(matrix)), matrix])
        return design @ self.coef_


class MeanRegressor:
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.mean_ = float(np.asarray(y).mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.repeat(self.mean_, len(X))


def test_public_version_matches_installed_distribution_metadata() -> None:
    assert __version__ == "0.3.1"
    assert version("jano") == __version__


def test_single_fraction_train_test_split() -> None:
    frame = build_frame(size=10)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size=0.7,
            test_size=0.3,
        ),
        step=0.1,
        strategy="single",
    )

    split = next(splitter.iter_splits(frame))

    assert list(split.segments.keys()) == ["train", "test"]
    assert len(split.segments["train"]) == 7
    assert len(split.segments["test"]) == 3


def test_rolling_duration_splits_with_gap() -> None:
    frame = build_frame(size=8)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="3D",
            test_size="2D",
            gap_before_test="1D",
        ),
        step="1D",
        strategy="rolling",
    )

    splits = list(splitter.iter_splits(frame))

    assert len(splits) == 2
    assert splits[0].boundaries["train"].start == pd.Timestamp("2024-01-01")
    assert splits[0].boundaries["test"].start == pd.Timestamp("2024-01-05")
    assert len(splits[0].segments["train"]) == 3
    assert len(splits[0].segments["test"]) == 2


def test_duration_splits_can_align_to_calendar_days() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 05:21",
                    "2024-01-01 12:00",
                    "2024-01-02 06:00",
                    "2024-01-02 18:00",
                    "2024-01-03 07:00",
                ]
            ),
            "feature": range(5),
            "target": range(5),
        }
    )
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="1D",
            test_size="1D",
            calendar_frequency="D",
        ),
        step="1D",
        strategy="single",
    )

    split = next(splitter.iter_splits(frame))

    assert split.boundaries["train"].start == pd.Timestamp("2024-01-01 00:00")
    assert split.boundaries["train"].end == pd.Timestamp("2024-01-02 00:00")
    assert split.boundaries["test"].start == pd.Timestamp("2024-01-02 00:00")
    assert split.boundaries["test"].end == pd.Timestamp("2024-01-03 00:00")
    assert split.segments["train"].tolist() == [0, 1]
    assert split.segments["test"].tolist() == [2, 3]


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


def test_split_returns_plain_index_tuples() -> None:
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

    train_idx, test_idx = next(splitter.split(frame))
    assert train_idx.tolist() == [0, 1, 2, 3, 4]
    assert test_idx.tolist() == [5, 6, 7]


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


def test_duration_split_keeps_original_positions_on_unsorted_frame() -> None:
    frame = build_frame(size=8).iloc[[4, 0, 5, 1, 6, 2, 7, 3]].reset_index(drop=True)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="3D",
            test_size="2D",
        ),
        step="1D",
        strategy="single",
    )

    train_idx, test_idx = next(splitter.split(frame))

    assert train_idx.tolist() == [1, 3, 5]
    assert test_idx.tolist() == [7, 0]


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


def test_get_n_splits_matches_generated_splits() -> None:
    frame = build_frame(size=12)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="3D",
            test_size="2D",
            gap_before_test="1D",
        ),
        step="1D",
        strategy="rolling",
    )

    assert splitter.get_n_splits(frame) == len(list(splitter.iter_splits(frame)))


def test_allow_partial_keeps_last_incomplete_test_segment() -> None:
    frame = build_frame(size=7)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="3D",
            test_size="3D",
        ),
        step="1D",
        strategy="single",
        allow_partial=True,
    )

    split = next(splitter.iter_splits(frame))

    assert len(split.segments["train"]) == 3
    assert len(split.segments["test"]) == 3


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


def test_invalid_frame_type_is_rejected() -> None:
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

    with pytest.raises(TypeError, match="pandas DataFrame, NumPy ndarray or polars DataFrame"):
        next(splitter.iter_splits([1, 2, 3]))


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


def test_legacy_import_surface_matches_public_splitter() -> None:
    assert LegacyTemporalBacktestSplitter is TemporalBacktestSplitter


def test_legacy_describe_import_surface_matches_public_summary() -> None:
    assert LegacySimulationSummary is SimulationSummary


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


def test_missing_time_column_is_rejected() -> None:
    frame = build_frame(size=10).rename(columns={"timestamp": "event_time"})
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=5, test_size=3),
        step=1,
        strategy="single",
    )

    with pytest.raises(ValueError, match="not found"):
        next(splitter.iter_splits(frame))


def test_empty_frame_is_rejected() -> None:
    frame = build_frame(size=0)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=5, test_size=3),
        step=1,
        strategy="single",
    )

    with pytest.raises(ValueError, match="at least one row"):
        next(splitter.iter_splits(frame))


def test_invalid_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="strategy"):
        TemporalBacktestSplitter(
            time_col="timestamp",
            partition=TemporalPartitionSpec(layout="train_test", train_size=5, test_size=3),
            step=1,
            strategy="diagonal",
        )


def test_get_n_splits_requires_dataset() -> None:
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=5, test_size=3),
        step=1,
        strategy="single",
    )

    with pytest.raises(ValueError, match="X is required"):
        splitter.get_n_splits()


@pytest.mark.parametrize("value", [True, 0, 1.2, "not-a-duration"])
def test_invalid_size_values_are_rejected(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        SizeSpec.from_value(value)


def test_allow_partial_for_row_based_segments_truncates_last_segment() -> None:
    frame = build_frame(size=9)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=5, test_size=5),
        step=1,
        strategy="single",
        allow_partial=True,
    )

    split = next(splitter.iter_splits(frame))

    assert len(split.segments["train"]) == 5
    assert len(split.segments["test"]) == 4


def test_duration_allow_partial_truncates_final_segment() -> None:
    frame = build_frame(size=7)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size="4D", test_size="4D"),
        step="1D",
        strategy="single",
        allow_partial=True,
    )

    split = next(splitter.iter_splits(frame))

    assert len(split.segments["train"]) == 4
    assert len(split.segments["test"]) == 3


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


def test_fractional_size_resolution_can_fail_when_rounding_to_zero() -> None:
    frame = build_frame(size=3)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size=0.2,
            test_size=0.8,
        ),
        step=0.1,
        strategy="single",
    )

    with pytest.raises(ValueError, match="resolved to zero rows"):
        list(splitter.iter_splits(frame))


def test_duration_single_fold_breaks_after_first_valid_split() -> None:
    frame = build_frame(size=20)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="5D",
            test_size="2D",
        ),
        step="1D",
        strategy="single",
    )

    splits = list(splitter.iter_splits(frame))

    assert len(splits) == 1


def test_positional_allow_partial_breaks_when_last_segment_starts_after_dataset() -> None:
    frame = build_frame(size=4)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size=4,
            test_size=3,
        ),
        step=1,
        strategy="single",
        allow_partial=True,
    )

    with pytest.raises(StopIteration):
        next(splitter.iter_splits(frame))


def test_time_indexer_slice_between_uses_bounds_correctly() -> None:
    frame = build_frame(size=6).iloc[[3, 0, 4, 1, 5, 2]].reset_index(drop=True)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="2D",
            test_size="2D",
        ),
        step="1D",
        strategy="single",
    )

    train_idx, test_idx = next(splitter.split(frame))

    assert train_idx.tolist() == [1, 3]
    assert test_idx.tolist() == [5, 0]


def test_numpy_array_input_is_supported_with_integer_time_column() -> None:
    frame = build_frame(size=8)
    values = np.column_stack(
        [
            frame["timestamp"].astype("datetime64[ns]").astype(str).to_numpy(),
            frame["feature"].to_numpy(),
            frame["target"].to_numpy(),
        ]
    )
    splitter = TemporalBacktestSplitter(
        time_col=0,
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="3D",
            test_size="2D",
        ),
        step="1D",
        strategy="single",
    )

    train_idx, test_idx = next(splitter.split(values))

    assert train_idx.tolist() == [0, 1, 2]
    assert test_idx.tolist() == [3, 4]


def test_pandas_integer_time_column_uses_position_not_label() -> None:
    frame = build_frame(size=8)
    splitter = TemporalBacktestSplitter(
        time_col=0,
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="3D",
            test_size="2D",
        ),
        step="1D",
        strategy="single",
    )

    split = next(splitter.iter_splits(frame))
    result = TemporalSimulation(
        time_col=0,
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="3D",
            test_size="2D",
        ),
        step="1D",
        strategy="single",
        start_at="2024-01-01",
    ).run(frame)

    assert split.segments["train"].tolist() == [0, 1, 2]
    assert result.summary.time_col == "timestamp"


def test_auto_engine_keeps_numpy_input_native_for_planning() -> None:
    frame = build_frame(size=8)
    values = np.column_stack(
        [
            frame["timestamp"].astype("datetime64[ns]").astype(str).to_numpy(),
            frame["feature"].to_numpy(),
            frame["target"].to_numpy(),
        ]
    )
    splitter = TemporalBacktestSplitter(
        time_col=0,
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="3D",
            test_size="2D",
        ),
        step="1D",
        strategy="single",
    )

    plan = splitter.plan(values)

    assert plan.engine_metadata.engine == "numpy"
    assert plan.engine_metadata.input_backend == "numpy"
    assert plan.engine_metadata.converted is False
    assert plan.to_frame()["train_rows"].tolist() == [3]


def test_polars_input_is_supported() -> None:
    frame = pl.DataFrame(build_frame(size=8))
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="3D",
            test_size="2D",
        ),
        step="1D",
        strategy="single",
    )

    split = next(splitter.iter_splits(frame))

    assert split.segments["train"].tolist() == [0, 1, 2]
    assert split.segments["test"].tolist() == [3, 4]


def test_auto_engine_keeps_polars_input_native_for_planning() -> None:
    frame = pl.DataFrame(build_frame(size=8))
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="3D",
            test_size="2D",
        ),
        step="1D",
        strategy="single",
    )

    plan = splitter.plan(frame)
    materialized = plan.materialize()

    assert plan.engine_metadata.engine == "polars"
    assert plan.engine_metadata.input_backend == "polars"
    assert plan.engine_metadata.converted is False
    assert materialized[0].segments["train"].tolist() == [0, 1, 2]


def test_engine_can_force_stable_pandas_path_for_polars_input() -> None:
    frame = pl.DataFrame(build_frame(size=8))
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="3D",
            test_size="2D",
        ),
        step="1D",
        strategy="single",
        engine="pandas",
    )

    plan = splitter.plan(frame)

    assert plan.engine_metadata.engine == "pandas"
    assert plan.engine_metadata.input_backend == "polars"
    assert plan.engine_metadata.converted is True


def test_temporal_simulation_accepts_numpy_input() -> None:
    frame = build_frame(size=12)
    values = np.column_stack(
        [
            frame["timestamp"].astype("datetime64[ns]").astype(str).to_numpy(),
            frame["feature"].to_numpy(),
            frame["target"].to_numpy(),
        ]
    )
    simulation = TemporalSimulation(
        time_col=0,
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="4D",
            test_size="2D",
        ),
        step="1D",
        strategy="rolling",
    )

    result = simulation.run(values)

    assert result.total_folds == 6
    assert result.frame.columns.tolist() == [0, 1, 2]
    assert result.engine_metadata.engine == "numpy"


def test_gap_before_train_and_gap_after_test_are_respected() -> None:
    frame = build_frame(size=12)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="3D",
            test_size="2D",
            gap_before_train="1D",
            gap_before_test="1D",
            gap_after_test="2D",
        ),
        step="1D",
        strategy="rolling",
    )

    splits = list(splitter.iter_splits(frame))

    assert len(splits) == 3
    assert splits[0].boundaries["train"].start == pd.Timestamp("2024-01-02")
    assert splits[0].boundaries["train"].end == pd.Timestamp("2024-01-05")
    assert splits[0].boundaries["test"].start == pd.Timestamp("2024-01-06")
    assert splits[-1].boundaries["test"].end == pd.Timestamp("2024-01-10")


def test_temporal_semantics_can_use_different_columns_per_segment() -> None:
    frame = pd.DataFrame(
        {
            "departured_at": pd.to_datetime(
                [
                    "2025-09-01",
                    "2025-09-02",
                    "2025-09-03",
                    "2025-09-04",
                    "2025-09-05",
                    "2025-09-06",
                    "2025-09-07",
                ]
            ),
            "arrived_at": pd.to_datetime(
                [
                    "2025-09-02",
                    "2025-09-03",
                    "2025-09-04",
                    "2025-09-08",
                    "2025-09-09",
                    "2025-09-10",
                    "2025-09-11",
                ]
            ),
            "feature": range(7),
            "target": range(10, 17),
        }
    )
    splitter = TemporalBacktestSplitter(
        time_col=TemporalSemanticsSpec(
            timeline_col="departured_at",
            segment_time_cols={"train": "arrived_at", "test": "departured_at"},
        ),
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="4D",
            test_size="2D",
        ),
        step="1D",
        strategy="single",
    )

    split = next(splitter.iter_splits(frame))

    assert split.segments["train"].tolist() == [0, 1, 2]
    assert split.segments["test"].tolist() == [4, 5]


def test_custom_segment_temporal_semantics_require_duration_sizes() -> None:
    with pytest.raises(ValueError, match="require duration-based sizes"):
        TemporalBacktestSplitter(
            time_col=TemporalSemanticsSpec(
                timeline_col="departured_at",
                segment_time_cols={"train": "arrived_at"},
            ),
            partition=TemporalPartitionSpec(
                layout="train_test",
                train_size=4,
                test_size=2,
            ),
            step=1,
            strategy="single",
        )


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

    result = policy.evaluate(frame, model=SimpleLinearRegressor(), target_col="target")
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
        metrics="mae",
    )
    best = policy.find_optimal_train_size(
        frame,
        model=SimpleLinearRegressor(),
        target_col="target",
        metric="mae",
        metrics="mae",
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

    result = policy.evaluate(frame, model=SimpleLinearRegressor(), target_col="target")
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
        threshold=5.0,
        relative=False,
    )

    assert onset is not None
    assert onset["window"] == 3


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

    with pytest.raises(ValueError, match="Unknown metric"):
        policy.evaluate(frame, model=SimpleLinearRegressor(), target_col="target", metrics="bogus")

    with pytest.raises(ValueError, match="metrics mapping must not be empty"):
        policy.evaluate(frame, model=SimpleLinearRegressor(), target_col="target", metrics={})

    with pytest.raises(ValueError, match="metrics must not be empty"):
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


def test_mcp_preview_dataset_reads_local_csv(tmp_path) -> None:
    path = write_csv_frame(tmp_path, build_frame(size=6))

    result = preview_dataset(path, sample_rows=3)

    assert result["sample_rows"] == 3
    assert result["columns"] == ["timestamp", "feature", "target"]
    assert len(result["preview"]) == 3


def test_mcp_load_dataset_frame_supports_csv(tmp_path) -> None:
    path = write_csv_frame(tmp_path, build_frame(size=5))

    frame = load_dataset_frame(path)

    assert list(frame.columns) == ["timestamp", "feature", "target"]
    assert len(frame) == 5


def test_mcp_plan_walk_forward_returns_preview_rows(tmp_path) -> None:
    path = write_csv_frame(tmp_path, build_frame(size=20))

    result = plan_walk_forward(
        path,
        partition={"layout": "train_test", "train_size": "4D", "test_size": "2D"},
        step="1D",
        time_col="timestamp",
        max_folds=4,
    )

    assert result["total_folds"] == 4
    assert result["engine"] == {
        "engine": "pandas",
        "input_backend": "pandas",
        "converted": False,
    }
    assert "iteration" in result["columns"]
    assert len(result["preview"]) == 4


def test_mcp_run_walk_forward_returns_summary_and_html(tmp_path) -> None:
    path = write_csv_frame(tmp_path, build_frame(size=18))

    result = run_walk_forward(
        path,
        partition={"layout": "train_test", "train_size": "4D", "test_size": "2D"},
        step="1D",
        time_col="timestamp",
        max_folds=3,
        title="MCP simulation",
    )

    assert result["total_folds"] == 3
    assert result["engine"] == {
        "engine": "pandas",
        "input_backend": "pandas",
        "converted": False,
    }
    assert len(result["summary_preview"]) == 3
    assert "<html" in result["html"].lower()
    assert "segment_stats" in result["chart_data"]


def test_mcp_server_build_is_lazy_about_optional_dependency() -> None:
    try:
        server = build_server()
    except RuntimeError as exc:
        assert "optional MCP dependency" in str(exc)
    else:
        assert server is not None


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
        metrics="rmse",
    )

    best = policy.find_optimal_train_size(
        frame,
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
        metrics="rmse",
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
        metrics="rmse",
    )
    onset = policy.find_drift_onset(
        frame,
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
        metrics="rmse",
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
        metrics="rmse",
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


def test_partition_engine_and_io_cover_native_and_error_branches(monkeypatch) -> None:
    frame = build_frame(size=4)
    structured = np.array([(1, "a"), (2, "b")], dtype=[("num", "i4"), ("txt", "U1")])

    pandas_engine = PartitionEngine.from_input(frame.to_numpy(), prefer="pandas")
    assert pandas_engine.metadata.converted is True

    numpy_engine = PartitionEngine.from_input(frame, prefer="numpy")
    assert numpy_engine.metadata.engine == "numpy"
    assert numpy_engine.metadata.converted is True
    assert numpy_engine.column_values(1).tolist() == frame.to_numpy()[:, 1].tolist()

    structured_engine = PartitionEngine.from_input(structured, prefer="numpy")
    assert structured_engine.columns == ["num", "txt"]
    assert structured_engine.column_values(0).tolist() == [1, 2]
    assert structured_engine.column_values("txt").tolist() == ["a", "b"]

    polars_engine = PartitionEngine.from_input(frame, prefer="polars")
    assert polars_engine.metadata.engine == "polars"
    assert polars_engine.metadata.converted is True
    assert PartitionEngine.from_input(pl.from_pandas(frame), prefer="polars").metadata.converted is False

    with pytest.raises(ValueError, match="engine must be one of"):
        PartitionEngine.from_input(frame, prefer="duckdb")
    with pytest.raises(ValueError, match="out of bounds"):
        numpy_engine.column_values(99)
    with pytest.raises(ValueError, match="was not found"):
        pandas_engine.column_values("missing")

    unsupported = PartitionEngine(frame, engine="unknown", input_backend="pandas")
    object.__setattr__(unsupported.metadata, "engine", "unknown")
    with pytest.raises(RuntimeError, match="Unsupported partition engine"):
        unsupported.column_values("feature")

    scalar_engine = PartitionEngine.__new__(PartitionEngine)
    scalar_engine.data = np.array(1)
    with pytest.raises(TypeError, match="NumPy scalar inputs are not supported"):
        PartitionEngine._resolve_total_rows(scalar_engine)

    monkeypatch.setattr(engines_module, "detect_backend", lambda _: "custom")
    with pytest.raises(ValueError, match="Polars engine can only be forced"):
        PartitionEngine.from_input(frame, prefer="polars")
    with pytest.raises(ValueError, match="NumPy engine can only be forced"):
        PartitionEngine.from_input(frame, prefer="numpy")

    monkeypatch.setattr(io_module, "pl", None)

    class FakePolarsFrame:
        __module__ = "polars.fake"

    with pytest.raises(ImportError, match="Polars input support requires"):
        io_module.coerce_tabular_input(FakePolarsFrame())
    with pytest.raises(TypeError, match="NumPy scalar inputs are not supported"):
        io_module.coerce_tabular_input(np.array(1))
    assert list(io_module.coerce_tabular_input(structured).columns) == ["num", "txt"]
    assert detect_backend(frame) == "pandas"
    assert detect_backend(frame.to_numpy()) == "numpy"
    assert missing_columns(["feature", "missing", 99], ["feature", "target"]) == ["missing", 99]


def test_mcp_tools_cover_formats_and_temporal_semantics(tmp_path) -> None:
    frame = build_frame(size=6)
    csv_path = tmp_path / "frame.csv"
    parquet_path = tmp_path / "frame.parquet"
    zip_path = tmp_path / "frame.zip"
    empty_csv = tmp_path / "empty.csv"
    unknown_path = tmp_path / "frame.data"
    no_csv_zip = tmp_path / "no_csv.zip"
    frame.to_csv(csv_path, index=False)
    pd.DataFrame(columns=frame.columns).to_csv(empty_csv, index=False)
    csv_path.replace(unknown_path)
    frame.to_csv(csv_path, index=False)

    import zipfile

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(csv_path, arcname="inside.csv")
    with zipfile.ZipFile(no_csv_zip, "w") as archive:
        archive.writestr("inside.txt", "hello")

    assert len(load_dataset_frame(str(csv_path), sample_rows=2)) == 2
    parquet_path.write_bytes(b"placeholder")
    original_read_parquet = mcp_tools_module.pd.read_parquet
    mcp_tools_module.pd.read_parquet = lambda *args, **kwargs: frame.copy()
    assert len(load_dataset_frame(str(parquet_path), dataset_format="parquet", sample_rows=3)) == 3
    mcp_tools_module.pd.read_parquet = original_read_parquet
    assert len(load_dataset_frame(str(zip_path), dataset_format="zip", sample_rows=4)) == 4

    with pytest.raises(FileNotFoundError, match="Dataset was not found"):
        load_dataset_frame(str(tmp_path / "missing.csv"))
    with pytest.raises(ValueError, match="did not contain any CSV files"):
        load_dataset_frame(str(no_csv_zip), dataset_format="zip")
    with pytest.raises(ValueError, match="Loaded dataset is empty"):
        load_dataset_frame(str(empty_csv))
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        load_dataset_frame(str(csv_path), dataset_format="json")
    with pytest.raises(ValueError, match="Could not infer dataset format"):
        mcp_tools_module._resolve_dataset_format(unknown_path, "auto")
    with pytest.raises(ValueError, match="time_col is required"):
        mcp_tools_module._build_temporal_semantics(
            time_col=None,
            order_col=None,
            train_time_col=None,
            validation_time_col=None,
            test_time_col=None,
        )

    assert mcp_tools_module._build_temporal_semantics(
        time_col="timestamp",
        order_col=None,
        train_time_col=None,
        validation_time_col=None,
        test_time_col=None,
    ) == "timestamp"
    semantics = mcp_tools_module._build_temporal_semantics(
        time_col="timestamp",
        order_col="feature",
        train_time_col="timestamp",
        validation_time_col=None,
        test_time_col="target",
    )
    assert semantics.order_col == "feature"
    assert semantics.segment_time_cols == {"train": "timestamp", "test": "target"}

    preview = preview_dataset(str(csv_path), sample_rows=3)
    assert preview["sample_rows"] == 3

    plan = plan_walk_forward(
        str(csv_path),
        partition={"layout": "train_test", "train_size": 4, "test_size": 2},
        step=2,
        time_col="timestamp",
        preview_rows=1,
    )
    run = run_walk_forward(
        str(csv_path),
        partition={"layout": "train_test", "train_size": 4, "test_size": 2},
        step=2,
        time_col="timestamp",
        preview_rows=1,
    )
    assert plan["total_folds"] == 1
    assert "html" in run


def test_mcp_server_builds_tools_and_main_runs(monkeypatch) -> None:
    tools = {}
    original_build_server = mcp_server_module.build_server

    class FakeMCP:
        def __init__(self, name, instructions):
            self.name = name
            self.instructions = instructions
            self.ran = False

        def tool(self):
            def decorator(fn):
                tools[fn.__name__] = fn
                return fn

            return decorator

        def run(self):
            self.ran = True

    fake_module = types.ModuleType("mcp.server.fastmcp")
    fake_module.FastMCP = FakeMCP
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_module)
    monkeypatch.setattr(mcp_server_module, "preview_dataset", lambda *args, **kwargs: {"kind": "preview", "args": args, "kwargs": kwargs})
    monkeypatch.setattr(mcp_server_module, "plan_walk_forward", lambda *args, **kwargs: {"kind": "plan", "args": args, "kwargs": kwargs})
    monkeypatch.setattr(mcp_server_module, "run_walk_forward", lambda *args, **kwargs: {"kind": "run", "args": args, "kwargs": kwargs})

    server = build_server()
    assert server.name == "Jano"
    assert tools["preview_local_dataset"]("data.csv")["kind"] == "preview"
    assert tools["plan_walk_forward_simulation"]("data.csv", {"layout": "train_test"}, "1D", "ts")["kind"] == "plan"
    assert tools["run_walk_forward_simulation"]("data.csv", {"layout": "train_test"}, "1D", "ts")["kind"] == "run"

    monkeypatch.setattr(mcp_server_module, "build_server", lambda: server)
    mcp_server_module.main()
    assert server.ran is True
    monkeypatch.setattr(mcp_server_module, "build_server", original_build_server)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mcp.server.fastmcp":
            raise ImportError("missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="optional MCP dependency"):
        mcp_server_module.build_server()


def test_planning_simulation_and_splitter_cover_remaining_helpers(tmp_path, monkeypatch) -> None:
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
    html_path = tmp_path / "plan.html"
    assert sim_plan.write_html(html_path) == html_path
    assert materialized.to_dict()["engine"]["engine"] == materialized.engine_metadata.engine
    assert list(materialized.iter_splits())
    assert materialized.write_html(tmp_path / "result.html").exists()

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
    assert isinstance(splitter.describe_simulation(frame, output="html"), str)
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

    assert policies_module._accuracy(np.array([1, 0]), np.array([1, 1])) == 0.5
    assert policies_module._resolve_column(frame, 1) == "feature"
    assert policies_module._resolve_columns(frame, [0, "feature"]) == ["timestamp", "feature"]
    with pytest.raises(ValueError, match="Unknown metric"):
        policies_module._normalize_metric_mapping("unknown")
    with pytest.raises(ValueError, match="must not be empty"):
        policies_module._normalize_metric_mapping({})
    with pytest.raises(ValueError, match="Unknown metric"):
        policies_module._normalize_metric_mapping(["rmse", "weird"])
    with pytest.raises(ValueError, match="metrics must not be empty"):
        policies_module._normalize_metric_mapping([])
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


def test_remaining_engine_and_type_branches(monkeypatch) -> None:
    frame = build_frame(size=4)

    monkeypatch.setattr(engines_module, "pl", None)
    with pytest.raises(ImportError, match="optional 'polars' dependency"):
        PartitionEngine.from_input(frame, prefer="polars")

    monkeypatch.setattr(engines_module, "detect_backend", lambda _: "custom")
    monkeypatch.setattr(engines_module, "coerce_tabular_input", lambda _: frame.copy())
    auto_engine = PartitionEngine.from_input(object(), prefer="auto")
    assert auto_engine.metadata.engine == "pandas"
    assert auto_engine.metadata.converted is True

    class CustomFrame:
        pass

    custom_engine = PartitionEngine(CustomFrame(), engine="pandas", input_backend="custom")
    assert custom_engine.columns == ["timestamp", "feature", "target"]
    assert custom_engine.total_rows == len(frame)

    class FakePolarsFrame:
        __module__ = "polars.fake"

    with pytest.raises(ImportError, match="optional 'polars' dependency"):
        detect_backend(FakePolarsFrame())

    vector_engine = PartitionEngine(np.array([1, 2, 3]), engine="numpy", input_backend="numpy")
    assert vector_engine.columns == [0]
    assert vector_engine.total_rows == 3

    with pytest.raises(TypeError, match="Size values must be"):
        SizeSpec.from_value(object())


def test_remaining_mcp_tool_and_server_branches(monkeypatch, tmp_path) -> None:
    assert mcp_tools_module._resolve_dataset_format(Path("frame.parquet"), "auto") == "parquet"
    assert mcp_tools_module._resolve_dataset_format(Path("frame.zip"), "auto") == "zip"

    class FakeMCP:
        def __init__(self, *args, **kwargs):
            self.ran = False

        def tool(self):
            def decorator(fn):
                return fn

            return decorator

        def run(self):
            self.ran = True

    fake_module = types.ModuleType("mcp.server.fastmcp")
    fake_module.FastMCP = FakeMCP
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_module)
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        runpy.run_module("jano.mcp_server", run_name="__main__")


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
