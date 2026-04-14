from __future__ import annotations

from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest
import polars as pl

from jano import (
    FeatureLookbackSpec,
    PartitionPlan,
    PerformanceDecayPolicy,
    PlannedFold,
    SimulationPlan,
    TemporalBacktestSplitter,
    TemporalPartitionSpec,
    TemporalSemanticsSpec,
    TemporalSimulation,
    TrainGrowthPolicy,
    __version__,
)
from jano.describe import SimulationSummary as LegacySimulationSummary
from jano.jano import TemporalBacktestSplitter as LegacyTemporalBacktestSplitter
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


def test_public_version_matches_installed_distribution_metadata() -> None:
    assert __version__ == "0.3.0"
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
