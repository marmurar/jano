from __future__ import annotations

from importlib.metadata import version

import pandas as pd
import pytest

from jano import (
    TemporalBacktestSplitter,
    TemporalPartitionSpec,
    TemporalSemanticsSpec,
    TemporalSimulation,
    __version__,
)
from jano.describe import SimulationSummary as LegacySimulationSummary
from jano.jano import TemporalBacktestSplitter as LegacyTemporalBacktestSplitter
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


def test_public_version_matches_installed_distribution_metadata() -> None:
    assert __version__ == "0.2.0"
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

    with pytest.raises(TypeError, match="pandas DataFrame"):
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
