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
