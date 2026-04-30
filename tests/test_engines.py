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
