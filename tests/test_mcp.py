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
