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
from conftest import build_frame, write_csv_frame, SimpleLinearRegressor, MeanRegressor, mae, rmse, accuracy
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
from jano.mcp_tools import (
    compare_partition_strategies,
    compare_retrain_policies,
    find_train_history_window,
    inspect_dataset,
    load_dataset_frame,
    monitor_decay,
    plan_walk_forward,
    preview_dataset,
    run_walk_forward,
    run_walk_forward_baseline,
    suggest_partition_policy,
    validate_temporal_policy,
)
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

def test_mcp_inspect_dataset_returns_schema_and_candidates(tmp_path) -> None:
    path = write_csv_frame(tmp_path, build_frame(size=12))

    result = inspect_dataset(path, sample_rows=10, preview_rows=2)

    assert result["rows_scanned"] == 10
    assert result["sampled"] is True
    assert result["columns"][0]["name"] == "timestamp"
    assert result["time_col_candidates"][0]["name"] == "timestamp"
    assert result["target_col_candidates"][0]["name"] == "target"
    assert "feature" in result["numeric_columns"]
    assert len(result["preview"]) == 2

def test_mcp_suggest_partition_policy_returns_temporal_and_online_configs(tmp_path) -> None:
    path = write_csv_frame(tmp_path, build_frame(size=40))

    temporal = suggest_partition_policy(path, objective="walk_forward")
    weekly = suggest_partition_policy(path, objective="weekly_retraining", time_col="timestamp")
    online = suggest_partition_policy(path, objective="online", time_col="timestamp")

    assert temporal["suggestion"]["mode"] == "temporal_walk_forward"
    assert temporal["suggestion"]["time_col"] == "timestamp"
    assert temporal["suggestion"]["partition"]["layout"] == "train_test"
    assert weekly["suggestion"]["step"] == "7D"
    assert weekly["suggestion"]["partition"]["test_size"] == "7D"
    assert online["suggestion"]["mode"] == "event_based_online"
    assert online["suggestion"]["update_size"] == 1

def test_mcp_validate_temporal_policy_returns_diagnostics(tmp_path) -> None:
    path = write_csv_frame(tmp_path, build_frame(size=18))

    result = validate_temporal_policy(
        path,
        partition={"layout": "train_test", "train_size": "4D", "test_size": "2D"},
        step="1D",
        time_col="timestamp",
        max_folds=3,
    )

    assert result["valid"] is True
    assert result["issues"] == []
    assert result["total_folds"] == 3
    assert result["summary"]["train_rows_min"] > 0
    assert len(result["preview"]) == 3

def test_mcp_compare_partition_strategies_returns_plan_level_comparison(tmp_path) -> None:
    path = write_csv_frame(tmp_path, build_frame(size=22))

    result = compare_partition_strategies(
        path,
        configs=[
            {
                "name": "daily",
                "partition": {"layout": "train_test", "train_size": "5D", "test_size": "1D"},
                "step": "1D",
                "time_col": "timestamp",
                "max_folds": 3,
            },
            {
                "name": "weekly",
                "partition": {"layout": "train_test", "train_size": "7D", "test_size": "2D"},
                "step": "2D",
                "time_col": "timestamp",
                "max_folds": 2,
            },
        ],
    )

    assert [row["name"] for row in result["comparison"]] == ["daily", "weekly"]
    assert result["comparison"][0]["valid"] is True
    assert result["details"]["weekly"]["total_folds"] == 2

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

def test_mcp_run_walk_forward_baseline_returns_runner_data(tmp_path) -> None:
    path = write_csv_frame(tmp_path, build_frame(size=18))

    result = run_walk_forward_baseline(
        path,
        partition={"layout": "train_test", "train_size": "4D", "test_size": "2D"},
        step="1D",
        time_col="timestamp",
        target_col="target",
        model="mean",
        metrics={"mae": mae, "rmse": rmse},
        retrain="periodic",
        retrain_interval=2,
        max_folds=3,
        include_predictions=True,
        prediction_preview_rows=2,
    )

    assert result["summary"]["folds"] == 3
    assert result["summary"]["metrics"] == ["mae", "rmse"]
    assert result["retrain_policy"] == "PeriodicRetrain"
    assert len(result["folds_preview"]) == 3
    assert len(result["metrics_preview"]) == 6
    assert result["metric_directions"] == {"mae": "min", "rmse": "min"}
    assert len(result["predictions_preview"]) == 2

def test_mcp_run_walk_forward_baseline_supports_classification_and_drift(tmp_path) -> None:
    frame = build_frame(size=18)
    frame["class_target"] = np.where(frame["target"] > frame["target"].median(), "high", "low")
    path = write_csv_frame(tmp_path, frame)

    result = run_walk_forward_baseline(
        path,
        partition={"layout": "train_test", "train_size": "4D", "test_size": "2D"},
        step="1D",
        time_col="timestamp",
        target_col="class_target",
        model="majority_class",
        metrics={"accuracy": accuracy},
        metric_directions={"accuracy": "max"},
        retrain="on_drift",
        drift_metric="accuracy",
        drift_threshold=0.1,
        max_folds=3,
    )

    assert result["summary"]["metrics"] == ["accuracy"]
    assert result["retrain_policy"] == "DriftBasedRetrain"
    assert result["metric_directions"] == {"accuracy": "max"}
    assert "events" in result["retraining"]

def test_mcp_run_walk_forward_baseline_validates_arguments(tmp_path) -> None:
    path = write_csv_frame(tmp_path, build_frame(size=8))

    with pytest.raises(ValueError, match="target_col is required"):
        run_walk_forward_baseline(
            path,
            partition={"layout": "train_test", "train_size": "4D", "test_size": "2D"},
            step="1D",
            time_col="timestamp",
        )

    with pytest.raises(ValueError, match="model must be either"):
        run_walk_forward_baseline(
            path,
            partition={"layout": "train_test", "train_size": "4D", "test_size": "2D"},
            step="1D",
            time_col="timestamp",
            target_col="target",
            model="unsupported",
        )

def test_mcp_compare_retrain_policies_returns_agent_ready_comparison(tmp_path) -> None:
    path = write_csv_frame(tmp_path, build_frame(size=18))

    default_result = compare_retrain_policies(
        path,
        partition={"layout": "train_test", "train_size": "4D", "test_size": "2D"},
        step="1D",
        time_col="timestamp",
        target_col="target",
        model="mean",
        metrics={"mae": mae},
        max_folds=2,
    )
    assert [row["policy"] for row in default_result["comparison"]] == [
        "always",
        "never",
        "periodic_2",
    ]

    result = compare_retrain_policies(
        path,
        partition={"layout": "train_test", "train_size": "4D", "test_size": "2D"},
        step="1D",
        time_col="timestamp",
        target_col="target",
        model="mean",
        metrics={"mae": mae},
        policies=[
            {"name": "always", "retrain": "always"},
            {"name": "periodic_2", "retrain": "periodic", "retrain_interval": 2},
        ],
        max_folds=3,
    )

    assert [row["policy"] for row in result["comparison"]] == ["always", "periodic_2"]
    assert result["metric_directions"] == {"mae": "min"}
    assert result["details"]["always"]["summary"]["folds"] == 3
    assert result["details"]["periodic_2"]["retrain_policy"] == "PeriodicRetrain"

def test_mcp_train_history_and_decay_tools_return_json_ready_results(tmp_path) -> None:
    path = write_csv_frame(tmp_path, build_frame(size=18))

    history = find_train_history_window(
        path,
        time_col="timestamp",
        cutoff="2024-01-10",
        train_sizes=["3D", "5D", "7D"],
        test_size="2D",
        target_col="target",
        model="mean",
        metrics={"mae": mae, "rmse": rmse},
        metric="mae",
    )
    assert history["total_variants"] == 3
    assert history["metric_directions"] == {"mae": "min", "rmse": "min"}
    assert history["optimal"]["train_start"].startswith("2024-")
    assert len(history["records_preview"]) == 3

    decay = monitor_decay(
        path,
        time_col="timestamp",
        cutoff="2024-01-08",
        train_size="5D",
        test_size="2D",
        step="2D",
        target_col="target",
        model="mean",
        metrics={"mae": mae},
        metric="mae",
        threshold=0.0,
        relative=False,
        max_windows=3,
    )
    assert decay["total_windows"] == 3
    assert decay["metric_directions"] == {"mae": "min"}
    assert decay["drift_onset"] is not None
    assert decay["windows_preview"][0]["test_start"].startswith("2024-")

def test_mcp_study_tools_validate_required_arguments(tmp_path) -> None:
    path = write_csv_frame(tmp_path, build_frame(size=8))

    with pytest.raises(ValueError, match="target_col is required"):
        compare_retrain_policies(
            path,
            partition={"layout": "train_test", "train_size": "4D", "test_size": "2D"},
            step="1D",
            time_col="timestamp",
        )
    with pytest.raises(ValueError, match="policies must not be empty"):
        compare_retrain_policies(
            path,
            partition={"layout": "train_test", "train_size": "4D", "test_size": "2D"},
            step="1D",
            time_col="timestamp",
            target_col="target",
            policies=[],
        )
    with pytest.raises(ValueError, match="Each policy must be a dictionary"):
        compare_retrain_policies(
            path,
            partition={"layout": "train_test", "train_size": "4D", "test_size": "2D"},
            step="1D",
            time_col="timestamp",
            target_col="target",
            policies=["always"],
        )
    with pytest.raises(ValueError, match="policy retrain must be"):
        compare_retrain_policies(
            path,
            partition={"layout": "train_test", "train_size": "4D", "test_size": "2D"},
            step="1D",
            time_col="timestamp",
            target_col="target",
            policies=[{"retrain": "sometimes"}],
        )
    with pytest.raises(ValueError, match="retrain_interval is required"):
        compare_retrain_policies(
            path,
            partition={"layout": "train_test", "train_size": "4D", "test_size": "2D"},
            step="1D",
            time_col="timestamp",
            target_col="target",
            policies=[{"retrain": "periodic"}],
        )
    with pytest.raises(ValueError, match="train_sizes must not be empty"):
        find_train_history_window(
            path,
            time_col="timestamp",
            cutoff="2024-01-05",
            train_sizes=[],
            test_size="1D",
            target_col="target",
        )
    with pytest.raises(ValueError, match="cutoff is required"):
        find_train_history_window(
            path,
            time_col="timestamp",
            train_sizes=["2D"],
            test_size="1D",
            target_col="target",
        )
    with pytest.raises(ValueError, match="test_size is required"):
        find_train_history_window(
            path,
            time_col="timestamp",
            cutoff="2024-01-05",
            train_sizes=["2D"],
            target_col="target",
        )
    with pytest.raises(ValueError, match="target_col is required"):
        find_train_history_window(
            path,
            time_col="timestamp",
            cutoff="2024-01-05",
            train_sizes=["2D"],
            test_size="1D",
        )
    with pytest.raises(ValueError, match="cutoff is required"):
        monitor_decay(
            path,
            time_col="timestamp",
            train_size="2D",
            test_size="1D",
            step="1D",
            target_col="target",
        )
    with pytest.raises(ValueError, match="train_size is required"):
        monitor_decay(
            path,
            time_col="timestamp",
            cutoff="2024-01-05",
            test_size="1D",
            step="1D",
            target_col="target",
        )
    with pytest.raises(ValueError, match="test_size is required"):
        monitor_decay(
            path,
            time_col="timestamp",
            cutoff="2024-01-05",
            train_size="2D",
            step="1D",
            target_col="target",
        )
    with pytest.raises(ValueError, match="step is required"):
        monitor_decay(
            path,
            time_col="timestamp",
            cutoff="2024-01-05",
            train_size="2D",
            test_size="1D",
            target_col="target",
        )
    with pytest.raises(ValueError, match="target_col is required"):
        monitor_decay(
            path,
            time_col="timestamp",
            cutoff="2024-01-05",
            train_size="2D",
            test_size="1D",
            step="1D",
        )

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
    assert len(load_dataset_frame(str(parquet_path), dataset_format="parquet")) == 6
    mcp_tools_module.pd.read_parquet = original_read_parquet
    assert len(load_dataset_frame(str(zip_path), dataset_format="zip", sample_rows=4)) == 4
    assert len(load_dataset_frame(str(zip_path), sample_rows=4)) == 4

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

def test_mcp_baseline_helpers_cover_errors_and_json_ready() -> None:
    with pytest.raises(ValueError, match="mean baseline requires"):
        mcp_tools_module._MeanBaselineModel().fit(pd.DataFrame({"x": [1]}), ["not numeric"])
    with pytest.raises(ValueError, match="majority_class baseline requires"):
        mcp_tools_module._MajorityClassBaselineModel().fit(pd.DataFrame({"x": []}), [])

    records = mcp_tools_module._frame_records(
        pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01")],
                "duration": [pd.Timedelta(days=1)],
                "value": [np.int64(7)],
            }
        )
    )
    assert records == [
        {"timestamp": "2024-01-01T00:00:00", "duration": "1 days 00:00:00", "value": 7}
    ]
    assert mcp_tools_module._json_ready_object([np.float64(1.5)]) == [1.5]

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
    monkeypatch.setattr(mcp_server_module, "inspect_dataset", lambda *args, **kwargs: {"kind": "inspect", "args": args, "kwargs": kwargs})
    monkeypatch.setattr(mcp_server_module, "suggest_partition_policy", lambda *args, **kwargs: {"kind": "suggest", "args": args, "kwargs": kwargs})
    monkeypatch.setattr(mcp_server_module, "validate_temporal_policy", lambda *args, **kwargs: {"kind": "validate", "args": args, "kwargs": kwargs})
    monkeypatch.setattr(mcp_server_module, "compare_partition_strategies", lambda *args, **kwargs: {"kind": "compare_partitions", "args": args, "kwargs": kwargs})
    monkeypatch.setattr(mcp_server_module, "plan_walk_forward", lambda *args, **kwargs: {"kind": "plan", "args": args, "kwargs": kwargs})
    monkeypatch.setattr(mcp_server_module, "run_walk_forward", lambda *args, **kwargs: {"kind": "run", "args": args, "kwargs": kwargs})
    monkeypatch.setattr(mcp_server_module, "run_walk_forward_baseline", lambda *args, **kwargs: {"kind": "baseline", "args": args, "kwargs": kwargs})
    monkeypatch.setattr(mcp_server_module, "compare_retrain_policies", lambda *args, **kwargs: {"kind": "compare", "args": args, "kwargs": kwargs})
    monkeypatch.setattr(mcp_server_module, "find_train_history_window", lambda *args, **kwargs: {"kind": "history", "args": args, "kwargs": kwargs})
    monkeypatch.setattr(mcp_server_module, "monitor_decay", lambda *args, **kwargs: {"kind": "decay", "args": args, "kwargs": kwargs})

    server = build_server()
    assert server.name == "Jano"
    assert tools["preview_local_dataset"]("data.csv")["kind"] == "preview"
    assert tools["inspect_local_dataset"]("data.csv")["kind"] == "inspect"
    assert tools["suggest_temporal_partition_policy"]("data.csv")["kind"] == "suggest"
    assert tools["validate_temporal_partition_policy"](
        "data.csv",
        {"layout": "train_test"},
        "1D",
        "ts",
    )["kind"] == "validate"
    assert tools["compare_temporal_partition_strategies"](
        "data.csv",
        [{"partition": {"layout": "train_test"}, "step": "1D", "time_col": "ts"}],
    )["kind"] == "compare_partitions"
    assert tools["plan_walk_forward_simulation"]("data.csv", {"layout": "train_test"}, "1D", "ts")["kind"] == "plan"
    assert tools["run_walk_forward_simulation"]("data.csv", {"layout": "train_test"}, "1D", "ts")["kind"] == "run"
    assert tools["run_walk_forward_baseline_model"](
        "data.csv",
        {"layout": "train_test"},
        "1D",
        "ts",
        "target",
    )["kind"] == "baseline"
    assert tools["compare_retrain_policy_baselines"](
        "data.csv",
        {"layout": "train_test"},
        "1D",
        "ts",
        "target",
    )["kind"] == "compare"
    assert tools["find_train_history_window_baseline"](
        "data.csv",
        "ts",
        "2024-01-01",
        ["7D"],
        "1D",
        "target",
    )["kind"] == "history"
    assert tools["monitor_decay_baseline"](
        "data.csv",
        "ts",
        "2024-01-01",
        "7D",
        "1D",
        "1D",
        "target",
    )["kind"] == "decay"

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
    assert mcp_tools_module._duration_string(pd.Timedelta(0)) == "1D"

    no_time_path = write_csv_frame(
        tmp_path,
        pd.DataFrame({"feature": [1, 2, 3], "value": [4, 5, 6]}),
    )
    with pytest.raises(ValueError, match="Could not infer a time column"):
        suggest_partition_policy(no_time_path)
    with pytest.raises(ValueError, match="could not be parsed"):
        suggest_partition_policy(no_time_path, time_col="feature")
    valid_time_path = write_csv_frame(tmp_path, build_frame(size=6))
    with pytest.raises(ValueError, match="objective must be"):
        suggest_partition_policy(valid_time_path, time_col="timestamp", objective="unsupported")
    with pytest.raises(ValueError, match="configs must not be empty"):
        compare_partition_strategies(no_time_path, configs=[])
    with pytest.raises(ValueError, match="Each config must be a dictionary"):
        compare_partition_strategies(no_time_path, configs=["bad"])
    duplicate_time_path = write_csv_frame(
        tmp_path,
        pd.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-01-01", "2024-01-01"],
                "target": [1, 2, 3],
            }
        ),
    )
    assert suggest_partition_policy(duplicate_time_path, time_col="timestamp")["suggestion"]["step"] == "1D"

    typed = pd.DataFrame(
        {
            "event_time": pd.date_range("2024-01-01", periods=3, freq="D"),
            "flag": [True, False, True],
            "description": ["long free text a", "long free text b", "long free text c"],
            "label": ["a", "b", "a"],
        }
    )
    profile = inspect_dataset(write_csv_frame(tmp_path, typed), full_scan=True)
    kinds = {column["name"]: column["kind"] for column in profile["columns"]}
    assert profile["time_col_candidates"][0]["name"] == "event_time"
    assert mcp_tools_module._profile_column(typed, "flag")["kind"] == "boolean"
    datetime_profile = mcp_tools_module._profile_column(typed, "event_time")
    assert datetime_profile["kind"] == "datetime"
    assert mcp_tools_module._time_candidate_from_profile(datetime_profile)["score"] > 1.0
    assert mcp_tools_module._target_candidate_from_profile({"name": "notes", "kind": "text", "unique_count": 3}) is None
    invalid_time_profile = mcp_tools_module._profile_column(
        pd.DataFrame({"event_time": ["not a date", "still not a date"]}),
        "event_time",
    )
    assert "datetime_parse_ratio" not in invalid_time_profile
    assert profile["target_col_candidates"][0]["name"] == "label"

    issues, warnings = mcp_tools_module._diagnose_plan_frame(
        pd.DataFrame(
            {
                "train_rows": [0, 4],
                "test_rows": [3, 4],
                "is_partial": [True, False],
                "train_end": ["2024-01-02", "2024-01-04"],
                "test_start": ["2024-01-01", "2024-01-04"],
            }
        )
    )
    assert "train has one or more empty folds" in issues
    assert "one or more folds have test_start before train_end" in issues
    assert "plan contains partial folds" in warnings
    assert "one or more folds start test at the same timestamp train ends" in warnings
    assert mcp_tools_module._diagnose_plan_frame(pd.DataFrame())[0] == ["policy produced no folds"]
    clean_issues, clean_warnings = mcp_tools_module._diagnose_plan_frame(
        pd.DataFrame({"train_rows": [20], "test_rows": [8], "is_partial": [False]})
    )
    assert clean_issues == []
    assert clean_warnings == []
    assert mcp_tools_module._diagnose_plan_frame(pd.DataFrame({"validation_rows": [0]}))[0] == [
        "validation has one or more empty folds"
    ]
    assert mcp_tools_module._summarize_plan_frame(pd.DataFrame({"train_rows": []})) == {}

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
