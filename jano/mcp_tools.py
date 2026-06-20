from __future__ import annotations

from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd

from ._serialization import _frame_records, _json_ready, _json_ready_object
from .campaigns import SimulationCampaign, SimulationVariant
from .policies import PerformanceDecayPolicy, TrainGrowthPolicy
from .runner import DriftBasedRetrain, WalkForwardRunner
from .simulation import TemporalSimulation
from .types import TemporalPartitionSpec, TemporalSemanticsSpec


def load_dataset_frame(
    dataset_path: str,
    *,
    dataset_format: str = "auto",
    sample_rows: int | None = None,
) -> pd.DataFrame:
    """Load a local tabular dataset for MCP tools.

    Args:
        dataset_path: Local path to a CSV file, Parquet file or ZIP archive containing
            a CSV file.
        dataset_format: Explicit format (``"csv"``, ``"parquet"`` or ``"zip"``) or
            ``"auto"`` to infer it from the file extension.
        sample_rows: Optional maximum number of rows to read.

    Returns:
        Loaded dataset as a pandas DataFrame.
    """

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset was not found: {path}")

    resolved_format = _resolve_dataset_format(path, dataset_format)

    if resolved_format == "csv":
        frame = pd.read_csv(path, nrows=sample_rows)
    elif resolved_format == "parquet":
        frame = pd.read_parquet(path)
        if sample_rows is not None:
            frame = frame.head(sample_rows).copy()
    elif resolved_format == "zip":
        with zipfile.ZipFile(path) as archive:
            csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_members:
                raise ValueError("ZIP dataset did not contain any CSV files")
            member = csv_members[0]
            frame = pd.read_csv(archive.open(member), nrows=sample_rows)
    else:
        raise ValueError(f"Unsupported dataset format '{resolved_format}'")

    if frame.empty:
        raise ValueError("Loaded dataset is empty")
    return frame


def preview_dataset(
    dataset_path: str,
    *,
    dataset_format: str = "auto",
    sample_rows: int = 5,
) -> dict[str, Any]:
    """Return a compact preview of a local dataset for an MCP client.

    Args:
        dataset_path: Local dataset path.
        dataset_format: Explicit format or ``"auto"``.
        sample_rows: Number of rows to include in the preview.

    Returns:
        JSON-ready dictionary with dataset path, column names and preview rows.
    """

    frame = load_dataset_frame(dataset_path, dataset_format=dataset_format, sample_rows=sample_rows)
    return {
        "dataset_path": str(Path(dataset_path)),
        "summary": {
            "rows_previewed": int(len(frame)),
            "column_count": int(len(frame.columns)),
        },
        "warnings": [],
        "recommendations": [
            "Use inspect_dataset next if you need schema hints or candidate time/target columns.",
        ],
        "columns": [str(column) for column in frame.columns],
        "sample_rows": int(len(frame)),
        "preview": frame.head(sample_rows).to_dict(orient="records"),
    }


def inspect_dataset(
    dataset_path: str,
    *,
    dataset_format: str = "auto",
    sample_rows: int = 5_000,
    preview_rows: int = 5,
    full_scan: bool = False,
) -> dict[str, Any]:
    """Inspect a local dataset and return agent-ready schema hints.

    Args:
        dataset_path: Local dataset path.
        dataset_format: Explicit format or ``"auto"``.
        sample_rows: Number of rows to scan when ``full_scan`` is false.
        preview_rows: Number of example records to include.
        full_scan: Whether to scan the full dataset.

    Returns:
        JSON-ready dictionary with column profiles and candidate time/target columns.
    """

    rows_to_read = None if full_scan else sample_rows
    frame = load_dataset_frame(
        dataset_path,
        dataset_format=dataset_format,
        sample_rows=rows_to_read,
    )
    column_profiles = [_profile_column(frame, column) for column in frame.columns]
    time_candidates = sorted(
        (
            _time_candidate_from_profile(profile)
            for profile in column_profiles
            if _time_candidate_from_profile(profile) is not None
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    target_candidates = sorted(
        (
            _target_candidate_from_profile(profile)
            for profile in column_profiles
            if _target_candidate_from_profile(profile) is not None
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    warnings = []
    if not time_candidates:
        warnings.append("No obvious time column candidates were detected.")
    if not target_candidates:
        warnings.append("No obvious target column candidates were detected.")
    return {
        "dataset_path": str(Path(dataset_path)),
        "dataset_format": _resolve_dataset_format(Path(dataset_path), dataset_format),
        "summary": {
            "rows_scanned": int(len(frame)),
            "column_count": int(len(frame.columns)),
            "time_candidate_count": int(len(time_candidates)),
            "target_candidate_count": int(len(target_candidates)),
        },
        "warnings": warnings,
        "recommendations": [
            "Use suggest_partition_policy next to get a conservative starting policy.",
            "Use validate_temporal_policy before materializing folds.",
        ],
        "rows_scanned": int(len(frame)),
        "sampled": not full_scan,
        "columns": column_profiles,
        "time_col_candidates": time_candidates,
        "target_col_candidates": target_candidates,
        "numeric_columns": [
            profile["name"] for profile in column_profiles if profile["kind"] == "numeric"
        ],
        "categorical_columns": [
            profile["name"] for profile in column_profiles if profile["kind"] == "categorical"
        ],
        "preview": _frame_records(frame.head(preview_rows)),
    }


def suggest_partition_policy(
    dataset_path: str,
    *,
    dataset_format: str = "auto",
    time_col: str | int | None = None,
    objective: str = "walk_forward",
    sample_rows: int = 5_000,
) -> dict[str, Any]:
    """Suggest a conservative Jano partition policy from a local dataset sample.

    The suggestion is intentionally heuristic. It should be validated with
    ``validate_temporal_policy`` before being used for model evaluation.
    """

    profile = inspect_dataset(
        dataset_path,
        dataset_format=dataset_format,
        sample_rows=sample_rows,
        full_scan=False,
    )
    selected_time_col = time_col
    if selected_time_col is None:
        candidates = profile["time_col_candidates"]
        if not candidates:
            raise ValueError("Could not infer a time column; pass time_col explicitly")
        selected_time_col = candidates[0]["name"]
    selected_profile = next(
        (column for column in profile["columns"] if column["name"] == str(selected_time_col)),
        None,
    )
    if selected_profile is not None and _time_candidate_from_profile(selected_profile) is None:
        raise ValueError(f"time_col '{selected_time_col}' could not be parsed as datetimes")

    frame = load_dataset_frame(dataset_path, dataset_format=dataset_format, sample_rows=sample_rows)
    timestamps = pd.to_datetime(frame[selected_time_col], errors="coerce").dropna().sort_values()
    if timestamps.empty:
        raise ValueError(f"time_col '{selected_time_col}' could not be parsed as datetimes")

    span = timestamps.iloc[-1] - timestamps.iloc[0]
    median_step = timestamps.diff().dropna().median() if len(timestamps) > 1 else pd.Timedelta(days=1)
    if not isinstance(median_step, pd.Timedelta) or median_step <= pd.Timedelta(0):
        median_step = pd.Timedelta(days=1)

    if objective not in {"walk_forward", "daily_retraining", "weekly_retraining", "online"}:
        raise ValueError(
            "objective must be 'walk_forward', 'daily_retraining', 'weekly_retraining' or 'online'"
        )

    if objective == "online":
        suggestion = {
            "mode": "event_based_online",
            "time_col": selected_time_col,
            "initial_train_size": _duration_string(max(span * 0.3, pd.Timedelta(days=1))),
            "update_size": 1,
            "update_size_alternatives": [1, 100, "1D"],
            "notes": [
                "Use this mode when the operational unit is an observed event or micro-batch.",
                "Validate candidate update sizes with OnlineUpdatePolicyStudy before choosing a policy.",
            ],
        }
    else:
        test_size = pd.Timedelta(days=1)
        step = pd.Timedelta(days=1)
        if objective == "weekly_retraining":
            test_size = pd.Timedelta(days=7)
            step = pd.Timedelta(days=7)
        train_size = max(span * 0.6, test_size * 3, median_step * 10)
        suggestion = {
            "mode": "temporal_walk_forward",
            "time_col": selected_time_col,
            "partition": {
                "layout": "train_test",
                "train_size": _duration_string(train_size),
                "test_size": _duration_string(test_size),
            },
            "step": _duration_string(step),
            "strategy": "rolling",
            "allow_partial": False,
            "max_folds": 20,
            "notes": [
                "This is a conservative starting point, not an optimal policy.",
                "Run validate_temporal_policy before materializing folds.",
            ],
        }

    warnings = []
    if objective == "online":
        warnings.append("Online policies require user-defined retrain checkpoints or update triggers.")
    else:
        warnings.append("This is a heuristic starting point; validate it before running models.")
    return {
        "dataset_path": str(Path(dataset_path)),
        "objective": objective,
        "summary": {
            "rows_scanned": int(profile["rows_scanned"]),
            "time_candidate_count": int(len(profile["time_col_candidates"])),
            "selected_time_col": str(selected_time_col),
        },
        "warnings": warnings,
        "recommendations": [
            "Run validate_temporal_policy before using this suggestion for evaluation.",
        ],
        "rows_scanned": profile["rows_scanned"],
        "time_col_candidates": profile["time_col_candidates"],
        "suggestion": suggestion,
    }


def inspect_and_recommend_dataset(
    dataset_path: str,
    *,
    dataset_format: str = "auto",
    time_col: str | int | None = None,
    objective: str = "walk_forward",
    sample_rows: int = 5_000,
    preview_rows: int = 5,
    full_scan: bool = False,
) -> dict[str, Any]:
    """Return a single agent-friendly inspection and recommendation payload."""

    inspection = inspect_dataset(
        dataset_path,
        dataset_format=dataset_format,
        sample_rows=sample_rows,
        preview_rows=preview_rows,
        full_scan=full_scan,
    )
    suggestion = suggest_partition_policy(
        dataset_path,
        dataset_format=dataset_format,
        time_col=time_col,
        objective=objective,
        sample_rows=sample_rows,
    )
    warnings = [*inspection["warnings"], *suggestion["warnings"]]
    recommendations = [
        "Validate the suggested policy with validate_temporal_policy.",
        "Compare alternatives with compare_partition_strategies if there is ambiguity.",
    ]
    return {
        "dataset_path": str(Path(dataset_path)),
        "dataset_format": inspection["dataset_format"],
        "summary": {
            **inspection["summary"],
            "objective": objective,
            "selected_time_col": suggestion["summary"]["selected_time_col"],
        },
        "warnings": warnings,
        "recommendations": recommendations,
        "inspection": inspection,
        "suggestion": suggestion,
        "preview": inspection["preview"],
    }


def validate_temporal_policy(
    dataset_path: str,
    *,
    partition: dict[str, Any],
    step: object,
    time_col: str | int | None = None,
    strategy: str = "rolling",
    allow_partial: bool = False,
    engine: str = "auto",
    start_at: object | None = None,
    end_at: object | None = None,
    max_folds: int | None = None,
    dataset_format: str = "auto",
    order_col: str | int | None = None,
    train_time_col: str | int | None = None,
    validation_time_col: str | int | None = None,
    test_time_col: str | int | None = None,
    preview_rows: int = 20,
) -> dict[str, Any]:
    """Validate a walk-forward partition policy without running a model."""

    plan_payload = plan_walk_forward(
        dataset_path,
        partition=partition,
        step=step,
        time_col=time_col,
        strategy=strategy,
        allow_partial=allow_partial,
        engine=engine,
        start_at=start_at,
        end_at=end_at,
        max_folds=max_folds,
        dataset_format=dataset_format,
        order_col=order_col,
        train_time_col=train_time_col,
        validation_time_col=validation_time_col,
        test_time_col=test_time_col,
        preview_rows=10_000,
    )
    plan_frame = pd.DataFrame(plan_payload["preview"])
    issues, warnings = _diagnose_plan_frame(plan_frame)
    recommendations = []
    if issues:
        recommendations.append("Fix the reported issues before training a model.")
    else:
        recommendations.append("If this geometry looks right, move on to plan_walk_forward_simulation or run_walk_forward_simulation.")
    return {
        "dataset_path": str(Path(dataset_path)),
        "valid": not issues,
        "overview": {
            "valid": not issues,
            "issue_count": int(len(issues)),
            "warning_count": int(len(warnings)),
            "total_folds": int(plan_payload["total_folds"]),
        },
        "warnings": warnings,
        "recommendations": recommendations,
        "issues": issues,
        "total_folds": plan_payload["total_folds"],
        "engine": plan_payload["engine"],
        "summary": _summarize_plan_frame(plan_frame),
        "preview": plan_payload["preview"][:preview_rows],
    }


def compare_partition_strategies(
    dataset_path: str,
    *,
    configs: list[dict[str, Any]],
    dataset_format: str = "auto",
    preview_rows: int = 5,
) -> dict[str, Any]:
    """Compare multiple temporal partition configurations at the plan level."""

    if not configs:
        raise ValueError("configs must not be empty")

    comparisons = []
    details = {}
    for index, config in enumerate(configs):
        if not isinstance(config, dict):
            raise ValueError("Each config must be a dictionary")
        name = str(config.get("name") or f"config_{index}")
        plan_config = {key: value for key, value in config.items() if key != "name"}
        validation = validate_temporal_policy(
            dataset_path,
            dataset_format=dataset_format,
            preview_rows=preview_rows,
            **plan_config,
        )
        row = {
            "name": name,
            "valid": validation["valid"],
            "total_folds": validation["total_folds"],
            "issues": validation["issues"],
            "warnings": validation["warnings"],
        }
        row.update(validation["summary"])
        comparisons.append(_json_ready_object(row))
        details[name] = validation

    return {
        "dataset_path": str(Path(dataset_path)),
        "summary": {
            "comparison_count": int(len(comparisons)),
            "valid_count": int(sum(1 for row in comparisons if row["valid"])),
        },
        "warnings": [],
        "recommendations": [
            "Pick the configuration with valid folds and the most useful geometry before running models.",
        ],
        "comparison": comparisons,
        "details": details,
    }


def plan_walk_forward(
    dataset_path: str,
    *,
    partition: dict[str, Any],
    step: object,
    time_col: str | int | None = None,
    strategy: str = "rolling",
    allow_partial: bool = False,
    engine: str = "auto",
    start_at: object | None = None,
    end_at: object | None = None,
    max_folds: int | None = None,
    dataset_format: str = "auto",
    order_col: str | int | None = None,
    train_time_col: str | int | None = None,
    validation_time_col: str | int | None = None,
    test_time_col: str | int | None = None,
    title: str | None = None,
    preview_rows: int = 20,
) -> dict[str, Any]:
    """Return a precomputed walk-forward plan as JSON-ready data.

    Args:
        dataset_path: Local dataset path.
        partition: Dictionary accepted by ``TemporalPartitionSpec``.
        step: Step size used by the walk-forward simulation.
        time_col: Timeline column name or position.
        strategy: Movement strategy: ``"single"``, ``"rolling"`` or ``"expanding"``.
        allow_partial: Whether to keep a final partial fold.
        engine: Internal partition engine preference: ``"auto"``, ``"pandas"``,
            ``"polars"`` or ``"numpy"``.
        start_at: Optional lower timestamp bound.
        end_at: Optional upper timestamp bound.
        max_folds: Optional maximum number of folds.
        dataset_format: Explicit format or ``"auto"``.
        order_col: Optional column used to sort the dataset.
        train_time_col: Optional timestamp column used to assign train rows.
        validation_time_col: Optional timestamp column used to assign validation rows.
        test_time_col: Optional timestamp column used to assign test rows.
        title: Optional report title.
        preview_rows: Number of planned folds to include in the returned preview.

    Returns:
        JSON-ready dictionary with fold count, plan columns and preview rows.
    """

    frame = load_dataset_frame(dataset_path, dataset_format=dataset_format)
    simulation = _build_simulation(
        partition=partition,
        step=step,
        time_col=time_col,
        strategy=strategy,
        allow_partial=allow_partial,
        engine=engine,
        start_at=start_at,
        end_at=end_at,
        max_folds=max_folds,
        order_col=order_col,
        train_time_col=train_time_col,
        validation_time_col=validation_time_col,
        test_time_col=test_time_col,
    )
    plan = simulation.plan(frame, title=title)
    plan_frame = plan.to_frame()
    return {
        "dataset_path": str(Path(dataset_path)),
        "summary": {
            "total_folds": int(plan.total_folds),
            "engine": plan.partition_plan.engine_metadata.to_dict(),
        },
        "warnings": [],
        "recommendations": [
            "Review the preview before calling run_walk_forward_simulation.",
        ],
        "total_folds": int(plan.total_folds),
        "engine": plan.partition_plan.engine_metadata.to_dict(),
        "columns": [str(column) for column in plan_frame.columns],
        "preview": plan_frame.head(preview_rows).to_dict(orient="records"),
    }


def run_walk_forward(
    dataset_path: str,
    *,
    partition: dict[str, Any],
    step: object,
    time_col: str | int | None = None,
    strategy: str = "rolling",
    allow_partial: bool = False,
    engine: str = "auto",
    start_at: object | None = None,
    end_at: object | None = None,
    max_folds: int | None = None,
    dataset_format: str = "auto",
    order_col: str | int | None = None,
    train_time_col: str | int | None = None,
    validation_time_col: str | int | None = None,
    test_time_col: str | int | None = None,
    title: str | None = None,
    preview_rows: int = 20,
) -> dict[str, Any]:
    """Run a walk-forward simulation and return a compact summary.

    Args:
        dataset_path: Local dataset path.
        partition: Dictionary accepted by ``TemporalPartitionSpec``.
        step: Step size used by the walk-forward simulation.
        time_col: Timeline column name or position.
        strategy: Movement strategy: ``"single"``, ``"rolling"`` or ``"expanding"``.
        allow_partial: Whether to keep a final partial fold.
        engine: Internal partition engine preference: ``"auto"``, ``"pandas"``,
            ``"polars"`` or ``"numpy"``.
        start_at: Optional lower timestamp bound.
        end_at: Optional upper timestamp bound.
        max_folds: Optional maximum number of folds.
        dataset_format: Explicit format or ``"auto"``.
        order_col: Optional column used to sort the dataset.
        train_time_col: Optional timestamp column used to assign train rows.
        validation_time_col: Optional timestamp column used to assign validation rows.
        test_time_col: Optional timestamp column used to assign test rows.
        title: Optional report title.
        preview_rows: Number of summary rows to include in the response.

    Returns:
        JSON-ready dictionary with fold summary and chart data.
    """

    frame = load_dataset_frame(dataset_path, dataset_format=dataset_format)
    simulation = _build_simulation(
        partition=partition,
        step=step,
        time_col=time_col,
        strategy=strategy,
        allow_partial=allow_partial,
        engine=engine,
        start_at=start_at,
        end_at=end_at,
        max_folds=max_folds,
        order_col=order_col,
        train_time_col=train_time_col,
        validation_time_col=validation_time_col,
        test_time_col=test_time_col,
    )
    result = simulation.run(frame, title=title)
    summary_frame = result.to_frame()
    return {
        "dataset_path": str(Path(dataset_path)),
        "summary": {
            "total_folds": int(result.total_folds),
            "engine": result.engine_metadata.to_dict(),
        },
        "warnings": [],
        "recommendations": [
            "Use the chart data for external reporting, not for the core simulation contract.",
        ],
        "total_folds": int(result.total_folds),
        "engine": result.engine_metadata.to_dict(),
        "summary_preview": summary_frame.head(preview_rows).to_dict(orient="records"),
        "chart_data": result.chart_data.to_dict(),
    }


def run_simulation_campaign(
    dataset_path: str,
    *,
    variants: list[dict[str, Any]],
    dataset_format: str = "auto",
    max_workers: int | None = None,
    preview_rows: int = 20,
) -> dict[str, Any]:
    """Run several walk-forward simulations in parallel and compare them.

    Each variant is a compact simulation specification that must include ``name``,
    ``partition``, ``step`` and ``time_col``. The rest of the fields mirror
    ``run_walk_forward``.
    """

    if not variants:
        raise ValueError("variants must not be empty")

    frame = load_dataset_frame(dataset_path, dataset_format=dataset_format)
    campaign_variants: list[SimulationVariant] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError("Each variant must be a dictionary")
        name = str(variant.get("name") or f"variant_{index}")
        if "partition" not in variant:
            raise ValueError(f"variant '{name}' is missing partition")
        if "step" not in variant:
            raise ValueError(f"variant '{name}' is missing step")
        if "time_col" not in variant:
            raise ValueError(f"variant '{name}' is missing time_col")
        simulation = _build_simulation(
            partition=variant["partition"],
            step=variant["step"],
            time_col=variant.get("time_col"),
            strategy=variant.get("strategy", "rolling"),
            allow_partial=bool(variant.get("allow_partial", False)),
            engine=str(variant.get("engine", "auto")),
            start_at=variant.get("start_at"),
            end_at=variant.get("end_at"),
            max_folds=variant.get("max_folds"),
            order_col=variant.get("order_col"),
            train_time_col=variant.get("train_time_col"),
            validation_time_col=variant.get("validation_time_col"),
            test_time_col=variant.get("test_time_col"),
        )
        campaign_variants.append(
            SimulationVariant(
                name=name,
                simulation=simulation,
                title=variant.get("title"),
                metadata=dict(variant.get("metadata") or {}),
            )
        )

    result = SimulationCampaign(campaign_variants).run(frame, max_workers=max_workers)
    payload = result.to_dict()
    return {
        "dataset_path": str(Path(dataset_path)),
        **payload,
        "warnings": [],
        "recommendations": [
            "Use the comparison table to decide which partition or cadence to keep.",
            "Run the chosen variant again on the Python API if you need a full object graph.",
        ],
        "runs_preview": result.to_frame().head(preview_rows).to_dict(orient="records"),
    }


def run_walk_forward_baseline(
    dataset_path: str,
    *,
    partition: dict[str, Any],
    step: object,
    time_col: str | int | None = None,
    target_col: str | int | None = None,
    feature_cols: list[str | int] | None = None,
    model: str = "mean",
    metrics: dict[str, Any] | None = None,
    metric_directions: dict[str, str] | None = None,
    retrain: bool | str = "always",
    retrain_interval: int | None = None,
    drift_metric: str = "rmse",
    drift_threshold: float = 0.05,
    drift_baseline: str = "last_retrain",
    drift_relative: bool = True,
    strategy: str = "rolling",
    allow_partial: bool = False,
    engine: str = "auto",
    start_at: object | None = None,
    end_at: object | None = None,
    max_folds: int | None = None,
    dataset_format: str = "auto",
    order_col: str | int | None = None,
    train_time_col: str | int | None = None,
    validation_time_col: str | int | None = None,
    test_time_col: str | int | None = None,
    include_predictions: bool = False,
    preview_rows: int = 20,
    prediction_preview_rows: int = 20,
) -> dict[str, Any]:
    """Run a model-free baseline over temporal folds for MCP clients.

    Args:
        dataset_path: Local dataset path.
        partition: Dictionary accepted by ``TemporalPartitionSpec``.
        step: Step size used by the walk-forward simulation.
        time_col: Timeline column name or position.
        target_col: Target column name or position.
        feature_cols: Optional feature columns. Baseline models do not require
            features, but the argument keeps the MCP surface aligned with
            ``WalkForwardRunner``.
        model: Baseline estimator: ``"mean"`` for numeric regression or
            ``"majority_class"`` for classification.
        metrics: Mapping of metric names to user-provided callables.
        metric_directions: Optional mapping declaring ``"min"`` or ``"max"`` per
            metric.
        retrain: Retraining policy: ``"always"``, ``"never"``, ``"periodic"``,
            ``"on_drift"``, ``True`` or ``False``.
        retrain_interval: Fold interval required by ``retrain="periodic"``.
        drift_metric: Metric monitored when ``retrain="on_drift"``.
        drift_threshold: Degradation threshold for drift-based retraining.
        drift_baseline: Drift baseline: ``"last_retrain"``, ``"first"``,
            ``"best"`` or ``"previous_fold"``.
        drift_relative: Whether drift threshold is relative or absolute.
        strategy: Movement strategy: ``"single"``, ``"rolling"`` or ``"expanding"``.
        allow_partial: Whether to keep a final partial fold.
        engine: Internal partition engine preference.
        start_at: Optional lower timestamp bound.
        end_at: Optional upper timestamp bound.
        max_folds: Optional maximum number of folds.
        dataset_format: Explicit format or ``"auto"``.
        order_col: Optional column used to sort the dataset.
        train_time_col: Optional timestamp column used to assign train rows.
        validation_time_col: Optional timestamp column used to assign validation rows.
        test_time_col: Optional timestamp column used to assign test rows.
        include_predictions: Whether to include a bounded prediction preview.
        preview_rows: Number of fold/metric rows returned in previews.
        prediction_preview_rows: Number of predictions returned when requested.

    Returns:
        JSON-ready dictionary with runner summary, fold preview, metric trajectory
        preview, retraining events and optional prediction preview.
    """

    if target_col is None:
        raise ValueError("target_col is required")

    frame = load_dataset_frame(dataset_path, dataset_format=dataset_format)
    simulation = _build_simulation(
        partition=partition,
        step=step,
        time_col=time_col,
        strategy=strategy,
        allow_partial=allow_partial,
        engine=engine,
        start_at=start_at,
        end_at=end_at,
        max_folds=max_folds,
        order_col=order_col,
        train_time_col=train_time_col,
        validation_time_col=validation_time_col,
        test_time_col=test_time_col,
    )

    runner_kwargs: dict[str, Any] = {
        "model": _build_baseline_model(model),
        "target_col": target_col,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "metric_directions": metric_directions,
    }
    if retrain == "on_drift":
        runner_kwargs["retrain_policy"] = DriftBasedRetrain(
            metric=drift_metric,
            threshold=drift_threshold,
            baseline=drift_baseline,
            relative=drift_relative,
        )
    else:
        runner_kwargs["retrain"] = retrain
        runner_kwargs["retrain_interval"] = retrain_interval

    result = WalkForwardRunner(**runner_kwargs).run(simulation, frame)
    payload = result.report_data(include_predictions=False)
    response: dict[str, Any] = {
        "dataset_path": str(Path(dataset_path)),
        "model": model,
        "overview": {
            "folds": int(payload["summary"]["folds"]),
            "metric_count": int(len(payload["summary"]["metrics"])),
            "retrain_policy": result.retrain_policy,
        },
        "warnings": [],
        "recommendations": [
            "If the baseline is only for triage, move custom metrics and models to the Python API.",
        ],
        "retrain_policy": result.retrain_policy,
        "summary": payload["summary"],
        "folds_preview": payload["folds"][:preview_rows],
        "metrics_preview": payload["metrics"][:preview_rows],
        "retraining": payload["retraining"],
        "metric_directions": payload["metric_directions"],
    }
    if include_predictions:
        response["predictions_preview"] = _frame_records(
            result.predictions_frame().head(prediction_preview_rows)
        )
    return response


def compare_retrain_policies(
    dataset_path: str,
    *,
    partition: dict[str, Any],
    step: object,
    time_col: str | int | None = None,
    target_col: str | int | None = None,
    feature_cols: list[str | int] | None = None,
    model: str = "mean",
    metrics: dict[str, Any] | None = None,
    policies: list[dict[str, Any]] | None = None,
    strategy: str = "rolling",
    allow_partial: bool = False,
    engine: str = "auto",
    start_at: object | None = None,
    end_at: object | None = None,
    max_folds: int | None = None,
    dataset_format: str = "auto",
    order_col: str | int | None = None,
    train_time_col: str | int | None = None,
    validation_time_col: str | int | None = None,
    test_time_col: str | int | None = None,
    preview_rows: int = 20,
) -> dict[str, Any]:
    """Compare retraining policies over the same walk-forward fold geometry.

    The MCP surface intentionally uses built-in baseline models instead of
    accepting arbitrary Python estimators. Use the Python API for custom models.
    """

    if target_col is None:
        raise ValueError("target_col is required")
    normalized_policies = _normalize_retrain_policy_specs(policies)
    details: dict[str, Any] = {}
    comparison: list[dict[str, object]] = []
    metric_directions: dict[str, str] = {}

    for policy in normalized_policies:
        name = str(policy["name"])
        result = run_walk_forward_baseline(
            dataset_path,
            partition=partition,
            step=step,
            time_col=time_col,
            target_col=target_col,
            feature_cols=feature_cols,
            model=model,
            metrics=metrics,
            retrain=policy.get("retrain", "always"),
            retrain_interval=policy.get("retrain_interval"),
            drift_metric=policy.get("drift_metric", "rmse"),
            drift_threshold=policy.get("drift_threshold", 0.05),
            drift_baseline=policy.get("drift_baseline", "last_retrain"),
            drift_relative=policy.get("drift_relative", True),
            strategy=strategy,
            allow_partial=allow_partial,
            engine=engine,
            start_at=start_at,
            end_at=end_at,
            max_folds=max_folds,
            dataset_format=dataset_format,
            order_col=order_col,
            train_time_col=train_time_col,
            validation_time_col=validation_time_col,
            test_time_col=test_time_col,
            include_predictions=False,
            preview_rows=preview_rows,
        )
        summary = dict(result["summary"])
        row = {"policy": name, "model": model, "retrain_policy": result["retrain_policy"]}
        row.update(summary)
        comparison.append(_json_ready_object(row))
        details[name] = result
        metric_directions = dict(result["metric_directions"])

    return {
        "dataset_path": str(Path(dataset_path)),
        "model": model,
        "summary": {
            "comparison_count": int(len(comparison)),
            "policy_count": int(len(normalized_policies)),
        },
        "warnings": [],
        "recommendations": [
            "Use these comparisons to choose one retraining policy, then run the full Python flow.",
        ],
        "policies": normalized_policies,
        "metric_directions": metric_directions,
        "comparison": comparison,
        "details": details,
    }


def find_train_history_window(
    dataset_path: str,
    *,
    time_col: str | int | None = None,
    cutoff: object | None = None,
    train_sizes: list[object] | None = None,
    test_size: object | None = None,
    target_col: str | int | None = None,
    feature_cols: list[str | int] | None = None,
    model: str = "mean",
    metrics: dict[str, Any] | None = None,
    metric: str = "rmse",
    tolerance: float = 0.0,
    relative: bool = True,
    gap_before_test: object | None = None,
    dataset_format: str = "auto",
    order_col: str | int | None = None,
    train_time_col: str | int | None = None,
    validation_time_col: str | int | None = None,
    test_time_col: str | int | None = None,
    preview_rows: int = 20,
) -> dict[str, Any]:
    """Evaluate train-history candidates against one fixed test window."""

    if cutoff is None:
        raise ValueError("cutoff is required")
    if not train_sizes:
        raise ValueError("train_sizes must not be empty")
    if test_size is None:
        raise ValueError("test_size is required")
    if target_col is None:
        raise ValueError("target_col is required")

    frame = load_dataset_frame(dataset_path, dataset_format=dataset_format)
    semantics = _build_temporal_semantics(
        time_col=time_col,
        order_col=order_col,
        train_time_col=train_time_col,
        validation_time_col=validation_time_col,
        test_time_col=test_time_col,
    )
    policy = TrainGrowthPolicy(
        semantics,
        cutoff=cutoff,
        train_sizes=train_sizes,
        test_size=test_size,
        gap_before_test=gap_before_test,
    )
    result = policy.evaluate(
        frame,
        model=_build_baseline_model(model),
        target_col=target_col,
        feature_cols=feature_cols,
        metrics=metrics,
    )
    records = result.to_frame()
    optimal = result.find_optimal_train_size(
        metric=metric,
        tolerance=tolerance,
        relative=relative,
    )
    return {
        "dataset_path": str(Path(dataset_path)),
        "model": model,
        "metric": metric,
        "summary": {
            "variant_count": int(len(records)),
            "optimal_train_size": optimal.get("train_size"),
        },
        "warnings": [],
        "recommendations": [
            "Use the optimal window as a starting point, then verify it with a full runner or study.",
        ],
        "metric_directions": result.metric_directions,
        "total_variants": int(len(records)),
        "optimal": _json_ready_object(optimal),
        "records_preview": _frame_records(records.head(preview_rows)),
    }


def monitor_decay(
    dataset_path: str,
    *,
    time_col: str | int | None = None,
    cutoff: object | None = None,
    train_size: object | None = None,
    test_size: object | None = None,
    step: object | None = None,
    target_col: str | int | None = None,
    feature_cols: list[str | int] | None = None,
    model: str = "mean",
    metrics: dict[str, Any] | None = None,
    metric: str = "rmse",
    threshold: float = 0.1,
    baseline: str | float = "first",
    relative: bool = True,
    gap_before_test: object | None = None,
    max_windows: int | None = None,
    dataset_format: str = "auto",
    order_col: str | int | None = None,
    train_time_col: str | int | None = None,
    validation_time_col: str | int | None = None,
    test_time_col: str | int | None = None,
    preview_rows: int = 20,
) -> dict[str, Any]:
    """Evaluate when fixed-train performance decay crosses a threshold."""

    if cutoff is None:
        raise ValueError("cutoff is required")
    if train_size is None:
        raise ValueError("train_size is required")
    if test_size is None:
        raise ValueError("test_size is required")
    if step is None:
        raise ValueError("step is required")
    if target_col is None:
        raise ValueError("target_col is required")

    frame = load_dataset_frame(dataset_path, dataset_format=dataset_format)
    semantics = _build_temporal_semantics(
        time_col=time_col,
        order_col=order_col,
        train_time_col=train_time_col,
        validation_time_col=validation_time_col,
        test_time_col=test_time_col,
    )
    policy = PerformanceDecayPolicy(
        semantics,
        cutoff=cutoff,
        train_size=train_size,
        test_size=test_size,
        step=step,
        gap_before_test=gap_before_test,
        max_windows=max_windows,
    )
    result = policy.evaluate(
        frame,
        model=_build_baseline_model(model),
        target_col=target_col,
        feature_cols=feature_cols,
        metrics=metrics,
    )
    records = result.to_frame()
    drift_onset = result.find_drift_onset(
        metric=metric,
        threshold=threshold,
        baseline=baseline,
        relative=relative,
    )
    return {
        "dataset_path": str(Path(dataset_path)),
        "model": model,
        "metric": metric,
        "threshold": float(threshold),
        "baseline": baseline,
        "relative": bool(relative),
        "summary": {
            "window_count": int(len(records)),
            "drift_onset": _json_ready_object(drift_onset),
        },
        "warnings": [],
        "recommendations": [
            "Use the drift onset as a checkpoint, not as an automatic retrain rule.",
        ],
        "metric_directions": result.metric_directions,
        "total_windows": int(len(records)),
        "drift_onset": _json_ready_object(drift_onset),
        "windows_preview": _frame_records(records.head(preview_rows)),
    }


def _profile_column(frame: pd.DataFrame, column) -> dict[str, Any]:
    series = frame[column]
    non_null = series.dropna()
    dtype = str(series.dtype)
    if pd.api.types.is_bool_dtype(series):
        kind = "boolean"
    elif pd.api.types.is_numeric_dtype(series):
        kind = "numeric"
    elif pd.api.types.is_datetime64_any_dtype(series):
        kind = "datetime"
    else:
        kind = "categorical" if non_null.nunique(dropna=True) <= max(20, len(series) * 0.2) else "text"

    profile: dict[str, Any] = {
        "name": str(column),
        "dtype": dtype,
        "kind": kind,
        "non_null": int(non_null.shape[0]),
        "nulls": int(series.isna().sum()),
        "null_ratio": float(series.isna().mean()),
        "unique_count": int(non_null.nunique(dropna=True)),
        "examples": [_json_ready(value) for value in non_null.head(3).tolist()],
    }
    if kind == "numeric" and not non_null.empty:
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if not numeric.empty:
            profile["min"] = _json_ready(numeric.min())
            profile["max"] = _json_ready(numeric.max())
            profile["mean"] = _json_ready(numeric.mean())

    name = str(column).lower()
    should_try_datetime = (
        kind == "datetime"
        or any(token in name for token in ("date", "time", "timestamp", "_at", "dt"))
    )
    if should_try_datetime:
        if kind == "datetime":
            parsed = pd.to_datetime(non_null, errors="coerce")
        else:
            parsed = pd.to_datetime(non_null.astype("string"), errors="coerce", format="mixed")
        parse_ratio = float(parsed.notna().mean()) if len(non_null) else 0.0
        if parse_ratio >= 0.5:
            parsed = parsed.dropna()
            profile["datetime_parse_ratio"] = parse_ratio
            profile["datetime_min"] = _json_ready(parsed.min())
            profile["datetime_max"] = _json_ready(parsed.max())
    return profile


def _time_candidate_from_profile(profile: dict[str, Any]) -> dict[str, Any] | None:
    name = str(profile["name"])
    lower = name.lower()
    parse_ratio = float(profile.get("datetime_parse_ratio", 0.0))
    score = parse_ratio
    if any(token in lower for token in ("date", "time", "timestamp", "_at", "dt")):
        score += 0.35
    if profile["kind"] == "datetime":
        score += 0.4
    if score < 0.7:
        return None
    return {
        "name": name,
        "score": round(score, 3),
        "parse_ratio": round(parse_ratio, 3),
        "min": profile.get("datetime_min"),
        "max": profile.get("datetime_max"),
    }


def _target_candidate_from_profile(profile: dict[str, Any]) -> dict[str, Any] | None:
    name = str(profile["name"])
    lower = name.lower()
    score = 0.0
    if lower in {"target", "y", "label", "class", "outcome"}:
        score += 0.8
    if any(token in lower for token in ("target", "label", "class", "outcome", "delay", "sales")):
        score += 0.45
    if profile["kind"] in {"numeric", "categorical", "boolean"}:
        score += 0.2
    if score < 0.45:
        return None
    return {
        "name": name,
        "score": round(score, 3),
        "kind": profile["kind"],
        "unique_count": profile["unique_count"],
    }


def _duration_string(value: pd.Timedelta) -> str:
    if value <= pd.Timedelta(0):
        value = pd.Timedelta(days=1)
    days = max(1, int(round(value / pd.Timedelta(days=1))))
    return f"{days}D"


def _diagnose_plan_frame(plan_frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    if plan_frame.empty:
        return ["policy produced no folds"], warnings

    for segment in ("train", "validation", "test"):
        rows_col = f"{segment}_rows"
        if rows_col in plan_frame and (plan_frame[rows_col] <= 0).any():
            issues.append(f"{segment} has one or more empty folds")

    if "is_partial" in plan_frame and plan_frame["is_partial"].astype(bool).any():
        warnings.append("plan contains partial folds")

    if {"train_end", "test_start"}.issubset(plan_frame.columns):
        train_end = pd.to_datetime(plan_frame["train_end"], errors="coerce")
        test_start = pd.to_datetime(plan_frame["test_start"], errors="coerce")
        if (test_start < train_end).any():
            issues.append("one or more folds have test_start before train_end")
        if (test_start == train_end).any():
            warnings.append("one or more folds start test at the same timestamp train ends")

    if "train_rows" in plan_frame:
        train_rows = pd.to_numeric(plan_frame["train_rows"], errors="coerce").dropna()
        if not train_rows.empty and train_rows.min() < 10:
            warnings.append("one or more train folds have fewer than 10 rows")
    if "test_rows" in plan_frame:
        test_rows = pd.to_numeric(plan_frame["test_rows"], errors="coerce").dropna()
        if not test_rows.empty and test_rows.min() < 5:
            warnings.append("one or more test folds have fewer than 5 rows")
    return issues, warnings


def _summarize_plan_frame(plan_frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for segment in ("train", "validation", "test"):
        rows_col = f"{segment}_rows"
        if rows_col in plan_frame:
            rows = pd.to_numeric(plan_frame[rows_col], errors="coerce").dropna()
            if not rows.empty:
                summary[f"{segment}_rows_min"] = int(rows.min())
                summary[f"{segment}_rows_max"] = int(rows.max())
                summary[f"{segment}_rows_mean"] = float(rows.mean())
    if "simulation_start" in plan_frame:
        summary["simulation_start"] = _json_ready(
            pd.to_datetime(plan_frame["simulation_start"], errors="coerce").min()
        )
    if "simulation_end" in plan_frame:
        summary["simulation_end"] = _json_ready(
            pd.to_datetime(plan_frame["simulation_end"], errors="coerce").max()
        )
    return summary


def _resolve_dataset_format(path: Path, dataset_format: str) -> str:
    if dataset_format != "auto":
        return dataset_format
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".zip":
        return "zip"
    raise ValueError(
        "Could not infer dataset format from file extension; pass dataset_format explicitly"
    )


def _build_temporal_semantics(
    *,
    time_col: str | int | None,
    order_col: str | int | None,
    train_time_col: str | int | None,
    validation_time_col: str | int | None,
    test_time_col: str | int | None,
) -> str | int | TemporalSemanticsSpec:
    if time_col is None:
        raise ValueError("time_col is required")

    segment_time_cols = {
        name: value
        for name, value in {
            "train": train_time_col,
            "validation": validation_time_col,
            "test": test_time_col,
        }.items()
        if value is not None
    }
    if order_col is None and not segment_time_cols:
        return time_col
    return TemporalSemanticsSpec(
        timeline_col=time_col,
        order_col=order_col,
        segment_time_cols=segment_time_cols,
    )


def _build_simulation(
    *,
    partition: dict[str, Any],
    step: object,
    time_col: str | int | None,
    strategy: str,
    allow_partial: bool,
    engine: str,
    start_at: object | None,
    end_at: object | None,
    max_folds: int | None,
    order_col: str | int | None,
    train_time_col: str | int | None,
    validation_time_col: str | int | None,
    test_time_col: str | int | None,
) -> TemporalSimulation:
    semantics = _build_temporal_semantics(
        time_col=time_col,
        order_col=order_col,
        train_time_col=train_time_col,
        validation_time_col=validation_time_col,
        test_time_col=test_time_col,
    )
    return TemporalSimulation(
        time_col=semantics,
        partition=TemporalPartitionSpec(**partition),
        step=step,
        strategy=strategy,
        allow_partial=allow_partial,
        engine=engine,
        start_at=start_at,
        end_at=end_at,
        max_folds=max_folds,
    )


class _MeanBaselineModel:
    def fit(self, X, y):
        values = pd.to_numeric(pd.Series(y), errors="coerce").dropna()
        if values.empty:
            raise ValueError("mean baseline requires at least one numeric target value")
        self.value_ = float(values.mean())
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


class _MajorityClassBaselineModel:
    def fit(self, X, y):
        values = pd.Series(y).dropna()
        if values.empty:
            raise ValueError("majority_class baseline requires at least one target value")
        self.value_ = values.mode().iloc[0]
        return self

    def predict(self, X):
        return np.asarray([self.value_] * len(X), dtype=object)


def _build_baseline_model(model: str):
    if model == "mean":
        return _MeanBaselineModel()
    if model == "majority_class":
        return _MajorityClassBaselineModel()
    raise ValueError("model must be either 'mean' or 'majority_class'")


def _normalize_retrain_policy_specs(policies: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if policies is None:
        return [
            {"name": "always", "retrain": "always"},
            {"name": "never", "retrain": "never"},
            {"name": "periodic_2", "retrain": "periodic", "retrain_interval": 2},
        ]
    if not policies:
        raise ValueError("policies must not be empty")

    normalized = []
    for index, policy in enumerate(policies):
        if not isinstance(policy, dict):
            raise ValueError("Each policy must be a dictionary")
        retrain = policy.get("retrain", "always")
        if retrain not in {True, False, "always", "never", "periodic", "on_drift"}:
            raise ValueError(
                "policy retrain must be True, False, 'always', 'never', 'periodic' or 'on_drift'"
            )
        if retrain == "periodic" and policy.get("retrain_interval") is None:
            raise ValueError("retrain_interval is required for periodic policies")
        name = policy.get("name") or f"policy_{index}"
        normalized_policy = dict(policy)
        normalized_policy["name"] = name
        normalized.append(normalized_policy)
    return normalized
