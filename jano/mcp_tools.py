from __future__ import annotations

from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd

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
        "columns": [str(column) for column in frame.columns],
        "sample_rows": int(len(frame)),
        "preview": frame.head(sample_rows).to_dict(orient="records"),
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
        JSON-ready dictionary with fold summary, chart data and rendered HTML.
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
        "total_folds": int(result.total_folds),
        "engine": result.engine_metadata.to_dict(),
        "summary_preview": summary_frame.head(preview_rows).to_dict(orient="records"),
        "chart_data": result.chart_data.to_dict(),
        "html": result.html,
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
    metrics: str | list[str] | None = None,
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
        metrics: Metric name or list of metric names accepted by
            ``WalkForwardRunner``.
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


def _frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {str(key): _json_ready(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _json_ready(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value
