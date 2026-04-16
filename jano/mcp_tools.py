from __future__ import annotations

from pathlib import Path
from typing import Any
import zipfile

import pandas as pd

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
        "summary_preview": summary_frame.head(preview_rows).to_dict(orient="records"),
        "chart_data": result.chart_data.to_dict(),
        "html": result.html,
    }


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
        start_at=start_at,
        end_at=end_at,
        max_folds=max_folds,
    )
