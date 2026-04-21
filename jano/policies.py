from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .engines import PartitionEngine
from .io import coerce_tabular_input
from .slicing import TimeIndexer
from .types import ColumnRef, SizeSpec, TemporalSemanticsSpec
from .validation import validate_temporal_semantics

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(_mse(y_true, y_pred)))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


_BUILTIN_METRICS: dict[str, MetricFn] = {
    "mae": _mae,
    "mse": _mse,
    "rmse": _rmse,
    "accuracy": _accuracy,
}

_METRIC_DIRECTIONS: dict[str, str] = {
    "mae": "min",
    "mse": "min",
    "rmse": "min",
    "accuracy": "max",
}


def _coerce_semantics(time_col: str | int | TemporalSemanticsSpec) -> TemporalSemanticsSpec:
    if isinstance(time_col, TemporalSemanticsSpec):
        return validate_temporal_semantics(time_col)
    return validate_temporal_semantics(TemporalSemanticsSpec(timeline_col=time_col))


def _resolve_column(frame: pd.DataFrame, ref: ColumnRef) -> object:
    if isinstance(ref, int):
        return frame.columns[ref]
    return ref


def _resolve_columns(frame: pd.DataFrame, refs: Sequence[ColumnRef]) -> list[object]:
    return [_resolve_column(frame, ref) for ref in refs]


def _normalize_metric_mapping(
    metrics: str | Sequence[str] | Mapping[str, MetricFn] | None,
) -> tuple[dict[str, MetricFn], dict[str, str]]:
    if metrics is None:
        names = ("mae", "rmse")
        return (
            {name: _BUILTIN_METRICS[name] for name in names},
            {name: _METRIC_DIRECTIONS[name] for name in names},
        )

    if isinstance(metrics, str):
        if metrics not in _BUILTIN_METRICS:
            raise ValueError(f"Unknown metric '{metrics}'")
        return {metrics: _BUILTIN_METRICS[metrics]}, {metrics: _METRIC_DIRECTIONS[metrics]}

    if isinstance(metrics, Mapping):
        if not metrics:
            raise ValueError("metrics mapping must not be empty")
        return dict(metrics), {name: _METRIC_DIRECTIONS.get(name, "min") for name in metrics}

    normalized: dict[str, MetricFn] = {}
    directions: dict[str, str] = {}
    for name in metrics:
        if name not in _BUILTIN_METRICS:
            raise ValueError(f"Unknown metric '{name}'")
        normalized[name] = _BUILTIN_METRICS[name]
        directions[name] = _METRIC_DIRECTIONS[name]
    if not normalized:
        raise ValueError("metrics must not be empty")
    return normalized, directions


def _clone_model(model):
    return deepcopy(model)


def _fit_predict(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    estimator = _clone_model(model)
    fitted = estimator.fit(X_train, y_train)
    predictor = fitted if fitted is not None else estimator
    predictions = predictor.predict(X_test)
    return np.asarray(predictions)


def _ensure_duration(name: str, value) -> SizeSpec:
    spec = SizeSpec.from_value(value)
    if spec.kind != "duration":
        raise ValueError(f"{name} must use a duration-based size")
    return spec


def _prepare_supervised_frame(
    X,
    *,
    time_col: str | int | TemporalSemanticsSpec,
    target_col: ColumnRef,
    feature_cols: Sequence[ColumnRef] | None,
) -> tuple[pd.DataFrame, TemporalSemanticsSpec, list[object], object]:
    frame = coerce_tabular_input(X)
    if frame.empty:
        raise ValueError("X must contain at least one row")

    semantics = _coerce_semantics(time_col)
    target_name = _resolve_column(frame, target_col)

    if target_name not in frame.columns:
        raise ValueError(f"target_col '{target_name}' was not found in the dataset")

    if feature_cols is None:
        excluded = {target_name, semantics.timeline_col, semantics.effective_order_col}
        excluded.update(semantics.segment_time_cols.values())
        resolved_features = [column for column in frame.columns if column not in excluded]
    else:
        resolved_features = _resolve_columns(frame, feature_cols)

    if not resolved_features:
        raise ValueError("feature_cols resolved to an empty feature set")

    return frame, semantics, resolved_features, target_name


@dataclass(frozen=True)
class TrainGrowthResult:
    """Evaluated records for a fixed-test, growing-train temporal hypothesis.

    Attributes:
        records: DataFrame with one row per candidate train window and metric columns.
        metric_directions: Mapping from metric name to optimization direction:
            ``"min"`` for lower-is-better metrics and ``"max"`` for higher-is-better
            metrics.
    """

    records: pd.DataFrame
    metric_directions: dict[str, str]

    def to_frame(self) -> pd.DataFrame:
        """Return the evaluated train variants as a pandas DataFrame."""
        return self.records.copy()

    def find_optimal_train_size(
        self,
        metric: str = "rmse",
        tolerance: float = 0.0,
        relative: bool = True,
    ) -> dict[str, object]:
        """Return the smallest train window whose score is within tolerance of the best.

        Args:
            metric: Metric column used to compare train variants.
            tolerance: Allowed distance from the best score.
            relative: Whether tolerance is interpreted proportionally instead of absolutely.
        """
        if metric not in self.records.columns:
            raise ValueError(f"Metric '{metric}' is not present in the result frame")
        if tolerance < 0:
            raise ValueError("tolerance must be greater than or equal to zero")

        direction = self.metric_directions.get(metric, "min")
        series = self.records[metric].astype(float)

        if direction == "min":
            best = float(series.min())
            threshold = best * (1 + tolerance) if relative else best + tolerance
            candidates = self.records.loc[series <= threshold]
            winner = candidates.sort_values(["train_rows", metric, "variant"]).iloc[0]
        else:
            best = float(series.max())
            threshold = best * (1 - tolerance) if relative else best - tolerance
            candidates = self.records.loc[series >= threshold]
            winner = candidates.sort_values(["train_rows", metric, "variant"], ascending=[True, False, True]).iloc[0]
        return winner.to_dict()


@dataclass(frozen=True)
class PerformanceDecayResult:
    """Evaluated records for a fixed-train, moving-test temporal hypothesis.

    Attributes:
        records: DataFrame with one row per moving test window and metric columns.
        metric_directions: Mapping from metric name to optimization direction.
    """

    records: pd.DataFrame
    metric_directions: dict[str, str]

    def to_frame(self) -> pd.DataFrame:
        """Return the evaluated test windows as a pandas DataFrame."""
        return self.records.copy()

    def find_drift_onset(
        self,
        metric: str = "rmse",
        threshold: float = 0.1,
        baseline: str | float = "first",
        relative: bool = True,
    ) -> dict[str, object] | None:
        """Return the first evaluation window where performance becomes problematic.

        Args:
            metric: Metric column used to detect degradation.
            threshold: Allowed degradation from the baseline before the window is flagged.
            baseline: ``"first"``, ``"best"`` or an explicit numeric baseline.
            relative: Whether threshold is interpreted proportionally instead of absolutely.
        """
        if metric not in self.records.columns:
            raise ValueError(f"Metric '{metric}' is not present in the result frame")
        if threshold < 0:
            raise ValueError("threshold must be greater than or equal to zero")

        series = self.records[metric].astype(float)
        direction = self.metric_directions.get(metric, "min")
        if baseline == "first":
            baseline_value = float(series.iloc[0])
        elif baseline == "best":
            baseline_value = float(series.min() if direction == "min" else series.max())
        else:
            baseline_value = float(baseline)

        if direction == "min":
            if relative:
                if baseline_value == 0:
                    mask = series > threshold
                else:
                    mask = (series - baseline_value) / abs(baseline_value) >= threshold
            else:
                mask = series - baseline_value >= threshold
        else:
            if relative:
                if baseline_value == 0:
                    mask = series < -threshold
                else:
                    mask = (baseline_value - series) / abs(baseline_value) >= threshold
            else:
                mask = baseline_value - series >= threshold

        flagged = self.records.loc[mask]
        if flagged.empty:
            return None
        return flagged.iloc[0].to_dict()


class TrainGrowthPolicy:
    """Evaluate whether adding more training history improves a fixed test slice.

    This policy keeps the test window fixed and grows the train window backward in time.
    It is useful when you want to understand how much historical data is actually needed
    to match the best achievable test performance.

    Args:
        time_col: Timeline column name, column position, or ``TemporalSemanticsSpec``.
        cutoff: Boundary where candidate train windows end and the fixed test horizon
            begins after any configured gap.
        train_sizes: Candidate duration windows evaluated backward from ``cutoff``.
        test_size: Duration of the fixed test window.
        gap_before_test: Optional duration gap between train end and test start.
    """

    def __init__(
        self,
        time_col: str | int | TemporalSemanticsSpec,
        *,
        cutoff,
        train_sizes: Sequence[object],
        test_size,
        gap_before_test=None,
    ) -> None:
        if not train_sizes:
            raise ValueError("train_sizes must not be empty")
        self.temporal_semantics = _coerce_semantics(time_col)
        self.cutoff = pd.Timestamp(cutoff)
        self.train_sizes = [_ensure_duration("train_sizes", value) for value in train_sizes]
        self.test_size = _ensure_duration("test_size", test_size)
        self.gap_before_test = (
            _ensure_duration("gap_before_test", gap_before_test)
            if gap_before_test is not None
            else None
        )

    def evaluate(
        self,
        X,
        *,
        model,
        target_col: ColumnRef,
        feature_cols: Sequence[ColumnRef] | None = None,
        metrics: str | Sequence[str] | Mapping[str, MetricFn] | None = None,
    ) -> TrainGrowthResult:
        """Run the fixed-test evaluation over all configured train sizes.

        Args:
            X: Input dataset as ``pandas.DataFrame``, ``numpy.ndarray`` or
                ``polars.DataFrame``.
            model: Estimator with ``fit`` and ``predict`` methods.
            target_col: Target column name or position.
            feature_cols: Optional feature column names or positions. If omitted, all
                non-temporal, non-target columns are used.
            metrics: Metric name, sequence of metric names or mapping of custom metric
                functions.

        Returns:
            A ``TrainGrowthResult`` containing one metric row per candidate train size.
        """
        frame, semantics, resolved_features, target_name = _prepare_supervised_frame(
            X,
            time_col=self.temporal_semantics,
            target_col=target_col,
            feature_cols=feature_cols,
        )
        indexer = TimeIndexer(
            engine=PartitionEngine.from_input(frame),
            semantics=semantics,
        )
        metric_mapping, metric_directions = _normalize_metric_mapping(metrics)

        gap = self.gap_before_test.value if self.gap_before_test is not None else pd.Timedelta(0)
        test_start = self.cutoff + gap
        test_end = test_start + self.test_size.value
        test_idx = indexer.slice_between_for_segment("test", test_start, test_end)
        if len(test_idx) == 0:
            raise ValueError("The configured test window did not produce any rows")

        X_test = frame.iloc[test_idx][resolved_features]
        y_test = frame.iloc[test_idx][target_name]
        records: list[dict[str, object]] = []

        for fold, train_size in enumerate(self.train_sizes):
            train_start = self.cutoff - train_size.value
            train_end = self.cutoff
            train_idx = indexer.slice_between_for_segment("train", train_start, train_end)
            if len(train_idx) == 0:
                continue

            X_train = frame.iloc[train_idx][resolved_features]
            y_train = frame.iloc[train_idx][target_name]
            predictions = _fit_predict(model, X_train, y_train, X_test)
            row = {
                "variant": fold,
                "train_size": str(train_size.value),
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
            }
            y_true = np.asarray(y_test)
            for name, metric_fn in metric_mapping.items():
                row[name] = metric_fn(y_true, predictions)
            records.append(row)

        if not records:
            raise ValueError("No valid train windows were produced for the configured policy")

        return TrainGrowthResult(records=pd.DataFrame(records), metric_directions=metric_directions)

    def find_optimal_train_size(self, X, *, model, target_col: ColumnRef, feature_cols: Sequence[ColumnRef] | None = None, metrics: str | Sequence[str] | Mapping[str, MetricFn] | None = None, metric: str = "rmse", tolerance: float = 0.0, relative: bool = True) -> dict[str, object]:
        """Return the smallest train size that stays within tolerance of the best score.

        Args:
            X: Input dataset.
            model: Estimator with ``fit`` and ``predict`` methods.
            target_col: Target column name or position.
            feature_cols: Optional feature column names or positions.
            metrics: Metric name, sequence of metric names or custom metric mapping.
            metric: Metric column used to choose the optimal train size.
            tolerance: Allowed distance from the best score.
            relative: Whether ``tolerance`` is proportional instead of absolute.
        """
        result = self.evaluate(
            X,
            model=model,
            target_col=target_col,
            feature_cols=feature_cols,
            metrics=metrics,
        )
        return result.find_optimal_train_size(metric=metric, tolerance=tolerance, relative=relative)


class PerformanceDecayPolicy:
    """Evaluate how long a fixed train window stays useful as test moves forward.

    This policy keeps train fixed and repeatedly shifts the test window into the future.
    It is useful when you want to estimate when performance decay or drift becomes
    operationally relevant without retraining the model at every step.

    Args:
        time_col: Timeline column name, column position, or ``TemporalSemanticsSpec``.
        cutoff: Boundary where the fixed train window ends.
        train_size: Duration of the fixed train window looking backward from ``cutoff``.
        test_size: Duration of each test window.
        step: Duration by which the test window advances.
        gap_before_test: Optional duration gap between train end and first test start.
        max_windows: Optional maximum number of test windows to evaluate.
    """

    def __init__(
        self,
        time_col: str | int | TemporalSemanticsSpec,
        *,
        cutoff,
        train_size,
        test_size,
        step,
        gap_before_test=None,
        max_windows: int | None = None,
    ) -> None:
        self.temporal_semantics = _coerce_semantics(time_col)
        self.cutoff = pd.Timestamp(cutoff)
        self.train_size = _ensure_duration("train_size", train_size)
        self.test_size = _ensure_duration("test_size", test_size)
        self.step = _ensure_duration("step", step)
        self.gap_before_test = (
            _ensure_duration("gap_before_test", gap_before_test)
            if gap_before_test is not None
            else None
        )
        if max_windows is not None and max_windows <= 0:
            raise ValueError("max_windows must be greater than zero")
        self.max_windows = max_windows

    def evaluate(
        self,
        X,
        *,
        model,
        target_col: ColumnRef,
        feature_cols: Sequence[ColumnRef] | None = None,
        metrics: str | Sequence[str] | Mapping[str, MetricFn] | None = None,
    ) -> PerformanceDecayResult:
        """Run the fixed-train evaluation over moving test windows.

        Args:
            X: Input dataset as ``pandas.DataFrame``, ``numpy.ndarray`` or
                ``polars.DataFrame``.
            model: Estimator with ``fit`` and ``predict`` methods.
            target_col: Target column name or position.
            feature_cols: Optional feature column names or positions. If omitted, all
                non-temporal, non-target columns are used.
            metrics: Metric name, sequence of metric names or mapping of custom metric
                functions.

        Returns:
            A ``PerformanceDecayResult`` containing one metric row per test window.
        """
        frame, semantics, resolved_features, target_name = _prepare_supervised_frame(
            X,
            time_col=self.temporal_semantics,
            target_col=target_col,
            feature_cols=feature_cols,
        )
        indexer = TimeIndexer(
            engine=PartitionEngine.from_input(frame),
            semantics=semantics,
        )
        metric_mapping, metric_directions = _normalize_metric_mapping(metrics)

        train_start = self.cutoff - self.train_size.value
        train_end = self.cutoff
        train_idx = indexer.slice_between_for_segment("train", train_start, train_end)
        if len(train_idx) == 0:
            raise ValueError("The configured train window did not produce any rows")

        X_train = frame.iloc[train_idx][resolved_features]
        y_train = frame.iloc[train_idx][target_name]
        gap = self.gap_before_test.value if self.gap_before_test is not None else pd.Timedelta(0)
        initial_test_start = self.cutoff + gap
        records: list[dict[str, object]] = []

        window = 0
        while True:
            if self.max_windows is not None and window >= self.max_windows:
                break

            test_start = initial_test_start + (window * self.step.value)
            test_end = test_start + self.test_size.value
            test_idx = indexer.slice_between_for_segment("test", test_start, test_end)
            if len(test_idx) == 0:
                break

            X_test = frame.iloc[test_idx][resolved_features]
            y_test = frame.iloc[test_idx][target_name]
            predictions = _fit_predict(model, X_train, y_train, X_test)
            row = {
                "window": window,
                "train_size": str(self.train_size.value),
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
            }
            y_true = np.asarray(y_test)
            for name, metric_fn in metric_mapping.items():
                row[name] = metric_fn(y_true, predictions)
            records.append(row)
            window += 1

        if not records:
            raise ValueError("No valid test windows were produced for the configured policy")

        return PerformanceDecayResult(
            records=pd.DataFrame(records),
            metric_directions=metric_directions,
        )

    def find_drift_onset(self, X, *, model, target_col: ColumnRef, feature_cols: Sequence[ColumnRef] | None = None, metrics: str | Sequence[str] | Mapping[str, MetricFn] | None = None, metric: str = "rmse", threshold: float = 0.1, baseline: str | float = "first", relative: bool = True) -> dict[str, object] | None:
        """Return the first test window whose metric crosses the degradation threshold.

        Args:
            X: Input dataset.
            model: Estimator with ``fit`` and ``predict`` methods.
            target_col: Target column name or position.
            feature_cols: Optional feature column names or positions.
            metrics: Metric name, sequence of metric names or custom metric mapping.
            metric: Metric column used to detect degradation.
            threshold: Allowed degradation before a window is flagged.
            baseline: ``"first"``, ``"best"`` or an explicit numeric baseline.
            relative: Whether ``threshold`` is proportional instead of absolute.
        """
        result = self.evaluate(
            X,
            model=model,
            target_col=target_col,
            feature_cols=feature_cols,
            metrics=metrics,
        )
        return result.find_drift_onset(
            metric=metric,
            threshold=threshold,
            baseline=baseline,
            relative=relative,
        )
