from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..evaluation import EvaluationProfile
from .._serialization import _frame_records
from .._workflow_inputs import _resolve_workflow, _resolve_workflow_inputs
from ..policies import MetricSpec, _clone_model, _prepare_supervised_frame
from ..splits import TimeSplit
from ..types import ColumnRef, TemporalPartitionSpec, TemporalSemanticsSpec


@dataclass(frozen=True)
class PredictionBandContext:
    """Context passed to a user-owned prediction-band estimator."""

    fold: int
    split: TimeSplit
    estimator: object
    fitted_estimator: object
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    predictions: np.ndarray
    history: pd.DataFrame


@dataclass(frozen=True)
class PredictionBandScenarioResult:
    """Result of a walk-forward prediction-band scenario."""

    records: pd.DataFrame
    predictions: pd.DataFrame
    artifacts: pd.DataFrame
    metric_directions: dict[str, str]

    def to_frame(self) -> pd.DataFrame:
        """Return one row per fold with metrics and band summary columns."""
        return self.records.copy()

    def predictions_frame(self) -> pd.DataFrame:
        """Return row-level predictions with lower and upper band bounds."""
        return self.predictions.copy()

    def artifacts_frame(self) -> pd.DataFrame:
        """Return fold-level artifacts returned by the user-owned band estimator."""
        return self.artifacts.copy()

    def band_summary(self) -> pd.DataFrame:
        """Return band-specific summary columns for each fold."""
        columns = [
            "fold",
            "prediction_band_width_mean",
            "prediction_band_width_median",
            "prediction_band_coverage",
        ]
        return self.records[[column for column in columns if column in self.records]].copy()

    @property
    def metric_names(self) -> list[str]:
        """Return user-provided metric columns recorded on each fold."""
        return list(self.metric_directions)

    def summary(self) -> dict[str, object]:
        """Return compact aggregate statistics for the scenario."""
        summary: dict[str, object] = {
            "folds": int(len(self.records)),
            "metrics": self.metric_names,
        }
        if "prediction_band_width_mean" in self.records:
            summary["prediction_band_width_mean"] = float(
                self.records["prediction_band_width_mean"].astype(float).mean()
            )
        if "prediction_band_coverage" in self.records:
            summary["prediction_band_coverage_mean"] = float(
                self.records["prediction_band_coverage"].astype(float).mean()
            )
        for metric in self.metric_names:
            values = self.records[metric].astype(float)
            direction = self.metric_directions.get(metric, "min")
            best_index = values.idxmin() if direction == "min" else values.idxmax()
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_best"] = float(values.loc[best_index])
            summary[f"{metric}_best_fold"] = int(self.records.loc[best_index, "fold"])
        return summary

    def report_data(
        self,
        *,
        include_predictions: bool = False,
        include_artifacts: bool = False,
    ) -> dict[str, object]:
        """Return JSON-ready scenario output for agents, notebooks or dashboards."""
        payload: dict[str, object] = {
            "summary": self.summary(),
            "folds": _frame_records(self.to_frame()),
            "bands": _frame_records(self.band_summary()),
            "metric_directions": dict(self.metric_directions),
        }
        if include_predictions:
            payload["predictions"] = _frame_records(self.predictions_frame())
        if include_artifacts:
            payload["artifacts"] = _frame_records(self.artifacts_frame())
        return payload

    def to_dict(
        self,
        *,
        include_predictions: bool = False,
        include_artifacts: bool = False,
    ) -> dict[str, object]:
        """Return a serializable representation of the scenario."""
        return self.report_data(
            include_predictions=include_predictions,
            include_artifacts=include_artifacts,
        )


def estimate_prediction_band_by_fold(
    X,
    *,
    target_col: ColumnRef,
    band_estimator,
    estimator=None,
    feature_cols: Sequence[ColumnRef] | None = None,
    workflow=None,
    time_col: str | int | TemporalSemanticsSpec | None = None,
    partition: TemporalPartitionSpec | None = None,
    train_size=None,
    test_size=None,
    step=None,
    strategy: str = "rolling",
    max_folds: int | None = None,
    metrics: MetricSpec = None,
    metric_directions: Mapping[str, str] | None = None,
    primary_metric: str | None = None,
    prediction_column: str = "prediction",
) -> PredictionBandScenarioResult:
    """Estimate walk-forward prediction bands with a user-owned band estimator.

    Jano owns the temporal simulation geometry. The caller owns the uncertainty
    method. ``band_estimator`` may be a callable or an object exposing
    ``estimate(context)``. It must return a mapping with ``lower`` and ``upper``
    arrays of length ``len(context.X_test)``. Optional ``fold``, ``predictions``
    and ``artifacts`` mappings are merged into the structured result.
    """

    if estimator is None:
        raise ValueError("estimator is required")
    workflow = _resolve_workflow(
        workflow=workflow,
        time_col=time_col,
        partition=partition,
        train_size=train_size,
        test_size=test_size,
        step=step,
        strategy=strategy,
        max_folds=max_folds,
    )
    selected_input, semantics, splits = _resolve_workflow_inputs(workflow, X)
    frame, _, resolved_features, target_name = _prepare_supervised_frame(
        selected_input,
        time_col=semantics,
        target_col=target_col,
        feature_cols=feature_cols,
    )
    evaluation = EvaluationProfile(
        metrics=metrics,
        metric_directions=metric_directions,
        primary_metric=primary_metric,
    ).resolve()

    records: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []

    for split in splits:
        if "train" not in split.segments or "test" not in split.segments:
            raise ValueError("prediction band scenarios require 'train' and 'test' segments")

        train_idx = split.segments["train"]
        test_idx = split.segments["test"]
        train_pos = _as_position_list(train_idx)
        test_pos = _as_position_list(test_idx)

        X_train = frame.iloc[train_pos][resolved_features]
        y_train = frame.iloc[train_pos][target_name]
        X_test = frame.iloc[test_pos][resolved_features]
        y_test = frame.iloc[test_pos][target_name]

        fitted_estimator = _fit_model(estimator, X_train, y_train)
        point_predictions = np.asarray(fitted_estimator.predict(X_test))
        y_true = np.asarray(y_test)
        context = PredictionBandContext(
            fold=split.fold,
            split=split,
            estimator=estimator,
            fitted_estimator=fitted_estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            predictions=point_predictions,
            history=pd.DataFrame(records),
        )
        band_payload = _estimate_band(band_estimator, context)
        lower, upper = _validate_band_payload(band_payload, expected_rows=len(X_test))
        width = upper - lower

        boundaries = split.boundaries
        row: dict[str, object] = {
            "fold": split.fold,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_start": boundaries["train"].start,
            "train_end": boundaries["train"].end,
            "test_start": boundaries["test"].start,
            "test_end": boundaries["test"].end,
            "prediction_band_width_mean": float(np.mean(width)),
            "prediction_band_width_median": float(np.median(width)),
            "prediction_band_coverage": float(np.mean((y_true >= lower) & (y_true <= upper))),
        }
        row.update(_coerce_mapping_values(band_payload.get("fold", {})))
        for name, metric_fn in evaluation.metrics.items():
            row[name] = metric_fn(y_true, point_predictions)
        records.append(row)

        for artifact_name, value in _coerce_mapping_values(band_payload.get("artifacts", {})).items():
            artifact_rows.append({"fold": split.fold, "artifact": artifact_name, "value": value})

        test_frame = frame.iloc[test_pos]
        extra_prediction_columns = _validate_prediction_columns(
            band_payload.get("predictions", {}),
            expected_rows=len(X_test),
        )
        for row_pos, (original_index, truth, prediction, lo, hi, band_width) in enumerate(
            zip(test_frame.index, y_true, point_predictions, lower, upper, width)
        ):
            prediction_row = {
                "fold": split.fold,
                "row_index": original_index,
                "y_true": truth,
                prediction_column: prediction,
                "prediction_lower": lo,
                "prediction_upper": hi,
                "prediction_band_width": band_width,
            }
            for name, values in extra_prediction_columns.items():
                prediction_row[name] = values[row_pos]
            prediction_rows.append(prediction_row)

    if not records:
        raise ValueError("The configured workflow did not produce any valid folds")

    return PredictionBandScenarioResult(
        records=pd.DataFrame(records),
        predictions=pd.DataFrame(prediction_rows),
        artifacts=pd.DataFrame(artifact_rows),
        metric_directions=evaluation.metric_directions,
    )


def _fit_model(estimator, X_train: pd.DataFrame, y_train: pd.Series):
    cloned = _clone_model(estimator)
    fitted = cloned.fit(X_train, y_train)
    return fitted if fitted is not None else cloned


def _estimate_band(band_estimator, context: PredictionBandContext) -> Mapping[str, object]:
    if hasattr(band_estimator, "estimate"):
        payload = band_estimator.estimate(context)
    elif callable(band_estimator):
        payload = band_estimator(context)
    else:
        raise TypeError("band_estimator must be callable or expose estimate(context)")
    if not isinstance(payload, Mapping):
        raise TypeError("band_estimator must return a mapping")
    return payload


def _validate_band_payload(payload: Mapping[str, object], *, expected_rows: int) -> tuple[np.ndarray, np.ndarray]:
    if "lower" not in payload or "upper" not in payload:
        raise ValueError("band_estimator output must include lower and upper")
    lower = _as_vector(payload["lower"], expected_rows=expected_rows, name="lower")
    upper = _as_vector(payload["upper"], expected_rows=expected_rows, name="upper")
    if np.any(lower > upper):
        raise ValueError("band lower values must be less than or equal to upper values")
    return lower, upper


def _validate_prediction_columns(payload, *, expected_rows: int) -> dict[str, np.ndarray]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TypeError("predictions payload must be a mapping")
    return {
        str(name): _as_vector(values, expected_rows=expected_rows, name=str(name))
        for name, values in payload.items()
    }


def _as_vector(values, *, expected_rows: int, name: str) -> np.ndarray:
    vector = np.asarray(values)
    if vector.ndim != 1 or len(vector) != expected_rows:
        raise ValueError(f"{name} must be a one-dimensional array with {expected_rows} rows")
    return vector


def _coerce_mapping_values(payload) -> dict[str, object]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TypeError("fold and artifacts payloads must be mappings")
    return {str(key): value for key, value in payload.items()}


def _as_position_list(indexer) -> list[int]:
    return [int(position) for position in np.asarray(indexer)]
