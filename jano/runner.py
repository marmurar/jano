from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .io import coerce_tabular_input
from .evaluation import EvaluationProfile
from ._serialization import _frame_records
from ._workflow_inputs import _resolve_workflow_inputs
from .policies import (
    MetricFn,
    _clone_model,
    _prepare_supervised_frame,
)
from .simulation import TemporalSimulation
from .splitters import TemporalBacktestSplitter
from .splits import TimeSplit
from .workflows import WalkForwardPolicy


@dataclass(frozen=True)
class RetrainContext:
    """Context available to retrain policies before each fold is executed."""

    fold: int
    split: TimeSplit
    history: pd.DataFrame
    metric_directions: Mapping[str, str]
    last_retrain_fold: int | None
    primary_metric: str | None = None


class RetrainPolicy:
    """Base interface for deciding whether a runner should retrain."""

    def should_retrain(self, context: RetrainContext) -> bool:
        raise NotImplementedError


class AlwaysRetrain(RetrainPolicy):
    """Retrain before every fold."""

    def should_retrain(self, context: RetrainContext) -> bool:
        return True


class NeverRetrain(RetrainPolicy):
    """Train once and reuse the fitted model across all folds."""

    def should_retrain(self, context: RetrainContext) -> bool:
        return False


class PeriodicRetrain(RetrainPolicy):
    """Retrain every ``every`` folds after the previous retrain."""

    def __init__(self, every: int) -> None:
        if every <= 0:
            raise ValueError("every must be greater than zero")
        self.every = every

    def should_retrain(self, context: RetrainContext) -> bool:
        if context.last_retrain_fold is None:
            return True
        return (context.fold - context.last_retrain_fold) >= self.every


class FunctionRetrainPolicy(RetrainPolicy):
    """Delegate retraining decisions to a user-provided callable."""

    def __init__(self, rule: Callable[[RetrainContext], bool]) -> None:
        self.rule = rule

    def should_retrain(self, context: RetrainContext) -> bool:
        return bool(self.rule(context))


class DriftBasedRetrain(RetrainPolicy):
    """Retrain when previously observed degradation crosses a threshold.

    The decision for fold ``k`` is based on metrics observed through fold ``k-1``.
    """

    def __init__(
        self,
        *,
        metric: str | None = None,
        threshold: float = 0.05,
        baseline: str = "last_retrain",
        relative: bool = True,
    ) -> None:
        if threshold < 0:
            raise ValueError("threshold must be greater than or equal to zero")
        if baseline not in {"last_retrain", "first", "best", "previous_fold"}:
            raise ValueError(
                "baseline must be one of 'last_retrain', 'first', 'best' or 'previous_fold'"
            )
        self.metric = metric
        self.threshold = threshold
        self.baseline = baseline
        self.relative = relative

    def should_retrain(self, context: RetrainContext) -> bool:
        if context.history.empty:
            return True
        metric = self._metric_name(context)
        if metric not in context.history.columns:
            raise ValueError(f"Metric '{metric}' is not present in runner history")

        latest = float(context.history.iloc[-1][metric])
        direction = context.metric_directions.get(metric, "min")
        baseline_value = self._baseline_value(context)

        if direction == "min":
            if self.relative:
                if baseline_value == 0:
                    return latest > self.threshold
                return (latest - baseline_value) / abs(baseline_value) >= self.threshold
            return latest - baseline_value >= self.threshold

        if self.relative:
            if baseline_value == 0:
                return latest < -self.threshold
            return (baseline_value - latest) / abs(baseline_value) >= self.threshold
        return baseline_value - latest >= self.threshold

    def _metric_name(self, context: RetrainContext) -> str:
        if self.metric is not None:
            return self.metric
        if context.primary_metric is None:
            raise ValueError(
                "DriftBasedRetrain requires metric or an evaluation primary_metric"
            )
        return context.primary_metric

    def _baseline_value(self, context: RetrainContext) -> float:
        metric = self._metric_name(context)
        metric_history = context.history[metric].astype(float)
        if self.baseline == "first":
            return float(metric_history.iloc[0])
        if self.baseline == "previous_fold":
            return float(metric_history.iloc[-1])
        if self.baseline == "best":
            direction = context.metric_directions.get(metric, "min")
            return float(metric_history.min() if direction == "min" else metric_history.max())
        if context.last_retrain_fold is None:
            return float(metric_history.iloc[0])
        baseline_rows = context.history.loc[
            (context.history["fold"].astype("Int64") == context.last_retrain_fold)
            & (context.history["retrained"].astype(bool))
        ]
        if baseline_rows.empty:
            return float(metric_history.iloc[0])
        return float(baseline_rows.iloc[-1][metric])


@dataclass(frozen=True)
class WalkForwardRunResult:
    """Materialized execution of a temporal workflow with an estimator."""

    records: pd.DataFrame
    predictions: pd.DataFrame
    metric_directions: dict[str, str]
    retrain_policy: str
    primary_metric: str | None = None

    def to_frame(self) -> pd.DataFrame:
        """Return one row per evaluated fold."""
        return self.records.copy()

    def predictions_frame(self) -> pd.DataFrame:
        """Return row-level predictions across all test folds."""
        return self.predictions.copy()

    @property
    def metric_names(self) -> list[str]:
        """Return metric columns recorded for each evaluated fold."""
        metadata_columns = {
            "fold",
            "retrained",
            "last_retrain_fold",
            "train_rows",
            "test_rows",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
        }
        return [column for column in self.records.columns if column not in metadata_columns]

    def fold_summary(self) -> pd.DataFrame:
        """Return fold geometry and retraining metadata without metric columns."""
        columns = [
            "fold",
            "retrained",
            "last_retrain_fold",
            "train_rows",
            "test_rows",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
        ]
        return self.records[[column for column in columns if column in self.records.columns]].copy()

    def metric_trajectory(self) -> pd.DataFrame:
        """Return metrics in long format, one row per fold and metric."""
        metric_names = self.metric_names
        if not metric_names:
            return pd.DataFrame(columns=["fold", "metric", "value", "direction", "retrained"])

        trajectory = self.records.melt(
            id_vars=["fold", "retrained"],
            value_vars=metric_names,
            var_name="metric",
            value_name="value",
        )
        trajectory["direction"] = trajectory["metric"].map(
            lambda metric: self.metric_directions.get(metric, "min")
        )
        return trajectory[["fold", "metric", "value", "direction", "retrained"]]

    def retrain_events(self) -> pd.DataFrame:
        """Return the subset of folds where the estimator was retrained."""
        return self.fold_summary().loc[lambda frame: frame["retrained"].astype(bool)].reset_index(
            drop=True
        )

    def summary(self) -> dict[str, object]:
        """Return compact aggregate execution statistics."""
        retrained = self.records["retrained"].astype(bool)
        summary = {
            "folds": int(len(self.records)),
            "retrain_policy": self.retrain_policy,
            "retrain_events": int(retrained.sum()),
            "retrain_ratio": float(retrained.mean()),
            "metrics": self.metric_names,
            "primary_metric": self.primary_metric,
        }
        for metric in self.metric_names:
            values = self.records[metric].astype(float)
            direction = self.metric_directions.get(metric, "min")
            best_index = values.idxmin() if direction == "min" else values.idxmax()
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_best"] = float(values.loc[best_index])
            summary[f"{metric}_best_fold"] = int(self.records.loc[best_index, "fold"])
        return summary

    def report_data(self, *, include_predictions: bool = False) -> dict[str, object]:
        """Return JSON-ready execution data for notebooks, agents or custom reports."""
        payload: dict[str, object] = {
            "summary": self.summary(),
            "folds": _frame_records(self.fold_summary()),
            "metrics": _frame_records(self.metric_trajectory()),
            "retraining": {
                "policy": self.retrain_policy,
                "events": _frame_records(self.retrain_events()),
            },
            "metric_directions": dict(self.metric_directions),
            "primary_metric": self.primary_metric,
        }
        if include_predictions:
            payload["predictions"] = _frame_records(self.predictions_frame())
        return payload

    def to_dict(self, *, include_predictions: bool = False) -> dict[str, object]:
        """Return a serializable representation of the execution result."""
        return self.report_data(include_predictions=include_predictions)


class WalkForwardRunner:
    """Run an estimator over temporal folds while applying a retrain policy."""

    def __init__(
        self,
        *,
        model,
        target_col=None,
        feature_cols: Sequence[object] | None = None,
        retrain: bool | str = True,
        retrain_interval: int | None = None,
        retrain_policy: RetrainPolicy | None = None,
        metrics: MetricSpec = None,
        metric_directions: Mapping[str, str] | None = None,
        primary_metric: str | None = None,
        evaluation: EvaluationProfile | None = None,
        prediction_column: str = "prediction",
    ) -> None:
        if evaluation is not None and (
            metrics is not None or metric_directions is not None or primary_metric is not None
        ):
            raise ValueError(
                "evaluation cannot be combined with metrics, metric_directions or primary_metric"
            )
        self.model = model
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.evaluation = evaluation or EvaluationProfile(
            metrics=metrics,
            metric_directions=metric_directions,
            primary_metric=primary_metric,
        )
        self.metrics = self.evaluation.metrics
        self.prediction_column = prediction_column
        self.retrain_policy = self._normalize_policy(
            retrain=retrain,
            retrain_interval=retrain_interval,
            retrain_policy=retrain_policy,
        )

    def run(self, workflow, X, y=None) -> WalkForwardRunResult:
        """Execute the configured estimator over a temporal workflow."""
        target_col = self.target_col
        frame_input = X
        if y is not None:
            frame_input = coerce_tabular_input(X).copy()
            y_values = np.asarray(y)
            if len(frame_input) != len(y_values):
                raise ValueError("X and y must contain the same number of rows")
            frame_input["__target__"] = y_values
            target_col = "__target__"
        if target_col is None:
            raise ValueError("target_col is required when y is not provided")

        selected_input, semantics, splits = _resolve_workflow_inputs(workflow, frame_input)
        frame, _, resolved_features, target_name = _prepare_supervised_frame(
            selected_input,
            time_col=semantics,
            target_col=target_col,
            feature_cols=self.feature_cols,
        )
        evaluation = self.evaluation.resolve()
        metric_mapping = evaluation.metrics
        metric_directions = evaluation.metric_directions

        current_model = None
        last_retrain_fold: int | None = None
        records: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []

        for split in splits:
            if "train" not in split.segments or "test" not in split.segments:
                raise ValueError("WalkForwardRunner requires folds with 'train' and 'test' segments")

            train_idx = split.segments["train"]
            test_idx = split.segments["test"]
            X_train = frame.iloc[train_idx][resolved_features]
            y_train = frame.iloc[train_idx][target_name]
            X_test = frame.iloc[test_idx][resolved_features]
            y_test = frame.iloc[test_idx][target_name]

            context = RetrainContext(
                fold=split.fold,
                split=split,
                history=pd.DataFrame(records),
                metric_directions=metric_directions,
                last_retrain_fold=last_retrain_fold,
                primary_metric=evaluation.primary_metric,
            )
            should_retrain = current_model is None or self.retrain_policy.should_retrain(context)
            if should_retrain:
                estimator = _clone_model(self.model)
                fitted = estimator.fit(X_train, y_train)
                current_model = fitted if fitted is not None else estimator
                last_retrain_fold = split.fold

            predictions = np.asarray(current_model.predict(X_test))
            boundaries = split.boundaries
            row = {
                "fold": split.fold,
                "retrained": bool(should_retrain),
                "last_retrain_fold": last_retrain_fold,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_start": boundaries["train"].start,
                "train_end": boundaries["train"].end,
                "test_start": boundaries["test"].start,
                "test_end": boundaries["test"].end,
            }
            y_true = np.asarray(y_test)
            for name, metric_fn in metric_mapping.items():
                row[name] = metric_fn(y_true, predictions)
            records.append(row)

            test_frame = frame.iloc[test_idx]
            for original_index, truth, prediction in zip(test_frame.index, y_true, predictions):
                prediction_rows.append(
                    {
                        "fold": split.fold,
                        "row_index": original_index,
                        "retrained": bool(should_retrain),
                        "y_true": truth,
                        self.prediction_column: prediction,
                    }
                )

        if not records:
            raise ValueError("The configured workflow did not produce any valid folds")

        return WalkForwardRunResult(
            records=pd.DataFrame(records),
            predictions=pd.DataFrame(prediction_rows),
            metric_directions=metric_directions,
            retrain_policy=type(self.retrain_policy).__name__,
            primary_metric=evaluation.primary_metric,
        )

    def _normalize_policy(
        self,
        *,
        retrain: bool | str,
        retrain_interval: int | None,
        retrain_policy: RetrainPolicy | None,
    ) -> RetrainPolicy:
        if retrain_policy is not None:
            if retrain_interval is not None:
                raise ValueError("retrain_interval cannot be used together with retrain_policy")
            return retrain_policy

        if isinstance(retrain, bool):
            if not retrain:
                return NeverRetrain()
            if retrain_interval is None or retrain_interval == 1:
                return AlwaysRetrain()
            return PeriodicRetrain(retrain_interval)

        if retrain not in {"always", "never", "periodic"}:
            raise ValueError("retrain must be True, False, 'always', 'never' or 'periodic'")
        if retrain == "always":
            return AlwaysRetrain()
        if retrain == "never":
            return NeverRetrain()
        if retrain_interval is None:
            raise ValueError("retrain_interval is required when retrain='periodic'")
        return PeriodicRetrain(retrain_interval)
