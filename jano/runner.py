from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .io import coerce_tabular_input
from .policies import (
    MetricFn,
    _clone_model,
    _normalize_metric_mapping,
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


class DriftBasedRetrain(RetrainPolicy):
    """Retrain when previously observed degradation crosses a threshold.

    The decision for fold ``k`` is based on metrics observed through fold ``k-1``.
    """

    def __init__(
        self,
        *,
        metric: str = "rmse",
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
        if self.metric not in context.history.columns:
            raise ValueError(f"Metric '{self.metric}' is not present in runner history")

        latest = float(context.history.iloc[-1][self.metric])
        direction = context.metric_directions.get(self.metric, "min")
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

    def _baseline_value(self, context: RetrainContext) -> float:
        metric_history = context.history[self.metric].astype(float)
        if self.baseline == "first":
            return float(metric_history.iloc[0])
        if self.baseline == "previous_fold":
            return float(metric_history.iloc[-1])
        if self.baseline == "best":
            direction = context.metric_directions.get(self.metric, "min")
            return float(metric_history.min() if direction == "min" else metric_history.max())
        if context.last_retrain_fold is None:
            return float(metric_history.iloc[0])
        baseline_rows = context.history.loc[
            (context.history["fold"].astype("Int64") == context.last_retrain_fold)
            & (context.history["retrained"].astype(bool))
        ]
        if baseline_rows.empty:
            return float(metric_history.iloc[0])
        return float(baseline_rows.iloc[-1][self.metric])


@dataclass(frozen=True)
class WalkForwardRunResult:
    """Materialized execution of a temporal workflow with an estimator."""

    records: pd.DataFrame
    predictions: pd.DataFrame
    metric_directions: dict[str, str]
    retrain_policy: str

    def to_frame(self) -> pd.DataFrame:
        """Return one row per evaluated fold."""
        return self.records.copy()

    def predictions_frame(self) -> pd.DataFrame:
        """Return row-level predictions across all test folds."""
        return self.predictions.copy()

    def summary(self) -> dict[str, object]:
        """Return compact aggregate execution statistics."""
        retrained = self.records["retrained"].astype(bool)
        return {
            "folds": int(len(self.records)),
            "retrain_policy": self.retrain_policy,
            "retrain_events": int(retrained.sum()),
            "retrain_ratio": float(retrained.mean()),
        }


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
        metrics: str | Sequence[str] | Mapping[str, MetricFn] | None = None,
        prediction_column: str = "prediction",
    ) -> None:
        self.model = model
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.metrics = metrics
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

        selected_input, semantics, splits = self._resolve_workflow_inputs(workflow, frame_input)
        frame, _, resolved_features, target_name = _prepare_supervised_frame(
            selected_input,
            time_col=semantics,
            target_col=target_col,
            feature_cols=self.feature_cols,
        )
        metric_mapping, metric_directions = _normalize_metric_mapping(self.metrics)

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

    def _resolve_workflow_inputs(self, workflow, frame_input):
        if isinstance(workflow, WalkForwardPolicy):
            simulation = workflow.simulation
            selected = simulation._select_input(frame_input)
            splits = list(simulation.as_splitter().iter_splits(selected))
            if simulation.max_folds is not None:
                splits = splits[: simulation.max_folds]
            return selected, simulation.as_splitter().temporal_semantics, splits

        if isinstance(workflow, TemporalSimulation):
            selected = workflow._select_input(frame_input)
            splits = list(workflow.as_splitter().iter_splits(selected))
            if workflow.max_folds is not None:
                splits = splits[: workflow.max_folds]
            return selected, workflow.as_splitter().temporal_semantics, splits

        if isinstance(workflow, TemporalBacktestSplitter):
            return frame_input, workflow.temporal_semantics, list(workflow.iter_splits(frame_input))

        if hasattr(workflow, "simulation") and isinstance(workflow.simulation, TemporalSimulation):
            simulation = workflow.simulation
            selected = simulation._select_input(frame_input)
            splits = list(simulation.as_splitter().iter_splits(selected))
            if simulation.max_folds is not None:
                splits = splits[: simulation.max_folds]
            return selected, simulation.as_splitter().temporal_semantics, splits

        if hasattr(workflow, "as_splitter"):
            splitter = workflow.as_splitter()
            return frame_input, splitter.temporal_semantics, list(splitter.iter_splits(frame_input))

        raise TypeError(
            "workflow must be a TemporalBacktestSplitter, WalkForwardPolicy, TemporalSimulation or compatible object"
        )
