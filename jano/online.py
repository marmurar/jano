from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .evaluation import EvaluationProfile
from ._serialization import _frame_records, _json_ready
from .policies import MetricFn, MetricSpec, _clone_model, _prepare_supervised_frame
from .types import ColumnRef, SizeSpec, TemporalSemanticsSpec


class OnlineUpdateStrategy:
    """Base interface for updating a model during online temporal evaluation."""

    name = "OnlineUpdateStrategy"

    def initialize(self, model, X_initial: pd.DataFrame, y_initial: pd.Series):
        """Fit the initial model state before the first prediction batch."""
        raise NotImplementedError

    def update(self, model, X_batch: pd.DataFrame, y_batch: pd.Series):
        """Update the model after a prediction batch has been observed."""
        raise NotImplementedError


UpdateStrategyFactory = Callable[[], OnlineUpdateStrategy]
OnlineRetrainTrigger = Callable[[pd.DataFrame, dict[str, object]], object]


class PartialFitUpdateStrategy(OnlineUpdateStrategy):
    """Update models that implement scikit-learn-style ``partial_fit``.

    Args:
        classes: Optional class labels passed to the first ``partial_fit`` call for
            classifiers that require the full label set up front.
    """

    name = "PartialFitUpdateStrategy"

    def __init__(self, classes: Sequence[object] | None = None) -> None:
        self.classes = None if classes is None else np.asarray(list(classes))
        self._initialized = False

    def initialize(self, model, X_initial: pd.DataFrame, y_initial: pd.Series):
        """Call ``partial_fit`` on the initial train window."""
        self._initialized = False
        return self._partial_fit(model, X_initial, y_initial)

    def update(self, model, X_batch: pd.DataFrame, y_batch: pd.Series):
        """Call ``partial_fit`` after predictions have been scored."""
        return self._partial_fit(model, X_batch, y_batch)

    def _partial_fit(self, model, X, y):
        if not hasattr(model, "partial_fit"):
            raise TypeError("PartialFitUpdateStrategy requires a model with partial_fit")
        kwargs = {}
        if self.classes is not None and not self._initialized:
            kwargs["classes"] = self.classes
        fitted = model.partial_fit(X, y, **kwargs)
        self._initialized = True
        return fitted if fitted is not None else model


class RefitUpdateStrategy(OnlineUpdateStrategy):
    """Update any fit/predict estimator by refitting on observed history.

    This strategy is slower than ``PartialFitUpdateStrategy`` but works with
    standard estimators that only implement ``fit``. After each prediction batch
    is observed, the batch is appended to the internal history and the estimator
    is fitted again.

    Args:
        max_train_rows: Optional rolling cap for the number of most recent
            observed rows used on each refit. When omitted, history expands.
    """

    name = "RefitUpdateStrategy"

    def __init__(self, max_train_rows: int | None = None) -> None:
        if max_train_rows is not None and max_train_rows <= 0:
            raise ValueError("max_train_rows must be greater than zero")
        self.max_train_rows = max_train_rows
        self._X_history: pd.DataFrame | None = None
        self._y_history: pd.Series | None = None

    def initialize(self, model, X_initial: pd.DataFrame, y_initial: pd.Series):
        """Fit the model on the initial train window."""
        self._X_history = X_initial.copy()
        self._y_history = y_initial.copy()
        return self._fit(model)

    def update(self, model, X_batch: pd.DataFrame, y_batch: pd.Series):
        """Append the observed batch and refit the model on retained history."""
        if self._X_history is None or self._y_history is None:
            raise RuntimeError("RefitUpdateStrategy must be initialized before update")
        self._X_history = pd.concat([self._X_history, X_batch], axis=0)
        self._y_history = pd.concat([self._y_history, y_batch], axis=0)
        return self._fit(model)

    def _fit(self, model):
        if not hasattr(model, "fit"):
            raise TypeError("RefitUpdateStrategy requires a model with fit")
        if self._X_history is None or self._y_history is None:
            raise RuntimeError("RefitUpdateStrategy must be initialized before fitting")
        X_train = self._X_history
        y_train = self._y_history
        if self.max_train_rows is not None:
            X_train = X_train.tail(self.max_train_rows)
            y_train = y_train.tail(self.max_train_rows)
            self._X_history = X_train
            self._y_history = y_train
        fitted = model.fit(X_train, y_train)
        return fitted if fitted is not None else model


@dataclass(frozen=True)
class OnlineUpdatePolicy:
    """Candidate online update policy evaluated by ``OnlineUpdatePolicyStudy``.

    Args:
        name: Stable label used in result frames.
        update_size: Event, row-batch, duration or fraction cadence passed to
            ``OnlineTemporalRunner``.
        update_strategy: Strategy instance or factory. When omitted, the runner
            defaults to ``PartialFitUpdateStrategy``.
        update_cost: Relative cost per update. Use this to compare predictive
            quality against operational cost.
    """

    name: str
    update_size: object
    update_strategy: OnlineUpdateStrategy | UpdateStrategyFactory | None = None
    update_cost: float = 1.0

    def build_strategy(self) -> OnlineUpdateStrategy | None:
        """Return a fresh update strategy for one candidate run."""
        if self.update_strategy is None:
            return None
        if isinstance(self.update_strategy, OnlineUpdateStrategy):
            return deepcopy(self.update_strategy)
        strategy = self.update_strategy()
        if not isinstance(strategy, OnlineUpdateStrategy):
            raise TypeError("update_strategy factory must return an OnlineUpdateStrategy")
        return strategy


@dataclass(frozen=True)
class OnlineRunResult:
    """Materialized online temporal evaluation result.

    Attributes:
        records: One row per evaluated event or micro-batch.
        predictions: Optional row-level predictions for all evaluated rows.
        metric_directions: Mapping from metric name to ``"min"`` or ``"max"``.
        update_strategy: Name of the strategy used to update the model.
        primary_metric: Primary metric used by downstream analysis.
    """

    records: pd.DataFrame
    predictions: pd.DataFrame
    metric_directions: dict[str, str]
    update_strategy: str
    primary_metric: str | None = None

    def to_frame(self) -> pd.DataFrame:
        """Return one row per evaluated event or micro-batch."""
        return self.records.copy()

    def predictions_frame(self) -> pd.DataFrame:
        """Return row-level predictions across all online evaluation batches."""
        return self.predictions.copy()

    @property
    def metric_names(self) -> list[str]:
        metadata_columns = {
            "batch",
            "updated",
            "retrain_checkpoint",
            "retrain_reason",
            "train_rows_seen",
            "batch_rows",
            "batch_start",
            "batch_end",
        }
        return [
            column
            for column in self.records.columns
            if column not in metadata_columns and not column.startswith("retrain_")
        ]

    def metric_trajectory(self) -> pd.DataFrame:
        """Return metrics in long format, one row per batch and metric."""
        if not self.metric_names:
            return pd.DataFrame(columns=["batch", "metric", "value", "direction", "updated"])
        trajectory = self.records.melt(
            id_vars=["batch", "updated"],
            value_vars=self.metric_names,
            var_name="metric",
            value_name="value",
        )
        trajectory["direction"] = trajectory["metric"].map(
            lambda metric: self.metric_directions.get(metric, "min")
        )
        return trajectory[["batch", "metric", "value", "direction", "updated"]]

    def summary(self) -> dict[str, object]:
        """Return compact aggregate statistics for the online run."""
        checkpoint_count = (
            int(self.records["retrain_checkpoint"].astype(bool).sum())
            if "retrain_checkpoint" in self.records
            else 0
        )
        summary: dict[str, object] = {
            "batches": int(len(self.records)),
            "update_strategy": self.update_strategy,
            "updates": int(self.records["updated"].astype(bool).sum()),
            "retrain_checkpoints": checkpoint_count,
            "rows_evaluated": int(self.records["batch_rows"].sum()),
            "metrics": self.metric_names,
            "primary_metric": self.primary_metric,
        }
        if checkpoint_count:
            first_checkpoint = self.retrain_checkpoints().iloc[0]
            summary["first_retrain_checkpoint_batch"] = int(first_checkpoint["batch"])
            summary["first_retrain_checkpoint_time"] = _json_ready(first_checkpoint["batch_end"])
        for metric in self.metric_names:
            values = self.records[metric].astype(float)
            direction = self.metric_directions.get(metric, "min")
            best_index = values.idxmin() if direction == "min" else values.idxmax()
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_best"] = float(values.loc[best_index])
            summary[f"{metric}_best_batch"] = int(self.records.loc[best_index, "batch"])
        return summary

    def report_data(self, *, include_predictions: bool = False) -> dict[str, object]:
        """Return JSON-ready data for notebooks, agents and custom reports."""
        payload: dict[str, object] = {
            "summary": self.summary(),
            "batches": _frame_records(self.to_frame()),
            "metrics": _frame_records(self.metric_trajectory()),
            "retrain_checkpoints": _frame_records(self.retrain_checkpoints()),
            "metric_directions": dict(self.metric_directions),
            "primary_metric": self.primary_metric,
            "update_strategy": self.update_strategy,
        }
        if include_predictions:
            payload["predictions"] = _frame_records(self.predictions_frame())
        return payload

    def to_dict(self, *, include_predictions: bool = False) -> dict[str, object]:
        """Return a serializable representation of the online run."""
        return self.report_data(include_predictions=include_predictions)

    def retrain_checkpoints(self) -> pd.DataFrame:
        """Return batches where the user-defined online retrain trigger fired."""
        if "retrain_checkpoint" not in self.records:
            return pd.DataFrame(columns=self.records.columns)
        return self.records.loc[self.records["retrain_checkpoint"].astype(bool)].copy()


@dataclass(frozen=True)
class OnlineUpdatePolicyStudyResult:
    """Comparison result for multiple online update policies."""

    records: pd.DataFrame
    runs: dict[str, OnlineRunResult]
    metric_directions: dict[str, str]
    primary_metric: str | None = None

    def to_frame(self) -> pd.DataFrame:
        """Return one row per evaluated online update policy."""
        return self.records.copy()

    def run(self, policy: str) -> OnlineRunResult:
        """Return the detailed run for a named policy."""
        try:
            return self.runs[policy]
        except KeyError as exc:
            raise ValueError(f"Unknown online update policy '{policy}'") from exc

    def metric_trajectory(self) -> pd.DataFrame:
        """Return long-format metric trajectories for all policies."""
        frames = []
        for name, run in self.runs.items():
            frame = run.metric_trajectory()
            if not frame.empty:
                frame.insert(0, "policy", name)
                frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=["policy", "batch", "metric", "value", "direction", "updated"])
        return pd.concat(frames, ignore_index=True)

    def find_optimal_policy(
        self,
        metric: str | None = None,
        *,
        update_cost_weight: float = 0.0,
    ) -> dict[str, object]:
        """Return the best policy after optional update-cost penalization.

        Args:
            metric: Metric column to optimize. Defaults to ``primary_metric``.
            update_cost_weight: Penalty applied to ``total_update_cost``. For
                lower-is-better metrics the penalty is added; for higher-is-better
                metrics it is subtracted.
        """
        metric = metric or self.primary_metric
        if metric is None:
            raise ValueError("metric must be provided when no primary_metric is available")
        if metric not in self.records.columns:
            raise ValueError(f"Metric '{metric}' is not present in the result frame")
        if update_cost_weight < 0:
            raise ValueError("update_cost_weight must be greater than or equal to zero")

        records = self.records.copy()
        direction = self.metric_directions.get(metric, "min")
        penalty = records["total_update_cost"].astype(float) * update_cost_weight
        if direction == "min":
            records["_objective"] = records[metric].astype(float) + penalty
            winner = records.sort_values(["_objective", "updates", "policy"]).iloc[0]
        else:
            records["_objective"] = records[metric].astype(float) - penalty
            winner = records.sort_values(
                ["_objective", "updates", "policy"],
                ascending=[False, True, True],
            ).iloc[0]
        result = winner.drop(labels=["_objective"]).to_dict()
        result["objective"] = float(winner["_objective"])
        result["objective_metric"] = metric
        result["objective_direction"] = direction
        result["update_cost_weight"] = float(update_cost_weight)
        return result


class OnlineTemporalRunner:
    """Run prequential online temporal evaluation over events or micro-batches.

    The runner first initializes a model on an initial train window. It then repeats
    the production-like sequence ``predict -> observe target -> update model`` for
    each future event or micro-batch.

    Args:
        model: Estimator implementing ``predict`` plus the method required by
            ``update_strategy``. Use ``PartialFitUpdateStrategy`` for incremental
            estimators or ``RefitUpdateStrategy`` for standard ``fit`` estimators.
        time_col: Timeline column used to order events.
        target_col: Target column name or position.
        initial_train_size: Initial history used before the first prediction. Supports
            duration strings, integer row counts and fractions.
        update_size: Event or micro-batch size. Use ``1`` for event-level updates, an
            integer for row batches, or a duration string such as ``"1D"``.
        feature_cols: Optional feature columns. If omitted, all non-temporal,
            non-target columns are used.
        update_strategy: Strategy that initializes and updates the model.
        metrics: Mapping of metric names to user-provided callables.
        metric_directions: Optional metric direction overrides.
        primary_metric: Primary metric used by downstream analysis.
        evaluation: Optional explicit ``EvaluationProfile``.
        include_predictions: Whether row-level predictions should be stored.
        retrain_trigger: Optional callable evaluated after each batch is scored.
            It receives ``history`` (all records up to the current batch) and
            ``latest`` (the current batch record). Return ``True``, a reason
            string, or a dictionary such as ``{"retrain": True, "reason": "..."}``
            to mark that batch as a retraining checkpoint.
    """

    def __init__(
        self,
        *,
        model,
        time_col: str | int | TemporalSemanticsSpec,
        target_col: ColumnRef,
        initial_train_size,
        update_size=1,
        feature_cols: Sequence[ColumnRef] | None = None,
        update_strategy: OnlineUpdateStrategy | None = None,
        metrics: MetricSpec = None,
        metric_directions: dict[str, str] | None = None,
        primary_metric: str | None = None,
        evaluation: EvaluationProfile | None = None,
        include_predictions: bool = True,
        prediction_column: str = "prediction",
        retrain_trigger: OnlineRetrainTrigger | None = None,
    ) -> None:
        if evaluation is not None and (
            metrics is not None or metric_directions is not None or primary_metric is not None
        ):
            raise ValueError(
                "evaluation cannot be combined with metrics, metric_directions or primary_metric"
            )
        self.model = model
        self.time_col = time_col
        self.target_col = target_col
        self.initial_train_size = SizeSpec.from_value(initial_train_size)
        self.update_size = SizeSpec.from_value(update_size)
        self.feature_cols = feature_cols
        self.update_strategy = update_strategy or PartialFitUpdateStrategy()
        self.evaluation = evaluation or EvaluationProfile(
            metrics=metrics,
            metric_directions=metric_directions,
            primary_metric=primary_metric,
        )
        self.include_predictions = include_predictions
        self.prediction_column = prediction_column
        self.retrain_trigger = retrain_trigger

    def run(self, X) -> OnlineRunResult:
        """Execute prequential evaluation over ``X``."""
        frame, semantics, resolved_features, target_name = _prepare_supervised_frame(
            X,
            time_col=self.time_col,
            target_col=self.target_col,
            feature_cols=self.feature_cols,
        )
        timeline_col = semantics.timeline_col
        if isinstance(timeline_col, int):
            timeline_col = frame.columns[timeline_col]
        ordered = frame.assign(__jano_time__=pd.to_datetime(frame[timeline_col])).sort_values(
            "__jano_time__",
            kind="mergesort",
        )

        train_positions, batches = self._build_positions(ordered)
        if len(train_positions) == 0:
            raise ValueError("initial_train_size did not select any rows")
        if not batches:
            raise ValueError("update_size did not produce any evaluation batches")

        evaluation = self.evaluation.resolve()
        metric_mapping = evaluation.metrics
        metric_directions = evaluation.metric_directions

        current_model = _clone_model(self.model)
        current_model = self.update_strategy.initialize(
            current_model,
            ordered.iloc[train_positions][resolved_features],
            ordered.iloc[train_positions][target_name],
        )
        rows_seen = int(len(train_positions))
        records: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []

        for batch_number, batch_positions in enumerate(batches):
            batch = ordered.iloc[batch_positions]
            X_batch = batch[resolved_features]
            y_batch = batch[target_name]
            predictions = np.asarray(current_model.predict(X_batch))
            y_true = np.asarray(y_batch)

            batch_start = pd.Timestamp(batch["__jano_time__"].min())
            batch_end = pd.Timestamp(batch["__jano_time__"].max())
            row = {
                "batch": batch_number,
                "updated": True,
                "train_rows_seen": rows_seen,
                "batch_rows": int(len(batch_positions)),
                "batch_start": batch_start,
                "batch_end": batch_end,
            }
            for name, metric_fn in metric_mapping.items():
                row[name] = metric_fn(y_true, predictions)
            trigger_payload = self._evaluate_retrain_trigger(records, row)
            row.update(trigger_payload)
            records.append(row)

            if self.include_predictions:
                for original_index, truth, prediction in zip(batch.index, y_true, predictions):
                    prediction_rows.append(
                        {
                            "batch": batch_number,
                            "row_index": original_index,
                            "y_true": truth,
                            self.prediction_column: prediction,
                        }
                    )

            current_model = self.update_strategy.update(current_model, X_batch, y_batch)
            rows_seen += int(len(batch_positions))

        return OnlineRunResult(
            records=pd.DataFrame(records),
            predictions=pd.DataFrame(prediction_rows),
            metric_directions=metric_directions,
            update_strategy=getattr(
                self.update_strategy,
                "name",
                type(self.update_strategy).__name__,
            ),
            primary_metric=evaluation.primary_metric,
        )

    def _evaluate_retrain_trigger(
        self,
        previous_records: list[dict[str, object]],
        current_row: dict[str, object],
    ) -> dict[str, object]:
        if self.retrain_trigger is None:
            return {"retrain_checkpoint": False, "retrain_reason": None}

        history = pd.DataFrame([*previous_records, current_row])
        signal = self.retrain_trigger(history.copy(), dict(current_row))
        normalized = _normalize_retrain_signal(signal)
        return normalized

    def _build_positions(self, ordered: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]:
        total_rows = len(ordered)
        timestamps = ordered["__jano_time__"].to_numpy(dtype="datetime64[ns]")
        train_positions, initial_end_position, initial_end_time = self._initial_train_positions(
            timestamps,
            total_rows,
        )
        batches = self._evaluation_batches(timestamps, initial_end_position, initial_end_time)
        return train_positions, batches

    def _initial_train_positions(
        self,
        timestamps: np.ndarray,
        total_rows: int,
    ) -> tuple[np.ndarray, int, pd.Timestamp | None]:
        if self.initial_train_size.kind == "rows":
            initial_end = int(self.initial_train_size.value)
            train_positions = np.arange(min(initial_end, total_rows), dtype=np.int64)
            return train_positions, initial_end, None

        if self.initial_train_size.kind == "fraction":
            initial_end = int(round(total_rows * float(self.initial_train_size.value)))
            if initial_end <= 0:
                raise ValueError("Fractional online sizes resolved to zero rows")
            train_positions = np.arange(min(initial_end, total_rows), dtype=np.int64)
            return train_positions, initial_end, None

        start = pd.Timestamp(timestamps[0])
        initial_end = start + self.initial_train_size.value
        train_right = int(
            np.searchsorted(timestamps, np.datetime64(initial_end.to_datetime64()), side="left")
        )
        train_positions = np.arange(train_right, dtype=np.int64)
        return train_positions, train_right, initial_end

    def _evaluation_batches(
        self,
        timestamps: np.ndarray,
        initial_end_position: int,
        initial_end_time: pd.Timestamp | None,
    ) -> list[np.ndarray]:
        total_rows = len(timestamps)
        if self.update_size.kind == "rows":
            return _row_batches(initial_end_position, total_rows, int(self.update_size.value))

        if self.update_size.kind == "fraction":
            update_rows = int(round(total_rows * float(self.update_size.value)))
            if update_rows <= 0:
                raise ValueError("Fractional online sizes resolved to zero rows")
            return _row_batches(initial_end_position, total_rows, update_rows)

        if initial_end_position >= total_rows:
            return []

        batches: list[np.ndarray] = []
        cursor = initial_end_time or pd.Timestamp(timestamps[initial_end_position])
        max_time = pd.Timestamp(timestamps[-1])
        while cursor <= max_time:
            batch_end = cursor + self.update_size.value
            left = int(
                np.searchsorted(timestamps, np.datetime64(cursor.to_datetime64()), side="left")
            )
            right = int(
                np.searchsorted(timestamps, np.datetime64(batch_end.to_datetime64()), side="left")
            )
            if right > left:
                batches.append(np.arange(left, right, dtype=np.int64))
            cursor = batch_end
        return batches


class OnlineUpdatePolicyStudy:
    """Compare online update policies over the same temporal stream.

    The study runs one ``OnlineTemporalRunner`` per candidate policy and returns
    policy-level metrics plus detailed per-policy runs. It is useful for comparing
    update cadences such as every event, every ``N`` rows, every day, or refit
    strategies with different retained-history caps.
    """

    def __init__(
        self,
        *,
        model,
        time_col: str | int | TemporalSemanticsSpec,
        target_col: ColumnRef,
        initial_train_size,
        policies: Sequence[OnlineUpdatePolicy],
        feature_cols: Sequence[ColumnRef] | None = None,
        metrics: MetricSpec = None,
        metric_directions: dict[str, str] | None = None,
        primary_metric: str | None = None,
        evaluation: EvaluationProfile | None = None,
    ) -> None:
        if not policies:
            raise ValueError("policies must contain at least one OnlineUpdatePolicy")
        names = [policy.name for policy in policies]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError("policy names must be unique: " + ", ".join(duplicates))
        self.model = model
        self.time_col = time_col
        self.target_col = target_col
        self.initial_train_size = initial_train_size
        self.policies = tuple(policies)
        self.feature_cols = feature_cols
        self.metrics = metrics
        self.metric_directions = metric_directions
        self.primary_metric = primary_metric
        self.evaluation = evaluation

    def run(self, X) -> OnlineUpdatePolicyStudyResult:
        """Evaluate all candidate online update policies over ``X``."""
        records: list[dict[str, object]] = []
        runs: dict[str, OnlineRunResult] = {}
        metric_directions: dict[str, str] | None = None
        primary_metric: str | None = None

        for policy in self.policies:
            runner = OnlineTemporalRunner(
                model=self.model,
                time_col=self.time_col,
                target_col=self.target_col,
                feature_cols=self.feature_cols,
                initial_train_size=self.initial_train_size,
                update_size=policy.update_size,
                update_strategy=policy.build_strategy(),
                metrics=self.metrics,
                metric_directions=self.metric_directions,
                primary_metric=self.primary_metric,
                evaluation=self.evaluation,
                include_predictions=False,
            )
            run = runner.run(X)
            summary = run.summary()
            row = {
                "policy": policy.name,
                "update_size": str(policy.update_size),
                "update_strategy": summary["update_strategy"],
                "updates": summary["updates"],
                "rows_evaluated": summary["rows_evaluated"],
                "update_cost": float(policy.update_cost),
                "total_update_cost": float(policy.update_cost) * float(summary["updates"]),
            }
            for metric in run.metric_names:
                row[metric] = summary[f"{metric}_mean"]
                row[f"{metric}_best"] = summary[f"{metric}_best"]
                row[f"{metric}_best_batch"] = summary[f"{metric}_best_batch"]
            records.append(row)
            runs[policy.name] = run
            metric_directions = run.metric_directions
            primary_metric = run.primary_metric

        return OnlineUpdatePolicyStudyResult(
            records=pd.DataFrame(records),
            runs=runs,
            metric_directions=metric_directions or {},
            primary_metric=primary_metric,
        )


def _row_batches(start: int, stop: int, size: int) -> list[np.ndarray]:
    if size <= 0:  # pragma: no cover - SizeSpec validates public inputs before this point.
        raise ValueError("update_size must resolve to at least one row")
    batches = []
    cursor = start
    while cursor < stop:
        end = min(cursor + size, stop)
        batches.append(np.arange(cursor, end, dtype=np.int64))
        cursor = end
    return batches


def _normalize_retrain_signal(signal: object) -> dict[str, object]:
    payload: dict[str, object] = {"retrain_checkpoint": False, "retrain_reason": None}
    if signal is None or signal is False:
        return payload
    if signal is True:
        payload["retrain_checkpoint"] = True
        return payload
    if isinstance(signal, str):
        payload["retrain_checkpoint"] = True
        payload["retrain_reason"] = signal
        return payload
    if isinstance(signal, dict):
        should_retrain = bool(signal.get("retrain", signal.get("checkpoint", True)))
        payload["retrain_checkpoint"] = should_retrain
        payload["retrain_reason"] = signal.get("reason")
        for key, value in signal.items():
            if key in {"retrain", "checkpoint", "reason"}:
                continue
            payload[f"retrain_{key}"] = value
        return payload
    raise TypeError(
        "retrain_trigger must return None, bool, str or a dictionary with retrain/checkpoint metadata"
    )

