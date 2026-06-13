from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from ._serialization import _frame_records, _json_ready_object
from ._workflow_inputs import _resolve_workflow_inputs
from .io import coerce_tabular_input
from .runner import (
    AlwaysRetrain,
    NeverRetrain,
    PeriodicRetrain,
    RetrainContext,
    RetrainPolicy,
)
from .systems import (
    UpdateableSystem,
    _normalize_system_evaluation_result,
    _normalize_system_update_result,
)


@dataclass(frozen=True)
class SystemRunResult:
    """Materialized execution of a temporal updateable system."""

    records: pd.DataFrame
    details: pd.DataFrame
    metric_directions: dict[str, str]
    update_policy: str
    primary_metric: str | None = None

    def to_frame(self) -> pd.DataFrame:
        return self.records.copy()

    @property
    def metric_names(self) -> list[str]:
        metadata_columns = {
            "fold",
            "updated",
            "last_update_fold",
            "train_rows",
            "test_rows",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
        }
        return [column for column in self.records.columns if column not in metadata_columns]

    def fold_summary(self) -> pd.DataFrame:
        columns = [
            "fold",
            "updated",
            "last_update_fold",
            "train_rows",
            "test_rows",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
        ]
        return self.records[[column for column in columns if column in self.records.columns]].copy()

    def metric_trajectory(self) -> pd.DataFrame:
        metric_names = self.metric_names
        if not metric_names:
            return pd.DataFrame(columns=["fold", "metric", "value", "direction", "updated"])

        trajectory = self.records.melt(
            id_vars=["fold", "updated"],
            value_vars=metric_names,
            var_name="metric",
            value_name="value",
        )
        trajectory["direction"] = trajectory["metric"].map(
            lambda metric: self.metric_directions.get(metric, "min")
        )
        return trajectory[["fold", "metric", "value", "direction", "updated"]]

    def update_events(self) -> pd.DataFrame:
        return self.fold_summary().loc[lambda frame: frame["updated"].astype(bool)].reset_index(
            drop=True
        )

    def evaluation_details(self) -> pd.DataFrame:
        return self.details.copy()

    def summary(self) -> dict[str, object]:
        updated = self.records["updated"].astype(bool)
        summary = {
            "folds": int(len(self.records)),
            "update_policy": self.update_policy,
            "update_events": int(updated.sum()),
            "update_ratio": float(updated.mean()),
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

    def report_data(self) -> dict[str, object]:
        return {
            "summary": self.summary(),
            "folds": _frame_records(self.fold_summary()),
            "metrics": _frame_records(self.metric_trajectory()),
            "updates": _frame_records(self.update_events()),
            "evaluations": [
                _json_ready_object(row)
                for row in self.evaluation_details().to_dict(orient="records")
            ],
            "metric_directions": dict(self.metric_directions),
            "primary_metric": self.primary_metric,
        }

    def to_dict(self) -> dict[str, object]:
        return self.report_data()


class TemporalSystemRunner:
    """Run an updateable system over temporal folds using an update policy."""

    def __init__(
        self,
        *,
        system: UpdateableSystem,
        update: bool | str = True,
        update_interval: int | None = None,
        update_policy: RetrainPolicy | None = None,
        metric_directions: Mapping[str, str] | None = None,
        primary_metric: str | None = None,
    ) -> None:
        self.system = system
        self.metric_directions = dict(metric_directions or {})
        self.primary_metric = primary_metric
        self.update_policy = self._normalize_policy(
            update=update,
            update_interval=update_interval,
            update_policy=update_policy,
        )

    def run(self, workflow, frame_input) -> SystemRunResult:
        selected_input, semantics, splits = _resolve_workflow_inputs(workflow, frame_input)
        frame = coerce_tabular_input(selected_input).copy()

        current_state = _MissingState
        last_update_fold: int | None = None
        records: list[dict[str, object]] = []
        detail_rows: list[dict[str, object]] = []

        for split in splits:
            if "train" not in split.segments or "test" not in split.segments:
                raise ValueError("TemporalSystemRunner requires folds with 'train' and 'test' segments")

            train_idx = split.segments["train"]
            test_idx = split.segments["test"]
            train_frame = frame.iloc[train_idx].copy()
            test_frame = frame.iloc[test_idx].copy()

            context = RetrainContext(
                fold=split.fold,
                split=split,
                history=pd.DataFrame(records),
                metric_directions=self.metric_directions,
                last_retrain_fold=last_update_fold,
                primary_metric=self.primary_metric,
            )
            should_update = current_state is _MissingState or self.update_policy.should_retrain(
                context
            )
            update_metadata: dict[str, Any] = {}
            if should_update:
                update_result = _normalize_system_update_result(self.system.update(train_frame))
                current_state = update_result.state
                update_metadata = dict(update_result.metadata)
                last_update_fold = split.fold

            evaluation_result = _normalize_system_evaluation_result(
                self.system.evaluate(current_state, test_frame)
            )
            metric_values = dict(evaluation_result.metrics)
            self._validate_metric_directions(metric_values)

            boundaries = split.boundaries
            row = {
                "fold": split.fold,
                "updated": bool(should_update),
                "last_update_fold": last_update_fold,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_start": boundaries["train"].start,
                "train_end": boundaries["train"].end,
                "test_start": boundaries["test"].start,
                "test_end": boundaries["test"].end,
                **metric_values,
            }
            records.append(row)
            detail_rows.append(
                {
                    "fold": split.fold,
                    "updated": bool(should_update),
                    "update_metadata": update_metadata,
                    "evaluation_metadata": dict(evaluation_result.metadata),
                }
            )

        if not records:
            raise ValueError("The configured workflow did not produce any valid folds")

        return SystemRunResult(
            records=pd.DataFrame(records),
            details=pd.DataFrame(detail_rows),
            metric_directions=dict(self.metric_directions),
            update_policy=type(self.update_policy).__name__,
            primary_metric=self.primary_metric,
        )

    def _normalize_policy(
        self,
        *,
        update: bool | str,
        update_interval: int | None,
        update_policy: RetrainPolicy | None,
    ) -> RetrainPolicy:
        if update_policy is not None:
            if update_interval is not None:
                raise ValueError("update_interval cannot be used together with update_policy")
            return update_policy

        if isinstance(update, bool):
            if not update:
                return NeverRetrain()
            if update_interval is None or update_interval == 1:
                return AlwaysRetrain()
            return PeriodicRetrain(update_interval)

        if update not in {"always", "never", "periodic"}:
            raise ValueError("update must be True, False, 'always', 'never' or 'periodic'")
        if update == "always":
            return AlwaysRetrain()
        if update == "never":
            return NeverRetrain()
        if update_interval is None:
            raise ValueError("update_interval is required when update='periodic'")
        return PeriodicRetrain(update_interval)

    def _validate_metric_directions(self, metrics: Mapping[str, float]) -> None:
        unknown = sorted(set(self.metric_directions) - set(metrics))
        if unknown:
            raise ValueError(
                "metric_directions contains unknown metrics: " + ", ".join(unknown)
            )
        if self.primary_metric is not None and self.primary_metric not in metrics:
            raise ValueError(
                f"primary_metric '{self.primary_metric}' is not present in system metrics"
            )
        for name, direction in self.metric_directions.items():
            if direction not in {"min", "max"}:
                raise ValueError(f"Metric direction for '{name}' must be either 'min' or 'max'")


_MissingState = object()
