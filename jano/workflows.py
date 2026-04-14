from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .planning import SimulationPlan
from .policies import (
    MetricFn,
    PerformanceDecayPolicy,
    PerformanceDecayResult,
    TrainGrowthPolicy,
    TrainGrowthResult,
)
from .simulation import SimulationResult, TemporalSimulation
from .splitters import TemporalBacktestSplitter
from .types import ColumnRef, TemporalPartitionSpec, TemporalSemanticsSpec


@dataclass(frozen=True)
class RollingTrainHistoryResult:
    """Per-iteration optimal training-history choices over a walk-forward plan."""

    records: pd.DataFrame
    metric: str

    def to_frame(self) -> pd.DataFrame:
        """Return one row per outer iteration with the chosen optimal train size."""
        return self.records.copy()

    def summary(self) -> dict[str, object]:
        """Return compact aggregate statistics for the chosen train windows."""
        frame = self.records
        train_rows = frame["optimal_train_rows"].astype(float)
        return {
            "iterations": int(len(frame)),
            "metric": self.metric,
            "mean_optimal_train_rows": float(train_rows.mean()),
            "median_optimal_train_rows": float(train_rows.median()),
            "min_optimal_train_rows": int(train_rows.min()),
            "max_optimal_train_rows": int(train_rows.max()),
            "mean_metric": float(frame[self.metric].astype(float).mean()),
        }


class WalkForwardPolicy:
    """Recommended high-level entry point for production-like walk-forward evaluation."""

    def __init__(
        self,
        time_col: str | int | TemporalSemanticsSpec,
        *,
        partition: TemporalPartitionSpec,
        step,
        strategy: str = "rolling",
        allow_partial: bool = False,
        start_at: object | None = None,
        end_at: object | None = None,
        max_folds: int | None = None,
    ) -> None:
        self._simulation = TemporalSimulation(
            time_col=time_col,
            partition=partition,
            step=step,
            strategy=strategy,
            allow_partial=allow_partial,
            start_at=start_at,
            end_at=end_at,
            max_folds=max_folds,
        )

    def plan(self, X, title: str | None = None) -> SimulationPlan:
        """Return the precomputed walk-forward geometry."""
        return self._simulation.plan(X, title=title)

    def run(
        self,
        X,
        output_path: str | None = None,
        title: str | None = None,
    ) -> SimulationResult:
        """Materialize the walk-forward simulation."""
        return self._simulation.run(X, output_path=output_path, title=title)

    def as_splitter(self) -> TemporalBacktestSplitter:
        """Expose the underlying splitter for manual control."""
        return self._simulation.as_splitter()

    @property
    def simulation(self) -> TemporalSimulation:
        """Expose the underlying simulation object."""
        return self._simulation


class TrainHistoryPolicy:
    """Recommended entry point for fixed-test, growing-train history studies."""

    def __init__(
        self,
        time_col: str | int | TemporalSemanticsSpec,
        *,
        cutoff,
        train_sizes: Sequence[object],
        test_size,
        gap_before_test=None,
    ) -> None:
        self._policy = TrainGrowthPolicy(
            time_col,
            cutoff=cutoff,
            train_sizes=train_sizes,
            test_size=test_size,
            gap_before_test=gap_before_test,
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
        """Evaluate all configured train-history variants against one fixed test slice."""
        return self._policy.evaluate(
            X,
            model=model,
            target_col=target_col,
            feature_cols=feature_cols,
            metrics=metrics,
        )

    def find_optimal_train_size(self, X, **kwargs) -> dict[str, object]:
        """Return the smallest train window that stays within tolerance of the best score."""
        return self._policy.find_optimal_train_size(X, **kwargs)


class DriftMonitoringPolicy:
    """Recommended entry point for fixed-train, moving-test decay monitoring."""

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
        self._policy = PerformanceDecayPolicy(
            time_col,
            cutoff=cutoff,
            train_size=train_size,
            test_size=test_size,
            step=step,
            gap_before_test=gap_before_test,
            max_windows=max_windows,
        )

    def evaluate(
        self,
        X,
        *,
        model,
        target_col: ColumnRef,
        feature_cols: Sequence[ColumnRef] | None = None,
        metrics: str | Sequence[str] | Mapping[str, MetricFn] | None = None,
    ) -> PerformanceDecayResult:
        """Evaluate how performance evolves as the test window moves forward."""
        return self._policy.evaluate(
            X,
            model=model,
            target_col=target_col,
            feature_cols=feature_cols,
            metrics=metrics,
        )

    def find_drift_onset(self, X, **kwargs) -> dict[str, object] | None:
        """Return the first test window whose performance crosses the chosen threshold."""
        return self._policy.find_drift_onset(X, **kwargs)


class RollingTrainHistoryPolicy:
    """Run train-history optimization inside each outer walk-forward iteration.

    This policy answers questions such as: how much training history is required on
    average if the optimal train window is allowed to vary over time?
    """

    def __init__(
        self,
        time_col: str | int | TemporalSemanticsSpec,
        *,
        partition: TemporalPartitionSpec,
        step,
        train_sizes: Sequence[object],
        strategy: str = "rolling",
        allow_partial: bool = False,
        start_at: object | None = None,
        end_at: object | None = None,
        max_folds: int | None = None,
    ) -> None:
        self._walk_forward = WalkForwardPolicy(
            time_col,
            partition=partition,
            step=step,
            strategy=strategy,
            allow_partial=allow_partial,
            start_at=start_at,
            end_at=end_at,
            max_folds=max_folds,
        )
        self.train_sizes = list(train_sizes)
        if not self.train_sizes:
            raise ValueError("train_sizes must not be empty")

    def plan(self, X, title: str | None = None) -> SimulationPlan:
        """Return the outer walk-forward plan used by the composed policy."""
        return self._walk_forward.plan(X, title=title)

    def evaluate(
        self,
        X,
        *,
        model,
        target_col: ColumnRef,
        feature_cols: Sequence[ColumnRef] | None = None,
        metrics: str | Sequence[str] | Mapping[str, MetricFn] | None = None,
        metric: str = "rmse",
        tolerance: float = 0.0,
        relative: bool = True,
        title: str | None = None,
    ) -> RollingTrainHistoryResult:
        """Choose an optimal train-history size for each outer walk-forward iteration."""
        plan = self._walk_forward.plan(X, title=title)
        rows: list[dict[str, object]] = []

        for fold in plan.partition_plan.folds:
            train_boundary = fold.boundaries["train"]
            test_boundary = fold.boundaries["test"]
            gap_before_test = test_boundary.start - train_boundary.end
            inner_policy = TrainGrowthPolicy(
                self._walk_forward.simulation.time_col,
                cutoff=train_boundary.end,
                train_sizes=self.train_sizes,
                test_size=test_boundary.end - test_boundary.start,
                gap_before_test=gap_before_test,
            )
            best = inner_policy.find_optimal_train_size(
                X,
                model=model,
                target_col=target_col,
                feature_cols=feature_cols,
                metrics=metrics,
                metric=metric,
                tolerance=tolerance,
                relative=relative,
            )
            rows.append(
                {
                    "iteration": fold.iteration,
                    "fold": fold.fold,
                    "outer_train_start": train_boundary.start,
                    "outer_train_end": train_boundary.end,
                    "outer_test_start": test_boundary.start,
                    "outer_test_end": test_boundary.end,
                    "optimal_train_size": best["train_size"],
                    "optimal_train_start": best["train_start"],
                    "optimal_train_end": best["train_end"],
                    "optimal_train_rows": int(best["train_rows"]),
                    "test_rows": int(best["test_rows"]),
                    metric: float(best[metric]),
                }
            )

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError("The configured policy did not produce any valid outer iterations")
        return RollingTrainHistoryResult(records=frame, metric=metric)
