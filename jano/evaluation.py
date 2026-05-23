from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .policies import MetricFn, MetricSpec, _normalize_metric_mapping

MetricDirection = str


@dataclass(frozen=True)
class ResolvedEvaluationProfile:
    """Normalized metrics and metadata consumed by execution layers."""

    metrics: dict[str, MetricFn]
    metric_directions: dict[str, MetricDirection]
    primary_metric: str | None


@dataclass(frozen=True)
class EvaluationProfile:
    """Define how a temporal run should be measured.

    Args:
        metrics: Mapping of metric names to user-defined callables. ``None`` means
            no metrics are computed by Jano.
        metric_directions: Optional mapping from metric name to ``"min"`` or ``"max"``.
            Custom metrics default to ``"min"`` unless explicitly overridden.
        primary_metric: Metric used as the default optimization or retraining signal.
    """

    metrics: MetricSpec = None
    metric_directions: Mapping[str, MetricDirection] | None = None
    primary_metric: str | None = None

    def resolve(self) -> ResolvedEvaluationProfile:
        """Return normalized metric functions, directions and primary metric."""
        metrics, directions = _normalize_metric_mapping(self.metrics)
        resolved_directions = dict(directions)

        if self.metric_directions is not None:
            unknown = sorted(set(self.metric_directions) - set(metrics))
            if unknown:
                raise ValueError(
                    "metric_directions contains unknown metrics: " + ", ".join(unknown)
                )
            for name, direction in self.metric_directions.items():
                if direction not in {"min", "max"}:
                    raise ValueError("metric directions must be either 'min' or 'max'")
                resolved_directions[name] = direction

        primary_metric = self.primary_metric
        if primary_metric is None:
            primary_metric = _default_primary_metric(metrics)
        if primary_metric is not None and primary_metric not in metrics:
            raise ValueError(f"primary_metric '{primary_metric}' is not present in metrics")

        return ResolvedEvaluationProfile(
            metrics=dict(metrics),
            metric_directions=resolved_directions,
            primary_metric=primary_metric,
        )


class RegressionProfile(EvaluationProfile):
    """Convenience profile for user-provided regression-style losses."""

    def __init__(
        self,
        metrics: MetricSpec = None,
        *,
        metric_directions: Mapping[str, MetricDirection] | None = None,
        primary_metric: str | None = None,
    ) -> None:
        super().__init__(
            metrics=metrics,
            metric_directions=metric_directions,
            primary_metric=primary_metric,
        )


class ClassificationProfile(EvaluationProfile):
    """Convenience profile for user-provided classification-style scores."""

    def __init__(
        self,
        metrics: MetricSpec = None,
        *,
        metric_directions: Mapping[str, MetricDirection] | None = None,
        primary_metric: str | None = None,
    ) -> None:
        super().__init__(
            metrics=metrics,
            metric_directions=metric_directions,
            primary_metric=primary_metric,
        )


class OrdinalClassificationProfile(EvaluationProfile):
    """Profile for ordered classes where user-defined costs usually matter."""

    def __init__(
        self,
        metrics: Mapping[str, MetricFn],
        *,
        metric_directions: Mapping[str, MetricDirection] | None = None,
        primary_metric: str | None = None,
    ) -> None:
        super().__init__(
            metrics=metrics,
            metric_directions=metric_directions,
            primary_metric=primary_metric,
        )


class RankingProfile(EvaluationProfile):
    """Profile for ranking or retrieval evaluations with custom metrics."""

    def __init__(
        self,
        metrics: Mapping[str, MetricFn],
        *,
        metric_directions: Mapping[str, MetricDirection] | None = None,
        primary_metric: str | None = None,
    ) -> None:
        super().__init__(
            metrics=metrics,
            metric_directions=metric_directions,
            primary_metric=primary_metric,
        )


def _default_primary_metric(metrics: Mapping[str, MetricFn]) -> str | None:
    return next(iter(metrics), None)
