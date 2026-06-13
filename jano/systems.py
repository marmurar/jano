from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import pandas as pd


@dataclass(frozen=True)
class SystemUpdateResult:
    """State and metadata returned by a temporal system update."""

    state: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemEvaluationResult:
    """Metrics and metadata returned by a temporal system evaluation."""

    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


class UpdateableSystem(Protocol):
    """Protocol for systems that can be updated and then evaluated over time."""

    def update(self, train_frame: pd.DataFrame) -> SystemUpdateResult | object:
        """Refresh the internal system state from the current train window."""

    def evaluate(
        self,
        state: Any,
        test_frame: pd.DataFrame,
    ) -> SystemEvaluationResult | Mapping[str, float]:
        """Return metrics for the current state on the test window."""


def _normalize_system_update_result(result) -> SystemUpdateResult:
    if isinstance(result, SystemUpdateResult):
        return result
    return SystemUpdateResult(state=result)


def _normalize_system_evaluation_result(result) -> SystemEvaluationResult:
    if isinstance(result, SystemEvaluationResult):
        metrics = {str(name): float(value) for name, value in result.metrics.items()}
        return SystemEvaluationResult(metrics=metrics, metadata=dict(result.metadata))

    if not isinstance(result, Mapping):
        raise TypeError(
            "system.evaluate() must return a mapping of metrics or a SystemEvaluationResult"
        )

    metrics = {str(name): float(value) for name, value in result.items()}
    return SystemEvaluationResult(metrics=metrics)
