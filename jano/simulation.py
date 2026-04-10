from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import pandas as pd

from .reporting import (
    SimulationChartData,
    SimulationSummary,
    build_simulation_summary,
)
from .splitters import TemporalBacktestSplitter
from .splits import TimeSplit
from .types import TemporalPartitionSpec, TemporalSemanticsSpec


@dataclass(frozen=True)
class SimulationResult:
    """Materialized result of running a temporal simulation over a dataset.

    Attributes:
        frame: Source dataset used to build the simulation.
        splits: Materialized fold objects.
        summary: Structured report for the simulation.
    """

    frame: pd.DataFrame
    splits: List[TimeSplit]
    summary: SimulationSummary

    @property
    def total_folds(self) -> int:
        """Return the number of materialized folds."""
        return len(self.splits)

    @property
    def chart_data(self) -> SimulationChartData:
        """Return plot-ready chart data for the simulation."""
        return self.summary.chart_data

    @property
    def html(self) -> str:
        """Return the rendered HTML report."""
        return self.summary.html

    def to_frame(self) -> pd.DataFrame:
        """Return fold-level simulation metadata as a pandas DataFrame."""
        return self.summary.to_frame()

    def to_dict(self) -> dict[str, object]:
        """Return a serializable dictionary representation."""
        return self.summary.to_dict()

    def write_html(self, path: str | Path) -> Path:
        """Write the rendered HTML report to disk."""
        return self.summary.write_html(path)

    def iter_splits(self) -> Iterator[TimeSplit]:
        """Iterate over materialized fold objects."""
        return iter(self.splits)


class TemporalSimulation:
    """High-level interface for executing a complete temporal simulation.

    Args:
        time_col: Either the name of the timeline column or a ``TemporalSemanticsSpec``
            describing the timeline, ordering column and per-segment eligibility columns.
        partition: High-level definition of the train/test or train/validation/test layout.
        step: Amount by which the simulation advances after each fold.
        strategy: Simulation policy. Use ``"single"``, ``"rolling"`` or ``"expanding"``.
        allow_partial: Whether to keep the last fold when the final evaluation segment
            would otherwise run past the end of the dataset.
        start_at: Optional lower bound for the simulation timeline. Rows strictly before
            this timestamp are excluded before folds are generated.
        end_at: Optional upper bound for the simulation timeline. Rows strictly after
            this timestamp are excluded before folds are generated.
        max_folds: Optional maximum number of folds to materialize.
    """

    def __init__(
        self,
        time_col: str | TemporalSemanticsSpec,
        partition: TemporalPartitionSpec,
        step,
        strategy: str = "rolling",
        allow_partial: bool = False,
        start_at: object | None = None,
        end_at: object | None = None,
        max_folds: int | None = None,
    ) -> None:
        self.splitter = TemporalBacktestSplitter(
            time_col=time_col,
            partition=partition,
            step=step,
            strategy=strategy,
            allow_partial=allow_partial,
        )
        self.start_at = pd.Timestamp(start_at) if start_at is not None else None
        self.end_at = pd.Timestamp(end_at) if end_at is not None else None
        if max_folds is not None and max_folds <= 0:
            raise ValueError("max_folds must be greater than zero")
        self.max_folds = max_folds

    @property
    def time_col(self):
        """Return the timeline column configured for the simulation."""
        return self.splitter.time_col

    @property
    def partition(self):
        """Return the validated partition configuration used by the simulation."""
        return self.splitter.partition

    @property
    def temporal_semantics(self) -> TemporalSemanticsSpec:
        """Return the temporal semantics used by the simulation."""
        return self.splitter.temporal_semantics

    def as_splitter(self) -> TemporalBacktestSplitter:
        """Return the underlying low-level splitter."""
        return self.splitter

    def run(
        self,
        X: pd.DataFrame,
        output_path: str | Path | None = None,
        title: str | None = None,
    ) -> SimulationResult:
        """Execute the configured simulation over ``X`` and materialize its folds.

        Args:
            X: Input dataset as ``pandas.DataFrame``, ``numpy.ndarray`` or
                ``polars.DataFrame``.
            output_path: Optional filesystem path where the rendered HTML report should
                be written.
            title: Optional title used in the returned report outputs.

        Returns:
            A ``SimulationResult`` containing the materialized folds and their summary.
        """
        frame = self._select_frame(X)
        splits = list(self.splitter.iter_splits(frame))
        if self.max_folds is not None:
            splits = splits[: self.max_folds]
        if not splits:
            raise ValueError("The current configuration did not produce any valid folds")
        summary = build_simulation_summary(
            splits=splits,
            frame=frame,
            time_col=self.time_col,
            title=title or "Jano simulation summary",
        )
        if output_path is not None:
            summary.write_html(output_path)
        return SimulationResult(frame=frame, splits=splits, summary=summary)

    def _select_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self.splitter._coerce_frame(X)
        if self.start_at is None and self.end_at is None:
            return frame

        timestamps = pd.to_datetime(frame[self.temporal_semantics.timeline_col])
        mask = pd.Series(True, index=frame.index)
        if self.start_at is not None:
            mask &= timestamps >= self.start_at
        if self.end_at is not None:
            mask &= timestamps <= self.end_at

        filtered = frame.loc[mask].copy()
        if filtered.empty:
            raise ValueError("The configured simulation window does not contain any rows")
        return filtered
