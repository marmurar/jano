from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .reporting import SimulationSummary, build_simulation_summary
from .slicing import TimeIndexer
from .splits import TimeSplit
from .types import SegmentBoundaries, TemporalSemanticsSpec


def _normalize_windows(windows: Sequence[tuple[object, object]] | None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if windows is None:
        return []
    normalized = []
    for start, end in windows:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if end_ts <= start_ts:
            raise ValueError("Excluded windows must have end greater than start")
        normalized.append((start_ts, end_ts))
    return normalized


def _overlaps(boundary: SegmentBoundaries, window: tuple[pd.Timestamp, pd.Timestamp]) -> bool:
    return boundary.start < window[1] and boundary.end > window[0]


@dataclass(frozen=True)
class PlannedFold:
    """Precomputed temporal geometry for one simulation iteration.

    Attributes:
        iteration: Zero-based simulation iteration.
        boundaries: Mapping from segment name to closed-open temporal boundaries.
        counts: Mapping from segment name to the number of rows in that segment.
        metadata: Additional planning metadata such as ``is_partial``.
    """

    iteration: int
    boundaries: Dict[str, SegmentBoundaries]
    counts: Dict[str, int]
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def fold(self) -> int:
        return self.iteration

    @property
    def is_partial(self) -> bool:
        return bool(self.metadata.get("is_partial", False))

    @property
    def simulation_start(self) -> pd.Timestamp:
        first = next(iter(self.boundaries.values()))
        return first.start

    @property
    def simulation_end(self) -> pd.Timestamp:
        last = list(self.boundaries.values())[-1]
        return last.end

    def to_dict(self) -> dict[str, object]:
        """Return the fold as one serializable row for DataFrame/report output."""
        row: dict[str, object] = {
            "iteration": self.iteration,
            "fold": self.fold,
            "simulation_start": self.simulation_start,
            "simulation_end": self.simulation_end,
            "is_partial": self.is_partial,
        }
        for name, boundary in self.boundaries.items():
            row[f"{name}_start"] = boundary.start
            row[f"{name}_end"] = boundary.end
            row[f"{name}_rows"] = self.counts.get(name, 0)
        return row


@dataclass(frozen=True)
class PartitionPlan:
    """Precomputed temporal plan that can be inspected and materialized later.

    A plan contains fold boundaries and row counts before the actual train/test slices
    are materialized. This makes it cheap to inspect, filter or subset a simulation.

    Attributes:
        frame: Source dataset used to compute row counts and later materialize folds.
        temporal_semantics: Timeline, ordering and per-segment timestamp semantics.
        strategy: Movement strategy used to generate folds.
        size_kind: Unit family used by the partition: ``"duration"``, ``"rows"`` or
            ``"fraction"``.
        folds: Precomputed fold geometry.
    """

    frame: pd.DataFrame
    temporal_semantics: TemporalSemanticsSpec
    strategy: str
    size_kind: str
    folds: List[PlannedFold]

    @property
    def total_folds(self) -> int:
        return len(self.folds)

    @property
    def time_col(self):
        return self.temporal_semantics.timeline_col

    def to_frame(self) -> pd.DataFrame:
        """Return one row per planned fold with boundaries and row counts."""
        return pd.DataFrame([fold.to_dict() for fold in self.folds])

    def select_iterations(self, iterations: Sequence[int]) -> "PartitionPlan":
        """Return a plan containing only the selected iteration numbers."""
        wanted = set(iterations)
        selected = [fold for fold in self.folds if fold.iteration in wanted]
        return self._clone_with(selected)

    def select_from_iteration(self, iteration: int) -> "PartitionPlan":
        """Return a plan containing iterations greater than or equal to ``iteration``."""
        selected = [fold for fold in self.folds if fold.iteration >= iteration]
        return self._clone_with(selected)

    def select_until_iteration(self, iteration: int) -> "PartitionPlan":
        """Return a plan containing iterations less than or equal to ``iteration``."""
        selected = [fold for fold in self.folds if fold.iteration <= iteration]
        return self._clone_with(selected)

    def exclude_windows(
        self,
        *,
        train: Sequence[tuple[object, object]] | None = None,
        validation: Sequence[tuple[object, object]] | None = None,
        test: Sequence[tuple[object, object]] | None = None,
    ) -> "PartitionPlan":
        """Return a plan with folds removed when segment boundaries overlap exclusions.

        Args:
            train: Optional excluded windows applied to train segment boundaries.
            validation: Optional excluded windows applied to validation boundaries.
            test: Optional excluded windows applied to test segment boundaries.
        """
        excluded: Mapping[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {
            "train": _normalize_windows(train),
            "validation": _normalize_windows(validation),
            "test": _normalize_windows(test),
        }
        kept: list[PlannedFold] = []
        for fold in self.folds:
            should_drop = False
            for segment_name, windows in excluded.items():
                if not windows or segment_name not in fold.boundaries:
                    continue
                boundary = fold.boundaries[segment_name]
                if any(_overlaps(boundary, window) for window in windows):
                    should_drop = True
                    break
            if not should_drop:
                kept.append(fold)
        return self._clone_with(kept)

    def materialize(self) -> list[TimeSplit]:
        """Materialize the planned fold boundaries into ``TimeSplit`` objects."""
        if not self.folds:
            raise ValueError("The current plan does not contain any folds")
        indexer = TimeIndexer(frame=self.frame, semantics=self.temporal_semantics)
        splits: list[TimeSplit] = []
        for fold in self.folds:
            segments = {
                name: indexer.slice_between_for_segment(name, boundary.start, boundary.end)
                for name, boundary in fold.boundaries.items()
            }
            splits.append(
                TimeSplit(
                    fold=fold.iteration,
                    segments=segments,
                    boundaries=fold.boundaries,
                    metadata={**fold.metadata, "strategy": self.strategy, "size_kind": self.size_kind},
                )
            )
        return splits

    def iter_splits(self) -> Iterator[TimeSplit]:
        """Iterate over materialized ``TimeSplit`` objects."""
        return iter(self.materialize())

    def _clone_with(self, folds: list[PlannedFold]) -> "PartitionPlan":
        return PartitionPlan(
            frame=self.frame,
            temporal_semantics=self.temporal_semantics,
            strategy=self.strategy,
            size_kind=self.size_kind,
            folds=folds,
        )


@dataclass(frozen=True)
class SimulationPlan:
    """High-level simulation plan with helpers for reporting and materialization.

    Attributes:
        partition_plan: Lower-level partition plan with fold boundaries and counts.
        title: Report title used when the plan is described or written as HTML.
    """

    partition_plan: PartitionPlan
    title: str

    @property
    def total_folds(self) -> int:
        return self.partition_plan.total_folds

    def to_frame(self) -> pd.DataFrame:
        """Return one row per planned fold with boundaries and row counts."""
        return self.partition_plan.to_frame()

    def select_iterations(self, iterations: Sequence[int]) -> "SimulationPlan":
        """Return a simulation plan containing only the selected iteration numbers."""
        return SimulationPlan(self.partition_plan.select_iterations(iterations), self.title)

    def select_from_iteration(self, iteration: int) -> "SimulationPlan":
        """Return a simulation plan starting at ``iteration``."""
        return SimulationPlan(self.partition_plan.select_from_iteration(iteration), self.title)

    def select_until_iteration(self, iteration: int) -> "SimulationPlan":
        """Return a simulation plan ending at ``iteration``."""
        return SimulationPlan(self.partition_plan.select_until_iteration(iteration), self.title)

    def exclude_windows(
        self,
        *,
        train: Sequence[tuple[object, object]] | None = None,
        validation: Sequence[tuple[object, object]] | None = None,
        test: Sequence[tuple[object, object]] | None = None,
    ) -> "SimulationPlan":
        """Return a simulation plan after removing folds that overlap excluded windows."""
        return SimulationPlan(
            self.partition_plan.exclude_windows(
                train=train,
                validation=validation,
                test=test,
            ),
            self.title,
        )

    def materialize(self) -> "SimulationResult":
        """Materialize the plan into a ``SimulationResult``."""
        from .simulation import SimulationResult

        splits = self.partition_plan.materialize()
        summary = build_simulation_summary(
            splits=splits,
            frame=self.partition_plan.frame,
            time_col=self.partition_plan.time_col,
            title=self.title,
        )
        return SimulationResult(
            frame=self.partition_plan.frame,
            splits=splits,
            summary=summary,
        )

    def describe(self) -> SimulationSummary:
        """Materialize the plan and return its structured summary."""
        return self.materialize().summary

    def write_html(self, path: str | Path) -> Path:
        """Materialize the plan and write its rendered HTML report."""
        return self.materialize().write_html(path)
