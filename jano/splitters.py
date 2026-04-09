from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd

from .reporting import SimulationChartData, SimulationSummary, build_simulation_summary
from .slicing import TimeIndexer
from .splits import TimeSplit
from .types import SegmentBoundaries, SizeSpec, TemporalPartitionSpec
from .validation import ValidatedPartitionSpec, validate_partition_spec, validate_strategy


@dataclass(frozen=True)
class _SegmentOffsets:
    name: str
    start: object
    end: object


class TemporalBacktestSplitter:
    """Flexible temporal splitter for single or repeated temporal backtests.

    Args:
        time_col: Name of the timestamp column used to order the dataset.
        partition: High-level definition of the train/test or train/validation/test layout.
        step: Amount by which the simulation advances after each fold. It must use the
            same unit family as the partition sizes.
        strategy: Simulation policy. Use ``"single"`` for one split, ``"rolling"`` for
            fixed-size windows or ``"expanding"`` for growing training history.
        allow_partial: Whether to keep the last fold when the final evaluation segment
            would otherwise run past the end of the dataset.
    """

    def __init__(
        self,
        time_col: str,
        partition: TemporalPartitionSpec,
        step,
        strategy: str = "rolling",
        allow_partial: bool = False,
    ) -> None:
        self.time_col = time_col
        self.partition = validate_partition_spec(partition)
        self.step = SizeSpec.from_value(step)
        self.strategy = validate_strategy(strategy)
        self.allow_partial = allow_partial

        if self.partition.size_kind != self.step.kind:
            raise ValueError("step must use the same unit family as the partition sizes")

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, ...]]:
        """Yield each fold as a plain tuple of positional index arrays.

        Args:
            X: Input dataset as a pandas ``DataFrame``.
            y: Unused placeholder for scikit-learn compatibility.
            groups: Unused placeholder for scikit-learn compatibility.

        Yields:
            Tuples of NumPy arrays ordered by the configured segment names.
        """
        for split in self.iter_splits(X, y=y, groups=groups):
            ordered_names = list(split.segments.keys())
            yield tuple(split.segments[name] for name in ordered_names)

    def iter_splits(self, X, y=None, groups=None) -> Iterator[TimeSplit]:
        """Yield rich ``TimeSplit`` objects for each fold in the simulation.

        Args:
            X: Input dataset as a pandas ``DataFrame``.
            y: Unused placeholder for scikit-learn compatibility.
            groups: Unused placeholder for scikit-learn compatibility.

        Yields:
            ``TimeSplit`` instances containing segment indices, boundaries and metadata.
        """
        frame = self._coerce_frame(X)
        indexer = TimeIndexer(frame=frame, time_col=self.time_col)

        if self.partition.size_kind == "duration":
            yield from self._iter_duration_splits(indexer)
        else:
            yield from self._iter_positional_splits(indexer)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of valid folds generated for ``X``.

        Args:
            X: Input dataset as a pandas ``DataFrame``.
            y: Unused placeholder for scikit-learn compatibility.
            groups: Unused placeholder for scikit-learn compatibility.

        Returns:
            Total number of folds that would be produced by ``iter_splits``.
        """
        if X is None:
            raise ValueError("X is required to compute the number of splits")
        return sum(1 for _ in self.iter_splits(X, y=y, groups=groups))

    def describe_simulation(
        self,
        X: pd.DataFrame,
        output_path: str | Path | None = None,
        title: str | None = None,
        output: str = "summary",
    ) -> SimulationSummary | SimulationChartData | str:
        """Describe a simulation over a concrete dataset.

        Args:
            X: Input dataset as a pandas ``DataFrame``.
            output_path: Optional filesystem path where the rendered HTML report should
                be written.
            title: Optional title used in the returned report outputs.
            output: Output mode. Use ``"summary"`` for ``SimulationSummary``,
                ``"html"`` for a rendered HTML string or ``"chart_data"`` for
                plot-ready Python data.

        Returns:
            A ``SimulationSummary``, raw HTML string or ``SimulationChartData``
            depending on ``output``.
        """
        frame = self._coerce_frame(X)
        splits = list(self.iter_splits(frame))
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

        if output == "summary":
            return summary
        if output == "html":
            return summary.html
        if output == "chart_data":
            return summary.chart_data
        raise ValueError("output must be one of 'summary', 'html' or 'chart_data'")

    @staticmethod
    def _coerce_frame(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TemporalBacktestSplitter currently expects a pandas DataFrame")
        if X.empty:
            raise ValueError("X must contain at least one row")
        return X

    def _iter_duration_splits(self, indexer: TimeIndexer) -> Iterator[TimeSplit]:
        segment_names = list(self.partition.segments.keys())
        segment_sizes = self.partition.segments
        gap_sizes = self.partition.gaps

        start = indexer.min_time
        fold = 0

        while True:
            cursor = start
            train_start = indexer.min_time if self.strategy == "expanding" else start
            boundaries: Dict[str, SegmentBoundaries] = {}

            for name in segment_names:
                if name == "train":
                    segment_start = train_start
                    if self.strategy == "expanding":
                        segment_end = start + segment_sizes[name].value
                        if segment_end <= segment_start:
                            segment_end = segment_start + segment_sizes[name].value
                    else:
                        segment_end = segment_start + segment_sizes[name].value
                else:
                    gap = gap_sizes.get(name)
                    if gap is not None:
                        cursor = cursor + gap.value
                    segment_start = cursor
                    segment_end = segment_start + segment_sizes[name].value
                boundaries[name] = SegmentBoundaries(start=segment_start, end=segment_end)
                cursor = segment_end

            last_end = boundaries[segment_names[-1]].end
            if last_end > indexer.max_time:
                if self.allow_partial and boundaries[segment_names[-1]].start < indexer.max_time:
                    boundaries[segment_names[-1]] = SegmentBoundaries(
                        start=boundaries[segment_names[-1]].start,
                        end=indexer.max_time + pd.Timedelta(microseconds=1),
                    )
                else:
                    break

            segments = {}
            for name, boundary in boundaries.items():
                left, right = indexer.bounds_between(boundary.start, boundary.end)
                segments[name] = indexer.slice_positional(left, right)

            if not self._is_valid_segments(segments):
                break

            yield TimeSplit(
                fold=fold,
                segments=segments,
                boundaries=boundaries,
                metadata={"strategy": self.strategy, "size_kind": self.partition.size_kind},
            )

            fold += 1
            if self.strategy == "single":
                break
            start = start + self.step.value

    def _iter_positional_splits(self, indexer: TimeIndexer) -> Iterator[TimeSplit]:
        total_rows = indexer.total_rows
        sizes = {
            name: self._resolve_position_size(spec, total_rows)
            for name, spec in self.partition.segments.items()
        }
        gaps = {
            name: self._resolve_position_size(spec, total_rows)
            for name, spec in self.partition.gaps.items()
        }
        step = self._resolve_position_size(self.step, total_rows)
        fold = 0
        start = 0
        segment_names = list(self.partition.segments.keys())

        while True:
            cursor = 0 if self.strategy == "expanding" else start
            boundaries: Dict[str, SegmentBoundaries] = {}
            positions: Dict[str, Tuple[int, int]] = {}

            for name in segment_names:
                if name == "train" and self.strategy == "expanding":
                    segment_start = 0
                    segment_end = sizes[name] + (fold * step)
                else:
                    if name != "train":
                        cursor += gaps.get(name, 0)
                    segment_start = cursor
                    segment_end = segment_start + sizes[name]
                if name != "train" or self.strategy != "expanding":
                    positions[name] = (segment_start, segment_end)
                else:
                    positions[name] = (segment_start, segment_end)
                boundaries[name] = SegmentBoundaries(
                    start=indexer.timestamp_at(min(segment_start, total_rows - 1)),
                    end=indexer.timestamp_at(min(segment_end - 1, total_rows - 1)),
                )
                cursor = segment_end

            if positions[segment_names[-1]][1] > total_rows:
                if self.allow_partial:
                    last_name = segment_names[-1]
                    segment_start, _ = positions[last_name]
                    if segment_start >= total_rows:
                        break
                    positions[last_name] = (segment_start, total_rows)
                    boundaries[last_name] = SegmentBoundaries(
                        start=indexer.timestamp_at(segment_start),
                        end=indexer.max_time,
                    )
                else:
                    break

            segments = {
                name: indexer.slice_positional(*pos)
                for name, pos in positions.items()
            }
            if not self._is_valid_segments(segments):
                break

            yield TimeSplit(
                fold=fold,
                segments=segments,
                boundaries=boundaries,
                metadata={"strategy": self.strategy, "size_kind": self.partition.size_kind},
            )

            fold += 1
            if self.strategy == "single":
                break
            start += step

    @staticmethod
    def _is_valid_segments(segments: Dict[str, np.ndarray]) -> bool:
        return all(len(index) > 0 for index in segments.values())

    @staticmethod
    def _resolve_position_size(spec: SizeSpec, total_rows: int) -> int:
        if spec.kind == "rows":
            return int(spec.value)
        resolved = int(round(total_rows * float(spec.value)))
        if resolved <= 0:
            raise ValueError("Fractional sizes resolved to zero rows; choose larger fractions")
        return resolved
