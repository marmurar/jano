from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .splits import TimeSplit

SEGMENT_COLORS = {
    "train": "#1d4ed8",
    "validation": "#d97706",
    "test": "#059669",
}


@dataclass(frozen=True)
class SimulationChartData:
    """Plot-ready description of a temporal simulation timeline.

    Attributes:
        title: Report title.
        time_col: Name of the timestamp column used in the dataset.
        dataset_start: Earliest timestamp present in the dataset.
        dataset_end: Latest timestamp present in the dataset.
        total_rows: Number of rows in the source dataset.
        total_folds: Number of simulated folds.
        strategy: Split strategy used to build the simulation.
        size_kind: Unit family used by the partition sizes.
        segment_order: Ordered list of segment names.
        segment_colors: Color associated with each segment.
        segment_stats: Aggregate per-segment row statistics across folds.
        folds: Fold-level timeline payload ready for plotting.
    """

    title: str
    time_col: str
    dataset_start: pd.Timestamp
    dataset_end: pd.Timestamp
    total_rows: int
    total_folds: int
    strategy: str
    size_kind: str
    segment_order: List[str]
    segment_colors: Dict[str, str]
    segment_stats: Dict[str, Dict[str, object]]
    folds: List[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        """Return a serializable dictionary representation."""
        return {
            "title": self.title,
            "time_col": self.time_col,
            "dataset_start": self.dataset_start,
            "dataset_end": self.dataset_end,
            "total_rows": self.total_rows,
            "total_folds": self.total_folds,
            "strategy": self.strategy,
            "size_kind": self.size_kind,
            "segment_order": self.segment_order,
            "segment_colors": self.segment_colors,
            "segment_stats": self.segment_stats,
            "folds": self.folds,
        }


@dataclass(frozen=True)
class SimulationSummary:
    """Structured description of a temporal simulation over a dataset.

    Attributes:
        title: Report title.
        time_col: Name of the timestamp column used in the dataset.
        dataset_start: Earliest timestamp present in the dataset.
        dataset_end: Latest timestamp present in the dataset.
        total_rows: Number of rows in the source dataset.
        total_folds: Number of simulated folds.
        strategy: Split strategy used to build the simulation.
        size_kind: Unit family used by the partition sizes.
        folds: Fold-by-fold segment metadata.
        segment_order: Ordered list of segment names.
        chart_data: Plot-ready representation of the same simulation.
    """

    title: str
    time_col: str
    dataset_start: pd.Timestamp
    dataset_end: pd.Timestamp
    total_rows: int
    total_folds: int
    strategy: str
    size_kind: str
    folds: List[Dict[str, object]]
    segment_order: List[str]
    chart_data: SimulationChartData

    def to_dict(self) -> Dict[str, object]:
        """Return a serializable dictionary representation."""
        return {
            "title": self.title,
            "time_col": self.time_col,
            "dataset_start": self.dataset_start,
            "dataset_end": self.dataset_end,
            "total_rows": self.total_rows,
            "total_folds": self.total_folds,
            "strategy": self.strategy,
            "size_kind": self.size_kind,
            "segment_order": self.segment_order,
            "folds": self.folds,
            "chart_data": self.chart_data.to_dict(),
        }

    def to_frame(self) -> pd.DataFrame:
        """Convert fold summaries into a tabular pandas DataFrame."""
        rows = []
        for fold in self.folds:
            row = {
                "fold": fold["fold"],
                "simulation_start": fold["simulation_start"],
                "simulation_end": fold["simulation_end"],
            }
            for segment_name, segment_info in fold["segments"].items():
                row[f"{segment_name}_start"] = segment_info["start"]
                row[f"{segment_name}_end"] = segment_info["end"]
                row[f"{segment_name}_rows"] = segment_info["rows"]
            rows.append(row)
        return pd.DataFrame(rows)


def build_simulation_summary(
    splits: List[TimeSplit],
    frame: pd.DataFrame,
    time_col: object,
    title: str,
) -> SimulationSummary:
    resolved_time_col = frame.columns[time_col] if isinstance(time_col, int) else time_col
    dataset_start = pd.to_datetime(frame[resolved_time_col]).min()
    dataset_end = pd.to_datetime(frame[resolved_time_col]).max()
    time_col_label = str(resolved_time_col)
    segment_order = list(splits[0].segments.keys()) if splits else []
    fold_rows = [
        _build_fold_summary(split=split, segment_order=segment_order) for split in splits
    ]
    strategy = splits[0].metadata.get("strategy", "unknown") if splits else "unknown"
    size_kind = splits[0].metadata.get("size_kind", "unknown") if splits else "unknown"
    chart_data = _build_chart_data(
        title=title,
        time_col=time_col_label,
        dataset_start=dataset_start,
        dataset_end=dataset_end,
        total_rows=len(frame),
        total_folds=len(splits),
        strategy=strategy,
        size_kind=size_kind,
        segment_order=segment_order,
        folds=fold_rows,
    )

    return SimulationSummary(
        title=title,
        time_col=time_col_label,
        dataset_start=dataset_start,
        dataset_end=dataset_end,
        total_rows=len(frame),
        total_folds=len(splits),
        strategy=strategy,
        size_kind=size_kind,
        folds=fold_rows,
        segment_order=segment_order,
        chart_data=chart_data,
    )


def _build_fold_summary(split: TimeSplit, segment_order: List[str]) -> Dict[str, object]:
    segments = {}
    for segment_name in segment_order:
        boundary = split.boundaries[segment_name]
        segments[segment_name] = {
            "start": boundary.start,
            "end": boundary.end,
            "rows": int(len(split.segments[segment_name])),
        }

    return {
        "fold": split.fold,
        "simulation_start": split.boundaries[segment_order[0]].start,
        "simulation_end": split.boundaries[segment_order[-1]].end,
        "segments": segments,
    }


def _build_chart_data(
    title: str,
    time_col: str,
    dataset_start: pd.Timestamp,
    dataset_end: pd.Timestamp,
    total_rows: int,
    total_folds: int,
    strategy: str,
    size_kind: str,
    segment_order: List[str],
    folds: List[Dict[str, object]],
) -> SimulationChartData:
    total_seconds = max((dataset_end - dataset_start).total_seconds(), 1.0)
    segment_colors = {name: SEGMENT_COLORS.get(name, "#64748b") for name in segment_order}
    segment_stats = {}
    chart_folds = []

    for name in segment_order:
        rows = [int(fold["segments"][name]["rows"]) for fold in folds]
        segment_stats[name] = {
            "color": segment_colors[name],
            "total_rows": int(sum(rows)),
            "avg_rows": float(sum(rows) / len(rows)),
            "min_rows": int(min(rows)),
            "max_rows": int(max(rows)),
        }

    for fold in folds:
        chart_segments = {}
        for segment_name, segment_info in fold["segments"].items():
            start_offset = (segment_info["start"] - dataset_start).total_seconds()
            end_offset = (segment_info["end"] - dataset_start).total_seconds()
            chart_segments[segment_name] = {
                **segment_info,
                "offset_pct": round(max((start_offset / total_seconds) * 100, 0.0), 4),
                "width_pct": round(
                    max(((end_offset - start_offset) / total_seconds) * 100, 0.8),
                    4,
                ),
                "color": segment_colors.get(segment_name, "#64748b"),
            }

        chart_folds.append(
            {
                **fold,
                "simulation_span": fold["simulation_end"] - fold["simulation_start"],
                "segments": chart_segments,
            }
        )

    return SimulationChartData(
        title=title,
        time_col=time_col,
        dataset_start=dataset_start,
        dataset_end=dataset_end,
        total_rows=total_rows,
        total_folds=total_folds,
        strategy=strategy,
        size_kind=size_kind,
        segment_order=segment_order,
        segment_colors=segment_colors,
        segment_stats=segment_stats,
        folds=chart_folds,
    )
