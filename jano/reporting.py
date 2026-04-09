from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from pathlib import Path
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
        html: Rendered HTML report.
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
    html: str = field(repr=False)

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

    def write_html(self, path: str | Path) -> Path:
        """Write the rendered HTML report to ``path``."""
        destination = Path(path)
        destination.write_text(self.html, encoding="utf-8")
        return destination


def build_simulation_summary(
    splits: List[TimeSplit],
    frame: pd.DataFrame,
    time_col: str,
    title: str,
) -> SimulationSummary:
    dataset_start = pd.to_datetime(frame[time_col]).min()
    dataset_end = pd.to_datetime(frame[time_col]).max()
    segment_order = list(splits[0].segments.keys()) if splits else []
    fold_rows = [
        _build_fold_summary(split=split, segment_order=segment_order) for split in splits
    ]
    strategy = splits[0].metadata.get("strategy", "unknown") if splits else "unknown"
    size_kind = splits[0].metadata.get("size_kind", "unknown") if splits else "unknown"
    chart_data = _build_chart_data(
        title=title,
        time_col=time_col,
        dataset_start=dataset_start,
        dataset_end=dataset_end,
        total_rows=len(frame),
        total_folds=len(splits),
        strategy=strategy,
        size_kind=size_kind,
        segment_order=segment_order,
        folds=fold_rows,
    )
    html = _render_simulation_html(chart_data)

    return SimulationSummary(
        title=title,
        time_col=time_col,
        dataset_start=dataset_start,
        dataset_end=dataset_end,
        total_rows=len(frame),
        total_folds=len(splits),
        strategy=strategy,
        size_kind=size_kind,
        folds=fold_rows,
        segment_order=segment_order,
        chart_data=chart_data,
        html=html,
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


def _render_simulation_html(chart_data: SimulationChartData) -> str:
    legend = "".join(
        (
            f'<div class="legend-item"><span class="legend-chip" '
            f'style="background:{chart_data.segment_colors.get(name, "#64748b")}"></span>'
            f"{escape(name.title())}</div>"
        )
        for name in chart_data.segment_order
    )
    stat_cards = "".join(
        (
            '<div class="stat-card">'
            f'<div class="stat-label">{escape(name.title())}</div>'
            f'<div class="stat-value">{stats["avg_rows"]:.1f}</div>'
            f'<div class="stat-meta">avg rows per fold · min {stats["min_rows"]} · '
            f'max {stats["max_rows"]}</div>'
            "</div>"
        )
        for name, stats in chart_data.segment_stats.items()
    )

    rows_html = []
    for fold in chart_data.folds:
        bars = []
        chips = []
        for segment_name, segment_info in fold["segments"].items():
            tooltip = (
                f"{segment_name}: {segment_info['start']} -> {segment_info['end']} "
                f"({segment_info['rows']} rows)"
            )
            bars.append(
                f'<div class="segment segment-{escape(segment_name)}" '
                f'style="left:{segment_info["offset_pct"]:.4f}%;'
                f'width:{segment_info["width_pct"]:.4f}%;'
                f'background:{segment_info["color"]}" '
                f'title="{escape(tooltip)}">'
                f'<span>{escape(segment_name.title())}</span>'
                "</div>"
            )
            chips.append(
                '<div class="metric-chip">'
                f'<span class="metric-chip-dot" style="background:{segment_info["color"]}"></span>'
                f'{escape(segment_name.title())}: {segment_info["rows"]} rows'
                "</div>"
            )

        rows_html.append(
            '<section class="fold-card">'
            '<div class="fold-header">'
            f'<div><h3>Fold {fold["fold"]}</h3><p>{escape(str(fold["simulation_start"]))} '
            f'to {escape(str(fold["simulation_end"]))}</p></div>'
            f'<div class="fold-span">{escape(str(fold["simulation_span"]))}</div>'
            "</div>"
            '<div class="fold-track">'
            '<div class="track-shell">'
            f'{"".join(bars)}'
            "</div>"
            "</div>"
            f'<div class="fold-metrics">{"".join(chips)}</div>'
            "</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{escape(chart_data.title)}</title>
    <style>
      :root {{
        --bg-top: #e0f2fe;
        --bg-bottom: #f8fafc;
        --panel: rgba(255, 255, 255, 0.92);
        --panel-strong: #ffffff;
        --text: #0f172a;
        --muted: #475569;
        --soft: #64748b;
        --track: #dbeafe;
        --track-border: #bfdbfe;
        --border: rgba(148, 163, 184, 0.28);
        --shadow: 0 24px 80px rgba(15, 23, 42, 0.12);
      }}
      * {{
        box-sizing: border-box;
      }}
      body {{
        margin: 0;
        font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
        background:
          radial-gradient(circle at top left, rgba(29, 78, 216, 0.14), transparent 32%),
          radial-gradient(circle at top right, rgba(5, 150, 105, 0.12), transparent 24%),
          linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 46%);
        color: var(--text);
      }}
      .page {{
        max-width: 1180px;
        margin: 0 auto;
        padding: 40px 24px 56px;
      }}
      .hero {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(239, 246, 255, 0.92));
        border: 1px solid var(--border);
        border-radius: 28px;
        padding: 32px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
      }}
      h1 {{
        margin: 0 0 8px;
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1;
        letter-spacing: -0.04em;
      }}
      p {{
        margin: 0;
        color: var(--muted);
      }}
      code {{
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      }}
      .hero-subtitle {{
        max-width: 760px;
        font-size: 1rem;
        line-height: 1.6;
      }}
      .meta-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 14px;
        margin-top: 26px;
      }}
      .meta-item {{
        padding: 14px 16px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid var(--border);
      }}
      .meta-label {{
        color: var(--soft);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      .meta-value {{
        margin-top: 6px;
        font-size: 1.05rem;
        font-weight: 600;
      }}
      .section {{
        margin-top: 26px;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 24px;
        box-shadow: var(--shadow);
      }}
      .section-header {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: end;
        margin-bottom: 18px;
      }}
      .section-header h2 {{
        margin: 0;
        font-size: 1.35rem;
      }}
      .legend {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }}
      .legend-item, .metric-chip {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(248, 250, 252, 0.9);
        border: 1px solid var(--border);
        font-size: 0.92rem;
      }}
      .legend-chip, .metric-chip-dot {{
        width: 11px;
        height: 11px;
        border-radius: 999px;
        flex: 0 0 auto;
      }}
      .axis {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 18px;
        color: var(--muted);
        font-size: 0.92rem;
      }}
      .stat-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
      }}
      .stat-card {{
        padding: 18px;
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 252, 0.95));
        border: 1px solid var(--border);
      }}
      .stat-label {{
        color: var(--soft);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.76rem;
      }}
      .stat-value {{
        margin-top: 8px;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.04em;
      }}
      .stat-meta {{
        margin-top: 6px;
        color: var(--muted);
        line-height: 1.5;
      }}
      .timeline {{
        display: grid;
        gap: 14px;
      }}
      .fold-card {{
        padding: 18px;
        border-radius: 20px;
        background: var(--panel-strong);
        border: 1px solid var(--border);
      }}
      .fold-header {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: baseline;
        margin-bottom: 14px;
      }}
      .fold-header h3 {{
        margin: 0 0 4px;
        font-size: 1.08rem;
      }}
      .fold-span {{
        color: var(--muted);
        font-size: 0.92rem;
        white-space: nowrap;
      }}
      .track-shell {{
        position: relative;
        height: 34px;
        border-radius: 999px;
        background:
          linear-gradient(90deg, rgba(191, 219, 254, 0.55), rgba(219, 234, 254, 0.8)),
          var(--track);
        border: 1px solid var(--track-border);
        overflow: hidden;
      }}
      .segment {{
        position: absolute;
        top: 4px;
        bottom: 4px;
        border-radius: 999px;
        min-width: 6px;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.38);
      }}
      .segment span {{
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 10px;
        color: white;
        font-size: 0.78rem;
        font-weight: 600;
        white-space: nowrap;
        mix-blend-mode: screen;
      }}
      .fold-metrics {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 14px;
      }}
      @media (max-width: 720px) {{
        .page {{
          padding: 24px 14px 40px;
        }}
        .hero,
        .section {{
          padding: 18px;
          border-radius: 20px;
        }}
        .fold-header,
        .section-header {{
          flex-direction: column;
          align-items: start;
        }}
        .track-shell {{
          height: 28px;
        }}
        .segment span {{
          font-size: 0.7rem;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="page">
      <section class="hero">
        <p class="meta-label">Temporal partition simulation overview</p>
        <h1>{escape(chart_data.title)}</h1>
        <p class="hero-subtitle">
          Fold-by-fold visualization for a {escape(chart_data.strategy)} strategy over
          {chart_data.total_rows} rows using the <code>{escape(chart_data.time_col)}</code> timeline.
        </p>
        <div class="meta-grid">
          <div class="meta-item"><div class="meta-label">Dataset window</div><div class="meta-value">{escape(str(chart_data.dataset_start))} to {escape(str(chart_data.dataset_end))}</div></div>
          <div class="meta-item"><div class="meta-label">Total folds</div><div class="meta-value">{chart_data.total_folds}</div></div>
          <div class="meta-item"><div class="meta-label">Strategy</div><div class="meta-value">{escape(chart_data.strategy.title())}</div></div>
          <div class="meta-item"><div class="meta-label">Sizing mode</div><div class="meta-value">{escape(chart_data.size_kind.title())}</div></div>
        </div>
      </section>
      <section class="section">
        <div class="section-header">
          <div>
            <h2>Segment profile</h2>
            <p>Average and range of rows allocated to each segment across all folds.</p>
          </div>
          <div class="legend">{legend}</div>
        </div>
        <div class="stat-grid">
          {stat_cards}
        </div>
      </section>
      <section class="section">
        <div class="section-header">
          <div>
            <h2>Timeline</h2>
            <p>Each fold is plotted over the full dataset span so you can inspect overlap, drift and coverage at a glance.</p>
          </div>
        </div>
        <div class="axis">
          <span>{escape(str(chart_data.dataset_start))}</span>
          <span>{escape(str(chart_data.dataset_end))}</span>
        </div>
        <div class="timeline">
          {''.join(rows_html)}
        </div>
      </section>
    </main>
  </body>
</html>
"""
