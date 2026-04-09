from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .splits import TimeSplit


@dataclass(frozen=True)
class SimulationSummary:
    """Structured description of a temporal simulation over a dataset."""

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
    html: str = field(repr=False)

    def to_dict(self) -> Dict[str, object]:
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
        }

    def to_frame(self) -> pd.DataFrame:
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
    html = _render_simulation_html(
        title=title,
        dataset_start=dataset_start,
        dataset_end=dataset_end,
        total_rows=len(frame),
        total_folds=len(splits),
        segment_order=segment_order,
        folds=fold_rows,
    )

    return SimulationSummary(
        title=title,
        time_col=time_col,
        dataset_start=dataset_start,
        dataset_end=dataset_end,
        total_rows=len(frame),
        total_folds=len(splits),
        strategy=splits[0].metadata.get("strategy", "unknown") if splits else "unknown",
        size_kind=splits[0].metadata.get("size_kind", "unknown") if splits else "unknown",
        folds=fold_rows,
        segment_order=segment_order,
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


def _render_simulation_html(
    title: str,
    dataset_start: pd.Timestamp,
    dataset_end: pd.Timestamp,
    total_rows: int,
    total_folds: int,
    segment_order: List[str],
    folds: List[Dict[str, object]],
) -> str:
    colors = {
        "train": "#1f77b4",
        "validation": "#ff7f0e",
        "test": "#2ca02c",
    }
    total_seconds = max((dataset_end - dataset_start).total_seconds(), 1.0)

    legend = "".join(
        (
            f'<div class="legend-item"><span class="legend-chip" '
            f'style="background:{colors.get(name, "#6b7280")}"></span>{escape(name.title())}</div>'
        )
        for name in segment_order
    )

    rows_html = []
    for fold in folds:
        bars = []
        for segment_name, segment_info in fold["segments"].items():
            start_offset = (segment_info["start"] - dataset_start).total_seconds()
            end_offset = (segment_info["end"] - dataset_start).total_seconds()
            left = max((start_offset / total_seconds) * 100, 0)
            width = max(((end_offset - start_offset) / total_seconds) * 100, 0.8)
            tooltip = (
                f"{segment_name}: {segment_info['start']} -> {segment_info['end']} "
                f"({segment_info['rows']} rows)"
            )
            bars.append(
                f'<div class="segment segment-{escape(segment_name)}" '
                f'style="left:{left:.4f}%;width:{width:.4f}%;'
                f'background:{colors.get(segment_name, "#6b7280")}" '
                f'title="{escape(tooltip)}"></div>'
            )

        metrics = " · ".join(
            f"{escape(name)}: {fold['segments'][name]['rows']} rows" for name in segment_order
        )
        rows_html.append(
            "<div class=\"fold-row\">"
            f"<div class=\"fold-label\">Fold {fold['fold']}</div>"
            f"<div class=\"fold-track\">{''.join(bars)}</div>"
            f"<div class=\"fold-metrics\">{escape(metrics)}</div>"
            "</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{escape(title)}</title>
    <style>
      :root {{
        --bg: #f8fafc;
        --panel: #ffffff;
        --text: #0f172a;
        --muted: #475569;
        --track: #e2e8f0;
        --border: #cbd5e1;
      }}
      body {{
        margin: 0;
        font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: linear-gradient(180deg, #eff6ff 0%, var(--bg) 100%);
        color: var(--text);
      }}
      .page {{
        max-width: 1120px;
        margin: 0 auto;
        padding: 32px 24px 48px;
      }}
      .card {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
      }}
      h1 {{
        margin: 0 0 8px;
        font-size: 2rem;
      }}
      p {{
        margin: 0;
        color: var(--muted);
      }}
      .meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 20px 0 24px;
      }}
      .meta-item, .legend-item {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        background: #f8fafc;
        border: 1px solid var(--border);
        font-size: 0.92rem;
      }}
      .legend {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 20px;
      }}
      .legend-chip {{
        width: 12px;
        height: 12px;
        border-radius: 999px;
      }}
      .axis {{
        display: flex;
        justify-content: space-between;
        color: var(--muted);
        font-size: 0.88rem;
        margin-bottom: 8px;
      }}
      .fold-row {{
        display: grid;
        grid-template-columns: 80px minmax(240px, 1fr) 240px;
        gap: 14px;
        align-items: center;
        padding: 10px 0;
      }}
      .fold-label {{
        font-weight: 600;
      }}
      .fold-track {{
        position: relative;
        height: 18px;
        background: var(--track);
        border-radius: 999px;
        overflow: hidden;
      }}
      .segment {{
        position: absolute;
        top: 0;
        bottom: 0;
        border-radius: 999px;
      }}
      .fold-metrics {{
        color: var(--muted);
        font-size: 0.9rem;
      }}
      @media (max-width: 860px) {{
        .fold-row {{
          grid-template-columns: 1fr;
          gap: 8px;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="page">
      <section class="card">
        <h1>{escape(title)}</h1>
        <p>Temporal partition simulation overview for the configured backtest.</p>
        <div class="meta">
          <div class="meta-item">Dataset span: {escape(str(dataset_start))} → {escape(str(dataset_end))}</div>
          <div class="meta-item">Rows: {total_rows}</div>
          <div class="meta-item">Folds: {total_folds}</div>
        </div>
        <div class="legend">{legend}</div>
        <div class="axis">
          <span>{escape(str(dataset_start))}</span>
          <span>{escape(str(dataset_end))}</span>
        </div>
        {''.join(rows_html)}
      </section>
    </main>
  </body>
</html>
"""
