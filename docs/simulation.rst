Simulation reporting
====================

Jano can describe a temporal simulation over a concrete dataset and expose it in three complementary ways:

- a structured ``SimulationSummary``,
- a standalone HTML timeline report,
- or plot-ready ``SimulationChartData`` that you can feed into your own Python visualizations.

The entry point is ``describe_simulation()`` on ``TemporalBacktestSplitter``.

Example
-------

.. code-block:: python

   import pandas as pd

   from jano import TemporalBacktestSplitter, TemporalPartitionSpec

   frame = pd.DataFrame(
       {
           "timestamp": pd.date_range("2024-01-01", periods=365, freq="D"),
           "feature": range(365),
           "target": range(100, 465),
       }
   )

   splitter = TemporalBacktestSplitter(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="10D",
           test_size="5D",
       ),
       step="5D",
       strategy="rolling",
   )

   summary = splitter.describe_simulation(
       frame,
       title="Walk-forward simulation",
   )

   print(summary.total_folds)
   print(summary.to_frame().head())

   html = splitter.describe_simulation(frame, output="html")
   chart_data = splitter.describe_simulation(frame, output="chart_data")

   print(html[:120])
   print(chart_data.segment_stats)

What it returns
---------------

By default, ``describe_simulation()`` returns a ``SimulationSummary`` object with:

- dataset span and row count,
- total number of folds,
- fold-by-fold segment boundaries,
- a tabular view through ``to_frame()``,
- a serializable structure through ``to_dict()``,
- plot-ready timeline metadata through ``chart_data``,
- and an HTML report accessible through ``html`` or ``write_html()``.

You can also request a specific output directly:

- ``output="summary"`` returns ``SimulationSummary``,
- ``output="html"`` returns the rendered HTML string,
- ``output="chart_data"`` returns ``SimulationChartData``.

What the HTML shows
-------------------

The generated report draws one line per fold over the full dataset timeline and now includes:

- a richer summary header with dataset span, fold count, strategy and sizing mode,
- segment profile cards with average, minimum and maximum row counts,
- a clearer per-fold timeline with labels and row-count chips.

Each segment is color-coded:

- train in blue,
- validation in orange,
- test in green.

This makes it easier to inspect how a proposed simulation will behave before plugging it into a model or evaluation pipeline.

Using chart data directly
-------------------------

``SimulationChartData`` is designed for downstream plotting without reparsing HTML.

It includes:

- fold-level segment positions in timeline percentages,
- original start and end timestamps,
- row counts per segment,
- segment colors and aggregate statistics.

Example:

.. code-block:: python

   chart_data = splitter.describe_simulation(frame, output="chart_data")

   first_fold = chart_data.folds[0]
   first_train = first_fold["segments"]["train"]

   print(first_train["offset_pct"], first_train["width_pct"])
   print(chart_data.segment_stats["train"])
