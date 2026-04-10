Simulation reporting
====================

Jano can describe a temporal simulation over a concrete dataset and expose it in three complementary ways:

- a structured ``SimulationSummary``,
- a standalone HTML timeline report,
- or plot-ready ``SimulationChartData`` that you can feed into your own Python visualizations.

The entry point is ``describe_simulation()`` on ``TemporalBacktestSplitter``.

If you want to run a full simulation without manual fold iteration, the recommended interface is ``TemporalSimulation``.

The same API accepts three tabular inputs:

- ``pandas.DataFrame``
- ``numpy.ndarray`` using integer column references such as ``time_col=0``
- ``polars.DataFrame`` when the optional Polars dependency is installed

That means the temporal configuration stays the same even if the upstream data source changes. The only thing that changes is how you reference columns:

- by name for pandas and Polars
- by integer position for NumPy arrays

Example
-------

.. code-block:: python

   import pandas as pd

   from jano import TemporalPartitionSpec, TemporalSimulation

   frame = pd.DataFrame(
       {
           "timestamp": pd.date_range("2024-01-01", periods=365, freq="D"),
           "feature": range(365),
           "target": range(100, 465),
       }
   )

   simulation = TemporalSimulation(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="10D",
           test_size="5D",
       ),
       step="5D",
       strategy="rolling",
   )

   result = simulation.run(
       frame,
       title="Walk-forward simulation",
   )

   print(result.total_folds)
   print(result.to_frame().head())

   html = result.html
   chart_data = result.chart_data

   print(html[:120])
   print(chart_data.segment_stats)

You can anchor the simulation to a specific point in time and cap the number of folds:

.. code-block:: python

   simulation = TemporalSimulation(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="15D",
           test_size="4D",
       ),
       step="1D",
       strategy="rolling",
       start_at="2025-09-01",
       max_folds=15,
   )

   result = simulation.run(frame, title="15 daily retraining iterations")

``TemporalSimulation`` also accepts ``end_at`` if you want to constrain the simulation to a bounded time window before folds are generated.

If your source data is a NumPy array, reference the time column by integer position:

.. code-block:: python

   import numpy as np

   values = np.array(
       [
           ["2025-09-01", 0.2, 1],
           ["2025-09-02", 0.4, 0],
           ["2025-09-03", 0.1, 1],
           ["2025-09-04", 0.3, 0],
       ],
       dtype=object,
   )

   simulation = TemporalSimulation(
       time_col=0,
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="2D",
           test_size="1D",
       ),
       step="1D",
       strategy="single",
   )

Low-level manual control
------------------------

When you need direct control over folds or want to integrate with an external training loop, use ``TemporalBacktestSplitter`` directly.

.. code-block:: python

   from jano import TemporalBacktestSplitter, TemporalPartitionSpec

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

   for split in splitter.iter_splits(frame):
       print(split.summary())

Temporal semantics and leakage control
--------------------------------------

When a single timestamp column is not enough, you can pass a ``TemporalSemanticsSpec`` instead of a plain ``time_col`` string.

This lets you separate:

- the timeline used for reporting and global simulation bounds,
- the internal ordering column,
- and the timestamp column used to decide whether each segment is eligible.

That matters in production-like datasets where availability and event time differ. For example, a flight may depart on one day but only become usable for supervised training when its arrival is known.

.. code-block:: python

   from jano import TemporalBacktestSplitter, TemporalPartitionSpec, TemporalSemanticsSpec

   splitter = TemporalBacktestSplitter(
       time_col=TemporalSemanticsSpec(
           timeline_col="departured_at",
           segment_time_cols={
               "train": "arrived_at",
               "test": "departured_at",
           },
       ),
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="14D",
           test_size="3D",
           gap_before_train="1D",
           gap_before_test="1D",
           gap_after_test="2D",
       ),
       step="1D",
       strategy="rolling",
   )

In that configuration, the simulation is reported over the ``departured_at`` timeline, while the train set only includes rows whose ``arrived_at`` falls inside the train window. This prevents rows from entering train before the target would actually be available in production.

Simple HTML preview
-------------------

Below is a compact mock of the kind of timeline the generated HTML report shows.

.. raw:: html

   <div class="simulation-preview">
     <div class="preview-top">
       <div class="preview-kicker">Simulation report</div>
       <div class="preview-title">Walk-forward simulation</div>
       <div class="preview-meta">
         <span class="preview-chip">Rows: 365</span>
         <span class="preview-chip">Folds: 6</span>
         <span class="preview-chip">Strategy: rolling</span>
       </div>
     </div>
     <div class="preview-body">
       <div class="preview-row">
         <div class="preview-label">Fold 0</div>
         <div class="preview-track">
           <span class="preview-segment train" style="left: 0%; width: 44%;"></span>
           <span class="preview-segment validation" style="left: 48%; width: 12%;"></span>
           <span class="preview-segment test" style="left: 64%; width: 16%;"></span>
         </div>
       </div>
       <div class="preview-row">
         <div class="preview-label">Fold 1</div>
         <div class="preview-track">
           <span class="preview-segment train" style="left: 8%; width: 44%;"></span>
           <span class="preview-segment validation" style="left: 56%; width: 12%;"></span>
           <span class="preview-segment test" style="left: 72%; width: 16%;"></span>
         </div>
       </div>
       <div class="preview-row">
         <div class="preview-label">Fold 2</div>
         <div class="preview-track">
           <span class="preview-segment train" style="left: 16%; width: 44%;"></span>
           <span class="preview-segment validation" style="left: 64%; width: 12%;"></span>
           <span class="preview-segment test" style="left: 80%; width: 14%;"></span>
         </div>
       </div>
     </div>
   </div>

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
