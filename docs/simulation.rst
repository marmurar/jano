Simulation reporting
====================

Jano can describe a temporal simulation over a concrete dataset and render it as an HTML timeline.

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
       output_path="simulation.html",
       title="Walk-forward simulation",
   )

   print(summary.total_folds)
   print(summary.to_frame().head())

What it returns
---------------

``describe_simulation()`` returns a ``SimulationSummary`` object with:

- dataset span and row count,
- total number of folds,
- fold-by-fold segment boundaries,
- a tabular view through ``to_frame()``,
- a serializable structure through ``to_dict()``,
- and an HTML report accessible through ``html`` or ``write_html()``.

What the HTML shows
-------------------

The generated report draws one line per fold over the full dataset timeline.

Each segment is color-coded:

- train in blue,
- validation in orange,
- test in green.

The chart also includes row counts for each segment in every fold, which makes it easier to inspect how a proposed simulation will behave before plugging it into a model or evaluation pipeline.
