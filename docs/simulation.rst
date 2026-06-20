Simulation reporting
====================

Jano can describe a temporal simulation over a concrete dataset and expose it in two complementary ways:

- a structured ``SimulationSummary``,
- or plot-ready ``SimulationChartData`` that you can feed into your own Python visualizations.

The entry point is ``describe_simulation()`` on ``TemporalBacktestSplitter``.

If you want to run a full simulation without manual fold iteration, the recommended interface is ``WalkForwardPolicy``.

The overall workflow is deliberately layered:

- use high-level classes when the question is already encapsulated,
- inspect or prune iterations through ``plan()`` when needed,
- and fall back to manual fold iteration when you want to compose everything yourself.

The same API accepts three tabular inputs:

- ``pandas.DataFrame``
- ``numpy.ndarray`` using integer column references such as ``time_col=0``
- ``polars.DataFrame`` when the optional Polars dependency is installed

That means the temporal configuration stays the same even if the upstream data source changes. The only thing that changes is how you reference columns:

- by name for pandas and Polars
- by integer position for NumPy arrays

Example
-------

.. container:: example-block

   pandas.DataFrame

.. code-block:: python

   import pandas as pd

   from jano import TemporalPartitionSpec, WalkForwardPolicy

   frame = pd.DataFrame(
       {
           "timestamp": pd.date_range("2024-01-01", periods=365, freq="D"),
           "feature": range(365),
           "target": range(100, 465),
       }
   )

   policy = WalkForwardPolicy(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="10D",
           test_size="5D",
       ),
       step="5D",
       strategy="rolling",
   )

   result = policy.run(
       frame,
       title="Walk-forward simulation",
   )

   print(result.total_folds)
   print(result.to_frame().head())

   chart_data = result.chart_data

   print(chart_data.segment_stats)

If you want to inspect the simulation before materializing folds, use ``plan()``:

.. container:: example-block

   Planned simulation

.. code-block:: python

   plan = policy.plan(frame, title="Walk-forward plan")
   print(plan.total_folds)
   print(plan.to_frame().head())

   filtered = plan.exclude_windows(
       train=[("2025-12-20", "2026-01-05")],
   ).select_from_iteration(5)

   result = filtered.materialize()

The plan frame includes the iteration index plus segment boundaries and row counts, so you can inspect the structure first and only materialize the folds you actually want.

You can anchor the simulation to a specific point in time and cap the number of folds:

.. container:: example-block

   Anchored simulation

.. code-block:: python

   policy = WalkForwardPolicy(
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

   result = policy.run(frame, title="15 daily retraining iterations")

``WalkForwardPolicy`` also accepts ``end_at`` if you want to constrain the simulation to a bounded time window before folds are generated.

Running a model with retrain policies
-------------------------------------

When you do not want to write a manual ``for train_idx, test_idx in splitter`` loop,
use ``WalkForwardRunner`` on top of the temporal workflow. The runner keeps Jano's
responsibilities separated:

- ``WalkForwardPolicy`` still defines fold geometry
- ``WalkForwardRunner`` executes the estimator over those folds
- a retrain policy decides whether the estimator should be refit before each fold

.. code-block:: python

   import numpy as np

   from jano import TemporalPartitionSpec, WalkForwardPolicy, WalkForwardRunner

   def mae(y_true, y_pred):
       return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

   def rmse(y_true, y_pred):
       return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

   policy = WalkForwardPolicy(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="30D",
           test_size="7D",
       ),
       step="7D",
       strategy="rolling",
   )

   runner = WalkForwardRunner(
       model=model,
       target_col="target",
       feature_cols=["feature_a", "feature_b"],
       retrain="always",
       metrics={"mae": mae, "rmse": rmse},
   )

   run = runner.run(policy, frame)
   print(run.to_frame().head())
   print(run.summary())
   print(run.metric_trajectory().head())
   print(run.retrain_events())

Runner results are intentionally data-first. Jano does not need to own the final
dashboard layer; it exposes structured evidence that notebooks, agents,
presentation tools or applications can visualize in their own style:

- ``run.fold_summary()`` returns temporal fold geometry and retraining metadata.
- ``run.metric_trajectory()`` returns metrics in long format, ready for plotting.
- ``run.retrain_events()`` returns only folds where the estimator was refit.
- ``run.predictions_frame()`` returns row-level test predictions.
- ``run.report_data()`` and ``run.to_dict()`` return structured dictionaries for
  external reporting layers.

The shorthand retrain modes are:

- ``retrain="always"`` or ``retrain=True`` to refit on every fold
- ``retrain="never"`` or ``retrain=False`` to fit once and reuse the same model
- ``retrain="periodic"`` plus ``retrain_interval=K`` to refit every ``K`` folds

Simulation campaigns
--------------------

If you want to compare multiple temporal geometries over the same dataset,
bundle them into a ``SimulationCampaign`` and run the variants in parallel.
This is useful for sensitivity analysis, fold-count sweeps and policy
calibration. Each variant remains an independent simulation; Jano only
parallelizes the execution and aggregates the results.

.. code-block:: python

   from jano import SimulationCampaign, SimulationVariant, TemporalPartitionSpec, TemporalSimulation

   campaign = SimulationCampaign(
       [
           SimulationVariant(
               name="daily",
               simulation=TemporalSimulation(
                   time_col="timestamp",
                   partition=TemporalPartitionSpec(
                       layout="train_test",
                       train_size="30D",
                       test_size="1D",
                   ),
                   step="1D",
               ),
           ),
           SimulationVariant(
               name="weekly",
               simulation=TemporalSimulation(
                   time_col="timestamp",
                   partition=TemporalPartitionSpec(
                       layout="train_test",
                       train_size="42D",
                       test_size="7D",
                   ),
                   step="7D",
               ),
           ),
       ]
   )

   batch = campaign.run(frame, max_workers=2)
   print(batch.to_frame())
   print(batch.result_for("daily").summary.to_frame().head())

Evaluation profiles
-------------------

``EvaluationProfile`` separates how a temporal run is measured from when the
runner should retrain the estimator. Jano does not implement metric formulas;
the main contract is that users pass the metric or loss function that matches
their problem.

.. code-block:: python

   import numpy as np

   from jano import EvaluationProfile, FunctionRetrainPolicy, WalkForwardRunner

   def daily_cost(y_true, y_pred):
       return float(np.mean(np.abs(y_true - y_pred)))

   def retrain_rule(context):
       if context.history.empty:
           return True
       latest = context.history["daily_cost"].iloc[-1]
       limit = limit_for_date(context.split.boundaries["test"].end)
       return latest > limit

   runner = WalkForwardRunner(
       model=model,
       target_col="target",
       feature_cols=["feature_a", "feature_b"],
       evaluation=EvaluationProfile(
           metrics={"daily_cost": daily_cost},
           metric_directions={"daily_cost": "min"},
           primary_metric="daily_cost",
       ),
       retrain_policy=FunctionRetrainPolicy(retrain_rule),
   )

The profile tells Jano which metrics exist, whether each one should be minimized
or maximized and which metric is the primary operational signal. ``FunctionRetrainPolicy``
then gives the user full control over the retrain decision, including dynamic
thresholds, date-dependent losses or business rules.

Convenience profiles are available when the problem type helps make intent clear.
They do not add metric formulas; they keep user-provided metrics grouped by
problem style:

- ``RegressionProfile`` labels regression-style losses supplied by the user.
- ``ClassificationProfile`` labels classification-style scores supplied by the user.
- ``OrdinalClassificationProfile`` is intended for ordered classes with custom costs.
- ``RankingProfile`` is intended for ranking or retrieval metrics supplied by the user.

You can also pass an explicit retrain policy object:

.. code-block:: python

   from jano import DriftBasedRetrain, WalkForwardRunner

   runner = WalkForwardRunner(
       model=model,
       target_col="target",
       retrain_policy=DriftBasedRetrain(
           metric="mae",
           threshold=0.05,
           baseline="last_retrain",
       ),
       metrics={"mae": mae},
   )

``DriftBasedRetrain`` uses previously observed fold metrics to decide whether the next
fold should trigger a retrain. That makes it useful as a first operational benchmark,
without forcing drift logic into the splitter itself.

When ``DriftBasedRetrain`` is created without an explicit metric, it uses the
``primary_metric`` from the evaluation profile.

Running temporal systems with update policies
---------------------------------------------

Not every temporal system updates itself through ``fit()`` and ``predict()``.
RAG pipelines, prompt configurations and fine-tuning jobs often behave more like
"update state, then evaluate the current state on the next window".

``TemporalSystemRunner`` covers that case without changing Jano's temporal core.
It keeps the same fold geometry but replaces the estimator contract with an
``UpdateableSystem`` protocol:

- ``update(train_frame)`` refreshes the system state for the current train window
- ``evaluate(state, test_frame)`` returns user-defined metrics for the next test window

.. code-block:: python

   import numpy as np
   import pandas as pd

   from jano import (
       PeriodicRetrain,
       SystemEvaluationResult,
       SystemUpdateResult,
       TemporalPartitionSpec,
       TemporalSystemRunner,
       WalkForwardPolicy,
   )

   class MeanTargetSystem:
       def update(self, train_frame: pd.DataFrame):
           mean_target = float(train_frame["target"].mean())
           return SystemUpdateResult(
               state=mean_target,
               metadata={"train_target_mean": mean_target},
           )

       def evaluate(self, state, test_frame: pd.DataFrame):
           predictions = np.repeat(float(state), len(test_frame))
           mae = float(np.mean(np.abs(test_frame["target"] - predictions)))
           return SystemEvaluationResult(
               metrics={"mae": mae},
               metadata={"prediction_mean": float(state)},
           )

   policy = WalkForwardPolicy(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="30D",
           test_size="7D",
       ),
       step="7D",
       strategy="rolling",
   )

   runner = TemporalSystemRunner(
       system=MeanTargetSystem(),
       update_policy=PeriodicRetrain(2),
       metric_directions={"mae": "min"},
       primary_metric="mae",
   )

   run = runner.run(policy, frame)
   print(run.to_frame().head())
   print(run.metric_trajectory().head())
   print(run.update_events())

The important distinction is conceptual rather than technical: the update step can
mean retraining a model, rebuilding a retrieval index, refreshing a prompt set or
re-running a fine-tuning job. Jano still owns temporal partitioning and policy
simulation; the system object owns the operational update logic.

Built-in scenarios
------------------

Some operational questions are common enough to deserve a ready-to-use workflow,
but not generic enough to become runner behavior. Jano exposes those workflows as
``jano.scenarios``. Scenarios are built on top of the core primitives; they do not
change ``WalkForwardRunner``.

Prediction bands by fold
~~~~~~~~~~~~~~~~~~~~~~~~

``estimate_prediction_band_by_fold`` answers:

"For each temporal fold, what prediction band does my own uncertainty method
produce for the future test window?"

Jano does not implement cross-validation, bootstrap, conformal prediction or
confidence-interval formulas in this scenario. The user provides a
``band_estimator`` object or callable. That object receives the fold context and
returns ``lower`` and ``upper`` arrays for the current test fold.

.. code-block:: python

   import numpy as np

   from jano import estimate_prediction_band_by_fold

   def mae(y_true, y_pred):
       return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

   class FixedWidthBand:
       def estimate(self, context):
           return {
               "lower": context.predictions - 5.0,
               "upper": context.predictions + 5.0,
               "artifacts": {"method": "fixed_width"},
           }

   result = estimate_prediction_band_by_fold(
       frame,
       estimator=model,
       band_estimator=FixedWidthBand(),
       time_col="timestamp",
       target_col="target",
       feature_cols=["feature_a", "feature_b"],
       train_size="90D",
       test_size="7D",
       step="7D",
       strategy="rolling",
       metrics={"mae": mae},
   )

   print(result.to_frame().head())
   print(result.predictions_frame().head())
   print(result.band_summary())

The result exposes fold-level metrics and band summaries through
``to_frame()``, row-level lower and upper bounds through ``predictions_frame()``,
and user-owned artifacts through ``artifacts_frame()``. A real ``band_estimator``
can wrap scikit-learn ``KFold``, a custom resampling method, conformal prediction
or any other technique.

Calendar-aligned duration windows
---------------------------------

By default, duration windows start from the first observed timestamp. If the first row is
``2024-01-01 05:21`` and ``train_size="7D"``, the first train window ends at
``2024-01-08 05:21``.

Sometimes that is not the desired behavior. In operational datasets, you may want whole
calendar days instead: train through Jan 7 and test from Jan 8.

Use ``calendar_frequency="D"`` in ``TemporalPartitionSpec`` for that:

.. code-block:: python

   simulation = WalkForwardPolicy(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="7D",
           test_size="1D",
           calendar_frequency="D",
       ),
       step="1D",
       strategy="rolling",
   )

Jano uses closed-open boundaries: ``[start, end)``. A train boundary ending at
``2024-01-08 00:00:00`` means the train segment contains rows before Jan 8, while the
test segment can start exactly at Jan 8.

If your source data is a NumPy array, reference the time column by integer position:

.. container:: example-block

   NumPy input

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

If your source data is a Polars frame, the same configuration works with named columns:

.. container:: example-block

   polars.DataFrame

.. code-block:: python

   import polars as pl

   frame = pl.DataFrame(
       {
           "timestamp": ["2025-09-01", "2025-09-02", "2025-09-03", "2025-09-04"],
           "feature": [0.2, 0.4, 0.1, 0.3],
           "target": [1, 0, 1, 0],
       }
   ).with_columns(pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d"))

   simulation = TemporalSimulation(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="2D",
           test_size="1D",
       ),
       step="1D",
       strategy="single",
   )

   result = simulation.run(frame)

Choosing the partition engine
-----------------------------

All high-level simulation APIs accept ``engine``. The default, ``engine="auto"``, chooses
the internal representation used to compute temporal boundaries and row indices:

.. code-block:: python

   simulation = TemporalSimulation(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="7D",
           test_size="1D",
       ),
       step="1D",
       strategy="rolling",
       engine="auto",
   )

   result = simulation.run(frame)
   print(result.engine_metadata.to_dict())

``engine="auto"`` keeps Polars and NumPy inputs native for planning when safe. Use
``engine="pandas"`` to force the stable pandas path, or ``engine="polars"`` /
``engine="numpy"`` when you want to force a specific partition engine.

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

The same splitter can also precompute the full partition geometry:

.. code-block:: python

   plan = splitter.plan(frame)
   print(plan.to_frame()[["iteration", "train_start", "train_end", "test_start", "test_end"]])

This is the fully manual mode. It is the right place when you want to compose the full process yourself: partition layouts, temporal gaps, special date exclusions, feature lookback windows, model training loops or any custom evaluation logic that should not be hidden behind a higher-level helper.

Fixed cutoff studies
--------------------

These are special use cases on top of the basic simulation workflow.

Jano now exposes them as dedicated temporal policies instead of leaving them as manual recipes.

.. container:: example-block

   Fixed test, expanding train

.. code-block:: python

   from jano import TrainHistoryPolicy

   policy = TrainHistoryPolicy(
       "timestamp",
       cutoff="2025-09-15",
       train_sizes=["7D", "14D", "21D", "28D"],
       test_size="4D",
   )

   result = policy.evaluate(
       frame,
       model=model,
       target_col="target",
       feature_cols=["feature_1", "feature_2"],
       metrics={"mae": mae, "rmse": rmse},
   )

   print(result.to_frame()[["train_size", "mae", "rmse"]])
   print(result.find_optimal_train_size(metric="rmse", tolerance=0.01))

This keeps the same test slice fixed while train expands toward the past. It is the right shape for questions about training-history sufficiency, data efficiency and whether more historical data is really worth carrying into production training jobs.

The opposite special case is also common: keep train fixed, move test forward day by day and measure for how long a model or rule keeps its performance without retraining. That pattern answers a different operational question:

- how many days can this object stay in production before it degrades?
- how quickly does performance decay after the training cutoff?
- how often should retraining happen?

In other words:

- fixed test + growing train helps study training-history sufficiency
- fixed train + moving test helps study performance durability after deployment

.. container:: example-block

   Fixed train, moving test

.. code-block:: python

   from jano import DriftMonitoringPolicy

   policy = DriftMonitoringPolicy(
       "timestamp",
       cutoff="2025-09-15",
       train_size="30D",
       test_size="3D",
       step="1D",
       max_windows=10,
   )

   result = policy.evaluate(
       frame,
       model=model,
       target_col="target",
       feature_cols=["feature_1", "feature_2"],
       metrics={"mae": mae, "rmse": rmse},
   )

   print(result.to_frame()[["window", "test_start", "rmse"]])
   print(result.find_drift_onset(metric="rmse", threshold=0.15, baseline="first"))

This keeps the same training history fixed while the evaluation window moves forward over time. It is the right shape when you want to estimate how long an object can stay in production before retraining becomes necessary.

Composed policy: optimize train history inside each walk-forward iteration
--------------------------------------------------------------------------

When the question is more complex, you can still stay on the recommended surface.

``RollingTrainHistoryPolicy`` runs an outer walk-forward loop and, inside each iteration,
chooses the smallest train window that stays within tolerance of the best score for that
iteration's fixed test slice.

.. container:: example-block

   Walk-forward with inner train-history optimization

.. code-block:: python

   from jano import RollingTrainHistoryPolicy, TemporalPartitionSpec

   policy = RollingTrainHistoryPolicy(
       "timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="30D",
           test_size="1D",
       ),
       step="1D",
       strategy="rolling",
       max_folds=10,
       train_sizes=["5D", "10D", "15D", "30D"],
   )

   result = policy.evaluate(
       frame,
       model=model,
       target_col="target",
       feature_cols=["feature_1", "feature_2"],
       metrics={"rmse": rmse},
       metric="rmse",
       tolerance=0.01,
   )

   print(result.to_frame().head())
   print(result.summary())

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

Feature-specific lookback windows
---------------------------------

Some pipelines need another layer beyond the fold definition itself: different feature groups
may need different amounts of past data even when the supervised ``train`` segment is fixed.

.. code-block:: python

   from jano import FeatureLookbackSpec

   split = next(splitter.iter_splits(frame))
   lookbacks = FeatureLookbackSpec(
       default_lookback="15D",
       group_lookbacks={"lag_features": "65D"},
       feature_groups={"lag_features": ["lag_30", "lag_60"]},
   )

   history = split.slice_feature_history(
       frame,
       lookbacks,
       time_col="timestamp",
       segment_name="train",
   )

   recent_context = history["__default__"]
   lag_context = history["lag_features"]

This is useful when recent features only need a short context window while lagged or
seasonal features need a much deeper historical slice for the same model.

What it returns
---------------

By default, ``describe_simulation()`` returns a ``SimulationSummary`` object with:

- dataset span and row count,
- total number of folds,
- fold-by-fold segment boundaries,
- a tabular view through ``to_frame()``,
- a serializable structure through ``to_dict()``,
- plot-ready timeline metadata through ``chart_data``.

You can also request a specific output directly:

- ``output="summary"`` returns ``SimulationSummary``,
- ``output="chart_data"`` returns ``SimulationChartData``.

Using chart data directly
-------------------------

``SimulationChartData`` is designed for downstream plotting without an embedded reporting layer.

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
