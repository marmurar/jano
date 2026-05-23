Partitioning Modes
==================

Jano separates two related ideas that are often mixed together:

- **Temporal partitioning** divides a historical dataset into train, validation
  and test windows ordered by time.
- **Event-based online partitioning** divides an observed stream into events or
  micro-batches, then evaluates how a model behaves as new observations arrive.

Both modes are causal: data observed later must not influence decisions that
would have been made earlier. The difference is the unit that moves the
evaluation forward.

Temporal Partitioning
---------------------

Temporal partitioning is the default mode for backtesting tabular ML systems.
It answers questions such as:

- What would performance have looked like if the model had been retrained every
  day?
- How much history should the train window contain?
- How does a fixed model decay across future time windows?

Use ``TemporalBacktestSplitter`` directly when you want manual control over the
fold loop:

.. code-block:: python

   from jano import TemporalBacktestSplitter

   splitter = TemporalBacktestSplitter(
       time_col="timestamp",
       train_size="30D",
       test_size="7D",
       step="7D",
       strategy="rolling",
   )

   for train_idx, test_idx in splitter.split(frame):
       train = frame.iloc[train_idx]
       test = frame.iloc[test_idx]

Use ``WalkForwardPolicy`` or ``TemporalSimulation`` when you want Jano to build a
plan, run the folds, and expose auditable outputs.

Event-Based Online Partitioning
-------------------------------

Event-based online partitioning is not a walk-forward simulation over fixed
historical folds. It is a causal online evaluation pattern: initialize a model,
predict the next event or micro-batch, observe the target, update the model, and
repeat.

This is useful when the operational question is not only *when should I retrain
by calendar time?*, but also *how many new observations should I wait for before
updating the model?*

Use ``OnlineTemporalRunner`` with ``PartialFitUpdateStrategy`` when the model
supports real incremental updates through ``partial_fit``:

.. code-block:: python

   from jano import OnlineTemporalRunner, PartialFitUpdateStrategy

   runner = OnlineTemporalRunner(
       model=model,
       time_col="timestamp",
       target_col="target",
       feature_cols=["feature_a", "feature_b"],
       initial_train_size="30D",
       update_size=1,
       metrics={"mae": mae, "rmse": rmse},
       update_strategy=PartialFitUpdateStrategy(),
   )

   run = runner.run(frame)
   print(run.to_frame().head())
   print(run.metric_trajectory().head())
   print(run.summary())

The sequence is deliberately causal:

- initialize the model on the initial train window,
- predict the next event or micro-batch,
- score the prediction once the target is observed,
- update the model with that observed batch,
- repeat.

``update_size=1`` means event-level updates. You can also use row batches such as
``update_size=100`` or duration batches such as ``update_size="1D"``. This lets
you compare event-level, row-batch and time-batch update policies without
changing the rest of the runner configuration.

Not every estimator supports ``partial_fit``. For regular ``fit/predict`` models,
use ``RefitUpdateStrategy`` instead:

.. code-block:: python

   from jano import OnlineTemporalRunner, RefitUpdateStrategy

   runner = OnlineTemporalRunner(
       model=model,
       time_col="timestamp",
       target_col="target",
       feature_cols=["feature_a", "feature_b"],
       initial_train_size="30D",
       update_size="1D",
       metrics={"mae": mae},
       update_strategy=RefitUpdateStrategy(max_train_rows=10_000),
   )

This strategy refits after each observed batch. It is more expensive than
``partial_fit``, but it works with standard estimators and can keep bounded
history through ``max_train_rows``.

Finding an Observation-Driven Update Policy
-------------------------------------------

``OnlineUpdatePolicyStudy`` compares multiple update cadences over the same
temporal stream. That lets you ask whether model updates should be triggered by
calendar time, row count, or accumulated evidence:

.. code-block:: python

   from jano import OnlineUpdatePolicy, OnlineUpdatePolicyStudy, RefitUpdateStrategy

   study = OnlineUpdatePolicyStudy(
       model=model,
       time_col="timestamp",
       target_col="target",
       feature_cols=["feature_a", "feature_b"],
       initial_train_size="30D",
       policies=[
           OnlineUpdatePolicy("every-event", update_size=1, update_strategy=RefitUpdateStrategy()),
           OnlineUpdatePolicy("every-100-events", update_size=100, update_strategy=RefitUpdateStrategy()),
           OnlineUpdatePolicy("daily", update_size="1D", update_strategy=RefitUpdateStrategy()),
       ],
       metrics={"mae": mae},
   )

   comparison = study.run(frame)

   print(comparison.to_frame())
   print(comparison.metric_trajectory().head())
   print(comparison.find_optimal_policy(metric="mae", update_cost_weight=0.01))

The optional ``update_cost_weight`` penalizes policies that update too often. This
keeps the output data-first while making the tradeoff explicit: a policy can win
because it predicts better, because it updates less, or because it offers the best
cost-adjusted compromise.
