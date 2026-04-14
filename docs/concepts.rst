Concepts
========

Temporal partitioning
---------------------

Jano models evaluation as a temporal partitioning problem instead of a random sampling problem.

That framing is also useful when you want to evidence drift in simulation results, since changes over time remain visible instead of being blurred by random splits.

The important distinction is that Jano does not treat partitioning as a one-time preprocessing step. It treats it as a temporal process. A partition policy defines:

- how much history belongs in train,
- how large the evaluation horizon is,
- how far the simulation moves at each step,
- and which temporal gaps must exist to avoid leakage.

In that sense, a simulation is better understood as a sequence of causally valid folds than as a single train/test decomposition.

.. math::

   \left\{(\mathcal{D}_{train}^{(k)}, \mathcal{D}_{test}^{(k)})\right\}_{k=1}^K

This is why Jano is useful both for backtesting and for operational questions such as retraining cadence, stability over time and evidence of regime change.

Internally, the engine operates on pandas objects. At the public boundary, though, Jano accepts:

- ``pandas.DataFrame`` with named columns,
- ``numpy.ndarray`` with integer column references such as ``time_col=0``,
- ``polars.DataFrame`` converted internally before fold generation.

Instead of asking for a random share of rows, you define a partition policy:

- how large the train segment is,
- how large the validation or test segments are,
- whether there should be temporal gaps,
- and how the split should move over time.

Strategies
----------

``single``
  Produce one partition only. This is the temporal equivalent of a single split, but still respects chronological ordering.

``rolling``
  Move a fixed-size training window and evaluate repeatedly as time advances.

``expanding``
  Keep growing the training history while validation and test continue moving forward.

Layouts
-------

``train_test``
  Produce a train segment and a test segment.

``train_val_test``
  Produce train, validation and test segments in order.

Segment sizes
-------------

Jano currently accepts three unit families:

- durations such as ``"30D"`` or ``"12H"``,
- row counts such as ``5000``,
- fractions such as ``0.7``.

Within a partition, sizes and gaps should belong to the same unit family.

Outputs
-------

Jano exposes two complementary views:

- ``plan()`` precomputes the simulation geometry as an inspectable object before materializing folds.
- ``TemporalSimulation.run()`` materializes a full simulation and returns a reusable result object.
- ``split()`` yields plain index tuples, which keeps usage lightweight and easy to integrate.
- ``iter_splits()`` yields ``TimeSplit`` objects with segment metadata and helper methods.
- ``describe_simulation()`` yields either a ``SimulationSummary``, an HTML report string or ``SimulationChartData`` for custom Python plotting.

Planning before materialization
-------------------------------

Jano now exposes a planning layer between configuration and execution.

That means you can first compute the geometry of all subsequent partitions, inspect it,
filter it, and only then materialize the folds you actually want.

``plan()`` is useful when you want to:

- inspect the full list of iterations before training anything
- understand how many rows each segment would contain
- start from iteration ``N`` instead of the beginning
- exclude folds whose train or test windows overlap special dates
- work from a precomputed simulation plan rather than slicing the dataset immediately

At the low level:

.. code-block:: python

   plan = splitter.plan(frame)
   print(plan.to_frame().head())

At the high level:

.. code-block:: python

   plan = simulation.plan(frame, title="Policy preview")
   filtered = plan.exclude_windows(
       train=[("2025-12-20", "2026-01-05")],
   ).select_from_iteration(10)

   result = filtered.materialize()

The plan frame includes an explicit ``iteration`` column, segment boundaries and row counts.
That makes it possible to reason about the simulation as a first-class object instead of only
as a generator of folds.

Temporal hypotheses
-------------------

The previous sections describe the mechanics of temporal partitioning. On top of that base,
Jano can also encode higher-level evaluation hypotheses about how a model behaves across time.

The progression is meant to stay incremental:

- start with explicit partitions,
- then run walk-forward simulations,
- and finally test operational hypotheses about history sufficiency or performance decay.

Two core hypothesis policies are now part of the package.

``TrainGrowthPolicy``
  Keep the same test window fixed and expand train backward in time.

  This answers questions such as:

  - does adding more history improve the same test slice?
  - can a smaller train sample match the best observed test quality?
  - where does extra training history stop being useful?

  .. code-block:: python

     from jano import TrainGrowthPolicy

     policy = TrainGrowthPolicy(
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
         metrics=["mae", "rmse"],
     )

     best = result.find_optimal_train_size(metric="rmse", tolerance=0.01)

``PerformanceDecayPolicy``
  Keep train fixed and shift test forward over time.

  This answers questions such as:

  - how long can the current model stay in production before degradation becomes material?
  - when does drift start becoming a practical problem?
  - how often should retraining happen if retraining is expensive?

  .. code-block:: python

     from jano import PerformanceDecayPolicy

     policy = PerformanceDecayPolicy(
         "timestamp",
         cutoff="2025-09-15",
         train_size="30D",
         test_size="3D",
         step="1D",
         max_windows=14,
     )

     result = policy.evaluate(
         frame,
         model=model,
         target_col="target",
         feature_cols=["feature_1", "feature_2"],
         metrics=["mae", "rmse"],
     )

     onset = result.find_drift_onset(metric="rmse", threshold=0.15, baseline="first")

These policies are not just visual variations of the splitter. They encapsulate different
temporal questions about the system under evaluation:

- walk-forward simulation asks how the system would have behaved over time under a retraining policy
- train growth asks whether more historical data is actually worth using
- performance decay asks how long the current train set remains operationally safe

Feature lookback policies
-------------------------

Some temporal problems need one more layer of realism: not all feature groups use the same
historical depth.

For example:

- recent behavioral features may only need the last ``15D``
- long-lag or seasonal features may need ``65D`` or more

That does not necessarily mean the supervised train window itself should become longer.
It means the feature engineering pipeline needs different amounts of historical context
for different groups of variables.

Jano models that with ``FeatureLookbackSpec`` on top of a fold:

.. code-block:: python

   from jano import FeatureLookbackSpec

   lookbacks = FeatureLookbackSpec(
       default_lookback="15D",
       group_lookbacks={"lag_features": "65D"},
       feature_groups={"lag_features": ["lag_30", "lag_60"]},
   )

   split = next(splitter.iter_splits(frame))
   history = split.slice_feature_history(
       frame,
       lookbacks,
       time_col="timestamp",
       segment_name="train",
   )

   recent_context = history["__default__"]
   lag_context = history["lag_features"]

This keeps the fold geometry unchanged while making the required historical context explicit.
It is useful when feature computation and model training do not share the same temporal depth.
