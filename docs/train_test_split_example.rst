Random Splits vs Temporal Validation
====================================

``sklearn.model_selection.train_test_split`` is useful when observations can be
treated as approximately independent and identically distributed. That is not the
question Jano is designed to answer.

For time-correlated data, the question is usually operational:

   How would the model have behaved if it had only seen the past and then had to
   predict the future?

A random split can hide that question because it mixes dates across train and
test.

The first snippet assumes scikit-learn is installed only to illustrate the common
baseline. Jano itself does not require scikit-learn.

The scikit-learn way
--------------------

Imagine a daily dataset where the target distribution changes near the end of the
period:

.. code-block:: python

   import pandas as pd
   from sklearn.model_selection import train_test_split

   frame = pd.DataFrame(
       {
           "timestamp": pd.date_range("2025-01-01", periods=120, freq="D"),
           "feature": range(120),
           "target": [0] * 80 + [1] * 40,
       }
   )

   train_random, test_random = train_test_split(
       frame,
       test_size=0.2,
       shuffle=True,
       random_state=7,
   )

   temporal_leakage = (
       train_random["timestamp"].max() > test_random["timestamp"].min()
   )

   print(temporal_leakage)
   # True

The problem is not that scikit-learn is wrong. ``train_test_split`` is doing what
it is designed to do: random sampling. The problem is that random sampling is the
wrong abstraction for production-like temporal validation.

In this setup, train can contain observations from dates that are later than some
test observations. If the target changes over time, the evaluation can become too
optimistic because the model has already seen part of the future regime.

The Jano Version
----------------

With Jano, the split is not defined as a random share of rows. It is defined as a
temporal policy:

.. code-block:: python

   import pandas as pd

   from jano import TemporalPartitionSpec, WalkForwardPolicy

   frame = pd.DataFrame(
       {
           "timestamp": pd.date_range("2025-01-01", periods=120, freq="D"),
           "feature": range(120),
           "target": [0] * 80 + [1] * 40,
       }
   )

   policy = WalkForwardPolicy(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="60D",
           test_size="14D",
           gap_before_test="1D",
       ),
       step="14D",
       strategy="rolling",
   )

   plan = policy.plan(frame, title="Production-like temporal validation")

   print(
       plan.to_frame()[
           [
               "iteration",
               "train_start",
               "train_end",
               "train_rows",
               "test_start",
               "test_end",
               "test_rows",
           ]
       ].head()
   )

The plan makes the temporal contract explicit before any model is trained:

.. code-block:: text

    iteration train_start  train_end  train_rows test_start   test_end  test_rows
            0  2025-01-01 2025-03-02          60 2025-03-03 2025-03-17         14
            1  2025-01-15 2025-03-16          60 2025-03-17 2025-03-31         14
            2  2025-01-29 2025-03-30          60 2025-03-31 2025-04-14         14
            3  2025-02-12 2025-04-13          60 2025-04-14 2025-04-28         14

What Changes
------------

The difference is the evaluation contract:

- ``train_test_split`` answers: can this model generalize to a random sample from
  the same mixed period?
- Jano answers: how would this model behave as time advances under a specific
  training and evaluation policy?

That gives you:

- ordered train and test windows,
- explicit train/test duration,
- explicit gaps to model label or data availability latency,
- repeated folds instead of one static estimate,
- a ``plan()`` object that can be inspected, filtered and audited before slicing
  the dataset.

This is the point where Jano enters: not as a replacement for scikit-learn, but
as the temporal validation layer that sits before model training.
