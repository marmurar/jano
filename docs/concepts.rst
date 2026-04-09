Concepts
========

Temporal partitioning
---------------------

Jano models evaluation as a temporal partitioning problem instead of a random sampling problem.

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

- ``split()`` yields plain index tuples, which keeps usage lightweight and easy to integrate.
- ``iter_splits()`` yields ``TimeSplit`` objects with segment metadata and helper methods.
