Overview
========

Why Jano exists
---------------

Most train/test tools answer a simple question:

"How do I split this dataset once?"

Jano is designed to answer a more operational one:

"How would this system have behaved over time if I had trained, retrained and evaluated it under an explicit temporal policy?"

That also makes it useful for evidencing drift in simulation results, because temporal changes in behavior or performance become explicit fold after fold.

That makes it useful for:

- Backtesting predictive systems over transactional data.
- Simulating daily, weekly or custom retraining cadences.
- Comparing rolling and expanding windows.
- Introducing temporal gaps between train and evaluation segments.
- Defining ``train/test`` or ``train/validation/test`` layouts using durations, row counts or percentages.
- Surfacing drift in simulation outcomes by making temporal changes explicit across folds.

Design goals
------------

Jano is being reshaped around a few clear ideas:

- Explicit temporal partition definitions.
- Minimal hidden state.
- Predictable, inspectable folds.
- A pandas-friendly workflow with an interface inspired by ``sklearn.model_selection``.
- Rich fold objects that can be inspected, summarized and sliced.

Current status
--------------

The project is in an early redesign phase. The new core already supports:

- ``TemporalSimulation`` as a higher-level interface for running full simulations.
- Optional simulation window controls such as ``start_at``, ``end_at`` and ``max_folds``.
- ``single``, ``rolling`` and ``expanding`` strategies.
- ``train_test`` and ``train_val_test`` layouts.
- Sizes expressed as durations, row counts or fractions.
- Optional gaps before validation or test segments.
- Simulation reporting as summary objects, HTML reports or plot-ready Python data.
- A numpy-first temporal indexing path that trims overhead on large datasets.

The API is stable enough for experimentation and active design work, while still small enough to refine before broader publication.
