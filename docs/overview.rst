Overview
========

Why Jano exists
---------------

Jano is designed to structure, execute and analyze temporal simulations for machine learning systems operating on time-correlated data. The goal is not only to split a dataset once, but to create a disciplined way to reason about how machine learning systems behave when chronology, retraining cadence and data availability are treated as real constraints.

In many real systems, data is naturally organized across entities and time. A more faithful representation looks like:

.. math::

   \mathcal{D} = \{(x_{i,t}, y_{i,t})\}_{i=1,\dots,N;\; t=1,\dots,T}

where :math:`i` indexes entities such as users, routes or sellers, and :math:`t` denotes time.

Despite that structure, many standard evaluation workflows still act as if observations were independent and identically distributed:

.. math::

   (x_{i,t}, y_{i,t}) \sim \text{i.i.d.}

That mismatch is the source of the problem. Once temporal ordering is ignored, information from the future can leak into training, reported performance becomes optimistic, and offline validation stops matching the conditions under which models actually run in production.

Most train/test tooling answers a static question:

"How do I split this dataset once?"

Jano is built to answer a dynamic one:

"How would this system behave over time if I trained, retrained and evaluated it under an explicit temporal policy?"

That framing matters because chronology is not treated as noise to average away. In Jano, time is a first-class constraint: the split definition, fold boundaries and simulation outputs all preserve the distinction between what was known in the past and what only becomes available later.

Tools such as ``TimeSeriesSplit`` already improve on random sampling by enforcing a basic ordering between past and future. But they still mostly answer "how should I split this dataset?" Jano is aimed at a more operational question: how would a system have behaved over time under a given retraining and evaluation policy?

Instead of a single split, Jano works with an explicit temporal policy:

.. math::

   \pi = (\Delta_{train}, \Delta_{test}, s, g)

where :math:`\Delta_{train}` is the training window, :math:`\Delta_{test}` is the evaluation horizon, :math:`s` is the shift between iterations, and :math:`g` is an optional temporal gap for leakage control.

Under such a policy, evaluation becomes a sequence of temporally consistent experiments:

.. math::

   \left\{(\mathcal{D}_{train}^{(k)}, \mathcal{D}_{test}^{(k)})\right\}_{k=1}^K

rather than a one-shot estimate. Each fold preserves causality and contributes to a trajectory of system behavior over time, which is precisely what is missing from static validation.

This also makes the toolkit useful for evidencing drift in simulation results. Jano does not estimate drift metrics directly, but it makes temporal changes in behavior, calibration or performance visible fold after fold. That makes it possible to inspect degradation, instability and regime changes over time instead of collapsing them into a single aggregate metric.

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
- Time-aware evaluation as a reproducible process rather than a one-off split.
- Minimal hidden state.
- Predictable, inspectable folds.
- A pandas-friendly workflow with an interface inspired by ``sklearn.model_selection``.
- Input normalization for ``pandas``, ``numpy`` and ``polars`` tabular data.
- Rich fold objects that can be inspected, summarized and sliced.
- Auditability as a design constraint, so simulations remain reproducible and traceable.

Current status
--------------

The project is in an early redesign phase. The new core already supports:

- ``TemporalSimulation`` as a higher-level interface for running full simulations.
- Optional simulation window controls such as ``start_at``, ``end_at`` and ``max_folds``.
- ``single``, ``rolling`` and ``expanding`` strategies.
- ``train_test`` and ``train_val_test`` layouts.
- Sizes expressed as durations, row counts or fractions.
- Optional gaps before train, validation or test, plus a trailing gap after test.
- Temporal semantics that let each segment use a different timestamp column for eligibility.
- Input normalization for ``pandas``, ``numpy`` and optional ``polars`` data.
- Simulation reporting as summary objects, HTML reports or plot-ready Python data.
- A numpy-first temporal indexing path that trims overhead on large datasets.

The API is stable enough for experimentation and active design work, while still small enough to refine before broader publication. In practice, Jano already works as an experimental framework for reasoning about machine learning systems as time-evolving processes rather than static train/test artifacts.
