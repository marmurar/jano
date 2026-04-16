Jano
====

.. container:: language-switch

   **Language:** English | :doc:`Español <es/index>`

.. raw:: html

   <div class="landing-hero">
     <p class="landing-lead">
       Temporal Simulation and Backtesting Toolkit for Time-Dependent Machine Learning Systems
     </p>
     <p class="landing-tagline">
       The missing layer between ML models and production temporal validation.
     </p>
     <div class="landing-grid">
       <div class="landing-card">
         <h3>Explicit partition policies</h3>
         <p>Define train/test or train/validation/test layouts with durations, row counts or fractions.</p>
       </div>
       <div class="landing-card">
         <h3>Operational simulations</h3>
         <p>Model rolling, expanding or single-window evaluation with optional temporal gaps.</p>
       </div>
       <div class="landing-card">
         <h3>Drift becomes visible</h3>
         <p>By keeping folds anchored in time, changes in outcomes, calibration or behavior are easier to spot.</p>
       </div>
       <div class="landing-card">
         <h3>Flexible tabular inputs</h3>
         <p>Run the same API on pandas, NumPy or Polars data while keeping one temporal engine underneath.</p>
       </div>
     </div>
   </div>

.. container:: landing-visual

   .. image:: /_static/jano_viz.png
      :alt: Jano temporal partition visualization
      :class: landing-visual-image

   .. container:: landing-visual-caption

      A visual summary of how Jano lays out temporal partitions, folds and reporting across time.

Jano is a Python toolkit designed to structure, execute and analyze temporal simulations for machine learning systems operating on time-correlated data. It provides a formal framework to define time-aware partitioning policies, run walk-forward evaluations and generate auditable reports that reflect how models behave under realistic, production-like temporal dynamics.

Unlike traditional random splits that implicitly assume i.i.d. data, Jano treats chronology as a first-class constraint. It is built for scenarios where leakage must be tightly controlled and where system performance is expected to evolve over time because of drift, retraining cycles or changing data distributions.

At its core, Jano introduces explicit temporal partitioning abstractions. Users can define train, validation and test segments through durations, row counts or proportions, compose them into rolling, expanding or fixed-window strategies, and use temporal gaps to model the latency that often exists between training, prediction and label availability.

The recommended public surface centers on ``TemporalSimulation`` for end-to-end simulation workflows, while ``TemporalBacktestSplitter`` remains available for manual fold iteration and lower-level control. Jano does not compute drift metrics directly; instead, it exposes temporal structure in evaluation results so drift, regime changes and model decay become easier to inspect fold by fold.

Typical use cases include:

- Walk-forward validation for forecasting and time-aware classification.
- Simulation of retraining policies and deployment strategies.
- Monitoring model stability across time slices.
- Evaluating decision policies under evolving data conditions.

Supported input backends:

- ``pandas.DataFrame`` with named columns
- ``numpy.ndarray`` with integer column references
- ``polars.DataFrame`` through the optional ``jano[polars]`` extra

.. toctree::
   :hidden:
 
   es/index

Supported input backends:

- ``pandas.DataFrame`` with named columns
- ``numpy.ndarray`` with integer column references
- ``polars.DataFrame`` through the optional ``jano[polars]`` extra

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   concepts
   simulation
   mcp
   benchmark
   api
   release
   about
