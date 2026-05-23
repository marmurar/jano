Jano
====

.. container:: language-switch

   **Language:** English | :doc:`Español <es/index>`

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.20301006.svg
   :target: https://doi.org/10.5281/zenodo.20301006
   :alt: DOI

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
         <h3>Temporal and event partitions</h3>
         <p>Partition by calendar time, row counts or online event batches depending on how the system observes data.</p>
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

Jano is a Python toolkit designed to structure, execute and analyze temporal simulations for machine learning systems operating on time-correlated data. It provides a formal framework to define time-aware partitioning policies, run walk-forward evaluations, execute models under explicit retraining rules and generate auditable reports that reflect how systems behave under realistic, production-like temporal dynamics.

Unlike traditional random splits that implicitly assume i.i.d. data, Jano treats chronology as a first-class constraint. It is built for scenarios where leakage must be tightly controlled and where system performance is expected to evolve over time because of drift, retraining cycles or changing data distributions.

At its core, Jano introduces explicit partitioning abstractions. Users can define train, validation and test segments through durations, row counts or proportions, compose them into rolling, expanding or fixed-window strategies, and use temporal gaps to model the latency that often exists between training, prediction and label availability. For online settings, Jano can also partition the observed stream by events or micro-batches.

The recommended public surface centers on ``TemporalSimulation`` and ``WalkForwardPolicy`` for fold-level simulation, ``WalkForwardRunner`` for model execution over those folds, and ``TemporalBacktestSplitter`` for manual iteration and lower-level control. Jano does not compute drift metrics directly; instead, it exposes temporal structure in evaluation results so drift, regime changes and model decay become easier to inspect fold by fold.

Typical use cases include:

- Walk-forward validation for forecasting and time-aware classification.
- Simulation of retraining policies and deployment strategies.
- Execution of model benchmarks under explicit retraining rules.
- Monitoring model stability across time slices.
- Evaluating decision policies under evolving data conditions.

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
   train_test_split_example
   partitioning
   concepts
   simulation
   mcp
   ai
   benchmark
   datasets
   api
   release
   about
