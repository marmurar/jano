Temporal Backtesting for Time-Correlated Data
=============================================

.. raw:: html

   <div class="landing-hero">
     <span class="landing-kicker">Jano</span>
     <h1>Temporal partitions that make simulations auditable.</h1>
     <p class="landing-lead">
       Jano is a Python toolkit for defining temporal partition policies, running walk-forward simulations
       and inspecting how systems behave as time advances. It is built for datasets where chronology is part
       of the problem, not noise to average away.
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

Jano is useful when a single random split is too weak a proxy for reality: production retraining, walk-forward validation, model monitoring, policy evaluation or any workflow where the past should not leak into the future.

It also works well as a way to evidence drift in simulation results. Jano does not compute drift metrics directly, but it makes temporal shifts in outcomes explicit by preserving chronology across folds and reports.

The recommended public surface now centers on ``TemporalSimulation`` for full simulation runs, while ``TemporalBacktestSplitter`` remains available for manual fold iteration and lower-level control.

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
   benchmark
   api
   release
   about
