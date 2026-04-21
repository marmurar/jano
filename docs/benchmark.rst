Benchmark
=========

This page summarizes a local benchmark of Jano's adaptive partition engine.

The goal is to measure the cost of temporal partitioning itself: computing fold
boundaries, row counts and split indices. It does not include model training time.

What Was Measured
-----------------

Two operations were measured:

.. code-block:: python

   splitter.plan(data)
   sum(1 for _ in splitter.split(data))

``plan()`` precomputes fold geometry and row counts. ``split()`` yields the positional
index arrays for every fold.

Configuration Used
------------------

The same splitter configuration was used for every input backend:

- strategy: ``rolling``
- layout: ``train_test``
- ``train_size="2D"``
- ``test_size="12h"``
- ``step="12h"``
- dataset frequency: one row per minute
- repetitions: median of 7 timed runs after 2 warmups
- dataset sizes: 10k, 100k and 500k rows

The benchmark compares:

- ``engine="auto"``: Jano chooses the native safe path for the input.
- ``engine="pandas"``: Jano forces the stable pandas path.

For Polars and NumPy inputs, ``engine="pandas"`` is a useful baseline for the previous
behavior because it includes converting the input to pandas before partitioning.

Raw Results
-----------

.. list-table::
   :header-rows: 1

   * - Rows
     - Input
     - Engine arg
     - Selected engine
     - Converted
     - Folds
     - Plan ms
     - Split ms
   * - 10,000
     - pandas
     - auto
     - pandas
     - no
     - 9
     - 0.26
     - 0.33
   * - 10,000
     - pandas
     - pandas
     - pandas
     - no
     - 9
     - 0.26
     - 0.32
   * - 10,000
     - polars
     - auto
     - polars
     - no
     - 9
     - 0.26
     - 0.32
   * - 10,000
     - polars
     - pandas
     - pandas
     - yes
     - 9
     - 6.09
     - 6.07
   * - 10,000
     - numpy
     - auto
     - numpy
     - no
     - 9
     - 0.26
     - 0.31
   * - 10,000
     - numpy
     - pandas
     - pandas
     - yes
     - 9
     - 0.36
     - 0.41
   * - 100,000
     - pandas
     - auto
     - pandas
     - no
     - 134
     - 1.71
     - 2.53
   * - 100,000
     - pandas
     - pandas
     - pandas
     - no
     - 134
     - 1.70
     - 2.53
   * - 100,000
     - polars
     - auto
     - polars
     - no
     - 134
     - 1.71
     - 2.51
   * - 100,000
     - polars
     - pandas
     - pandas
     - yes
     - 134
     - 56.82
     - 58.95
   * - 100,000
     - numpy
     - auto
     - numpy
     - no
     - 134
     - 1.86
     - 2.59
   * - 100,000
     - numpy
     - pandas
     - pandas
     - yes
     - 134
     - 1.90
     - 2.72
   * - 500,000
     - pandas
     - auto
     - pandas
     - no
     - 690
     - 8.65
     - 12.40
   * - 500,000
     - pandas
     - pandas
     - pandas
     - no
     - 690
     - 8.61
     - 12.33
   * - 500,000
     - polars
     - auto
     - polars
     - no
     - 690
     - 8.58
     - 12.37
   * - 500,000
     - polars
     - pandas
     - pandas
     - yes
     - 690
     - 296.48
     - 304.73
   * - 500,000
     - numpy
     - auto
     - numpy
     - no
     - 690
     - 9.14
     - 13.00
   * - 500,000
     - numpy
     - pandas
     - pandas
     - yes
     - 690
     - 9.42
     - 13.40

Speedup Versus Forced Pandas
----------------------------

.. list-table::
   :header-rows: 1

   * - Rows
     - Input
     - Plan speedup
     - Split speedup
     - Engine path
   * - 10,000
     - pandas
     - 0.98x
     - 0.98x
     - pandas -> pandas
   * - 10,000
     - polars
     - 23.28x
     - 19.17x
     - pandas -> polars
   * - 10,000
     - numpy
     - 1.38x
     - 1.32x
     - pandas -> numpy
   * - 100,000
     - pandas
     - 0.99x
     - 1.00x
     - pandas -> pandas
   * - 100,000
     - polars
     - 33.29x
     - 23.45x
     - pandas -> polars
   * - 100,000
     - numpy
     - 1.02x
     - 1.05x
     - pandas -> numpy
   * - 500,000
     - pandas
     - 1.00x
     - 0.99x
     - pandas -> pandas
   * - 500,000
     - polars
     - 34.57x
     - 24.64x
     - pandas -> polars
   * - 500,000
     - numpy
     - 1.03x
     - 1.03x
     - pandas -> numpy

Visual Summary
--------------

The bars below show split-time speedup for ``engine="auto"`` against the forced pandas
baseline. Higher is better.

.. raw:: html

   <div class="benchmark-grid">
     <div class="benchmark-row">
       <div class="benchmark-label">10k rows</div>
       <div class="benchmark-bars">
         <div class="benchmark-bar">
           <span class="benchmark-name">pandas</span>
           <span class="benchmark-track"><span class="benchmark-fill pandas" style="width: 5.1%;"></span></span>
           <span class="benchmark-value">0.98x</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">numpy</span>
           <span class="benchmark-track"><span class="benchmark-fill numpy" style="width: 6.9%;"></span></span>
           <span class="benchmark-value">1.32x</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">polars</span>
           <span class="benchmark-track"><span class="benchmark-fill polars" style="width: 100%;"></span></span>
           <span class="benchmark-value">19.17x</span>
         </div>
       </div>
     </div>
     <div class="benchmark-row">
       <div class="benchmark-label">100k rows</div>
       <div class="benchmark-bars">
         <div class="benchmark-bar">
           <span class="benchmark-name">pandas</span>
           <span class="benchmark-track"><span class="benchmark-fill pandas" style="width: 4.3%;"></span></span>
           <span class="benchmark-value">1.00x</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">numpy</span>
           <span class="benchmark-track"><span class="benchmark-fill numpy" style="width: 4.5%;"></span></span>
           <span class="benchmark-value">1.05x</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">polars</span>
           <span class="benchmark-track"><span class="benchmark-fill polars" style="width: 100%;"></span></span>
           <span class="benchmark-value">23.45x</span>
         </div>
       </div>
     </div>
     <div class="benchmark-row">
       <div class="benchmark-label">500k rows</div>
       <div class="benchmark-bars">
         <div class="benchmark-bar">
           <span class="benchmark-name">pandas</span>
           <span class="benchmark-track"><span class="benchmark-fill pandas" style="width: 4.0%;"></span></span>
           <span class="benchmark-value">0.99x</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">numpy</span>
           <span class="benchmark-track"><span class="benchmark-fill numpy" style="width: 4.2%;"></span></span>
           <span class="benchmark-value">1.03x</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">polars</span>
           <span class="benchmark-track"><span class="benchmark-fill polars" style="width: 100%;"></span></span>
           <span class="benchmark-value">24.64x</span>
         </div>
       </div>
     </div>
   </div>

How To Read These Results
-------------------------

The main improvement is for Polars inputs. Before the adaptive engine, Polars inputs had
to be fully converted to pandas before partitioning. With ``engine="auto"``, Jano keeps
Polars column extraction native for planning and split-index generation.

The pandas path is intentionally unchanged. Pandas remains the stable baseline and the
engine selected for pandas input.

The NumPy path improves modestly in this benchmark because the measured structured-array
input already converts cheaply. NumPy remains useful as a low-overhead input path, but the
largest observed gain is avoiding the Polars-to-pandas conversion on larger datasets.

These timings are local and should be read as directional. They are most useful for
understanding partition-engine overhead; full model simulations will also include feature
engineering, model fitting and prediction time.
