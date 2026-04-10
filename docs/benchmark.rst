Benchmark
=========

This page summarizes a local benchmark of temporal partition generation across the currently supported tabular backends.

What was measured
-----------------

The benchmark measures the total time needed to materialize all folds from:

.. code-block:: python

   list(splitter.iter_splits(data))

Configuration used
------------------

The same splitter configuration was used for every backend:

- strategy: ``rolling``
- layout: ``train_test``
- ``train_size="3D"``
- ``test_size="12h"``
- ``gap_before_test="30min"``
- ``step="6h"``
- dataset frequency: one row per minute
- metric: wall-clock runtime over the full split iteration
- repetitions: 3 per backend and dataset size

The benchmark was run locally on the current implementation, where pandas is still the internal execution engine. That means the ``numpy`` and ``polars`` timings include the cost of normalizing those inputs to pandas before splitting.

Results
-------

.. list-table::
   :header-rows: 1

   * - Backend
     - Rows
     - Folds
     - Mean ms
     - Min ms
     - Max ms
   * - pandas
     - 10,000
     - 14
     - 7.581
     - 5.478
     - 11.634
   * - numpy
     - 10,000
     - 14
     - 4.536
     - 4.208
     - 5.181
   * - polars
     - 10,000
     - 14
     - 10.767
     - 10.657
     - 10.825
   * - pandas
     - 100,000
     - 264
     - 10.544
     - 8.264
     - 14.778
   * - numpy
     - 100,000
     - 264
     - 19.789
     - 18.930
     - 20.940
   * - polars
     - 100,000
     - 264
     - 65.366
     - 62.823
     - 69.806
   * - pandas
     - 500,000
     - 1,375
     - 24.592
     - 22.083
     - 29.403
   * - numpy
     - 500,000
     - 1,375
     - 94.801
     - 91.859
     - 97.771
   * - polars
     - 500,000
     - 1,375
     - 294.843
     - 289.612
     - 299.288
   * - pandas
     - 1,000,000
     - 2,764
     - 44.719
     - 38.910
     - 47.781
   * - numpy
     - 1,000,000
     - 2,764
     - 183.353
     - 177.574
     - 190.390
   * - polars
     - 1,000,000
     - 2,764
     - 587.886
     - 583.358
     - 592.276

Visual summary
--------------

Below, each bar shows the mean runtime for a given dataset size. The scale is relative within each row so the shape remains readable.

.. raw:: html

   <div class="benchmark-grid">
     <div class="benchmark-row">
       <div class="benchmark-label">10k rows</div>
       <div class="benchmark-bars">
         <div class="benchmark-bar">
           <span class="benchmark-name">pandas</span>
           <span class="benchmark-track"><span class="benchmark-fill pandas" style="width: 70.4%;"></span></span>
           <span class="benchmark-value">7.6 ms</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">numpy</span>
           <span class="benchmark-track"><span class="benchmark-fill numpy" style="width: 42.1%;"></span></span>
           <span class="benchmark-value">4.5 ms</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">polars</span>
           <span class="benchmark-track"><span class="benchmark-fill polars" style="width: 100%;"></span></span>
           <span class="benchmark-value">10.8 ms</span>
         </div>
       </div>
     </div>
     <div class="benchmark-row">
       <div class="benchmark-label">100k rows</div>
       <div class="benchmark-bars">
         <div class="benchmark-bar">
           <span class="benchmark-name">pandas</span>
           <span class="benchmark-track"><span class="benchmark-fill pandas" style="width: 16.1%;"></span></span>
           <span class="benchmark-value">10.5 ms</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">numpy</span>
           <span class="benchmark-track"><span class="benchmark-fill numpy" style="width: 30.3%;"></span></span>
           <span class="benchmark-value">19.8 ms</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">polars</span>
           <span class="benchmark-track"><span class="benchmark-fill polars" style="width: 100%;"></span></span>
           <span class="benchmark-value">65.4 ms</span>
         </div>
       </div>
     </div>
     <div class="benchmark-row">
       <div class="benchmark-label">500k rows</div>
       <div class="benchmark-bars">
         <div class="benchmark-bar">
           <span class="benchmark-name">pandas</span>
           <span class="benchmark-track"><span class="benchmark-fill pandas" style="width: 8.3%;"></span></span>
           <span class="benchmark-value">24.6 ms</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">numpy</span>
           <span class="benchmark-track"><span class="benchmark-fill numpy" style="width: 32.2%;"></span></span>
           <span class="benchmark-value">94.8 ms</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">polars</span>
           <span class="benchmark-track"><span class="benchmark-fill polars" style="width: 100%;"></span></span>
           <span class="benchmark-value">294.8 ms</span>
         </div>
       </div>
     </div>
     <div class="benchmark-row">
       <div class="benchmark-label">1M rows</div>
       <div class="benchmark-bars">
         <div class="benchmark-bar">
           <span class="benchmark-name">pandas</span>
           <span class="benchmark-track"><span class="benchmark-fill pandas" style="width: 7.6%;"></span></span>
           <span class="benchmark-value">44.7 ms</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">numpy</span>
           <span class="benchmark-track"><span class="benchmark-fill numpy" style="width: 31.2%;"></span></span>
           <span class="benchmark-value">183.4 ms</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">polars</span>
           <span class="benchmark-track"><span class="benchmark-fill polars" style="width: 100%;"></span></span>
           <span class="benchmark-value">587.9 ms</span>
         </div>
       </div>
     </div>
   </div>

How to read these results
-------------------------

The current benchmark should be read as:

- the partition engine itself is fast on large datasets,
- pandas is currently the fastest path end to end,
- numpy and polars are compatible public inputs, but not yet native optimized execution backends,
- their extra cost mostly comes from boundary normalization into pandas before fold generation.

So, if raw split speed matters most today, the best-performing input remains ``pandas.DataFrame``.
