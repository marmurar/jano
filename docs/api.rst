API reference
=============

.. raw:: html

   <p class="api-lead">
     The API is intentionally compact. Most workflows start with a <code>TemporalPartitionSpec</code>,
     pass it into <code>TemporalSimulation</code> for a full run, and optionally drop down to
     <code>TemporalBacktestSplitter</code> when they need manual control over folds.
   </p>

   <p class="api-lead">
     Public inputs can come from <code>pandas</code>, <code>numpy</code> or
     <code>polars</code>. When the source is not pandas, Jano normalizes it at the boundary
     and keeps the same split and reporting surface.
   </p>

Main workflow
-------------

.. autoclass:: jano.simulation.TemporalSimulation
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.simulation.SimulationResult
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.types.TemporalPartitionSpec
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.types.TemporalSemanticsSpec
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.splitters.TemporalBacktestSplitter
   :members:
   :undoc-members:
   :no-index:

Fold objects
------------

.. autoclass:: jano.splits.TimeSplit
   :members:
   :undoc-members:
   :no-index:

Reporting objects
-----------------

.. autoclass:: jano.reporting.SimulationSummary
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.reporting.SimulationChartData
   :members:
   :undoc-members:
   :no-index:

Type and validation helpers
---------------------------

.. autoclass:: jano.types.SizeSpec
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.types.SegmentBoundaries
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.validation.ValidatedPartitionSpec
   :members:
   :undoc-members:
   :no-index:

.. autofunction:: jano.validation.validate_strategy
   :no-index:

.. autofunction:: jano.validation.validate_partition_spec
   :no-index:
