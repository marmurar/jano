API reference
=============

.. raw:: html

   <p class="api-lead">
     The recommended surface is intentionally small. Most workflows start with
     <code>WalkForwardPolicy</code>, <code>TrainHistoryPolicy</code> or
     <code>DriftMonitoringPolicy</code>, then drop down to explicit simulation,
     planning or splitter objects only when lower-level control is needed.
   </p>

   <p class="api-lead">
     Public inputs can come from <code>pandas</code>, <code>numpy</code> or
     <code>polars</code>. When the source is not pandas, Jano normalizes it at the boundary
     and keeps the same split and reporting surface.
   </p>

Main workflow
-------------

.. autoclass:: jano.workflows.WalkForwardPolicy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.workflows.TrainHistoryPolicy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.workflows.DriftMonitoringPolicy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.workflows.RollingTrainHistoryPolicy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.workflows.RollingTrainHistoryResult
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.simulation.TemporalSimulation
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.simulation.SimulationResult
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.planning.SimulationPlan
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

.. autoclass:: jano.types.FeatureLookbackSpec
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.splitters.TemporalBacktestSplitter
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.planning.PartitionPlan
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.planning.PlannedFold
   :members:
   :undoc-members:
   :no-index:

Temporal policies
-----------------

.. autoclass:: jano.policies.TrainGrowthPolicy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.policies.TrainGrowthResult
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.policies.PerformanceDecayPolicy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.policies.PerformanceDecayResult
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

MCP helper functions
--------------------

These functions power the optional local MCP server and are useful for understanding
the exact tool contract exposed to AI clients.

.. autofunction:: jano.mcp_tools.preview_dataset
   :no-index:

.. autofunction:: jano.mcp_tools.plan_walk_forward
   :no-index:

.. autofunction:: jano.mcp_tools.run_walk_forward
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
