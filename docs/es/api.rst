Referencia de API
=================

.. raw:: html

   <p class="api-lead">
     La superficie recomendada es deliberadamente chica. La mayoría de los workflows
     empiezan con <code>WalkForwardPolicy</code>, <code>TrainHistoryPolicy</code> o
     <code>DriftMonitoringPolicy</code>, y recién bajan a simulación explícita,
     planning o splitter cuando necesitan control de nivel más bajo.
   </p>

   <p class="api-lead">
     Los inputs públicos pueden venir de <code>pandas</code>, <code>numpy</code> o
     <code>polars</code>. Cuando la fuente no es pandas, Jano la normaliza en el borde
     y mantiene la misma superficie de split y reporting.
   </p>

Workflow principal
------------------

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

Policies temporales
-------------------

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

Objetos de fold
---------------

.. autoclass:: jano.splits.TimeSplit
   :members:
   :undoc-members:
   :no-index:

Objetos de reporting
--------------------

.. autoclass:: jano.reporting.SimulationSummary
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.reporting.SimulationChartData
   :members:
   :undoc-members:
   :no-index:

Helpers de tipos y validación
-----------------------------

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
