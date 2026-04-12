Referencia de API
=================

.. raw:: html

   <p class="api-lead">
     La API es deliberadamente compacta. La mayoría de los workflows empiezan con un
     <code>TemporalPartitionSpec</code>, lo pasan a <code>TemporalSimulation</code> para
     una corrida completa, y opcionalmente bajan a <code>TemporalBacktestSplitter</code>
     cuando necesitan control manual de folds.
   </p>

   <p class="api-lead">
     Los inputs públicos pueden venir de <code>pandas</code>, <code>numpy</code> o
     <code>polars</code>. Cuando la fuente no es pandas, Jano la normaliza en el borde
     y mantiene la misma superficie de split y reporting.
   </p>

Workflow principal
------------------

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

.. autoclass:: jano.types.FeatureLookbackSpec
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.splitters.TemporalBacktestSplitter
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
