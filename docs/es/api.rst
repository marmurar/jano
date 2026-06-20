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

.. autoclass:: jano.campaigns.SimulationVariant
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.campaigns.SimulationCampaign
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.campaigns.BatchSimulationResult
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.runner.WalkForwardRunner
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.runner.WalkForwardRunResult
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.system_runner.TemporalSystemRunner
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.system_runner.SystemRunResult
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.systems.SystemUpdateResult
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.systems.SystemEvaluationResult
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.systems.UpdateableSystem
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.online.OnlineTemporalRunner
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.online.OnlineRunResult
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.online.OnlineUpdatePolicyStudy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.online.OnlineUpdatePolicyStudyResult
   :members:
   :undoc-members:
   :no-index:

Scenarios built-in
------------------

.. autofunction:: jano.scenarios.estimate_prediction_band_by_fold
   :no-index:

.. autoclass:: jano.scenarios.PredictionBandContext
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.scenarios.PredictionBandScenarioResult
   :members:
   :undoc-members:
   :no-index:

Perfiles de evaluación
----------------------

.. autoclass:: jano.evaluation.EvaluationProfile
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.evaluation.ResolvedEvaluationProfile
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.evaluation.RegressionProfile
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.evaluation.ClassificationProfile
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.evaluation.OrdinalClassificationProfile
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.evaluation.RankingProfile
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

Policies de retraining
----------------------

.. autoclass:: jano.runner.RetrainPolicy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.runner.AlwaysRetrain
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.runner.NeverRetrain
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.runner.PeriodicRetrain
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.runner.FunctionRetrainPolicy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.runner.DriftBasedRetrain
   :members:
   :undoc-members:
   :no-index:

Estrategias de actualización online
-----------------------------------

.. autoclass:: jano.online.OnlineUpdateStrategy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.online.OnlineUpdatePolicy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.online.PartialFitUpdateStrategy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: jano.online.RefitUpdateStrategy
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

Funciones helper MCP
--------------------

Estas funciones sostienen el servidor MCP local opcional y sirven para entender el
contrato exacto de tools que se expone a clientes de IA.

.. autofunction:: jano.mcp_tools.preview_dataset
   :no-index:

.. autofunction:: jano.mcp_tools.inspect_and_recommend_dataset
   :no-index:

.. autofunction:: jano.mcp_tools.plan_walk_forward
   :no-index:

.. autofunction:: jano.mcp_tools.run_walk_forward
   :no-index:

.. autofunction:: jano.mcp_tools.run_walk_forward_baseline
   :no-index:

.. autofunction:: jano.mcp_tools.compare_retrain_policies
   :no-index:

.. autofunction:: jano.mcp_tools.find_train_history_window
   :no-index:

.. autofunction:: jano.mcp_tools.monitor_decay
   :no-index:

Helpers de tipos y validación
-----------------------------

.. autoclass:: jano.engines.PartitionEngineMetadata
   :members:
   :undoc-members:
   :no-index:

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
