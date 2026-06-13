Simulación y reporting
======================

Jano puede describir una simulación temporal sobre un dataset concreto y exponerla de dos formas complementarias:

- un ``SimulationSummary`` estructurado
- ``SimulationChartData`` listo para graficar en tus propias visualizaciones Python

El entry point principal es ``describe_simulation()`` sobre ``TemporalBacktestSplitter``.

Si querés correr una simulación completa sin iterar folds manualmente, la interfaz recomendada es ``WalkForwardPolicy``.

El workflow general está pensado por capas:

- usar clases high-level cuando la pregunta ya está encapsulada
- inspeccionar o recortar iteraciones con ``plan()`` cuando haga falta
- y caer al modo manual de folds cuando querés componer todo por tu cuenta

La misma API acepta tres inputs tabulares:

- ``pandas.DataFrame``
- ``numpy.ndarray`` usando referencias enteras como ``time_col=0``
- ``polars.DataFrame`` cuando está instalado el extra opcional de Polars

Eso significa que la configuración temporal permanece igual aunque cambie la fuente upstream. Lo único que cambia es cómo referenciás las columnas:

- por nombre para pandas y Polars
- por posición entera para arrays NumPy

Ejemplo
-------

.. container:: example-block

   pandas.DataFrame

.. code-block:: python

   import pandas as pd

   from jano import TemporalPartitionSpec, WalkForwardPolicy

   frame = pd.DataFrame(
       {
           "timestamp": pd.date_range("2024-01-01", periods=365, freq="D"),
           "feature": range(365),
           "target": range(100, 465),
       }
   )

   policy = WalkForwardPolicy(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="10D",
           test_size="5D",
       ),
       step="5D",
       strategy="rolling",
   )

   result = policy.run(frame, title="Walk-forward simulation")

   print(result.total_folds)
   print(result.to_frame().head())
   print(result.chart_data.segment_stats)

Si querés inspeccionar la simulación antes de materializar folds, usá ``plan()``:

.. container:: example-block

   Simulación planificada

.. code-block:: python

   plan = policy.plan(frame, title="Plan walk-forward")
   print(plan.total_folds)
   print(plan.to_frame().head())

   filtered = plan.exclude_windows(
       train=[("2025-12-20", "2026-01-05")],
   ).select_from_iteration(5)

   result = filtered.materialize()

El frame del plan incluye el índice de iteración, boundaries por segmento y conteos de filas, de modo que podés inspeccionar la estructura primero y materializar después sólo los folds que te interesan.

Podés anclar la simulación a un punto de tiempo específico y limitar el número de folds:

.. container:: example-block

   Simulación anclada

.. code-block:: python

   policy = WalkForwardPolicy(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="15D",
           test_size="4D",
       ),
       step="1D",
       strategy="rolling",
       start_at="2025-09-01",
       max_folds=15,
   )

   result = policy.run(frame, title="15 iteraciones diarias de retraining")

``WalkForwardPolicy`` también acepta ``end_at`` cuando querés restringir la simulación a una ventana temporal acotada.

Correr un modelo con policies de retraining
-------------------------------------------

Cuando no querés escribir un loop manual tipo ``for train_idx, test_idx in splitter``,
podés usar ``WalkForwardRunner`` por encima del workflow temporal. El runner mantiene
las responsabilidades separadas:

- ``WalkForwardPolicy`` sigue definiendo la geometría de folds
- ``WalkForwardRunner`` ejecuta el estimador sobre esos folds
- una policy de retraining decide si el modelo debe refittearse antes de cada fold

.. code-block:: python

   import numpy as np

   from jano import TemporalPartitionSpec, WalkForwardPolicy, WalkForwardRunner

   def mae(y_true, y_pred):
       return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

   def rmse(y_true, y_pred):
       return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

   policy = WalkForwardPolicy(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="30D",
           test_size="7D",
       ),
       step="7D",
       strategy="rolling",
   )

   runner = WalkForwardRunner(
       model=model,
       target_col="target",
       feature_cols=["feature_a", "feature_b"],
       retrain="always",
       metrics={"mae": mae, "rmse": rmse},
   )

   run = runner.run(policy, frame)
   print(run.to_frame().head())
   print(run.summary())
   print(run.metric_trajectory().head())
   print(run.retrain_events())

Los resultados del runner son data-first. Jano no necesita ser dueño de la capa
final de dashboard; expone evidencia estructurada para que notebooks, agentes,
herramientas de presentación o aplicaciones la visualicen con su propio estilo:

- ``run.fold_summary()`` devuelve geometría temporal y metadata de retraining.
- ``run.metric_trajectory()`` devuelve métricas en formato long, listas para graficar.
- ``run.retrain_events()`` devuelve solo los folds donde el estimador se refitteó.
- ``run.predictions_frame()`` devuelve predicciones row-level sobre los tests.
- ``run.report_data()`` y ``run.to_dict()`` devuelven diccionarios estructurados
  para capas externas de reporting.

Los modos shorthand de retraining son:

- ``retrain="always"`` o ``retrain=True`` para refittear en cada fold
- ``retrain="never"`` o ``retrain=False`` para entrenar una vez y reutilizar el mismo modelo
- ``retrain="periodic"`` más ``retrain_interval=K`` para refittear cada ``K`` folds

Perfiles de evaluación
----------------------

``EvaluationProfile`` separa cómo se mide una corrida temporal de cuándo el runner
debería reentrenar el estimador. Jano no implementa fórmulas de métricas; el
contrato principal es que el usuario pase la función de métrica o pérdida que
corresponde a su problema.

.. code-block:: python

   import numpy as np

   from jano import EvaluationProfile, FunctionRetrainPolicy, WalkForwardRunner

   def daily_cost(y_true, y_pred):
       return float(np.mean(np.abs(y_true - y_pred)))

   def retrain_rule(context):
       if context.history.empty:
           return True
       latest = context.history["daily_cost"].iloc[-1]
       limit = limit_for_date(context.split.boundaries["test"].end)
       return latest > limit

   runner = WalkForwardRunner(
       model=model,
       target_col="target",
       feature_cols=["feature_a", "feature_b"],
       evaluation=EvaluationProfile(
           metrics={"daily_cost": daily_cost},
           metric_directions={"daily_cost": "min"},
           primary_metric="daily_cost",
       ),
       retrain_policy=FunctionRetrainPolicy(retrain_rule),
   )

El profile le dice a Jano qué métricas existen, si cada una debe minimizarse o
maximizarse y cuál es la señal operativa principal. ``FunctionRetrainPolicy`` le
da al usuario control total sobre la decisión de reentrenar, incluyendo thresholds
dinámicos, losses que cambian por fecha o reglas de negocio.

También hay perfiles convenientes cuando el tipo de problema ayuda a explicitar
la intención. No agregan fórmulas de métricas; agrupan métricas provistas por el
usuario según el estilo del problema:

- ``RegressionProfile`` etiqueta pérdidas de regresión provistas por el usuario.
- ``ClassificationProfile`` etiqueta scores de clasificación provistos por el usuario.
- ``OrdinalClassificationProfile`` está pensado para clases ordenadas con costos custom.
- ``RankingProfile`` está pensado para métricas de ranking o retrieval provistas por el usuario.

También podés pasar una policy explícita:

.. code-block:: python

   from jano import DriftBasedRetrain, WalkForwardRunner

   runner = WalkForwardRunner(
       model=model,
       target_col="target",
       retrain_policy=DriftBasedRetrain(
           metric="mae",
           threshold=0.05,
           baseline="last_retrain",
       ),
       metrics={"mae": mae},
   )

``DriftBasedRetrain`` usa métricas observadas en folds previos para decidir si el fold
siguiente debería disparar un retraining. Eso lo vuelve útil como benchmark operativo
inicial, sin meter lógica de drift dentro del splitter.

Cuando ``DriftBasedRetrain`` se crea sin una métrica explícita, usa el
``primary_metric`` del perfil de evaluación.

Ejecutar sistemas temporales con policies de actualización
----------------------------------------------------------

No todo sistema temporal se actualiza con ``fit()`` y ``predict()``. Un pipeline
RAG, una configuración de prompts o un job de fine-tuning suelen comportarse más
como "actualizar estado y luego evaluar ese estado sobre la siguiente ventana".

``TemporalSystemRunner`` cubre ese caso sin cambiar el core temporal de Jano.
Mantiene la misma geometría de folds, pero reemplaza el contrato del estimador por
el protocolo ``UpdateableSystem``:

- ``update(train_frame)`` refresca el estado del sistema para la ventana de train actual
- ``evaluate(state, test_frame)`` devuelve métricas definidas por el usuario sobre la siguiente ventana de test

.. code-block:: python

   import numpy as np
   import pandas as pd

   from jano import (
       PeriodicRetrain,
       SystemEvaluationResult,
       SystemUpdateResult,
       TemporalPartitionSpec,
       TemporalSystemRunner,
       WalkForwardPolicy,
   )

   class MeanTargetSystem:
       def update(self, train_frame: pd.DataFrame):
           mean_target = float(train_frame["target"].mean())
           return SystemUpdateResult(
               state=mean_target,
               metadata={"train_target_mean": mean_target},
           )

       def evaluate(self, state, test_frame: pd.DataFrame):
           predictions = np.repeat(float(state), len(test_frame))
           mae = float(np.mean(np.abs(test_frame["target"] - predictions)))
           return SystemEvaluationResult(
               metrics={"mae": mae},
               metadata={"prediction_mean": float(state)},
           )

   policy = WalkForwardPolicy(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="30D",
           test_size="7D",
       ),
       step="7D",
       strategy="rolling",
   )

   runner = TemporalSystemRunner(
       system=MeanTargetSystem(),
       update_policy=PeriodicRetrain(2),
       metric_directions={"mae": "min"},
       primary_metric="mae",
   )

   run = runner.run(policy, frame)
   print(run.to_frame().head())
   print(run.metric_trajectory().head())
   print(run.update_events())

La distinción importante es conceptual más que técnica: el paso de actualización
puede significar reentrenar un modelo, reconstruir un índice de retrieval,
refrescar un set de prompts o relanzar un fine-tuning. Jano sigue siendo dueño del
particionado temporal y de la simulación de policies; el objeto sistema es dueño
de la lógica operativa de actualización.

Scenarios built-in
------------------

Algunas preguntas operativas aparecen lo suficiente como para tener un workflow
prearmado, pero no son tan generales como para convertirse en comportamiento del
runner. Jano expone esos workflows como ``jano.scenarios``. Los scenarios se
construyen sobre las primitivas core; no modifican ``WalkForwardRunner``.

Bandas de predicción por fold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``estimate_prediction_band_by_fold`` responde:

"En cada fold temporal, qué banda de predicción produce mi propio método de
incertidumbre para el test futuro?"

Jano no implementa cross-validation, bootstrap, conformal prediction ni fórmulas
de intervalos de confianza en este scenario. El usuario provee un
``band_estimator`` como objeto o callable. Ese objeto recibe el contexto del fold
y devuelve arrays ``lower`` y ``upper`` para el test actual.

.. code-block:: python

   import numpy as np

   from jano import estimate_prediction_band_by_fold

   def mae(y_true, y_pred):
       return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

   class FixedWidthBand:
       def estimate(self, context):
           return {
               "lower": context.predictions - 5.0,
               "upper": context.predictions + 5.0,
               "artifacts": {"method": "fixed_width"},
           }

   result = estimate_prediction_band_by_fold(
       frame,
       estimator=model,
       band_estimator=FixedWidthBand(),
       time_col="timestamp",
       target_col="target",
       feature_cols=["feature_a", "feature_b"],
       train_size="90D",
       test_size="7D",
       step="7D",
       strategy="rolling",
       metrics={"mae": mae},
   )

   print(result.to_frame().head())
   print(result.predictions_frame().head())
   print(result.band_summary())

El resultado expone métricas y resumen de bandas por fold vía
``to_frame()``, bounds inferiores y superiores por fila vía
``predictions_frame()``, y artifacts definidos por el usuario vía
``artifacts_frame()``. Un ``band_estimator`` real puede envolver ``KFold`` de
scikit-learn, un método custom de remuestreo, conformal prediction u otra técnica.

Alineación a días calendario
----------------------------

Por defecto, las ventanas por duración arrancan desde el primer timestamp observado. Si la
primera fila es ``2024-01-01 05:21`` y ``train_size="7D"``, la primera ventana de train
termina en ``2024-01-08 05:21``.

A veces eso no es lo buscado. En datasets operativos, podés querer días calendario completos:
train hasta Jan 7 y test desde Jan 8.

Usá ``calendar_frequency="D"`` en ``TemporalPartitionSpec`` para eso:

.. code-block:: python

   simulation = WalkForwardPolicy(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="7D",
           test_size="1D",
           calendar_frequency="D",
       ),
       step="1D",
       strategy="rolling",
   )

Jano usa boundaries cerrados-abiertos: ``[start, end)``. Un train que termina en
``2024-01-08 00:00:00`` contiene filas anteriores a Jan 8, mientras que test puede empezar
exactamente en Jan 8.

Si el source data es un array NumPy, referenciá la columna temporal por posición entera:

.. container:: example-block

   Input NumPy

.. code-block:: python

   import numpy as np

   values = np.array(
       [
           ["2025-09-01", 0.2, 1],
           ["2025-09-02", 0.4, 0],
           ["2025-09-03", 0.1, 1],
           ["2025-09-04", 0.3, 0],
       ],
       dtype=object,
   )

   simulation = TemporalSimulation(
       time_col=0,
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="2D",
           test_size="1D",
       ),
       step="1D",
       strategy="single",
   )

Si el source data es un frame de Polars, la misma configuración funciona con columnas nombradas:

.. container:: example-block

   polars.DataFrame

.. code-block:: python

   import polars as pl

   frame = pl.DataFrame(
       {
           "timestamp": ["2025-09-01", "2025-09-02", "2025-09-03", "2025-09-04"],
           "feature": [0.2, 0.4, 0.1, 0.3],
           "target": [1, 0, 1, 0],
       }
   ).with_columns(pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d"))

   simulation = TemporalSimulation(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="2D",
           test_size="1D",
       ),
       step="1D",
       strategy="single",
   )

   result = simulation.run(frame)

Elegir el motor de particionado
-------------------------------

Todas las APIs de simulación de alto nivel aceptan ``engine``. El default,
``engine="auto"``, elige la representación interna usada para calcular boundaries
temporales e índices de filas:

.. code-block:: python

   simulation = TemporalSimulation(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="7D",
           test_size="1D",
       ),
       step="1D",
       strategy="rolling",
       engine="auto",
   )

   result = simulation.run(frame)
   print(result.engine_metadata.to_dict())

``engine="auto"`` mantiene inputs Polars y NumPy nativos para planning cuando es seguro.
Usá ``engine="pandas"`` para forzar el camino pandas estable, o ``engine="polars"`` /
``engine="numpy"`` cuando quieras forzar un motor específico de particionado.

Control manual low-level
------------------------

Cuando necesitás control directo de los folds o integrar con un training loop externo, usá ``TemporalBacktestSplitter`` directamente.

.. code-block:: python

   from jano import TemporalBacktestSplitter, TemporalPartitionSpec

   splitter = TemporalBacktestSplitter(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="10D",
           test_size="5D",
       ),
       step="5D",
       strategy="rolling",
   )

   for split in splitter.iter_splits(frame):
       print(split.summary())

El mismo splitter también puede precalcular la geometría completa de la partición:

.. code-block:: python

   plan = splitter.plan(frame)
   print(plan.to_frame()[["iteration", "train_start", "train_end", "test_start", "test_end"]])

Este es el modo manual completo. Es el lugar correcto cuando querés componer por tu cuenta todo el proceso: layouts de partición, gaps temporales, exclusión de fechas especiales, lookbacks por grupo de features, training loops del modelo o cualquier lógica de evaluación que no convenga esconder detrás de un helper high-level.

Estudios con cutoff fijo
------------------------

Estos son casos especiales encima del workflow básico de simulación.

Jano los expone como policies temporales dedicadas en lugar de dejarlos como recetas manuales.

.. container:: example-block

   Test fijo, train creciente

.. code-block:: python

   from jano import TrainHistoryPolicy

   policy = TrainHistoryPolicy(
       "timestamp",
       cutoff="2025-09-15",
       train_sizes=["7D", "14D", "21D", "28D"],
       test_size="4D",
   )

   result = policy.evaluate(
       frame,
       model=model,
       target_col="target",
       feature_cols=["feature_1", "feature_2"],
       metrics={"mae": mae, "rmse": rmse},
   )

   print(result.to_frame()[["train_size", "mae", "rmse"]])
   print(result.find_optimal_train_size(metric="rmse", tolerance=0.01))

Esto mantiene fijo el mismo test mientras train se expande hacia el pasado. Es la forma correcta para preguntas sobre suficiencia de historia y eficiencia de datos.

El caso opuesto también es común: dejar train fijo y mover test día a día para medir cuánto tiempo un modelo o regla mantiene su performance sin retraining.

.. container:: example-block

   Train fijo, test móvil

.. code-block:: python

   from jano import DriftMonitoringPolicy

   policy = DriftMonitoringPolicy(
       "timestamp",
       cutoff="2025-09-15",
       train_size="30D",
       test_size="3D",
       step="1D",
       max_windows=10,
   )

   result = policy.evaluate(
       frame,
       model=model,
       target_col="target",
       feature_cols=["feature_1", "feature_2"],
       metrics={"mae": mae, "rmse": rmse},
   )

   print(result.to_frame()[["window", "test_start", "rmse"]])
   print(result.find_drift_onset(metric="rmse", threshold=0.15, baseline="first"))

Policy compuesta: optimizar historia de train dentro de cada iteración walk-forward
-----------------------------------------------------------------------------------

Cuando la pregunta es más compleja, podés seguir dentro de la superficie recomendada.

``RollingTrainHistoryPolicy`` ejecuta un loop walk-forward externo y, dentro de cada
iteración, elige la menor ventana de train que queda dentro de la tolerancia del mejor
score para el test fijo de esa iteración.

.. code-block:: python

   from jano import RollingTrainHistoryPolicy, TemporalPartitionSpec

   policy = RollingTrainHistoryPolicy(
       "timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="30D",
           test_size="1D",
       ),
       step="1D",
       strategy="rolling",
       max_folds=10,
       train_sizes=["5D", "10D", "15D", "30D"],
   )

   result = policy.evaluate(
       frame,
       model=model,
       target_col="target",
       feature_cols=["feature_1", "feature_2"],
       metrics={"rmse": rmse},
       metric="rmse",
       tolerance=0.01,
   )

   print(result.to_frame().head())
   print(result.summary())

Semántica temporal y control de leakage
---------------------------------------

Cuando una sola columna temporal no alcanza, podés pasar un ``TemporalSemanticsSpec`` en lugar de un simple string en ``time_col``.

Esto permite separar:

- la timeline usada para reporting y bounds globales
- la columna de orden interno
- la columna temporal usada para decidir la elegibilidad de cada segmento

Eso importa en datasets más parecidos a producción, donde tiempo de evento y tiempo de disponibilidad no son iguales.

.. code-block:: python

   from jano import TemporalBacktestSplitter, TemporalPartitionSpec, TemporalSemanticsSpec

   splitter = TemporalBacktestSplitter(
       time_col=TemporalSemanticsSpec(
           timeline_col="departured_at",
           segment_time_cols={
               "train": "arrived_at",
               "test": "departured_at",
           },
       ),
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="14D",
           test_size="3D",
           gap_before_train="1D",
           gap_before_test="1D",
           gap_after_test="2D",
       ),
       step="1D",
       strategy="rolling",
   )

Lookback por grupo de features
------------------------------

Algunos pipelines necesitan una capa adicional más allá del fold: distintos grupos de features pueden requerir diferente cantidad de historia, aun cuando el segmento supervisado de ``train`` sea fijo.

.. code-block:: python

   from jano import FeatureLookbackSpec

   split = next(splitter.iter_splits(frame))
   lookbacks = FeatureLookbackSpec(
       default_lookback="15D",
       group_lookbacks={"lag_features": "65D"},
       feature_groups={"lag_features": ["lag_30", "lag_60"]},
   )

   history = split.slice_feature_history(
       frame,
       lookbacks,
       time_col="timestamp",
       segment_name="train",
   )

   recent_context = history["__default__"]
   lag_context = history["lag_features"]

Los resultados quedan expuestos como estructura tabular y como ``chart_data`` listo para
graficar desde notebooks, dashboards o cualquier capa de visualización externa.
