Simulación y reporting
======================

Jano puede describir una simulación temporal sobre un dataset concreto y exponerla de tres formas complementarias:

- un ``SimulationSummary`` estructurado
- un reporte HTML standalone con timeline
- ``SimulationChartData`` listo para graficar en tus propias visualizaciones Python

El entry point principal es ``describe_simulation()`` sobre ``TemporalBacktestSplitter``.

Si querés correr una simulación completa sin iterar folds manualmente, la interfaz recomendada es ``TemporalSimulation``.

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

   from jano import TemporalPartitionSpec, TemporalSimulation

   frame = pd.DataFrame(
       {
           "timestamp": pd.date_range("2024-01-01", periods=365, freq="D"),
           "feature": range(365),
           "target": range(100, 465),
       }
   )

   simulation = TemporalSimulation(
       time_col="timestamp",
       partition=TemporalPartitionSpec(
           layout="train_test",
           train_size="10D",
           test_size="5D",
       ),
       step="5D",
       strategy="rolling",
   )

   result = simulation.run(frame, title="Walk-forward simulation")

   print(result.total_folds)
   print(result.to_frame().head())
   print(result.html[:120])
   print(result.chart_data.segment_stats)

Si querés inspeccionar la simulación antes de materializar folds, usá ``plan()``:

.. container:: example-block

   Simulación planificada

.. code-block:: python

   plan = simulation.plan(frame, title="Plan walk-forward")
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

   simulation = TemporalSimulation(
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

   result = simulation.run(frame, title="15 iteraciones diarias de retraining")

``TemporalSimulation`` también acepta ``end_at`` cuando querés restringir la simulación a una ventana temporal acotada.

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

Estudios con cutoff fijo
------------------------

Estos son casos especiales encima del workflow básico de simulación.

Jano los expone como policies temporales dedicadas en lugar de dejarlos como recetas manuales.

.. container:: example-block

   Test fijo, train creciente

.. code-block:: python

   from jano import TrainGrowthPolicy

   policy = TrainGrowthPolicy(
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
       metrics=["mae", "rmse"],
   )

   print(result.to_frame()[["train_size", "mae", "rmse"]])
   print(result.find_optimal_train_size(metric="rmse", tolerance=0.01))

Esto mantiene fijo el mismo test mientras train se expande hacia el pasado. Es la forma correcta para preguntas sobre suficiencia de historia y eficiencia de datos.

El caso opuesto también es común: dejar train fijo y mover test día a día para medir cuánto tiempo un modelo o regla mantiene su performance sin retraining.

.. container:: example-block

   Train fijo, test móvil

.. code-block:: python

   from jano import PerformanceDecayPolicy

   policy = PerformanceDecayPolicy(
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
       metrics=["mae", "rmse"],
   )

   print(result.to_frame()[["window", "test_start", "rmse"]])
   print(result.find_drift_onset(metric="rmse", threshold=0.15, baseline="first"))

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

Preview simple de HTML
----------------------

Debajo hay un mock compacto del tipo de timeline que muestra el reporte HTML generado.

.. raw:: html

   <div class="simulation-preview">
     <div class="preview-top">
       <div class="preview-kicker">Simulation report</div>
       <div class="preview-title">Walk-forward simulation</div>
       <div class="preview-meta">
         <span class="preview-chip">Rows: 365</span>
         <span class="preview-chip">Folds: 6</span>
         <span class="preview-chip">Strategy: rolling</span>
       </div>
     </div>
     <div class="preview-body">
       <div class="preview-row">
         <div class="preview-label">Fold 0</div>
         <div class="preview-track">
           <div class="preview-segment train" style="left: 4%; width: 44%;"></div>
           <div class="preview-segment validation" style="left: 53%; width: 14%;"></div>
           <div class="preview-segment test" style="left: 72%; width: 18%;"></div>
         </div>
       </div>
       <div class="preview-row">
         <div class="preview-label">Fold 1</div>
         <div class="preview-track">
           <div class="preview-segment train" style="left: 9%; width: 44%;"></div>
           <div class="preview-segment validation" style="left: 58%; width: 14%;"></div>
           <div class="preview-segment test" style="left: 77%; width: 18%;"></div>
         </div>
       </div>
     </div>
   </div>
