Modos de Partición
==================

Jano separa dos ideas relacionadas que suelen mezclarse:

- **Partición temporal**: divide un dataset histórico en ventanas de train,
  validation y test ordenadas por tiempo.
- **Partición online basada en eventos**: divide un stream observado en eventos
  o micro-batches, y evalúa cómo se comporta un modelo a medida que llegan nuevas
  observaciones.

Ambos modos son causales: los datos observados después no deben influir en
decisiones que se habrían tomado antes. La diferencia es la unidad que hace
avanzar la evaluación.

Partición Temporal
------------------

La partición temporal es el modo base para backtesting de sistemas tabulares de
machine learning. Responde preguntas como:

- ¿cómo habría sido la performance si el modelo se hubiese reentrenado todos los
  días?
- ¿cuánta historia debería contener la ventana de train?
- ¿cómo se degrada un modelo fijo sobre ventanas futuras?

Usá ``TemporalBacktestSplitter`` directamente cuando querés controlar manualmente
el loop de folds:

.. code-block:: python

   from jano import TemporalBacktestSplitter

   splitter = TemporalBacktestSplitter(
       time_col="timestamp",
       train_size="30D",
       test_size="7D",
       step="7D",
       strategy="rolling",
   )

   for train_idx, test_idx in splitter.split(frame):
       train = frame.iloc[train_idx]
       test = frame.iloc[test_idx]

Usá ``WalkForwardPolicy`` o ``TemporalSimulation`` cuando querés que Jano genere
un plan, ejecute los folds y exponga resultados auditables.

Partición Online Basada en Eventos
----------------------------------

La partición online basada en eventos no es una simulación walk-forward sobre
folds históricos fijos. Es un patrón causal de evaluación online: inicializar un
modelo, predecir el próximo evento o micro-batch, observar el target, actualizar
el modelo y repetir.

Sirve cuando la pregunta operativa no es solo *¿cuándo debería reentrenar por
calendario?*, sino también *¿cuántas observaciones nuevas debería esperar antes
de actualizar el modelo?*

Usá ``OnlineTemporalRunner`` con ``PartialFitUpdateStrategy`` cuando el modelo
soporta actualización incremental real vía ``partial_fit``:

.. code-block:: python

   from jano import OnlineTemporalRunner, PartialFitUpdateStrategy

   runner = OnlineTemporalRunner(
       model=model,
       time_col="timestamp",
       target_col="target",
       feature_cols=["feature_a", "feature_b"],
       initial_train_size="30D",
       update_size=1,
       metrics={"mae": mae, "rmse": rmse},
       update_strategy=PartialFitUpdateStrategy(),
   )

   run = runner.run(frame)
   print(run.to_frame().head())
   print(run.metric_trajectory().head())
   print(run.summary())

La secuencia es causal por diseño:

- inicializa el modelo sobre la ventana inicial de train
- predice el próximo evento o micro-batch
- mide la predicción cuando se observa el target
- actualiza el modelo con ese batch observado
- repite

``update_size=1`` significa actualización por evento. También podés usar batches
por filas como ``update_size=100`` o por duración como ``update_size="1D"``. Eso
permite comparar políticas de actualización por evento, por batch de filas o por
batch temporal sin cambiar el resto de la configuración.

No todos los estimadores soportan ``partial_fit``. Para modelos clásicos
``fit/predict``, usá ``RefitUpdateStrategy``:

.. code-block:: python

   from jano import OnlineTemporalRunner, RefitUpdateStrategy

   runner = OnlineTemporalRunner(
       model=model,
       time_col="timestamp",
       target_col="target",
       feature_cols=["feature_a", "feature_b"],
       initial_train_size="30D",
       update_size="1D",
       metrics={"mae": mae},
       update_strategy=RefitUpdateStrategy(max_train_rows=10_000),
   )

Esta estrategia refittea después de cada batch observado. Es más costosa que
``partial_fit``, pero funciona con estimadores estándar y puede mantener historia
acotada con ``max_train_rows``.

Encontrar una Policy de Actualización por Observaciones
-------------------------------------------------------

``OnlineUpdatePolicyStudy`` compara varias cadencias de actualización sobre el
mismo stream temporal. Eso permite preguntar si las actualizaciones del modelo
deberían dispararse por calendario, por cantidad de filas o por evidencia
acumulada:

.. code-block:: python

   from jano import OnlineUpdatePolicy, OnlineUpdatePolicyStudy, RefitUpdateStrategy

   study = OnlineUpdatePolicyStudy(
       model=model,
       time_col="timestamp",
       target_col="target",
       feature_cols=["feature_a", "feature_b"],
       initial_train_size="30D",
       policies=[
           OnlineUpdatePolicy("every-event", update_size=1, update_strategy=RefitUpdateStrategy()),
           OnlineUpdatePolicy("every-100-events", update_size=100, update_strategy=RefitUpdateStrategy()),
           OnlineUpdatePolicy("daily", update_size="1D", update_strategy=RefitUpdateStrategy()),
       ],
       metrics={"mae": mae},
   )

   comparison = study.run(frame)

   print(comparison.to_frame())
   print(comparison.metric_trajectory().head())
   print(comparison.find_optimal_policy(metric="mae", update_cost_weight=0.01))

El parámetro opcional ``update_cost_weight`` penaliza policies que actualizan muy
seguido. Así el output sigue siendo data-first, pero el tradeoff queda explícito:
una policy puede ganar porque predice mejor, porque actualiza menos o porque
ofrece el mejor compromiso ajustado por costo.
