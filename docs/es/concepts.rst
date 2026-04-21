Conceptos
=========

Particionado temporal
---------------------

Jano modela la evaluación como un problema de particionado temporal, no como uno de muestreo aleatorio.

Ese enfoque también es útil cuando querés evidenciar drift en resultados de simulación, porque los cambios a través del tiempo permanecen visibles en lugar de quedar diluidos por splits aleatorios.

La distinción importante es que Jano no trata el particionado como un paso de preprocesamiento de una sola vez. Lo trata como un proceso temporal. Una política de partición define:

- cuánta historia entra en train
- qué tan grande es el horizonte de evaluación
- cuánto se mueve la simulación en cada paso
- qué gaps temporales deben existir para evitar leakage

En ese sentido, una simulación se entiende mejor como una secuencia de folds causalmente válidos que como una única descomposición train/test.

.. math::

   \left\{(\mathcal{D}_{train}^{(k)}, \mathcal{D}_{test}^{(k)})\right\}_{k=1}^K

Por eso Jano es útil tanto para backtesting como para preguntas operativas sobre retraining, estabilidad temporal y cambios de régimen.

Internamente, el motor trabaja sobre pandas. En el borde público, sin embargo, Jano acepta:

- ``pandas.DataFrame`` con columnas nombradas
- ``numpy.ndarray`` con referencias enteras como ``time_col=0``
- ``polars.DataFrame`` convertido internamente antes de generar folds

En lugar de pedir un share aleatorio de filas, definís una política de partición:

- qué tan grande es train
- qué tan grandes son validation o test
- si debe haber gaps temporales
- y cómo debe moverse el split a lo largo del tiempo

Workflow composicional
----------------------

Jano está pensado como una herramienta composicional.

La progresión buscada es:

- empezar con una partición temporal simple
- agregar movimiento con ``single``, ``rolling`` o ``expanding``
- inspeccionar la geometría con ``plan()``
- subir a policies high-level cuando la pregunta ya está encapsulada
- y bajar al modo manual completo cuando necesitás control total

En otras palabras, Jano ofrece tres niveles de uso:

- una superficie recomendada y chica como ``WalkForwardPolicy``, ``TrainHistoryPolicy`` o ``DriftMonitoringPolicy``
- workflows explícitos de nivel más bajo como ``TemporalSimulation``, ``TrainGrowthPolicy`` o ``PerformanceDecayPolicy``
- una capa intermedia de planning con ``plan()``
- un modo manual a través de ``TemporalBacktestSplitter`` e ``iter_splits()`` cuando querés componer a gusto particiones, gaps, historia de features y loops externos de entrenamiento

Ese último nivel importa porque no toda evaluación productiva entra en una clase predefinida. Jano debería ayudarte cuando el caso común alcanza, pero también dejarte componer tu propia lógica temporal cuando el problema lo exige.

Estrategias
-----------

``single``
  Produce una sola partición. Es el equivalente temporal de un split único, pero respetando el orden cronológico.

``rolling``
  Mueve una ventana fija de entrenamiento y evalúa repetidamente a medida que el tiempo avanza.

``expanding``
  Hace crecer la historia de entrenamiento mientras validation y test siguen avanzando hacia adelante.

Layouts
-------

``train_test``
  Produce un segmento de train y uno de test.

``train_val_test``
  Produce train, validation y test en ese orden.

Tamaños de segmento
-------------------

Jano acepta hoy tres familias de unidades:

- duraciones como ``"30D"`` o ``"12h"``
- conteos de filas como ``5000``
- fracciones como ``0.7``

Dentro de una misma partición, tamaños y gaps deben pertenecer a la misma familia de unidades.

En particiones basadas en duración, las ventanas usan tiempo transcurrido por defecto. Si
necesitás buckets calendario completos, configurá ``calendar_frequency`` en
``TemporalPartitionSpec``. Por ejemplo, ``calendar_frequency="D"`` alinea las ventanas a
días de medianoche a medianoche, en lugar de anclarlas al primer timestamp observado.

Salidas
-------

Jano expone dos vistas complementarias:

- ``plan()`` precalcula la geometría de la simulación como un objeto inspeccionable antes de materializar folds
- ``TemporalSimulation.run()`` materializa una simulación completa y devuelve un resultado reusable
- ``split()`` entrega tuplas de índices, útil para integración liviana
- ``iter_splits()`` entrega objetos ``TimeSplit`` con metadata y helpers
- ``describe_simulation()`` entrega ``SimulationSummary``, HTML o ``SimulationChartData`` para plots custom

Motor adaptativo de particionado
--------------------------------

Jano separa la representación usada para particionar de la representación usada para
reporting. Con ``engine="auto"``, el splitter elige el camino rápido seguro:

- inputs pandas siguen por el camino pandas
- inputs NumPy usan indexación de arrays cuando las columnas se referencian por posición
- inputs Polars mantienen extracción de columnas Polars para planning y generación de índices

Esto evita convertir un dataset grande de Polars o NumPy a pandas solo para calcular
boundaries y conteos de folds. Jano sigue materializando objetos pandas cuando un reporte
o slice de cara al usuario necesita semántica pandas.

Podés forzar un camino con ``engine="pandas"``, ``engine="polars"`` o ``engine="numpy"``.
La prioridad sigue siendo correctitud temporal primero, estabilidad de API después, y
velocidad en tercer lugar.

Planificación antes de materializar
-----------------------------------

Jano ahora expone una capa de planning entre la configuración y la ejecución.

Eso significa que primero podés calcular la geometría de todas las particiones futuras, inspeccionarla, filtrarla y recién después materializar los folds que realmente te interesan.

``plan()`` es útil cuando querés:

- inspeccionar la lista completa de iteraciones antes de entrenar nada
- entender cuántas filas tendría cada segmento
- arrancar desde la iteración ``N`` en vez del comienzo
- excluir folds cuyo train o test caen sobre fechas especiales
- trabajar sobre un plan precomputado en vez de slice del dataset inmediatamente

A nivel low-level:

.. code-block:: python

   plan = splitter.plan(frame)
   print(plan.to_frame().head())

A nivel high-level:

.. code-block:: python

   plan = simulation.plan(frame, title="Vista previa de la policy")
   filtered = plan.exclude_windows(
       train=[("2025-12-20", "2026-01-05")],
   ).select_from_iteration(10)

   result = filtered.materialize()

El frame del plan incluye una columna explícita ``iteration``, boundaries por segmento y conteos de filas. Eso permite razonar sobre la simulación como objeto de primer nivel, no solo como generador de folds.

Hipótesis temporales
--------------------

Las secciones anteriores describen la mecánica de particionado temporal. Encima de esa base,
Jano también puede codificar hipótesis de evaluación sobre cómo se comporta un modelo en el tiempo.

La progresión está pensada para ser incremental:

- primero particiones explícitas
- luego simulaciones walk-forward
- finalmente hipótesis operativas sobre suficiencia de historia o degradación temporal

Dos policies centrales ya forman parte del paquete, y cada una además tiene un wrapper recomendado más chico.

``TrainHistoryPolicy`` / ``TrainGrowthPolicy``
  Mantiene fijo el mismo test y expande train hacia atrás en el tiempo.

  Responde preguntas como:

  - ¿agregar más historia mejora el mismo test?
  - ¿puede una muestra más chica igualar la mejor calidad observada?
  - ¿dónde deja de ser útil seguir sumando historia?

``DriftMonitoringPolicy`` / ``PerformanceDecayPolicy``
  Mantiene train fijo y desplaza test hacia adelante.

  Responde preguntas como:

  - ¿cuánto tiempo puede permanecer el modelo en producción antes de degradarse materialmente?
  - ¿cuándo empieza a ser un problema práctico el drift?
  - ¿cada cuánto conviene reentrenar si reentrenar es costoso?

Estas policies no son sólo variaciones visuales del splitter. Encapsulan preguntas temporales distintas sobre el sistema que estás evaluando:

- la simulación walk-forward pregunta cómo se habría comportado el sistema bajo una política de retraining
- el crecimiento de train pregunta si realmente vale la pena usar más historia
- la degradación temporal pregunta cuánto tiempo sigue siendo operativamente seguro el train actual

También hay una hipótesis compuesta construida encima de esas piezas.

``RollingTrainHistoryPolicy``
  Ejecuta un loop walk-forward externo y elige el tamaño óptimo de train dentro de cada iteración.

  Esto responde preguntas como:

  - ¿cuánta historia de entrenamiento necesito en promedio a lo largo del tiempo?
  - ¿el tamaño óptimo de train se mantiene estable o cambia entre iteraciones?
  - ¿se puede bajar costo de entrenamiento adaptando la profundidad histórica en lugar de usar siempre la ventana máxima?

Policies de lookback por features
---------------------------------

Algunos problemas temporales necesitan una capa adicional de realismo: no todos los grupos de features usan la misma profundidad histórica.

Por ejemplo:

- features de comportamiento reciente pueden necesitar sólo ``15D``
- features con lags largos o estacionalidad pueden necesitar ``65D`` o más

Eso no significa necesariamente que la ventana supervisada de train deba ser más grande. Significa que el pipeline de features necesita distintas cantidades de contexto histórico para distintos grupos de variables.

Jano modela eso con ``FeatureLookbackSpec`` sobre un fold ya definido:

.. code-block:: python

   from jano import FeatureLookbackSpec

   lookbacks = FeatureLookbackSpec(
       default_lookback="15D",
       group_lookbacks={"lag_features": "65D"},
       feature_groups={"lag_features": ["lag_30", "lag_60"]},
   )

   split = next(splitter.iter_splits(frame))
   history = split.slice_feature_history(
       frame,
       lookbacks,
       time_col="timestamp",
       segment_name="train",
   )

   recent_context = history["__default__"]
   lag_context = history["lag_features"]

Esto mantiene fija la geometría del fold, pero hace explícito el contexto histórico requerido por cada grupo de features. Es útil cuando el cómputo de features y el entrenamiento supervisado no comparten la misma profundidad temporal.
