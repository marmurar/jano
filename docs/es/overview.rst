Descripción general
===================

Por qué existe Jano
-------------------

Jano está diseñado para estructurar, ejecutar y analizar simulaciones temporales para sistemas de machine learning que operan sobre datos correlacionados en el tiempo. El objetivo no es solamente partir un dataset una vez, sino crear una forma disciplinada de razonar sobre cómo se comporta un sistema cuando la cronología, la cadencia de retraining y la disponibilidad de datos se tratan como restricciones reales.

En muchos sistemas reales, los datos están organizados por entidades y tiempo. Una representación más fiel se parece a:

.. math::

   \mathcal{D} = \{(x_{i,t}, y_{i,t})\}_{i=1,\dots,N;\; t=1,\dots,T}

donde :math:`i` indexa entidades como usuarios, rutas o sellers, y :math:`t` denota tiempo.

Aun así, muchos workflows de evaluación siguen actuando como si las observaciones fueran independientes e idénticamente distribuidas:

.. math::

   (x_{i,t}, y_{i,t}) \sim \text{i.i.d.}

Ese desajuste es el problema de fondo. Cuando se ignora el orden temporal, información del futuro puede filtrarse al train, la performance reportada se vuelve optimista y la validación offline deja de reflejar lo que realmente pasa en producción.

La mayoría de las herramientas de train/test responden una pregunta estática:

"¿Cómo parto este dataset una vez?"

Jano está construido para responder una dinámica:

"¿Cómo se habría comportado este sistema a lo largo del tiempo si entrenara, reentrenara y evaluara bajo una política temporal explícita?"

Herramientas como ``TimeSeriesSplit`` ya mejoran el muestreo aleatorio al imponer un orden básico entre pasado y futuro. Pero siguen respondiendo principalmente "cómo partir" los datos. Jano apunta a una pregunta más operativa: cómo se habría comportado un sistema bajo una política concreta de evaluación y retraining.

En lugar de un único split, Jano trabaja con una política temporal explícita:

.. math::

   \pi = (\Delta_{train}, \Delta_{test}, s, g)

donde :math:`\Delta_{train}` es la ventana de entrenamiento, :math:`\Delta_{test}` el horizonte de evaluación, :math:`s` el shift entre iteraciones y :math:`g` un gap temporal opcional para controlar leakage.

Bajo esa política, la evaluación se convierte en una secuencia de experimentos temporalmente consistentes:

.. math::

   \left\{(\mathcal{D}_{train}^{(k)}, \mathcal{D}_{test}^{(k)})\right\}_{k=1}^K

en lugar de una estimación única. Cada fold preserva la causalidad y contribuye a una trayectoria de comportamiento del sistema a lo largo del tiempo.

Esto también vuelve al toolkit útil para evidenciar drift en los resultados de simulación. Jano no estima métricas de drift directamente, pero hace visibles los cambios temporales en comportamiento, calibración o performance fold por fold. Eso permite inspeccionar degradación, inestabilidad y cambios de régimen sin colapsarlos en una sola métrica agregada.

Eso lo hace útil para:

- backtesting de sistemas predictivos sobre datos transaccionales
- simulación de cadencias diarias, semanales o custom de retraining
- comparación entre ventanas rolling y expanding
- introducción de gaps temporales entre train y evaluación
- definición de layouts ``train/test`` o ``train/validation/test`` con duraciones, filas o proporciones
- exposición de drift en los resultados de simulación al hacer explícitos los cambios temporales entre folds

Objetivos de diseño
-------------------

Jano se está ordenando alrededor de algunas ideas claras:

- definiciones explícitas de partición temporal
- evaluación time-aware como proceso reproducible, no como split único
- estado oculto mínimo
- folds predecibles e inspeccionables
- workflow amigable con pandas e inspirado en ``sklearn.model_selection``
- normalización de inputs tabulares ``pandas``, ``numpy`` y ``polars``
- objetos de fold ricos, que puedan inspeccionarse, resumirse y cortarse
- auditabilidad como restricción de diseño

Estado actual
-------------

El core actual ya soporta:

- ``TemporalSimulation`` como interfaz high-level para correr simulaciones completas
- controles de ventana como ``start_at``, ``end_at`` y ``max_folds``
- estrategias ``single``, ``rolling`` y ``expanding``
- layouts ``train_test`` y ``train_val_test``
- tamaños expresados como duraciones, conteos de filas o fracciones
- gaps opcionales antes de train, validation o test, y un trailing gap después de test
- semántica temporal que permite usar distintas columnas por segmento para elegibilidad
- normalización de inputs ``pandas``, ``numpy`` y ``polars``
- reporting como summary, HTML o datos listos para graficar en Python
- una ruta de indexado temporal más numpy-first para reducir overhead en datasets grandes

La API ya es suficientemente estable para experimentar y seguir refinando el diseño. En la práctica, Jano ya funciona como un framework experimental para razonar sobre sistemas de machine learning como procesos que evolucionan en el tiempo, y no como artefactos estáticos de train/test.
