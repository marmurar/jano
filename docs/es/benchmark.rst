Benchmark
=========

Esta página resume un benchmark local de generación de particiones temporales sobre los backends tabulares actualmente soportados.

Qué se midió
------------

El benchmark mide el tiempo total necesario para materializar todos los folds de:

.. code-block:: python

   list(splitter.iter_splits(data))

Configuración usada
-------------------

Se utilizó la misma configuración de splitter para todos los backends:

- strategy: ``rolling``
- layout: ``train_test``
- ``train_size="3D"``
- ``test_size="12h"``
- ``gap_before_test="30min"``
- ``step="6h"``
- frecuencia del dataset: una fila por minuto
- métrica: wall-clock runtime sobre la iteración completa de splits
- repeticiones: 3 por backend y tamaño de dataset

El benchmark se corrió localmente sobre la implementación actual, donde pandas sigue siendo el motor interno de ejecución. Eso significa que los tiempos de ``numpy`` y ``polars`` incluyen el costo de normalizar esos inputs a pandas antes de partir.

Por eso este benchmark debe leerse como un benchmark end-to-end de la API pública tal como se comporta hoy, no como una comparación justa entre engines nativos de cada backend.

Resultados
----------

.. list-table::
   :header-rows: 1

   * - Backend
     - Filas
     - Folds
     - Mean ms
     - Min ms
     - Max ms
   * - pandas
     - 10,000
     - 14
     - 7.581
     - 5.478
     - 11.634
   * - numpy
     - 10,000
     - 14
     - 4.536
     - 4.208
     - 5.181
   * - polars
     - 10,000
     - 14
     - 10.767
     - 10.657
     - 10.825
   * - pandas
     - 100,000
     - 264
     - 10.544
     - 8.264
     - 14.778
   * - numpy
     - 100,000
     - 264
     - 19.789
     - 18.930
     - 20.940
   * - polars
     - 100,000
     - 264
     - 65.366
     - 62.823
     - 69.806
   * - pandas
     - 500,000
     - 1,375
     - 24.592
     - 22.083
     - 29.403
   * - numpy
     - 500,000
     - 1,375
     - 94.801
     - 91.859
     - 97.771
   * - polars
     - 500,000
     - 1,375
     - 294.843
     - 289.612
     - 299.288
   * - pandas
     - 1,000,000
     - 2,764
     - 44.719
     - 38.910
     - 47.781
   * - numpy
     - 1,000,000
     - 2,764
     - 183.353
     - 177.574
     - 190.390
   * - polars
     - 1,000,000
     - 2,764
     - 587.886
     - 583.358
     - 592.276

Cómo leer estos resultados
--------------------------

Hoy este benchmark debe interpretarse así:

- el motor de particionado es rápido sobre datasets grandes
- pandas sigue siendo el camino más rápido end-to-end
- numpy y polars son inputs públicos compatibles, pero todavía no backends de ejecución nativa optimizados
- su costo extra proviene principalmente de la normalización a pandas antes de generar folds

Más explícitamente:

- ``pandas`` mide el camino directo
- ``numpy`` mide conversión a pandas más generación de particiones
- ``polars`` mide conversión a pandas más generación de particiones

Si hoy lo más importante es la velocidad bruta de split, el input que mejor rinde sigue siendo ``pandas.DataFrame``.
