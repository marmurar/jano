Benchmark
=========

Esta página resume un benchmark local del motor adaptativo de particionado de Jano.

El objetivo es medir el costo del particionado temporal en sí: calcular boundaries de
folds, conteos de filas e índices de split. No incluye tiempo de entrenamiento de modelos.

Qué Se Midió
------------

Se midieron dos operaciones:

.. code-block:: python

   splitter.plan(data)
   sum(1 for _ in splitter.split(data))

``plan()`` precalcula la geometría de folds y conteos de filas. ``split()`` devuelve los
arrays de índices posicionales para cada fold.

Configuración Usada
-------------------

Se usó la misma configuración de splitter para todos los backends de entrada:

- strategy: ``rolling``
- layout: ``train_test``
- ``train_size="2D"``
- ``test_size="12h"``
- ``step="12h"``
- frecuencia del dataset: una fila por minuto
- repeticiones: mediana de 7 corridas medidas después de 2 warmups
- tamaños de dataset: 10k, 100k y 500k filas

El benchmark compara:

- ``engine="auto"``: Jano elige el camino nativo seguro para el input.
- ``engine="pandas"``: Jano fuerza el camino pandas estable.

Para inputs Polars y NumPy, ``engine="pandas"`` sirve como baseline del comportamiento
anterior porque incluye convertir el input a pandas antes de particionar.

Resultados
----------

.. list-table::
   :header-rows: 1

   * - Filas
     - Input
     - Engine arg
     - Engine elegido
     - Convertido
     - Folds
     - Plan ms
     - Split ms
   * - 10,000
     - pandas
     - auto
     - pandas
     - no
     - 9
     - 0.26
     - 0.33
   * - 10,000
     - pandas
     - pandas
     - pandas
     - no
     - 9
     - 0.26
     - 0.32
   * - 10,000
     - polars
     - auto
     - polars
     - no
     - 9
     - 0.26
     - 0.32
   * - 10,000
     - polars
     - pandas
     - pandas
     - sí
     - 9
     - 6.09
     - 6.07
   * - 10,000
     - numpy
     - auto
     - numpy
     - no
     - 9
     - 0.26
     - 0.31
   * - 10,000
     - numpy
     - pandas
     - pandas
     - sí
     - 9
     - 0.36
     - 0.41
   * - 100,000
     - pandas
     - auto
     - pandas
     - no
     - 134
     - 1.71
     - 2.53
   * - 100,000
     - pandas
     - pandas
     - pandas
     - no
     - 134
     - 1.70
     - 2.53
   * - 100,000
     - polars
     - auto
     - polars
     - no
     - 134
     - 1.71
     - 2.51
   * - 100,000
     - polars
     - pandas
     - pandas
     - sí
     - 134
     - 56.82
     - 58.95
   * - 100,000
     - numpy
     - auto
     - numpy
     - no
     - 134
     - 1.86
     - 2.59
   * - 100,000
     - numpy
     - pandas
     - pandas
     - sí
     - 134
     - 1.90
     - 2.72
   * - 500,000
     - pandas
     - auto
     - pandas
     - no
     - 690
     - 8.65
     - 12.40
   * - 500,000
     - pandas
     - pandas
     - pandas
     - no
     - 690
     - 8.61
     - 12.33
   * - 500,000
     - polars
     - auto
     - polars
     - no
     - 690
     - 8.58
     - 12.37
   * - 500,000
     - polars
     - pandas
     - pandas
     - sí
     - 690
     - 296.48
     - 304.73
   * - 500,000
     - numpy
     - auto
     - numpy
     - no
     - 690
     - 9.14
     - 13.00
   * - 500,000
     - numpy
     - pandas
     - pandas
     - sí
     - 690
     - 9.42
     - 13.40

Speedup Contra Pandas Forzado
-----------------------------

.. list-table::
   :header-rows: 1

   * - Filas
     - Input
     - Speedup plan
     - Speedup split
     - Camino de engine
   * - 10,000
     - pandas
     - 0.98x
     - 0.98x
     - pandas -> pandas
   * - 10,000
     - polars
     - 23.28x
     - 19.17x
     - pandas -> polars
   * - 10,000
     - numpy
     - 1.38x
     - 1.32x
     - pandas -> numpy
   * - 100,000
     - pandas
     - 0.99x
     - 1.00x
     - pandas -> pandas
   * - 100,000
     - polars
     - 33.29x
     - 23.45x
     - pandas -> polars
   * - 100,000
     - numpy
     - 1.02x
     - 1.05x
     - pandas -> numpy
   * - 500,000
     - pandas
     - 1.00x
     - 0.99x
     - pandas -> pandas
   * - 500,000
     - polars
     - 34.57x
     - 24.64x
     - pandas -> polars
   * - 500,000
     - numpy
     - 1.03x
     - 1.03x
     - pandas -> numpy

Resumen Visual
--------------

Las barras muestran el speedup de ``split()`` con ``engine="auto"`` contra el baseline
``engine="pandas"``. Más alto es mejor.

.. raw:: html

   <div class="benchmark-grid">
     <div class="benchmark-row">
       <div class="benchmark-label">10k filas</div>
       <div class="benchmark-bars">
         <div class="benchmark-bar">
           <span class="benchmark-name">pandas</span>
           <span class="benchmark-track"><span class="benchmark-fill pandas" style="width: 5.1%;"></span></span>
           <span class="benchmark-value">0.98x</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">numpy</span>
           <span class="benchmark-track"><span class="benchmark-fill numpy" style="width: 6.9%;"></span></span>
           <span class="benchmark-value">1.32x</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">polars</span>
           <span class="benchmark-track"><span class="benchmark-fill polars" style="width: 100%;"></span></span>
           <span class="benchmark-value">19.17x</span>
         </div>
       </div>
     </div>
     <div class="benchmark-row">
       <div class="benchmark-label">100k filas</div>
       <div class="benchmark-bars">
         <div class="benchmark-bar">
           <span class="benchmark-name">pandas</span>
           <span class="benchmark-track"><span class="benchmark-fill pandas" style="width: 4.3%;"></span></span>
           <span class="benchmark-value">1.00x</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">numpy</span>
           <span class="benchmark-track"><span class="benchmark-fill numpy" style="width: 4.5%;"></span></span>
           <span class="benchmark-value">1.05x</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">polars</span>
           <span class="benchmark-track"><span class="benchmark-fill polars" style="width: 100%;"></span></span>
           <span class="benchmark-value">23.45x</span>
         </div>
       </div>
     </div>
     <div class="benchmark-row">
       <div class="benchmark-label">500k filas</div>
       <div class="benchmark-bars">
         <div class="benchmark-bar">
           <span class="benchmark-name">pandas</span>
           <span class="benchmark-track"><span class="benchmark-fill pandas" style="width: 4.0%;"></span></span>
           <span class="benchmark-value">0.99x</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">numpy</span>
           <span class="benchmark-track"><span class="benchmark-fill numpy" style="width: 4.2%;"></span></span>
           <span class="benchmark-value">1.03x</span>
         </div>
         <div class="benchmark-bar">
           <span class="benchmark-name">polars</span>
           <span class="benchmark-track"><span class="benchmark-fill polars" style="width: 100%;"></span></span>
           <span class="benchmark-value">24.64x</span>
         </div>
       </div>
     </div>
   </div>

Cómo Leer Estos Resultados
--------------------------

La mejora principal aparece en inputs Polars. Antes del motor adaptativo, los inputs
Polars tenían que convertirse completamente a pandas antes de particionar. Con
``engine="auto"``, Jano mantiene extracción de columnas Polars nativa para planning y
generación de índices de split.

El camino pandas queda intencionalmente igual. Pandas sigue siendo el baseline estable y
el engine elegido para inputs pandas.

El camino NumPy mejora de forma moderada en este benchmark porque el input medido usa
structured arrays con conversión relativamente barata. NumPy sigue siendo útil como
camino de bajo overhead, pero la mayor ganancia observada es evitar la conversión
Polars-a-pandas en datasets grandes.

Estos tiempos son locales y deben leerse como direccionales. Sirven principalmente para
entender overhead del motor de particionado; simulaciones completas también incluyen
feature engineering, entrenamiento y predicción.
