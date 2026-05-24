Jano
=====

.. container:: language-switch

   **Idioma:** :doc:`English <../index>` | Español

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.20301006.svg
   :target: https://doi.org/10.5281/zenodo.20301006
   :alt: DOI

.. raw:: html

   <div class="landing-hero">
     <p class="landing-lead">
       Toolkit de simulación temporal y backtesting para sistemas de machine learning dependientes del tiempo
     </p>
     <p class="landing-tagline">
       La capa faltante entre los modelos de ML y la validación temporal en producción.
     </p>
     <div class="landing-grid">
       <div class="landing-card">
         <h3>Políticas explícitas de partición</h3>
         <p>Definí layouts train/test o train/validation/test con duraciones, filas o proporciones.</p>
       </div>
       <div class="landing-card">
         <h3>Relojes temporales</h3>
         <p>Avanzá por calendario, filas, batches online o checkpoints de retraining definidos por el usuario.</p>
       </div>
       <div class="landing-card">
         <h3>El drift se vuelve visible</h3>
         <p>Al mantener los folds anclados en el tiempo, los cambios en resultados y comportamiento quedan expuestos.</p>
       </div>
       <div class="landing-card">
         <h3>Inputs tabulares flexibles</h3>
         <p>Usá la misma API con pandas, NumPy o Polars manteniendo un único motor temporal.</p>
       </div>
     </div>
   </div>

.. container:: landing-visual

   .. image:: /_static/jano_viz.png
      :alt: Visualización de particiones temporales en Jano
      :class: landing-visual-image

   .. container:: landing-visual-caption

      Resumen visual de cómo Jano organiza particiones temporales, folds y reporting a lo largo del tiempo.

Jano es un toolkit de Python para estructurar, ejecutar y analizar particiones y simulaciones temporales sobre sistemas de machine learning que operan con datos correlacionados en el tiempo. Proporciona un marco explícito para definir políticas de partición temporal, correr evaluaciones walk-forward, ejecutar modelos bajo reglas explícitas de retraining y generar reportes auditables que reflejan cómo se comportan los sistemas bajo dinámicas de producción más realistas. Para escenarios online, también permite avanzar sobre la misma línea temporal por eventos observados, micro-batches o checkpoints de retraining definidos por el usuario.

A diferencia de los splits aleatorios tradicionales, Jano trata la cronología como una restricción de primer nivel. Está pensado para escenarios donde el leakage debe controlarse estrictamente y donde la performance cambia en el tiempo por drift, ciclos de reentrenamiento o cambios en la distribución de los datos.

En la práctica, la superficie recomendada se reparte entre ``TemporalSimulation`` y ``WalkForwardPolicy`` para simulación de folds, ``WalkForwardRunner`` para ejecutar modelos sobre esos folds, y ``TemporalBacktestSplitter`` para control manual e iteración low-level. Jano no calcula métricas de drift directamente; expone la estructura temporal de los resultados para que el drift, los cambios de régimen y la degradación sean observables fold por fold.

Casos de uso típicos:

- validación walk-forward para forecasting o clasificación temporal
- simulación de políticas de retraining y despliegue
- ejecución de benchmarks de modelo bajo reglas explícitas de retraining
- monitoreo de estabilidad del modelo a lo largo del tiempo
- evaluación de políticas de decisión bajo condiciones de datos cambiantes

Backends soportados:

- ``pandas.DataFrame`` con columnas nombradas
- ``numpy.ndarray`` con referencias enteras de columnas
- ``polars.DataFrame`` a través del extra opcional ``jano[polars]``

.. toctree::
   :maxdepth: 2
   :caption: Contenidos

   overview
   train_test_split_example
   partitioning
   concepts
   simulation
   mcp
   ai
   benchmark
   datasets
   api
   release
   about
