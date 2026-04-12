Jano
=====

.. container:: language-switch

   **Idioma:** :doc:`English <../index>` | Español

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
         <h3>Simulaciones operativas</h3>
         <p>Modelá evaluaciones rolling, expanding o single-window con gaps temporales opcionales.</p>
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

Jano es un toolkit de Python para estructurar, ejecutar y analizar simulaciones temporales sobre sistemas de machine learning que operan con datos correlacionados en el tiempo. Proporciona un marco explícito para definir políticas de partición temporal, correr evaluaciones walk-forward y generar reportes auditables que reflejan cómo se comportan los modelos bajo dinámicas de producción más realistas.

A diferencia de los splits aleatorios tradicionales, Jano trata la cronología como una restricción de primer nivel. Está pensado para escenarios donde el leakage debe controlarse estrictamente y donde la performance cambia en el tiempo por drift, ciclos de reentrenamiento o cambios en la distribución de los datos.

En la práctica, la superficie recomendada es ``TemporalSimulation`` para corridas completas, mientras que ``TemporalBacktestSplitter`` sigue disponible para control manual de folds. Jano no calcula métricas de drift directamente; expone la estructura temporal de los resultados para que el drift, los cambios de régimen y la degradación sean observables fold por fold.

Casos de uso típicos:

- validación walk-forward para forecasting o clasificación temporal
- simulación de políticas de retraining y despliegue
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
   concepts
   simulation
   benchmark
   api
   release
   about
