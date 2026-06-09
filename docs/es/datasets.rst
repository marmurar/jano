Datasets externos
=================

Los ejemplos de Jano deberían ser reproducibles sin commitear datasets pesados en
Git. El repositorio versiona metadata y código de descarga, mientras que los
archivos descargados quedan siempre locales bajo ``data/raw/``.

El directorio ``data/`` está ignorado intencionalmente por Git.

Registry
--------

La metadata vive en ``datasets/registry.json``. Cada entrada registra la URL de
origen, página fuente, nota de licencia o términos, path local esperado, tipo de
tarea, columna temporal y target sugerido.

El registry actual incluye:

- ``bike_sharing_hourly`` para ejemplos chicos de regresión y walk-forward.
- ``bts_airline_2024_01`` para costos ordinales de demoras y retraining.
- ``nyc_tlc_yellow_2024_01`` para ejemplos grandes con Parquet y benchmarks.
- ``household_power`` para granularidad temporal por minuto.
- ``rossmann_store_sales`` como gold example para comparar split aleatorio,
  holdout cronológico, simulación walk-forward y policies de retraining.

Descarga local
--------------

Listar datasets disponibles:

.. code-block:: bash

   python scripts/download_dataset.py --list

Descargar un dataset sin guardarlo en Git:

.. code-block:: bash

   python scripts/download_dataset.py bike_sharing_hourly --extract

Algunos datasets requieren credenciales del proveedor. Rossmann está alojado en
Kaggle, por lo que primero hay que configurar la Kaggle CLI y luego correr:

.. code-block:: bash

   python scripts/download_dataset.py rossmann_store_sales --extract

Por defecto se guarda debajo de ``data/raw/``. Podés cambiar esa ubicación:

.. code-block:: bash

   python scripts/download_dataset.py nyc_tlc_yellow_2024_01 --data-root /tmp/jano-data

Gold example
------------

El notebook Rossmann es el ejemplo end-to-end recomendado:

.. code-block:: bash

   jupyter notebook notebooks/rossmann_temporal_validation.ipynb

Demuestra:

- por qué un split aleatorio puede responder la pregunta temporal equivocada,
- cómo un holdout cronológico mejora el baseline pero sigue dando una sola foto,
- cómo ``plan()`` expone la geometría de folds antes de entrenar,
- cómo ``WalkForwardRunner`` ejecuta el mismo modelo en varias fechas simuladas de despliegue,
- cómo comparar policies de retraining sobre la misma geometría temporal.

Si no hay credenciales de Kaggle, el notebook usa un fallback determinístico
similar a Rossmann. Eso lo mantiene ejecutable offline sin ocultar que el camino
principal es el dataset real.

Política
--------

- Commitear metadata, ejemplos y scripts de descarga.
- No commitear CSV, ZIP, Parquet ni archivos de cache descargados.
- Mantener notebooks ejecutables descargando o leyendo archivos locales desde ``data/raw/``.
- Mantener tests automatizados independientes de la red; usar fixtures sintéticas o descargas locales mockeadas.
- Marcar cualquier chequeo futuro con datos reales como opcional o external-data.
