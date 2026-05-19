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

Descarga local
--------------

Listar datasets disponibles:

.. code-block:: bash

   python scripts/download_dataset.py --list

Descargar un dataset sin guardarlo en Git:

.. code-block:: bash

   python scripts/download_dataset.py bike_sharing_hourly --extract

Por defecto se guarda debajo de ``data/raw/``. Podés cambiar esa ubicación:

.. code-block:: bash

   python scripts/download_dataset.py nyc_tlc_yellow_2024_01 --data-root /tmp/jano-data

Política
--------

- Commitear metadata, ejemplos y scripts de descarga.
- No commitear CSV, ZIP, Parquet ni archivos de cache descargados.
- Mantener notebooks ejecutables descargando o leyendo archivos locales desde ``data/raw/``.
- Mantener tests automatizados independientes de la red; usar fixtures sintéticas o descargas locales mockeadas.
- Marcar cualquier chequeo futuro con datos reales como opcional o external-data.
