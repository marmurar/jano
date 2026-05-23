Servidor MCP
============

Jano incluye un servidor MCP local opcional para que agentes de IA puedan usar la librería a través de una superficie chica y explícita.

Esto sirve cuando querés que un agente:

- inspeccione un dataset local,
- precalcule un plan walk-forward,
- corra una simulación temporal,
- ejecute un baseline simple,
- y corra estudios temporales baseline sin escribir Python manualmente.

La primera superficie MCP es deliberadamente angosta. Se enfoca en el workflow más estable y más legible para agentes:

- previsualizar un dataset,
- planificar una simulación walk-forward,
- correr una simulación walk-forward,
- correr un baseline sobre los mismos folds,
- comparar policies de reentrenamiento,
- evaluar ventanas de historia de entrenamiento,
- monitorear decay con train fijo.

Por qué MCP además de la librería Python
----------------------------------------

Instalar una librería Python no alcanza para garantizar que un agente de IA la use correctamente.

La capa MCP le da al agente:

- un conjunto chico de tools explícitas,
- inputs y outputs estructurados,
- y un workflow recomendado alineado con la superficie pública high-level de Jano.

Instalación
-----------

El servidor MCP depende del SDK oficial de MCP para Python y está pensado para entornos con Python 3.10+.

Instalalo con:

.. code-block:: bash

   python -m pip install "jano[mcp]"

Cómo correr el servidor local
-----------------------------

Corré el servidor MCP sobre stdio:

.. code-block:: bash

   jano-mcp

O directamente vía módulo:

.. code-block:: bash

   python -m jano.mcp_server

Tools MCP disponibles
---------------------

``preview_local_dataset``
  Lee un CSV local, un Parquet o un ZIP con CSV y devuelve una preview compacta.

``plan_walk_forward_simulation``
  Construye un ``plan()`` walk-forward y devuelve boundaries por iteración, conteos de filas
  y metadata del motor de particionado elegido.

``run_walk_forward_simulation``
  Materializa una simulación walk-forward y devuelve un resumen compacto, metadata del motor
  de particionado elegido y el HTML renderizado.

``run_walk_forward_baseline_model``
  Ejecuta un baseline incorporado sobre los folds walk-forward y devuelve datos del runner:
  resumen agregado, preview de folds, trayectoria de métricas, eventos de reentrenamiento
  y una preview acotada de predicciones opcional. Usá ``model="mean"`` para targets
  numéricos de regresión y ``model="majority_class"`` para targets de clasificación.

``compare_retrain_policy_baselines``
  Ejecuta el mismo baseline sobre la misma geometría temporal cambiando la policy de
  reentrenamiento. Devuelve una fila de comparación por policy y previews por policy
  de folds y métricas.

``find_train_history_window_baseline``
  Evalúa distintas ventanas de historia de entrenamiento contra un test fijo y devuelve
  la ventana más chica que queda dentro de la tolerancia configurada respecto del mejor
  score.

``monitor_decay_baseline``
  Mantiene fijo el train, mueve el test hacia adelante y devuelve la primera ventana
  donde la métrica elegida cruza el umbral de degradación configurado.

Las tools de planning y ejecución aceptan ``engine`` con los mismos valores que la API Python: ``"auto"``,
``"pandas"``, ``"polars"`` o ``"numpy"``.

Ejemplo de baseline runner
--------------------------

.. code-block:: json

   {
     "dataset_path": "data/bts/bts_ontime_2024_01.zip",
     "partition": {
       "layout": "train_test",
       "train_size": "7D",
       "test_size": "1D"
     },
     "step": "1D",
     "time_col": "FL_DATE",
     "target_col": "arrival_state",
     "model": "majority_class",
     "retrain": "periodic",
     "retrain_interval": 2,
     "max_folds": 5
   }

Esta tool es intencionalmente un baseline, no un ejecutor general de modelos arbitrarios.
MCP JSON no puede transportar callables de Python, por lo que las corridas
productivas con métricas deberían usar ``WalkForwardRunner`` directamente desde
Python. Así la construcción del modelo, feature engineering y métricas custom
quedan en código del usuario.

Ejemplos de estudios temporales
-------------------------------

Comparar policies de reentrenamiento sobre la misma geometría:

.. code-block:: json

   {
     "dataset_path": "data/flights.csv",
     "partition": {
       "layout": "train_test",
       "train_size": "14D",
       "test_size": "1D"
     },
     "step": "1D",
     "time_col": "scheduled_departure_at",
     "target_col": "arrival_delay",
     "model": "mean",
     "policies": [
       {"name": "always", "retrain": "always"},
       {"name": "never", "retrain": "never"},
       {"name": "weekly", "retrain": "periodic", "retrain_interval": 7}
     ]
   }

Buscar una ventana compacta de train history contra un test fijo:

.. code-block:: json

   {
     "dataset_path": "data/flights.csv",
     "time_col": "scheduled_departure_at",
     "cutoff": "2024-02-01",
     "train_sizes": ["7D", "14D", "30D"],
     "test_size": "3D",
     "target_col": "arrival_delay",
     "metric": "mae",
     "tolerance": 0.02
   }

Monitorear decay con train fijo:

.. code-block:: json

   {
     "dataset_path": "data/flights.csv",
     "time_col": "scheduled_departure_at",
     "cutoff": "2024-02-01",
     "train_size": "30D",
     "test_size": "1D",
     "step": "1D",
     "target_col": "arrival_delay",
     "metric": "mae",
     "threshold": 0.10,
     "relative": true
   }

Ejemplo de configuración del cliente MCP
----------------------------------------

Muchos clientes MCP aceptan una entrada de configuración como esta:

.. code-block:: json

   {
     "mcpServers": {
       "jano": {
         "command": "jano-mcp"
       }
     }
   }

Si preferís un comando Python explícito:

.. code-block:: json

   {
     "mcpServers": {
       "jano": {
         "command": "python",
         "args": ["-m", "jano.mcp_server"]
       }
     }
   }

Asistentes de código con IA
---------------------------

El servidor MCP está pensado para asistentes de código con soporte MCP, como Claude Code,
Claude Desktop, Cursor, runtimes de Codex con MCP y otros entornos locales de agentes.

Jano siempre puede usarse directamente como librería Python. El servidor MCP sirve cuando
querés que el asistente vea un conjunto chico de tools declaradas, en lugar de inferir imports
y componer código Python desde cero.

Usá la misma configuración local en cualquier cliente compatible con MCP:

.. code-block:: json

   {
     "mcpServers": {
       "jano": {
         "command": "python",
         "args": ["-m", "jano.mcp_server"]
       }
     }
   }

Modelo de privacidad
--------------------

El servidor corre localmente. Lee archivos locales a través del proceso iniciado por tu cliente MCP.
Jano no sube datasets a ningún lado por sí mismo.

El acceso a archivos depende del entorno del cliente y de los paths que le pases a las tools, así que
conviene usar rutas dentro del proyecto y evitar dar acceso amplio a carpetas no relacionadas.

Alcance actual
--------------

La primera versión MCP no intenta exponer todas las primitives de Jano.

Empieza deliberadamente con:

- preview de datasets,
- planning,
- simulación walk-forward,
- ejecución de baselines,
- estudios temporales baseline.

La composición low-level y las policies temporales más ligadas a modelos siguen disponibles en la librería Python.
