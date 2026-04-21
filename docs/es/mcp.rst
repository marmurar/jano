Servidor MCP
============

Jano incluye un servidor MCP local opcional para que agentes de IA puedan usar la librería a través de una superficie chica y explícita.

Esto sirve cuando querés que un agente:

- inspeccione un dataset local,
- precalcule un plan walk-forward,
- y corra una simulación temporal sin escribir Python manualmente.

La primera superficie MCP es deliberadamente angosta. Se enfoca en el workflow más estable y más legible para agentes:

- previsualizar un dataset,
- planificar una simulación walk-forward,
- correr una simulación walk-forward.

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

Ambas tools aceptan ``engine`` con los mismos valores que la API Python: ``"auto"``,
``"pandas"``, ``"polars"`` o ``"numpy"``.

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
- simulación walk-forward.

La composición low-level y las policies temporales más ligadas a modelos siguen disponibles en la librería Python.
