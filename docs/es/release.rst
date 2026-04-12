Instalación y Release
=====================

Instalación
-----------

Una vez publicado en PyPI, instalá el paquete con:

.. container:: example-block

   Instalación estándar

.. code-block:: bash

   python -m pip install jano

Para usar inputs de Polars directamente, instalá el extra opcional:

.. container:: example-block

   Extra opcional de Polars

.. code-block:: bash

   python -m pip install "jano[polars]"

Para desarrollo local, instalá el proyecto en modo editable con dependencias de desarrollo:

.. container:: example-block

   Setup de desarrollo

.. code-block:: bash

   python -m pip install -e ".[dev]"

Versionado
----------

Jano expone su versión pública a través de ``jano.__version__``. La versión de distribución sale de ``jano/_version.py``, de modo que la versión de runtime y la metadata del paquete publicado permanezcan alineadas.

Flujo de release
----------------

El repositorio incluye un workflow dedicado de GitHub Actions para publicación en PyPI:

1. Actualizá ``jano/_version.py`` a la versión de release.
2. Verificá localmente:

   .. container:: example-block

      Chequeos de release

   .. code-block:: bash

      python -m pytest -q
      python -m build
      python -m twine check dist/*

3. Commit y push de los cambios de release.
4. Creá y pusheá un tag Git que coincida con la versión, por ejemplo:

   .. container:: example-block

      Tag de release

   .. code-block:: bash

      git tag v0.3.0
      git push origin v0.3.0

5. El workflow ``Publish`` construye los artefactos, los valida con ``twine check`` y publica en PyPI vía trusted publishing.

Configuración de PyPI
---------------------

Para que el workflow publique correctamente, PyPI debe confiar en este repositorio de GitHub como publisher del proyecto ``jano``. Eso se configura una sola vez en las settings del proyecto en PyPI; luego las releases etiquetadas pueden publicarse sin guardar API tokens en GitHub Secrets.
