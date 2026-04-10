Installation and Release
========================

Installation
------------

Once the package is published to PyPI, install it with:

.. container:: example-block

   Standard installation

.. code-block:: bash

   python -m pip install jano

To use Polars inputs directly, install the optional extra:

.. container:: example-block

   Optional Polars extra

.. code-block:: bash

   python -m pip install "jano[polars]"

For local development, install the project in editable mode with the development dependencies:

.. container:: example-block

   Development setup

.. code-block:: bash

   python -m pip install -e ".[dev]"

Versioning
----------

Jano exposes its package version through ``jano.__version__``. The distribution version is sourced from ``jano/_version.py`` so the runtime version and the published package metadata stay aligned.

Release flow
------------

The repository now includes a dedicated GitHub Actions workflow for PyPI publication:

1. Update ``jano/_version.py`` to the release version.
2. Verify locally:

   .. container:: example-block

      Release checks

   .. code-block:: bash

      python -m pytest -q
      python -m build
      python -m twine check dist/*

3. Commit and push the release changes.
4. Create and push a Git tag that matches the release version, for example:

   .. container:: example-block

      Release tag

   .. code-block:: bash

      git tag v0.2.0
      git push origin v0.2.0

5. The ``Publish`` workflow builds the artifacts, validates them with ``twine check`` and publishes them to PyPI via trusted publishing.

PyPI configuration
------------------

To make the workflow publish successfully, PyPI must trust this GitHub repository as a publisher for the ``jano`` project. Configure that once in the PyPI project settings, then future tagged releases can publish without storing API tokens in GitHub secrets.
