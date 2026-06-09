External datasets
=================

Jano examples should be reproducible without committing large datasets to Git.
The repository version-controls dataset metadata and download code, while all
downloaded files stay local under ``data/raw/``.

The ``data/`` directory is intentionally ignored by Git.

Registry
--------

Dataset metadata lives in ``datasets/registry.json``. Each entry records the
source URL, source page, license or terms note, expected local path, task type,
time column and suggested target column.

The current registry includes:

- ``bike_sharing_hourly`` for small regression and walk-forward examples.
- ``bts_airline_2024_01`` for ordinal delay-cost and retraining examples.
- ``nyc_tlc_yellow_2024_01`` for larger Parquet-based performance examples.
- ``household_power`` for minute-level time-series examples.
- ``rossmann_store_sales`` for the gold example comparing random split,
  chronological holdout, walk-forward simulation and retraining policies.

Download locally
----------------

List available datasets:

.. code-block:: bash

   python scripts/download_dataset.py --list

Download a dataset without storing it in Git:

.. code-block:: bash

   python scripts/download_dataset.py bike_sharing_hourly --extract

Some datasets require provider-specific credentials. Rossmann is hosted on
Kaggle, so configure the Kaggle CLI first and then run:

.. code-block:: bash

   python scripts/download_dataset.py rossmann_store_sales --extract

By default the file is saved below ``data/raw/``. You can override that location:

.. code-block:: bash

   python scripts/download_dataset.py nyc_tlc_yellow_2024_01 --data-root /tmp/jano-data

Gold example
------------

The Rossmann notebook is the recommended end-to-end example:

.. code-block:: bash

   jupyter notebook notebooks/rossmann_temporal_validation.ipynb

It demonstrates:

- why a random split can answer the wrong temporal question,
- how a chronological holdout improves the baseline but still gives one snapshot,
- how Jano's ``plan()`` exposes fold geometry before training,
- how ``WalkForwardRunner`` executes the same model across multiple deployment dates,
- how retraining policies can be compared over the same fold geometry.

If Kaggle credentials are not available, the notebook uses a deterministic
Rossmann-like fallback. That keeps the notebook executable in offline
environments while making the real-data path explicit.

Policy
------

- Commit metadata, examples and download scripts.
- Do not commit downloaded CSV, ZIP, Parquet or cache files.
- Keep notebooks executable by downloading or reading local files from ``data/raw/``.
- Keep automated tests independent from network access; use synthetic fixtures or mocked local downloads.
- Mark any future real-data checks as optional or external-data tests.
