Jano documentation
==================

Jano is a Python library for temporal partitions and backtesting over time-correlated datasets.

It is meant for situations where a single random split is not enough and where evaluation has to respect the ordering of time: transactional data, production simulations, walk-forward validation, repeated retraining, or any experiment in which the past should never peek into the future.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   concepts
   simulation
   api
