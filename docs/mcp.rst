MCP server
==========

Jano ships an optional local MCP server so AI agents can use the library through a small, explicit tool surface.

This is useful when you want an agent to:

- inspect a local dataset,
- precompute a walk-forward plan,
- run a temporal simulation,
- execute a simple baseline model,
- and run baseline temporal studies without writing Python code manually.

The initial MCP surface is intentionally narrow. It focuses on the most stable, agent-friendly workflow:

- preview a dataset,
- plan a walk-forward simulation,
- run a walk-forward simulation,
- run a baseline model over the same folds,
- compare retraining policies,
- evaluate train-history windows,
- monitor fixed-train performance decay.

Why MCP instead of only the Python library?
-------------------------------------------

Installing a Python library is not enough to guarantee that an AI agent will use it correctly.

The MCP layer gives the agent:

- a small set of explicit tools,
- structured inputs and outputs,
- and a recommended workflow that mirrors the high-level public surface of Jano.

Installation
------------

The MCP server depends on the official Python MCP SDK and is intended for Python 3.10+ environments.

Install it with:

.. code-block:: bash

   python -m pip install "jano[mcp]"

Running the local server
------------------------

Run the MCP server over stdio:

.. code-block:: bash

   jano-mcp

Or directly via the module:

.. code-block:: bash

   python -m jano.mcp_server

Available MCP tools
-------------------

``preview_local_dataset``
  Read a local CSV, Parquet file or ZIP-wrapped CSV and return a compact preview.

``plan_walk_forward_simulation``
  Build a walk-forward ``plan()`` and return iteration boundaries, row counts and
  selected partition-engine metadata.

``run_walk_forward_simulation``
  Materialize a walk-forward simulation and return a compact summary, selected
  partition-engine metadata and rendered HTML.

``run_walk_forward_baseline_model``
  Execute a built-in baseline model over the walk-forward folds and return
  runner data: aggregate summary, fold preview, metric trajectory, retraining
  events and an optional bounded prediction preview. Use ``model="mean"`` for
  numeric regression targets and ``model="majority_class"`` for classification
  targets.

``compare_retrain_policy_baselines``
  Run the same baseline model over the same fold geometry while changing the
  retraining policy. The response includes one comparison row per policy plus
  per-policy fold and metric previews.

``find_train_history_window_baseline``
  Evaluate multiple training-history windows against one fixed test window and
  return the smallest train window that stays within the requested tolerance of
  the best score.

``monitor_decay_baseline``
  Keep a training window fixed, move the test window forward and return the first
  window where the chosen metric crosses the configured degradation threshold.

The planning and execution tools accept ``engine`` with the same values as the Python API: ``"auto"``,
``"pandas"``, ``"polars"`` or ``"numpy"``.

Baseline runner example
-----------------------

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

This tool is intentionally a baseline, not a general arbitrary-model executor.
MCP JSON cannot transport Python callables, so metric-evaluated production runs
should use the Python ``WalkForwardRunner`` directly. That keeps model
construction, feature engineering and custom metrics in user code.

Temporal study examples
-----------------------

Compare retraining policies over the same geometry:

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

Find a compact train-history window against a fixed test horizon:

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

Monitor decay with a fixed train window:

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

Example MCP client configuration
--------------------------------

Many MCP clients accept a configuration entry like this:

.. code-block:: json

   {
     "mcpServers": {
       "jano": {
         "command": "jano-mcp"
       }
     }
   }

If you prefer an explicit Python command:

.. code-block:: json

   {
     "mcpServers": {
       "jano": {
         "command": "python",
         "args": ["-m", "jano.mcp_server"]
       }
     }
   }

AI coding assistants
--------------------

The MCP server is intended for MCP-aware coding assistants such as Claude Code, Claude Desktop,
Cursor, Codex runtimes with MCP support and other local agent environments.

Jano can always be used directly as a Python library. The MCP server is useful when you want
the assistant to see a small set of declared tools instead of inferring imports and composing
Python code from scratch.

Use the same local server configuration in any MCP-aware client:

.. code-block:: json

   {
     "mcpServers": {
       "jano": {
         "command": "python",
         "args": ["-m", "jano.mcp_server"]
       }
     }
   }

Privacy model
-------------

The server runs locally. It reads local files through the process started by your MCP client.
Jano does not upload datasets anywhere by itself.

Access to files is still governed by the client environment and the paths you provide to the
tools, so prefer project-local paths and avoid giving agents broad access to unrelated folders.

Current scope
-------------

The first MCP release does not try to expose every Jano primitive.

It deliberately starts with:

- dataset preview,
- planning,
- walk-forward simulation,
- baseline-model execution,
- baseline temporal studies.

Lower-level composition and model-specific temporal hypothesis policies remain available in the Python library itself.
