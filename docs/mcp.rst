MCP server
==========

Jano ships an optional local MCP server so AI agents can use the library through a small, explicit tool surface.

This is useful when you want an agent to:

- inspect a local dataset,
- precompute a walk-forward plan,
- and run a temporal simulation without writing Python code manually.

The initial MCP surface is intentionally narrow. It focuses on the most stable, agent-friendly workflow:

- preview a dataset,
- plan a walk-forward simulation,
- run a walk-forward simulation.

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
  Build a walk-forward ``plan()`` and return iteration boundaries plus row counts.

``run_walk_forward_simulation``
  Materialize a walk-forward simulation and return a compact summary plus rendered HTML.

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
- walk-forward simulation.

Lower-level composition and model-specific temporal hypothesis policies remain available in the Python library itself.
