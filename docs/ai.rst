AI-ready usage
==============

Jano includes documentation and integration points designed for AI-assisted use.
These files help agents use the library correctly, execute stable workflows and
modify the repository without breaking architectural boundaries.

The three surfaces are:

- architecture notes for design context,
- an agent guide and tool-specific rule files,
- and an optional MCP server for local tool execution.

Architecture notes
------------------

The technical design map lives under ``docs/architecture/``.

It includes:

- ADRs for accepted decisions,
- specs for intended behavior,
- RFCs for open design proposals.

These files are useful when an agent is going to modify Jano itself. They explain
constraints such as:

- the splitter remains model-agnostic,
- manual fold iteration remains public,
- runner results are data-first,
- studies compose lower-level primitives.

Agent guide and adapters
------------------------

The canonical AI-facing guide is:

``docs/ai/jano-agent-guide.md``

It explains:

- when to use ``TemporalBacktestSplitter``,
- when to use ``WalkForwardPolicy`` and ``plan()``,
- when to use ``WalkForwardRunner``,
- how to consume ``metric_trajectory()``, ``fold_summary()`` and ``report_data()``,
- and what temporal leakage rules to respect.

Tool-specific adapters point back to that canonical guide:

- ``skills/jano/SKILL.md`` for Codex-style skill usage,
- ``CLAUDE.md`` for Claude Code or Claude Desktop repository guidance,
- ``.cursor/rules/jano.mdc`` for Cursor rules.

MCP server
----------

The MCP server is executable code, not just documentation. It exposes a small set
of local tools so MCP-aware clients can inspect datasets and run Jano workflows.

Use MCP when an agent should execute operations over local files:

- preview a local dataset,
- build a walk-forward plan,
- run a walk-forward simulation.

Use the agent guide or skill when an agent needs to reason about Jano or write
Python code with the library.

For repository changes, prefer opening a pull request into ``master`` so CI,
documentation checks and configured AI review tools can inspect the diff before
merge.

In short:

- architecture notes explain why and where the project is going,
- the agent guide explains how to use Jano correctly,
- MCP gives agents tools they can call locally.
