# Jano Architecture Notes

This folder contains design material for contributors and AI-assisted development.
It is intentionally written in Markdown so it is easy to read, diff, summarize and
use as context for coding agents.

These documents are not user-facing tutorials. They describe intent, boundaries,
tradeoffs and accepted decisions so future changes can stay aligned with Jano's
architecture.

## Document Types

- RFCs are proposals under discussion. They compare options and keep open questions visible.
- Specs define intended behavior for a feature, layer or workflow.
- ADRs record accepted architecture decisions and their consequences.

## Current Layer Model

Jano is organized as a set of layers that should remain separable:

- Splitter: temporal geometry and manual fold iteration.
- Plan: precomputed fold boundaries and row counts before materialization.
- Simulation: materialized temporal experiments and fold-level reporting.
- Runner: model execution over temporal folds with explicit retraining rules.
- Online runner: event-level or micro-batch prequential learning simulation.
- Study: higher-level operational hypotheses built from lower-level primitives.
- Scenario: ready-to-use production-style workflows that compose the core without changing it.
- MCP: local tool surface for AI agents and other MCP-aware clients.

## Active Design Principles

- Chronology is a first-class constraint.
- The splitter core remains model-agnostic.
- Manual fold iteration remains public and supported.
- Higher-level APIs reduce boilerplate without hiding temporal structure.
- Runner outputs are data-first and plot-ready, not dashboard-first.
- Studies should encode clear operational questions, not replace lower-level composition.
- Scenarios may be opinionated, but they must live above the runner and keep core contracts stable.
- Agent-facing APIs should return stable, structured objects that are easy to inspect.

## Use Cases

- [Use cases beyond supervised ML](use-cases.md)

## Starting Points

- [Spec: Walk-forward runner](specs/walk-forward-runner.md)
- [Spec: Online temporal runner](specs/online-temporal-runner.md)
- [Spec: Study layer](specs/study-layer.md)
- [RFC: Study layer](rfcs/0001-study-layer.md)
- [RFC: Online temporal runner](rfcs/0002-online-temporal-runner.md)
- [ADR 0001: Keep the splitter model-agnostic](adrs/0001-keep-splitter-model-agnostic.md)
- [ADR 0002: Keep runner reporting data-first](adrs/0002-runner-reporting-data-first.md)
- [ADR 0003: Keep manual fold iteration public](adrs/0003-manual-fold-iteration-remains-public.md)
- [ADR 0004: Add a temporal system runner above the core](adrs/0004-temporal-system-runner.md)
