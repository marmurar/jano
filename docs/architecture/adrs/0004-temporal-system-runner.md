# ADR 0004: Add a Temporal System Runner Above the Core

## Status

Accepted.

## Context

Jano originally focused on temporal partitioning, simulation and model execution
through `WalkForwardRunner`. That covers sklearn-like estimators well, but some
real systems update without fitting a classic model:

- retrieval indexes can be refreshed,
- prompts can be revised,
- routing policies can change,
- fine-tuning jobs can be wrapped in an operational update step.

Those workflows still need fold geometry, retraining or refresh decisions and
structured outputs, but they do not fit the `fit()` / `predict()` contract cleanly.

## Decision

Jano adds a `TemporalSystemRunner` layer for updateable systems.

The runner:

- consumes temporal folds produced by the existing simulation layer,
- calls a user-provided system `update()` step when policy says to update,
- calls a user-provided system `evaluate()` step on each fold,
- returns structured, data-first results.

The new layer is intentionally above the core splitter and simulation primitives.
It does not change fold geometry, and it does not move update logic into the
splitter or planning layers.

## Consequences

- Jano can model more than sklearn-style estimators without changing the splitter.
- LLM, RAG and prompt-refresh workflows can reuse the same temporal machinery.
- The runner surface grows, but the core temporal contract stays stable.
- Agent-facing and notebook-facing workflows can consume structured update and
  evaluation results without parsing HTML or bespoke logs.

## Invariants

- `TemporalBacktestSplitter` remains model-agnostic.
- `plan()` still describes geometry only.
- `TemporalSimulation` still materializes temporal folds.
- `TemporalSystemRunner` stays above the core primitives and does not modify
  partition semantics.
- Updateable systems are modeled through user-provided callables or protocols,
  not built-in LLM-specific assumptions.
