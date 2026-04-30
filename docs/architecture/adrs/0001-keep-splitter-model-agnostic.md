# ADR 0001: Keep the Splitter Model-Agnostic

## Status

Accepted.

## Context

Jano started as a temporal partitioning and simulation toolkit. As the project grew,
it added model execution through `WalkForwardRunner` and retraining policies. That
created a design question: should model fitting, prediction and metrics become part
of `TemporalBacktestSplitter`, or should they live in a higher layer?

## Decision

`TemporalBacktestSplitter` remains model-agnostic.

It defines temporal geometry, validates partition semantics and yields folds. It does
not fit models, compute predictions, evaluate metrics or decide retraining policies.

Model execution belongs in `WalkForwardRunner` and future execution/study layers.

## Consequences

- The splitter stays small, composable and close to `sklearn.model_selection` usage.
- Users can keep writing manual loops over folds when they need full control.
- Higher-level workflows can evolve without destabilizing the splitter contract.
- Some users will need one additional object when they want Jano to execute models.

## Invariants

- Splitter APIs should not require a model, target column or metric.
- Runner and study APIs may consume splitters, policies or simulations.
- Changes to execution behavior should not alter fold geometry.

