# Spec: Study Layer

## Purpose

Studies encode operational hypotheses on top of Jano's temporal primitives.

They are not generic simulations. They ask specific questions that users repeatedly
face when validating time-dependent ML systems.

## Current Study Questions

Train history sufficiency:

- Does adding more training history improve performance on the same test window?
- How much history is needed before performance stops improving materially?

Performance decay:

- If a model is trained once, how long does it remain useful?
- When does a fixed model begin showing unacceptable degradation?

Rolling train history:

- Inside each walk-forward iteration, how much train history is optimal?
- On average, how much history is needed over time?

## Relationship to Other Layers

Studies should compose lower-level primitives:

- splitters for temporal geometry
- plans for fold boundaries and counts
- simulations for materialized fold execution
- runners for model execution and metrics

Studies should not replace these primitives or hide them completely.

## Output Shape

Study outputs should provide:

- `to_frame()` for tabular records
- `summary()` when aggregate interpretation is useful
- structured dictionaries when agent consumption is useful
- direct references to relevant folds, windows or policies

## Non-Goals

Studies should not become:

- AutoML
- model registries
- generic orchestration systems
- opaque optimization engines

## Future Candidates

- retraining cadence search
- policy comparison over the same fold geometry
- target-aware decay studies
- cost-aware train history search
- feature-group lookback studies

