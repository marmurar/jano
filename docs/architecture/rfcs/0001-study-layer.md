# RFC 0001: Formal Study Layer

## Status

Open.

## Problem

Jano already has lower-level primitives for temporal geometry, planning, simulation
and model execution. Users also want higher-level workflows that answer operational
questions directly.

Examples:

- Is more training data actually better?
- How long does a model remain useful without retraining?
- How often should retraining happen?
- How much history is optimal across walk-forward iterations?

These questions are related to simulations, but they are more specific than generic
fold execution.

## Proposal

Introduce `Study` as a conceptual layer in the public architecture.

A study is an encapsulated temporal hypothesis. It composes existing primitives and
returns structured evidence for analysis.

Initial study families:

- train history sufficiency
- performance decay
- rolling train history optimization
- retraining cadence comparison

## Alternatives

Keep studies as standalone policy classes only.

This keeps the API small but makes documentation and mental models less clear.

Expose every study as a method on `TemporalSimulation`.

This makes discovery easier but risks turning simulation into a large object with
too many responsibilities.

Create a generic `Study` base class immediately.

This may be premature until more concrete study shapes are implemented.

## Open Questions

- Should public classes use the suffix `Policy`, `Study` or both?
- Should study outputs share a common base result interface?
- Should studies accept a `WalkForwardRunner`, or create one internally?
- Should comparison studies return one combined result or multiple named results?

## Recommendation

Keep the existing policy classes stable for now, but document them as part of a
study layer. Introduce a formal base class only after at least one more study type
is implemented and the repeated result shape is clear.

