# ADR 0003: Manual Fold Iteration Remains Public

## Status

Accepted.

## Context

Jano is adding higher-level APIs such as `WalkForwardPolicy`, `TemporalSimulation`,
`WalkForwardRunner` and study policies. These reduce boilerplate for common temporal
experiments, but some users need full control over model training, feature generation,
metrics, leakage handling and external orchestration.

## Decision

Manual fold iteration remains a supported public mode.

The low-level APIs must continue to support patterns like:

```python
for split in splitter.iter_splits(frame):
    ...
```

and:

```python
for train_idx, test_idx in splitter.split(frame):
    ...
```

Higher-level simulations, runners and studies are additive. They do not replace the
manual mode.

## Consequences

- Jano remains useful for advanced users and custom pipelines.
- Higher-level APIs can be opinionated without trapping users.
- Documentation should present examples from simple to advanced composition.

## Invariants

- `TemporalBacktestSplitter.split()` and `iter_splits()` stay public.
- Rich fold objects remain inspectable and sliceable.
- New abstractions must be reducible to lower-level temporal primitives.

