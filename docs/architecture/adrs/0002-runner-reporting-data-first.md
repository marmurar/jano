# ADR 0002: Keep Runner Reporting Data-First

## Status

Accepted.

## Context

`WalkForwardRunner` executes estimators over temporal folds and returns fold-level
metrics and predictions. A possible next step was to build an HTML reporting system
for runner results.

The project direction favors Jano as a temporal experimentation framework, not as a
dashboard generator. AI agents, notebooks, dashboards and presentation tools can
create visual outputs if Jano exposes structured evidence clearly.

## Decision

Runner reporting remains data-first.

`WalkForwardRunResult` exposes structured outputs such as:

- `to_frame()`
- `fold_summary()`
- `metric_trajectory()`
- `retrain_events()`
- `predictions_frame()`
- `report_data()`
- `to_dict()`

Jano may keep simple HTML for simulation geometry inspection, but runner execution
results should prioritize stable, plot-ready and JSON-ready data.

## Consequences

- Agents can consume runner outputs without parsing HTML.
- Users can choose their own visualization layer and style.
- The runner avoids assumptions about regression, binary classification, multiclass
classification or custom cost functions.
- The runner avoids implementing generic metric formulas; user code supplies the
  callable and Jano reports the resulting values.
- Future target-aware reporting should be implemented as structured metadata first.

## Invariants

- Runner output methods should not assume a target type by default.
- Metric names are output labels, not references to built-in Jano formulas.
- New reporting helpers should return pandas objects or serializable dictionaries.
- HTML should not become the primary execution-reporting contract.
