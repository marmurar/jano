# Spec: Online Temporal Runner

## Purpose

`OnlineTemporalRunner` evaluates models under prequential temporal learning:

1. fit on an initial train window;
2. predict the next event or micro-batch;
3. observe the target;
4. update the model;
5. record metrics over time.

This extends Jano's simulation layer without changing fold geometry or the
splitter core.

## Public Inputs

The runner accepts:

- an estimator;
- `time_col`;
- `target_col`;
- optional `feature_cols`;
- `initial_train_size`;
- `update_size`;
- an update strategy;
- an optional `EvaluationProfile`;
- metric names or custom metric functions.

`initial_train_size` and `update_size` must use the same unit family:

- row counts, for event-level or fixed-size micro-batches;
- duration strings, for calendar-like micro-batches;
- fractions, for proportion-based simulations.

## Update Strategies

The initial strategies are `PartialFitUpdateStrategy` and `RefitUpdateStrategy`.

`PartialFitUpdateStrategy` requires a model with `partial_fit` and optionally
accepts `classes` for classifiers that need the complete label set on the first
call.

`RefitUpdateStrategy` requires a standard `fit` method. It appends each observed
batch to retained history and refits the estimator. It can optionally keep a
bounded rolling history through `max_train_rows`.

Future strategies may include:

- warm-start continuation;
- refit from bounded history;
- custom user-owned update callables;
- model-specific wrappers outside the core package.

## Outputs

`OnlineRunResult` exposes:

- `to_frame()` for one row per event or micro-batch;
- `metric_trajectory()` for long-format metrics;
- `predictions_frame()` for row-level predictions when requested;
- `summary()` for aggregate execution statistics;
- `report_data()` and `to_dict()` for agent-friendly structured output.

`OnlineUpdatePolicyStudy` compares multiple `OnlineUpdatePolicy` candidates over
the same stream. A policy defines:

- a stable name;
- an `update_size`, such as one event, `N` rows or a duration;
- an update strategy;
- an optional relative update cost.

`OnlineUpdatePolicyStudyResult` exposes:

- `to_frame()` with one row per candidate policy;
- `metric_trajectory()` with all candidate metric trajectories;
- `run(policy)` for the detailed underlying `OnlineRunResult`;
- `find_optimal_policy()` for metric or cost-adjusted selection.

## Acceptance Criteria

- Predictions are made before the batch target is used for updating.
- Event-level updates are possible with `update_size=1`.
- Duration micro-batches are possible with values such as `update_size="1D"`.
- Initial history and update cadence can use different unit families, such as
  initial train by duration and updates by number of observations.
- Several observation-driven policies can be compared over the same stream.
- The splitter core remains unchanged.
- Output is data-first and plot-ready.
