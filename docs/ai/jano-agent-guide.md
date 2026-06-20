# Jano Agent Guide

This guide is the canonical AI-facing usage guide for Jano. Use it when an agent
needs to write code with Jano, analyze a temporal dataset, propose validation
experiments or modify the project.

## What Jano Is

Jano is a temporal simulation and backtesting toolkit for time-dependent machine
learning systems. It is designed for datasets where chronology matters and random
splits can leak future information into training.

Use Jano to:

- define temporal train/test or train/validation/test partitions
- inspect fold geometry before materializing data
- run walk-forward simulations
- execute models over temporal folds under explicit retraining policies
- structure evidence about drift, decay, retraining cadence and train history

Jano is not AutoML, a model registry, a feature store or a dashboard framework.

## Recommended API Selection

Use `TemporalBacktestSplitter` when the user wants low-level fold iteration or full
manual control.

Use `WalkForwardPolicy` when the user wants a production-like temporal simulation
with rolling, expanding or single-window movement.

Use `policy.plan(frame)` before running when the user wants to inspect fold dates,
row counts, iteration indices or exclude specific windows.

Use `WalkForwardRunner` when the user wants Jano to fit, predict and measure a model
over temporal folds.

Use `TemporalSystemRunner` when the user wants Jano to simulate update policies
over a system that is better described as `update(train_frame) -> state` and
`evaluate(state, test_frame) -> metrics`, such as RAG refreshes, prompt-set
updates or custom fine-tuning jobs.

Use `TrainHistoryPolicy` when the question is: does more training history improve
performance on the same fixed test window?

Use `DriftMonitoringPolicy` or `PerformanceDecayPolicy` when the question is: how
long does a model or rule remain useful after a fixed training window?

Use `RollingTrainHistoryPolicy` when the question is: how much history is optimal on
average across walk-forward iterations?

Use `jano.scenarios.estimate_prediction_band_by_fold` when the user wants a
ready-made temporal workflow that returns point predictions plus per-row
prediction bands for each Jano fold. Jano only provides the temporal fold context;
the user-owned `band_estimator` must decide how to compute the band, for example
with K-fold validation, bootstrap, conformal prediction, quantile models or a
domain-specific method.

## Metric Contract

Jano does not implement metric formulas. Agents should not pass strings such as
`"mae"`, `"rmse"` or `"accuracy"` expecting Jano to resolve them.

Define the metric in user code and pass a mapping:

```python
import numpy as np

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

metrics = {"mae": mae}
```

Metric names are labels used in result frames, trajectories and retraining rules.
Metric formulas belong to the user or to external metric libraries chosen by the
user.

## Core Patterns

### Inspect Before Running

Prefer planning before materialization when the user asks about fold geometry,
dataset windows, row counts or excluded dates.

```python
from jano import TemporalPartitionSpec, WalkForwardPolicy

policy = WalkForwardPolicy(
    time_col="timestamp",
    partition=TemporalPartitionSpec(
        layout="train_test",
        train_size="30D",
        test_size="7D",
    ),
    step="7D",
    strategy="rolling",
)

plan = policy.plan(frame)
plan_frame = plan.to_frame()
```

### Run a Simulation Without a Model

Use this when the user wants fold-level simulation and reporting but not model
execution.

```python
result = policy.run(frame, title="Walk-forward simulation")

summary_frame = result.to_frame()
chart_data = result.chart_data.to_dict()
```

### Execute a Model Over Folds

Use this when the user wants metrics, predictions or retraining behavior.

```python
import numpy as np

from jano import WalkForwardRunner

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

runner = WalkForwardRunner(
    model=model,
    target_col="target",
    feature_cols=["feature_a", "feature_b"],
    retrain="periodic",
    retrain_interval=2,
    metrics={"mae": mae, "rmse": rmse},
)

run = runner.run(policy, frame)

records = run.to_frame()
metrics = run.metric_trajectory()
retrain_events = run.retrain_events()
report_data = run.report_data()
```

### Execute a Temporal System Without Forcing `predict()`

Use this when the user owns a system that updates state and then evaluates that
state on the next test window.

```python
from jano import PeriodicRetrain, TemporalSystemRunner

runner = TemporalSystemRunner(
    system=my_system,
    update_policy=PeriodicRetrain(2),
    metric_directions={"groundedness": "max", "cost_usd": "min"},
    primary_metric="groundedness",
)

run = runner.run(policy, frame)

records = run.to_frame()
metrics = run.metric_trajectory()
update_events = run.update_events()
details = run.evaluation_details()
```

### Estimate Prediction Bands With User-Owned Uncertainty Logic

Use this when the temporal fold geometry should be standardized by Jano, but the
uncertainty method belongs to the project. Do not add cross-validation or
confidence-interval formulas to Jano for this scenario.

```python
from jano import estimate_prediction_band_by_fold


class MyBandEstimator:
    def estimate(self, context):
        lower, upper = compute_project_band(
            estimator=context.estimator,
            fitted_estimator=context.fitted_estimator,
            X_train=context.X_train,
            y_train=context.y_train,
            X_test=context.X_test,
            predictions=context.predictions,
        )
        return {"lower": lower, "upper": upper}


result = estimate_prediction_band_by_fold(
    frame,
    estimator=model,
    band_estimator=MyBandEstimator(),
    time_col="timestamp",
    target_col="target",
    feature_cols=["feature_a", "feature_b"],
    train_size="90D",
    test_size="7D",
    step="7D",
    metrics={"business_cost": business_cost},
)
```

## Data-First Reporting

Prefer structured outputs over generated HTML for model execution. Jano's runner is
designed to return evidence that notebooks, dashboards, slides or agents can
visualize externally.

Use:

- `run.fold_summary()` for temporal fold geometry and retraining metadata
- `run.metric_trajectory()` for long-format metric data
- `run.retrain_events()` for folds where the model was refit
- `run.predictions_frame()` for row-level test predictions
- `run.report_data()` or `run.to_dict()` for JSON-ready agent output
- `OnlineRunResult.retrain_checkpoints()` for online batches where a
  user-defined retrain trigger fired

Do not make HTML dashboards the primary contract for runner results.

## MCP Usage

Use the local MCP server when the agent should execute Jano against local files
through declared tools instead of writing Python ad hoc.

Available MCP tools cover:

- dataset preview with `preview_local_dataset`
- schema inspection and column hints with `inspect_local_dataset`
- conservative policy suggestions with `suggest_temporal_partition_policy`
- plan-level diagnostics with `validate_temporal_partition_policy`
- policy geometry comparisons with `compare_temporal_partition_strategies`
- fold planning with `plan_walk_forward_simulation`
- simulation execution with `run_walk_forward_simulation`
- parallel simulation sweeps with `run_simulation_campaign`
- simple baseline-model execution with `run_walk_forward_baseline_model`
- retraining policy comparison with `compare_retrain_policy_baselines`
- fixed-test train-history search with `find_train_history_window_baseline`
- fixed-train decay monitoring with `monitor_decay_baseline`

For unfamiliar datasets, inspect and validate before running models:

1. call `inspect_local_dataset`;
2. call `suggest_temporal_partition_policy`;
3. call `validate_temporal_partition_policy`;
4. compare alternatives with `compare_temporal_partition_strategies` if needed.

Use `run_walk_forward_baseline_model` for quick sanity checks over a dataset before
writing custom model code. It supports `model="mean"` for numeric regression targets
and `model="majority_class"` for classification targets. For production estimators,
write Python with `WalkForwardRunner` or `TemporalSystemRunner` so the project
controls feature engineering, model construction, update logic and custom
metrics explicitly.

Use `run_simulation_campaign` when the question is not a single fold geometry but
several competing simulation hypotheses that should be compared in parallel.

Use the baseline study tools when the agent needs fast evidence about temporal
hypotheses before writing custom model code. These MCP tools are for operational
triage and dataset inspection, not a substitute for project-owned estimators.

## Temporal Safety Rules

- Do not use random `train_test_split` for time-dependent validation.
- Do not train on rows whose target or label would not have been available in production.
- Use `gap_before_test`, `gap_before_validation` or `TemporalSemanticsSpec` when leakage is possible.
- Use calendar alignment with `calendar_frequency` when the user wants complete days or other fixed calendar periods.
- Keep model fitting and metrics outside `TemporalBacktestSplitter`.
- Keep prediction-band uncertainty logic user-defined. Pass a `band_estimator`
  to `estimate_prediction_band_by_fold`; do not add built-in K-fold, bootstrap
  or confidence-interval implementations to Jano core.
- Keep online drift/retrain checkpoint logic user-defined. Pass a callable
  `retrain_trigger` to `OnlineTemporalRunner`; do not add built-in drift formulas.
- Preserve manual fold iteration as a valid path for advanced users.

## Multiple Time Columns

Use `TemporalSemanticsSpec` when the dataset has different timestamps for ordering,
training eligibility or testing eligibility.

Example: flights may have `scheduled_departure_at` and `actual_arrival_at`. A train
row may only be eligible after arrival, while the timeline may remain anchored on
scheduled departure.

```python
from jano import TemporalSemanticsSpec

semantics = TemporalSemanticsSpec(
    timeline_col="scheduled_departure_at",
    order_col="scheduled_departure_at",
    segment_time_cols={
        "train": "actual_arrival_at",
        "test": "scheduled_departure_at",
    },
)
```

## Working on the Jano Repo

Before changing architecture or public APIs, read:

- `docs/architecture/README.md`
- `docs/architecture/adrs/`
- `docs/architecture/specs/`
- `docs/architecture/rfcs/`

Important architecture rules:

- the splitter remains model-agnostic
- runner execution does not redefine fold geometry
- metrics are user-owned callables, not Jano built-ins
- runner outputs are data-first
- manual fold iteration remains public
- studies compose lower-level primitives

## Verification

For code changes, run:

```bash
python3 -m pytest --cov=jano --cov-report=term-missing
python3 -m sphinx -b html docs docs/_build/html
```

The project currently has a 99% coverage gate.
