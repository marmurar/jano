# RFC 0002: Online Temporal Runner

## Status

Accepted for initial implementation on a feature branch.

## Context

Jano already supports temporal split policies, walk-forward simulation, model
execution over folds and retraining policies. Those layers answer questions such
as:

- What happens if the model retrains daily, weekly or every `K` folds?
- What happens if the model is trained once and reused?
- What happens if retraining depends on previously observed metrics?

There is another production-like learning policy that does not fit a classic
train/test fold: predict the next event, observe its target, update the model and
continue. This is commonly described as online or prequential evaluation.

## Decision

Add an `OnlineTemporalRunner` layer instead of changing `TemporalBacktestSplitter`.

The runner will simulate:

```text
initial train window
predict next event or micro-batch
observe target
update model
repeat
```

The first update strategies are:

- `PartialFitUpdateStrategy`, targeting estimators that implement
  scikit-learn-style `partial_fit`.
- `RefitUpdateStrategy`, targeting standard `fit/predict` estimators by refitting
  on retained observed history after each batch.

Add `OnlineUpdatePolicyStudy` to compare observation-driven policies over the
same stream. A policy can update every event, every `N` rows, every duration
window or with a different retained-history strategy. The study returns
policy-level metrics and can select the best policy with an optional update-cost
penalty.

## Rationale

This keeps the splitter core model-agnostic while expanding Jano into a broader
temporal simulation framework.

The new runner answers a different question than walk-forward folds:

- Walk-forward: how does a model behave when retrained and evaluated on temporal
  blocks?
- Online runner: how does a model behave when it learns continuously from events
  or micro-batches?
- Policy study: how much new evidence should trigger retraining, and is the
  predictive gain worth the update cost?

## Non-Goals

- Do not turn Jano into AutoML.
- Do not implement model-specific XGBoost, LightGBM or neural network update
  logic in the core.
- Do not make online learning part of `TemporalBacktestSplitter`.
- Do not hide the update policy from the user.

## Future Work

- Warm-start update strategies.
- Event weighting or recency weighting.
- Micro-batch policies based on event count or duration.
- Cost-aware comparison between online updates and periodic retraining.
