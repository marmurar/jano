# Jano

Use this skill when the user needs temporal partitioning, walk-forward validation,
temporal simulation, retraining policy evaluation or leakage-aware time splits with
Jano.

The canonical guide is `docs/ai/jano-agent-guide.md`. Read that file first when you
need examples, API selection rules or repo architecture constraints.

## When to Use

Use Jano when:

- the dataset has a time column
- random splitting would leak future information
- the user wants rolling, expanding or single-window evaluation
- the user wants to inspect temporal fold geometry
- the user wants to compare retraining policies
- the user wants structured fold, metric or prediction outputs for external plotting
- the user will provide project-specific metric or loss functions

## Preferred APIs

- Start with `WalkForwardPolicy` for production-like simulations.
- Use `policy.plan(frame)` before materializing folds when inspecting geometry.
- Use `WalkForwardRunner` when fitting and predicting over folds.
- Use `TemporalSystemRunner` when the user has a system that updates state and
  then evaluates the current state on the next test window, for example RAG
  refreshes, prompt updates or custom fine-tuning jobs.
- Use `TemporalBacktestSplitter` for manual control.
- Use `report_data()`, `metric_trajectory()` and `fold_summary()` for agent-readable outputs.
- Use `evaluation_details()` and `update_events()` when working with
  `TemporalSystemRunner`.
- Use `OnlineTemporalRunner(retrain_trigger=...)` when the user wants online
  event or micro-batch checkpoints for retraining inflection points.
- Use MCP tools `inspect_local_dataset`, `suggest_temporal_partition_policy` and
  `validate_temporal_partition_policy` before running models on an unfamiliar file.
- Use MCP tool `compare_temporal_partition_strategies` when multiple fold geometries
  are plausible.
- Use the MCP tool `run_walk_forward_baseline_model` for quick local sanity checks
  before writing custom model code.
- Use MCP study tools for baseline temporal hypotheses:
  `compare_retrain_policy_baselines`, `find_train_history_window_baseline` and
  `monitor_decay_baseline`.
- Use `estimate_prediction_band_by_fold` when the user wants a built-in temporal
  scenario for prediction bands, but keep the band computation in a user-owned
  `band_estimator`.
- Keep `TemporalSystemRunner` in Python. MCP is for dataset inspection,
  planning and baseline studies; it does not transport arbitrary
  `UpdateableSystem` implementations.

## Metric Contract

Jano does not implement or copy common metric formulas. Do not pass metric names
as strings and do not assume built-in `mae`, `rmse` or `accuracy` helpers exist.

Define metric functions in user code and pass them as a mapping:

```python
def business_cost(y_true, y_pred):
    ...

runner = WalkForwardRunner(
    model=model,
    target_col="target",
    metrics={"business_cost": business_cost},
    metric_directions={"business_cost": "min"},
)
```

Metric names are labels for output columns and policy decisions; metric formulas
belong to the user.

## Rules

- Do not use random `train_test_split` for time-dependent evaluation.
- Do not add model logic to `TemporalBacktestSplitter`.
- Do not add metric formulas to Jano core.
- Do not add built-in K-fold, bootstrap, conformal or confidence-interval
  formulas to Jano scenarios; the user-owned `band_estimator` computes bands.
- Do not add built-in online drift formulas; keep retrain checkpoint logic as
  user-provided callables.
- Prefer structured outputs over generated HTML for runner results.
- Read `docs/architecture/` before changing public APIs.
