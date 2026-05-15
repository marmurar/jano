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

## Preferred APIs

- Start with `WalkForwardPolicy` for production-like simulations.
- Use `policy.plan(frame)` before materializing folds when inspecting geometry.
- Use `WalkForwardRunner` when fitting and predicting over folds.
- Use `TemporalBacktestSplitter` for manual control.
- Use `report_data()`, `metric_trajectory()` and `fold_summary()` for agent-readable outputs.
- Use the MCP tool `run_walk_forward_baseline_model` for quick local sanity checks
  before writing custom model code.
- Use MCP study tools for baseline temporal hypotheses:
  `compare_retrain_policy_baselines`, `find_train_history_window_baseline` and
  `monitor_decay_baseline`.

## Rules

- Do not use random `train_test_split` for time-dependent evaluation.
- Do not add model logic to `TemporalBacktestSplitter`.
- Prefer structured outputs over generated HTML for runner results.
- Read `docs/architecture/` before changing public APIs.
