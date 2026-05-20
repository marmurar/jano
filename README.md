# Jano

<p align="center">
  <img src="https://raw.githubusercontent.com/marmurar/jano/master/imgs/jano_logo.png" alt="Jano logo" width="260" />
</p>

[![CI](https://github.com/marmurar/jano/actions/workflows/ci.yml/badge.svg)](https://github.com/marmurar/jano/actions/workflows/ci.yml)
[![Docs](https://github.com/marmurar/jano/actions/workflows/docs.yml/badge.svg)](https://github.com/marmurar/jano/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/marmurar/jano/graph/badge.svg)](https://codecov.io/gh/marmurar/jano)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20301006-blue.svg)](https://doi.org/10.5281/zenodo.20301006)
[![PyPI](https://img.shields.io/pypi/v/jano.svg)](https://pypi.org/project/jano/)
[![Python versions](https://img.shields.io/pypi/pyversions/jano.svg)](https://pypi.org/project/jano/)
[![PyPI Downloads](https://static.pepy.tech/badge/jano/month)](https://pepy.tech/projects/jano)
[![License](https://img.shields.io/pypi/l/jano.svg)](https://github.com/marmurar/jano/blob/master/LICENSE.txt)

Jano is a Python library for defining temporal partitions and backtesting schemes over time-correlated datasets.

The missing layer between ML models and production temporal validation.

Documentation: [marmurar.github.io/jano](https://marmurar.github.io/jano/)

It is designed for cases where a plain `train_test_split()` is not enough: transactional data, production simulations, repeated retraining, walk-forward validation, model monitoring, rule evaluation, or any experiment where the ordering of time matters.

The core accepts `pandas.DataFrame`, `numpy.ndarray` and `polars.DataFrame` inputs through a unified API. Jano keeps native pandas, NumPy and Polars paths for partition planning when that is safe, and falls back to pandas materialization for reporting and user-facing slices.

The project is named after Janus, the Roman god of beginnings, transitions and thresholds. That framing fits the library well: Jano helps define how a dataset moves from training periods into evaluation periods, fold after fold.

## MCP server

Jano also ships an optional local MCP server so AI agents can use the library through a small, explicit tool surface instead of generating Python ad hoc.

Current MCP tools:

- `preview_local_dataset`
- `plan_walk_forward_simulation`
- `run_walk_forward_simulation`
- `run_walk_forward_baseline_model`
- `compare_retrain_policy_baselines`
- `find_train_history_window_baseline`
- `monitor_decay_baseline`

Install it in a Python 3.10+ environment:

```bash
python -m pip install "jano[mcp]"
```

Run it locally over stdio:

```bash
jano-mcp
```

Or use the module entrypoint:

```bash
python -m jano.mcp_server
```

Example MCP client configuration:

```json
{
  "mcpServers": {
    "jano": {
      "command": "jano-mcp"
    }
  }
}
```

The MCP layer is intentionally opinionated: it exposes planning, walk-forward simulation, baseline-model execution and baseline temporal studies first, while the full Python library remains available when you need custom composition.

This is meant for MCP-aware coding assistants such as Claude Code, Claude Desktop, Cursor, Codex runtimes with MCP support, and other local agent environments. The server runs locally and reads only the file paths you provide to its tools; Jano does not upload datasets anywhere by itself.

## AI-ready usage

Jano includes three surfaces intended to make the project easier for AI agents to use and extend:

- Architecture notes in `docs/architecture/` explain the project layers, accepted decisions, specs and open RFCs.
- The canonical agent guide in `docs/ai/jano-agent-guide.md` explains which Jano API to use for common temporal validation tasks.
- Tool-specific adapters provide lightweight entry points for Codex, Claude and Cursor:
  - `skills/jano/SKILL.md`
  - `CLAUDE.md`
  - `.cursor/rules/jano.mdc`

Use the MCP server when an agent should execute Jano operations over local datasets. Use the skill or agent guide when an agent needs to reason about Jano, write code with the library or modify the repository safely.

## Reproducible external datasets

Jano examples should be reproducible without committing large datasets to Git.
Dataset metadata is versioned in `datasets/registry.json`, while downloaded files
stay local under `data/raw/`, which is ignored by the repository.

List available datasets:

```bash
python scripts/download_dataset.py --list
```

Download one locally:

```bash
python scripts/download_dataset.py bike_sharing_hourly --extract
```

The current registry includes Bike Sharing, BTS Airline On-Time Performance,
NYC TLC Yellow Taxi and Household Power datasets for regression, classification,
ordinal-cost and larger benchmark examples.

## Why Jano exists

Many machine learning datasets are not just tabular; they are structured over time and often across multiple entities such as users, routes, sellers or products. In those settings, a more faithful view of the data is not "a bag of independent rows" but a temporally ordered process.

Standard evaluation tooling usually assumes observations are i.i.d. enough that a static split is acceptable. That assumption breaks quickly when time matters: future information leaks into training, performance estimates become optimistic, and offline validation stops reflecting what really happens in production.

Most train/test utilities answer a simple question:

"How do I split this dataset once?"

Jano is meant to answer a richer one:

"How would this system have behaved over time if I had trained, retrained and evaluated it under a specific temporal policy?"

That difference is the core of the project. Jano treats evaluation as a temporal simulation rather than a static partition. Instead of defining one split, it defines a policy over time: train window, evaluation horizon, shift between iterations and optional leakage-control gaps. Running that policy produces a sequence of causally valid folds rather than one aggregate estimate.

That also makes it a useful way to evidence drift in simulation results, because temporal shifts in behavior, performance or calibration become visible fold after fold.

That makes it useful not only for machine learning, but for any workflow where the data is time-dependent:

- Backtesting predictive models on transactional data.
- Simulating daily or weekly retraining in production.
- Comparing rolling versus expanding windows.
- Introducing explicit gaps between training and evaluation periods.
- Defining `train/test` or `train/validation/test` partitions with durations, row counts or percentages.
- Surfacing drift in simulation outcomes by making temporal changes explicit across folds.

## Project direction

Jano is being reshaped as a small, explicit temporal partitioning toolkit with an interface inspired by `sklearn.model_selection`.

The design goals are:

- Clear, composable temporal partition definitions.
- Low hidden state and predictable behavior.
- Compatibility with pandas-first workflows.
- A splitter-style API that can evolve toward stronger scikit-learn interoperability.
- Rich split objects for inspection, auditability and simulation.

## Current API

The recommended high-level surface is intentionally small:

- `WalkForwardPolicy` for production-like walk-forward evaluation,
- `WalkForwardRunner` when you want Jano to execute a model over those folds and manage retraining cadence,
- `TrainHistoryPolicy` for fixed-test, growing-train questions,
- `DriftMonitoringPolicy` for fixed-train, moving-test questions.

Those classes sit on top of the lower-level building blocks that remain available:

- `TemporalSimulation` for explicit simulation objects,
- `TemporalBacktestSplitter` for manual fold iteration,
- `TrainGrowthPolicy` and `PerformanceDecayPolicy` for lower-level temporal hypothesis primitives.

The workflow is intentionally compositional:

- start simple with predefined layouts and strategies,
- move to `plan()` when you want to inspect or filter iterations before running them,
- use higher-level policies such as `TrainGrowthPolicy` or `PerformanceDecayPolicy` when the question is already encapsulated,
- and fall back to manual fold iteration when you want to compose everything yourself: partitions, gaps, feature history and model training logic.

The cleanest mental model is to treat Jano as five layers that can stay independent:

- `TemporalBacktestSplitter` for temporal geometry and manual fold iteration.
- `plan()` for inspecting and filtering that geometry before materialization.
- `TemporalSimulation` and `WalkForwardPolicy` for fold-level simulation and reporting.
- `WalkForwardRunner` for training, predicting and measuring over temporal folds with explicit retrain rules.
- higher-level studies and policies for operational questions such as train sufficiency, decay and retraining cadence.

That separation is deliberate. The splitter remains the free-form core. Runners and studies extend what Jano can do at the simulation layer, but they do not replace manual fold iteration.

It supports:

- `single`, `rolling` and `expanding` strategies.
- `train_test` and `train_val_test` layouts.
- Segment sizes defined as durations like `"30D"`, row counts like `5000`, or fractions like `0.7`.
- Calendar-aligned duration windows with `calendar_frequency="D"` when you want complete days instead of elapsed-time windows anchored at the first timestamp.
- Optional gaps before validation or test segments.
- Plain index output through `split()`.
- Rich fold objects through `iter_splits()`.
- Simulation summaries, HTML timeline reports and plot-ready chart data through `describe_simulation()`.
- An adaptive partition engine that keeps pandas, NumPy and Polars inputs native for planning when it is safe, and falls back to pandas when stability is more important.

## Example: random splits vs temporal validation

`sklearn.model_selection.train_test_split` is useful for random i.i.d.-style
evaluation. It is the wrong abstraction when the model will be trained on the
past and asked to predict the future.

The first snippet assumes scikit-learn is installed only to illustrate the common
baseline. Jano itself does not require scikit-learn.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

frame = pd.DataFrame(
    {
        "timestamp": pd.date_range("2025-01-01", periods=120, freq="D"),
        "feature": range(120),
        "target": [0] * 80 + [1] * 40,
    }
)

train_random, test_random = train_test_split(
    frame,
    test_size=0.2,
    shuffle=True,
    random_state=7,
)

print(train_random["timestamp"].max() > test_random["timestamp"].min())
# True: train contains dates later than some test rows.
```

With Jano, the evaluation is a temporal policy:

```python
from jano import TemporalPartitionSpec, WalkForwardPolicy

policy = WalkForwardPolicy(
    time_col="timestamp",
    partition=TemporalPartitionSpec(
        layout="train_test",
        train_size="60D",
        test_size="14D",
        gap_before_test="1D",
    ),
    step="14D",
    strategy="rolling",
)

plan = policy.plan(frame, title="Production-like temporal validation")
print(plan.to_frame()[["iteration", "train_end", "test_start", "test_end"]])
```

That makes the temporal contract inspectable before training: train only sees the
past, test moves forward, and optional gaps model label or data availability
latency.

## Example: run a full simulation without manual iteration

```python
import pandas as pd

from jano import TemporalPartitionSpec, WalkForwardPolicy

frame = pd.DataFrame(
    {
        "timestamp": pd.date_range("2024-01-01", periods=60, freq="D"),
        "feature": range(60),
        "target": range(100, 160),
    }
)

policy = WalkForwardPolicy(
    time_col="timestamp",
    partition=TemporalPartitionSpec(
        layout="train_test",
        train_size="30D",
        test_size="1D",
    ),
    step="1D",
    strategy="rolling",
)

result = policy.run(frame, title="One month in production")

print(result.total_folds)
print(result.engine_metadata.to_dict())
print(result.summary.to_frame().head())
print(result.chart_data.segment_stats)
```

By default, `engine="auto"` lets Jano choose the safest fast path for partitioning:

## Example: run a model over the walk-forward policy

```python
from jano import TemporalPartitionSpec, WalkForwardPolicy, WalkForwardRunner

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

runner = WalkForwardRunner(
    model=model,
    target_col="target",
    feature_cols=["feature"],
    retrain="periodic",
    retrain_interval=2,
    metrics=["mae", "rmse"],
)

run = runner.run(policy, frame)

print(run.to_frame().head())
print(run.summary())
print(run.metric_trajectory().head())
print(run.retrain_events())

report_data = run.report_data(include_predictions=False)
```

Supported retrain modes are:

- `retrain=True` or `retrain="always"` to refit on every fold.
- `retrain=False` or `retrain="never"` to train once and benchmark a fixed model.
- `retrain="periodic"` with `retrain_interval=K` to refit every `K` folds.
- `retrain_policy=DriftBasedRetrain(...)` when the next retrain decision should depend on previously observed fold metrics.
- `retrain_policy=FunctionRetrainPolicy(...)` when the retrain decision is a custom function of fold history, dates, costs or external thresholds.

Evaluation profiles separate how a run is measured from when a model is retrained.
Built-in metrics are convenience shortcuts; production-like validation can pass a
custom loss or score and declare whether lower or higher is better:

```python
import numpy as np

from jano import EvaluationProfile, FunctionRetrainPolicy, WalkForwardRunner

def daily_cost(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def retrain_rule(context):
    if context.history.empty:
        return True
    latest = context.history["daily_cost"].iloc[-1]
    limit = limit_for_date(context.split.boundaries["test"].end)
    return latest > limit

runner = WalkForwardRunner(
    model=model,
    target_col="target",
    feature_cols=["feature"],
    evaluation=EvaluationProfile(
        metrics={"daily_cost": daily_cost},
        metric_directions={"daily_cost": "min"},
        primary_metric="daily_cost",
    ),
    retrain_policy=FunctionRetrainPolicy(retrain_rule),
)
```

Convenience profiles such as `RegressionProfile`, `ClassificationProfile`,
`OrdinalClassificationProfile` and `RankingProfile` keep the API explicit without
splitting the runner into problem-specific classes.

Runner results are intentionally data-first rather than dashboard-first:

- `run.fold_summary()` returns temporal fold geometry and retraining metadata.
- `run.metric_trajectory()` returns metrics in long format, ready for plotting.
- `run.retrain_events()` returns only folds where the estimator was refit.
- `run.predictions_frame()` returns row-level test predictions.
- `run.report_data()` / `run.to_dict()` return structured dictionaries for notebooks, agents, dashboards or presentation tools.

pandas inputs stay pandas, Polars inputs use Polars column extraction, and NumPy arrays
use array indexing. You can force a path with `engine="pandas"`, `engine="polars"` or
`engine="numpy"` when you need deterministic behavior for a pipeline.

If you want to inspect the full simulation geometry before materializing folds, plan it first:

```python
plan = policy.plan(frame, title="One month in production")
print(plan.total_folds)
print(plan.to_frame().head())

filtered = plan.exclude_windows(
    train=[("2025-12-20", "2026-01-05")],
).select_from_iteration(5)

result = filtered.materialize()
```

That plan frame includes the explicit iteration index, segment boundaries and row counts for each fold.

You can also anchor a simulation to a specific date and limit how many folds are materialized:

```python
policy = WalkForwardPolicy(
    time_col="timestamp",
    partition=TemporalPartitionSpec(
        layout="train_test",
        train_size="15D",
        test_size="4D",
    ),
    step="1D",
    strategy="rolling",
    start_at="2025-09-01",
    max_folds=15,
)

result = policy.run(frame, title="15 daily retraining iterations")
```

The recommended walk-forward surface also supports `end_at` when you want to constrain the simulation to a bounded time window before folds are generated.

When a single timestamp is not enough, `WalkForwardPolicy`, `TemporalSimulation` and `TemporalBacktestSplitter` can also receive a `TemporalSemanticsSpec`. That lets you keep one column as the reported timeline while using different timestamp columns to decide whether `train`, `validation` or `test` rows are actually eligible. This is useful for production-style leakage control, for example when a target only becomes available at `arrived_at` even if the operational timeline is anchored on `departured_at`.

For `numpy.ndarray` inputs, use integer column references:

```python
import numpy as np

values = np.array(
    [
        ["2025-09-01", 1.2, 10],
        ["2025-09-02", 1.5, 11],
        ["2025-09-03", 1.1, 12],
    ],
    dtype=object,
)

splitter = TemporalBacktestSplitter(
    time_col=0,
    partition=TemporalPartitionSpec(
        layout="train_test",
        train_size="2D",
        test_size="1D",
    ),
    step="1D",
    strategy="single",
)
```

## Example: manual control with the low-level splitter

```python
from jano import TemporalBacktestSplitter, TemporalPartitionSpec

splitter = TemporalBacktestSplitter(
    time_col="timestamp",
    partition=TemporalPartitionSpec(
        layout="train_val_test",
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
    ),
    step=0.2,
    strategy="single",
)

for split in splitter.iter_splits(frame):
    print(split.summary())
```

## Example: keep the same test window and grow train backward

This is a special use case. It is useful when you want to study whether more training history really improves the same test slice.

```python
from jano import TrainHistoryPolicy

policy = TrainHistoryPolicy(
    "timestamp",
    cutoff="2025-09-15",
    train_sizes=["7D", "14D", "21D", "28D"],
    test_size="4D",
)

result = policy.evaluate(
    frame,
    model=model,
    target_col="target",
    feature_cols=["feature_1", "feature_2"],
    metrics=["mae", "rmse"],
)

print(result.to_frame()[["train_size", "rmse"]])
print(result.find_optimal_train_size(metric="rmse", tolerance=0.01))
```

That pattern keeps `test` fixed while `train` expands toward the past. It is a practical way to study data efficiency or to estimate how much history is actually needed.

The opposite special case is also common: keep `train` fixed and move `test` forward day by day to estimate how long a model or rule keeps its performance without retraining. The two patterns answer different questions:

- fixed `test` + growing `train`: how much history do I actually need?
- fixed `train` + moving `test`: for how long does performance hold after deployment?

Example of the second pattern:

```python
from jano import DriftMonitoringPolicy

policy = DriftMonitoringPolicy(
    "timestamp",
    cutoff="2025-09-15",
    train_size="30D",
    test_size="3D",
    step="1D",
    max_windows=10,
)

result = policy.evaluate(
    frame,
    model=model,
    target_col="target",
    feature_cols=["feature_1", "feature_2"],
    metrics=["mae", "rmse"],
)

print(result.to_frame()[["window", "test_start", "rmse"]])
print(result.find_drift_onset(metric="rmse", threshold=0.15, baseline="first"))
```

## Example: optimize training history inside each walk-forward iteration

This is the next-level composed question: if each outer test window is allowed to choose its own optimal training history, how much history is needed on average?

```python
from jano import RollingTrainHistoryPolicy, TemporalPartitionSpec

policy = RollingTrainHistoryPolicy(
    "timestamp",
    partition=TemporalPartitionSpec(
        layout="train_test",
        train_size="30D",
        test_size="1D",
    ),
    step="1D",
    strategy="rolling",
    max_folds=10,
    train_sizes=["5D", "10D", "15D", "30D"],
)

result = policy.evaluate(
    frame,
    model=model,
    target_col="target",
    feature_cols=["feature_1", "feature_2"],
    metrics="rmse",
    metric="rmse",
    tolerance=0.01,
)

print(result.to_frame().head())
print(result.summary())
```

## Example: different feature groups can require different history depths

The supervised fold can stay fixed while feature engineering still asks for different
lookback windows per feature group.

```python
from jano import FeatureLookbackSpec

split = next(splitter.iter_splits(frame))
lookbacks = FeatureLookbackSpec(
    default_lookback="15D",
    group_lookbacks={"lag_features": "65D"},
    feature_groups={"lag_features": ["lag_30", "lag_60"]},
)

history = split.slice_feature_history(
    frame,
    lookbacks,
    time_col="timestamp",
    segment_name="train",
)

recent_context = history["__default__"]
lag_context = history["lag_features"]
```

This is useful when recent features only need a short window while lagged or seasonal
features need much deeper historical context for the same model.

## Example: describe a simulation as HTML

```python
summary = splitter.describe_simulation(frame, title="Walk-forward simulation")
html = splitter.describe_simulation(frame, output="html")
chart_data = splitter.describe_simulation(frame, output="chart_data")

print(summary.total_folds)
print(summary.to_frame().head())
print(chart_data.segment_stats)
```

That gives you three ways to consume the same simulation:

- `summary` for tabular metadata and export helpers,
- `html` for a standalone visual report,
- `chart_data` for direct Python plotting without reparsing HTML.

The generated report shows each fold across the dataset timeline, with richer summary cards, clearer segment labels and row counts per partition.

## Installation

Install the current release from PyPI:

```bash
python -m pip install jano
```

To use Polars inputs directly:

```bash
python -m pip install "jano[polars]"
```

For local development:

```bash
python -m pip install -e ".[dev]"
python -m pytest --cov=jano --cov-report=term-missing
python -m sphinx -b html docs docs/_build/html
```

Jano also exposes its runtime version through `jano.__version__`.

## Release flow

The repository includes a dedicated GitHub Actions workflow for PyPI publication through trusted publishing.

The release path is:

1. Update `jano/_version.py`.
2. Run `python -m pytest -q`.
3. Run `python -m build` and `python -m twine check dist/*`.
4. Push a tag like `v0.4.0`.

That tag triggers the `Publish` workflow, which builds the wheel and source distribution and publishes them to PyPI.

In parallel, the repository also includes a `GitHub Release` workflow that can create a GitHub Release and attach the built wheel and source distribution for any `v*` tag.

## Zenodo DOI

Jano includes repository metadata for Zenodo in `.zenodo.json` and citation
metadata in `CITATION.cff`.

To mint a DOI for the project:

1. Log in to Zenodo with the GitHub account that owns or administers this repository.
2. Open the Zenodo GitHub integration page.
3. Click `Sync now`.
4. Enable the `marmurar/jano` repository.
5. Create a new GitHub Release for the next version tag.
6. Wait for Zenodo to archive the release and assign the DOI.
7. Add the generated DOI badge and DOI URL back to this README and the Sphinx docs.

Current Zenodo DOI: [10.5281/zenodo.20301006](https://doi.org/10.5281/zenodo.20301006).

## Continuous integration and coverage

The repository includes:

- GitHub Actions for tests across multiple Python versions.
- GitHub Pages publication for Sphinx documentation.
- Coverage reporting with `pytest-cov`.
- Codecov upload and status tracking.
- A coverage gate set to 99%.

## Status

Jano is an early public project with a usable core and an API that is still being refined as the simulation layer grows.

The low-level temporal partitioning surface is the most stable part of the library: `TemporalBacktestSplitter`, `TemporalPartitionSpec`, `TemporalSimulation`, `WalkForwardPolicy` and `plan()` are the foundation for manual fold iteration, auditability and simulation planning.

The higher-level execution and study APIs, including `WalkForwardRunner`, retrain policies, train-history studies and drift-monitoring helpers, are intentionally evolving. They are covered by tests and documented, but naming and ergonomics may still change while Jano is being shaped into a broader temporal experimentation framework.

Current distribution and quality signals:

- PyPI package: [jano](https://pypi.org/project/jano/).
- Latest tested release line: `0.4.x`.
- Test suite: `134 passed`.
- Coverage gate: `99%` minimum.
- Current measured coverage: `99.15%`.
- Documentation: [marmurar.github.io/jano](https://marmurar.github.io/jano/).

For production use, pin an explicit version and review release notes before upgrading. For experimentation, temporal validation design work and prototype evaluation pipelines, the project is ready to use.

## Citation

If you use Jano in research, technical reports, benchmarks or production validation work, please cite the project with this BibTeX entry:

```bibtex
@software{muraro_jano_2026,
  author       = {Muraro, Marcos Manuel},
  title        = {Jano: Temporal Simulation and Backtesting Toolkit for Time-Dependent Machine Learning Systems},
  year         = {2026},
  version      = {0.4.1},
  url          = {https://github.com/marmurar/jano},
  repository   = {https://github.com/marmurar/jano},
  doi          = {10.5281/zenodo.20301006},
  license      = {MIT},
  note         = {Python toolkit for temporal simulation, walk-forward validation, backtesting, and retraining-policy analysis for time-dependent machine learning systems}
}
```

The same citation metadata is also available in [CITATION.cff](CITATION.cff).

## Authors

- Marcos Manuel Muraro

## Contributing

Feedback and design discussion are especially valuable right now. If you are using temporal backtesting for ML, analytics, operations or experimentation, that context can help shape the API in the right direction.

## Star history

[![Star History Chart](https://api.star-history.com/svg?repos=marmurar/jano&type=Date)](https://star-history.com/#marmurar/jano&Date)
