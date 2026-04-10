# Jano

<p align="center">
  <img src="https://raw.githubusercontent.com/marmurar/jano/master/imgs/jano_logo.png" alt="Jano logo" width="260" />
</p>

[![CI](https://github.com/marmurar/jano/actions/workflows/ci.yml/badge.svg)](https://github.com/marmurar/jano/actions/workflows/ci.yml)
[![Docs](https://github.com/marmurar/jano/actions/workflows/docs.yml/badge.svg)](https://github.com/marmurar/jano/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/marmurar/jano/graph/badge.svg)](https://codecov.io/gh/marmurar/jano)

Jano is a Python library for defining temporal partitions and backtesting schemes over time-correlated datasets.

Documentation: [marmurar.github.io/jano](https://marmurar.github.io/jano/)

It is designed for cases where a plain `train_test_split()` is not enough: transactional data, production simulations, repeated retraining, walk-forward validation, model monitoring, rule evaluation, or any experiment where the ordering of time matters.

The project is named after Janus, the Roman god of beginnings, transitions and thresholds. That framing fits the library well: Jano helps define how a dataset moves from training periods into evaluation periods, fold after fold.

## Why Jano exists

Most train/test utilities answer a simple question:

"How do I split this dataset once?"

Jano is meant to answer a richer one:

"How would this system have behaved over time if I had trained, retrained and evaluated it under a specific temporal policy?"

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

The recommended high-level entry point is `TemporalSimulation`.

`TemporalBacktestSplitter` remains available as the lower-level primitive when you want direct control over folds and manual iteration.

It supports:

- `single`, `rolling` and `expanding` strategies.
- `train_test` and `train_val_test` layouts.
- Segment sizes defined as durations like `"30D"`, row counts like `5000`, or fractions like `0.7`.
- Optional gaps before validation or test segments.
- Plain index output through `split()`.
- Rich fold objects through `iter_splits()`.
- Simulation summaries, HTML timeline reports and plot-ready chart data through `describe_simulation()`.
- A numpy-first internal indexing path to reduce split overhead on large datasets.

## Example: run a full simulation without manual iteration

```python
import pandas as pd

from jano import TemporalPartitionSpec, TemporalSimulation

frame = pd.DataFrame(
    {
        "timestamp": pd.date_range("2024-01-01", periods=60, freq="D"),
        "feature": range(60),
        "target": range(100, 160),
    }
)

simulation = TemporalSimulation(
    time_col="timestamp",
    partition=TemporalPartitionSpec(
        layout="train_test",
        train_size="30D",
        test_size="1D",
    ),
    step="1D",
    strategy="rolling",
)

result = simulation.run(frame, title="One month in production")

print(result.total_folds)
print(result.summary.to_frame().head())
print(result.chart_data.segment_stats)
```

You can also anchor a simulation to a specific date and limit how many folds are materialized:

```python
simulation = TemporalSimulation(
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

result = simulation.run(frame, title="15 daily retraining iterations")
```

The high-level simulation layer also supports `end_at` when you want to constrain the simulation to a bounded time window before folds are generated.

When a single timestamp is not enough, both `TemporalSimulation` and `TemporalBacktestSplitter` can also receive a `TemporalSemanticsSpec`. That lets you keep one column as the reported timeline while using different timestamp columns to decide whether `train`, `validation` or `test` rows are actually eligible. This is useful for production-style leakage control, for example when a target only becomes available at `arrived_at` even if the operational timeline is anchored on `departured_at`.

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

After the first PyPI release, install the package with:

```bash
python -m pip install jano
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
4. Push a tag like `v0.2.0`.

That tag triggers the `Publish` workflow, which builds the wheel and source distribution and publishes them to PyPI.

In parallel, the repository also includes a `GitHub Release` workflow that can create a GitHub Release and attach the built wheel and source distribution for any `v*` tag. That gives the project a distribution channel even while PyPI access is still being recovered.

## Continuous integration and coverage

The repository includes:

- GitHub Actions for tests across multiple Python versions.
- GitHub Pages publication for Sphinx documentation.
- Coverage reporting with `pytest-cov`.
- Codecov upload and status tracking.

## Status

Jano is currently in an early redesign phase. The public API is stabilizing around temporal partition specs, reusable splitters and rich split objects.

That means the project is already usable for experimentation, but it is still a good moment to refine naming, ergonomics and compatibility guarantees before publishing broadly.

## Authors

- Marcos Manuel Muraro

## Contributing

Feedback and design discussion are especially valuable right now. If you are using temporal backtesting for ML, analytics, operations or experimentation, that context can help shape the API in the right direction.

## Star history

[![Star History Chart](https://api.star-history.com/svg?repos=marmurar/jano&type=Date)](https://star-history.com/#marmurar/jano&Date)
