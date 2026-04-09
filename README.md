# Jano

<p align="center">
  <img src="./imgs/jano_logo.png" alt="Jano logo" width="260" />
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

That makes it useful not only for machine learning, but for any workflow where the data is time-dependent:

- Backtesting predictive models on transactional data.
- Simulating daily or weekly retraining in production.
- Comparing rolling versus expanding windows.
- Introducing explicit gaps between training and evaluation periods.
- Defining `train/test` or `train/validation/test` partitions with durations, row counts or percentages.

## Project direction

Jano is being reshaped as a small, explicit temporal partitioning toolkit with an interface inspired by `sklearn.model_selection`.

The design goals are:

- Clear, composable temporal partition definitions.
- Low hidden state and predictable behavior.
- Compatibility with pandas-first workflows.
- A splitter-style API that can evolve toward stronger scikit-learn interoperability.
- Rich split objects for inspection, auditability and simulation.

## Current API

The main entry point is `TemporalBacktestSplitter`.

It supports:

- `single`, `rolling` and `expanding` strategies.
- `train_test` and `train_val_test` layouts.
- Segment sizes defined as durations like `"30D"`, row counts like `5000`, or fractions like `0.7`.
- Optional gaps before validation or test segments.
- Plain index output through `split()`.
- Rich fold objects through `iter_splits()`.
- Simulation summaries and HTML timeline reports through `describe_simulation()`.

## Example: rolling backtest by duration

```python
import pandas as pd

from jano import TemporalBacktestSplitter, TemporalPartitionSpec

frame = pd.DataFrame(
    {
        "timestamp": pd.date_range("2024-01-01", periods=12, freq="D"),
        "feature": range(12),
        "target": range(100, 112),
    }
)

splitter = TemporalBacktestSplitter(
    time_col="timestamp",
    partition=TemporalPartitionSpec(
        layout="train_test",
        train_size="5D",
        test_size="2D",
        gap_before_test="1D",
    ),
    step="1D",
    strategy="rolling",
)

for train_idx, test_idx in splitter.split(frame):
    train = frame.iloc[train_idx]
    test = frame.iloc[test_idx]
    print(train["timestamp"].min(), test["timestamp"].min())
```

## Example: `train/validation/test` by fraction

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
summary = splitter.describe_simulation(
    frame,
    output_path="simulation.html",
    title="Walk-forward simulation",
)

print(summary.total_folds)
print(summary.to_frame().head())
```

That produces an HTML report showing each fold across the dataset timeline, with colored train, validation and test segments plus row counts per partition.

## Installation

Once published, the package will be installable from PyPI.

For local development:

```bash
python -m pip install -e ".[dev]"
python -m pytest --cov=jano --cov-report=term-missing
python -m sphinx -b html docs docs/_build/html
```

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
