# Claude Guidance for Jano

Use `docs/ai/jano-agent-guide.md` as the primary guide for working with Jano.

## Project Intent

Jano is a temporal simulation and backtesting toolkit for time-dependent machine
learning systems. It helps define temporal partitions, inspect simulation plans,
run walk-forward evaluations and execute models under explicit retraining policies.

## Working Rules

- Prefer `WalkForwardPolicy` plus `plan()` for temporal simulation design.
- Use `WalkForwardRunner` for model execution over folds.
- Use `TemporalBacktestSplitter` when manual fold iteration is required.
- Keep the splitter model-agnostic.
- Keep runner outputs data-first and plot-ready.
- Read `docs/architecture/` before changing architecture or public APIs.

## Verification

Run these before considering code changes complete:

```bash
python3 -m pytest --cov=jano --cov-report=term-missing
python3 -m sphinx -b html docs docs/_build/html
```

