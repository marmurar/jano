from __future__ import annotations

from .types import TemporalPartitionSpec
from .simulation import TemporalSimulation
from .splitters import TemporalBacktestSplitter
from .workflows import WalkForwardPolicy


def _resolve_workflow(
    *,
    workflow,
    time_col,
    partition,
    train_size,
    test_size,
    step,
    strategy: str,
    max_folds: int | None,
):
    if workflow is not None:
        if any(value is not None for value in [time_col, partition, train_size, test_size, step]):
            raise ValueError(
                "workflow cannot be combined with time_col, partition, train_size, test_size or step"
            )
        return workflow
    if time_col is None:
        raise ValueError("time_col is required when workflow is not provided")
    if partition is None:
        if train_size is None or test_size is None:
            raise ValueError("train_size and test_size are required when partition is not provided")
        partition = TemporalPartitionSpec(
            layout="train_test",
            train_size=train_size,
            test_size=test_size,
        )
    if step is None:
        raise ValueError("step is required when workflow is not provided")
    return WalkForwardPolicy(
        time_col,
        partition=partition,
        step=step,
        strategy=strategy,
        max_folds=max_folds,
    )


def _resolve_workflow_inputs(workflow, frame_input):
    if isinstance(workflow, WalkForwardPolicy):
        simulation = workflow.simulation
        selected = simulation.select_input(frame_input)
        splits = list(simulation.as_splitter().iter_splits(selected))
        if simulation.max_folds is not None:
            splits = splits[: simulation.max_folds]
        return selected, simulation.as_splitter().temporal_semantics, splits

    if isinstance(workflow, TemporalSimulation):
        selected = workflow.select_input(frame_input)
        splits = list(workflow.as_splitter().iter_splits(selected))
        if workflow.max_folds is not None:
            splits = splits[: workflow.max_folds]
        return selected, workflow.as_splitter().temporal_semantics, splits

    if isinstance(workflow, TemporalBacktestSplitter):
        return frame_input, workflow.temporal_semantics, list(workflow.iter_splits(frame_input))

    if hasattr(workflow, "simulation") and isinstance(workflow.simulation, TemporalSimulation):
        simulation = workflow.simulation
        selected = simulation.select_input(frame_input)
        splits = list(simulation.as_splitter().iter_splits(selected))
        if simulation.max_folds is not None:
            splits = splits[: simulation.max_folds]
        return selected, simulation.as_splitter().temporal_semantics, splits

    if hasattr(workflow, "as_splitter"):
        splitter = workflow.as_splitter()
        return frame_input, splitter.temporal_semantics, list(splitter.iter_splits(frame_input))

    raise TypeError(
        "workflow must be a TemporalBacktestSplitter, WalkForwardPolicy, TemporalSimulation or compatible object"
    )
