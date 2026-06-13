from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import build_frame, mae
from jano import (
    DriftBasedRetrain,
    PeriodicRetrain,
    SystemEvaluationResult,
    SystemRunResult,
    SystemUpdateResult,
    TemporalPartitionSpec,
    TemporalSystemRunner,
    WalkForwardPolicy,
)
from jano.splits import TimeSplit
from jano.types import SegmentBoundaries, TemporalSemanticsSpec


class MeanTargetSystem:
    def __init__(self) -> None:
        self.updated_means: list[float] = []

    def update(self, train_frame: pd.DataFrame):
        mean_target = float(train_frame["target"].mean())
        self.updated_means.append(mean_target)
        return SystemUpdateResult(
            state=mean_target,
            metadata={"train_target_mean": mean_target},
        )

    def evaluate(self, state, test_frame: pd.DataFrame):
        predictions = np.repeat(float(state), len(test_frame))
        score = mae(test_frame["target"], predictions)
        return SystemEvaluationResult(
            metrics={"mae": score},
            metadata={"prediction_mean": float(state)},
        )


class MappingSystem:
    def update(self, train_frame: pd.DataFrame):
        return float(train_frame["target"].mean())

    def evaluate(self, state, test_frame: pd.DataFrame):
        predictions = np.repeat(float(state), len(test_frame))
        return {"mae": mae(test_frame["target"], predictions)}


class NoMetricSystem:
    def update(self, train_frame: pd.DataFrame):
        return SystemUpdateResult(state=None)

    def evaluate(self, state, test_frame: pd.DataFrame):
        return SystemEvaluationResult(metrics={})


class InvalidEvaluationSystem:
    def update(self, train_frame: pd.DataFrame):
        return 1.0

    def evaluate(self, state, test_frame: pd.DataFrame):
        return 1.0


def _policy() -> WalkForwardPolicy:
    return WalkForwardPolicy(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size=4,
            test_size=2,
        ),
        step=2,
        strategy="rolling",
    )


def test_temporal_system_runner_updates_every_fold_by_default() -> None:
    frame = build_frame(size=10)
    system = MeanTargetSystem()

    result = TemporalSystemRunner(
        system=system,
        metric_directions={"mae": "min"},
        primary_metric="mae",
    ).run(_policy(), frame)

    assert isinstance(result, SystemRunResult)
    assert result.update_policy == "AlwaysRetrain"
    assert result.to_frame()["updated"].tolist() == [True, True, True]
    assert result.metric_names == ["mae"]
    assert len(system.updated_means) == 3


def test_temporal_system_runner_supports_periodic_updates_and_report_data() -> None:
    frame = build_frame(size=10)
    result = TemporalSystemRunner(
        system=MeanTargetSystem(),
        update="periodic",
        update_interval=2,
        metric_directions={"mae": "min"},
        primary_metric="mae",
    ).run(_policy(), frame)

    assert result.to_frame()["updated"].tolist() == [True, False, True]
    assert result.update_events()["fold"].tolist() == [0, 2]
    assert result.metric_trajectory()["metric"].tolist() == ["mae", "mae", "mae"]
    assert result.summary()["mae_best_fold"] == 0
    assert result.report_data()["evaluations"][0]["update_metadata"]["train_target_mean"] == 101.5


def test_temporal_system_runner_accepts_mapping_style_systems() -> None:
    frame = build_frame(size=10)
    result = TemporalSystemRunner(system=MappingSystem()).run(_policy(), frame)

    assert "mae" in result.to_frame().columns
    assert result.summary()["update_events"] == 3
    assert "evaluations" in result.to_dict()


def test_temporal_system_runner_accepts_explicit_update_policy() -> None:
    frame = build_frame(size=10)
    result = TemporalSystemRunner(
        system=MeanTargetSystem(),
        update_policy=PeriodicRetrain(2),
        metric_directions={"mae": "min"},
    ).run(_policy(), frame)

    assert result.update_policy == "PeriodicRetrain"
    assert result.to_frame()["updated"].tolist() == [True, False, True]


def test_temporal_system_runner_uses_retrain_history_for_drift_based_updates() -> None:
    frame = build_frame(size=10)
    result = TemporalSystemRunner(
        system=MeanTargetSystem(),
        update_policy=DriftBasedRetrain(
            metric="mae",
            threshold=0.5,
            baseline="first",
            relative=False,
        ),
        metric_directions={"mae": "min"},
        primary_metric="mae",
    ).run(_policy(), frame)

    assert result.to_frame()["updated"].tolist() == [True, False, True]


def test_temporal_system_runner_validates_metric_directions_and_primary_metric() -> None:
    frame = build_frame(size=10)
    with pytest.raises(ValueError, match="unknown metrics"):
        TemporalSystemRunner(
            system=MappingSystem(),
            metric_directions={"rmse": "min"},
        ).run(_policy(), frame)

    with pytest.raises(ValueError, match="primary_metric 'rmse'"):
        TemporalSystemRunner(
            system=MappingSystem(),
            primary_metric="rmse",
        ).run(_policy(), frame)

    with pytest.raises(ValueError, match="must be either 'min' or 'max'"):
        TemporalSystemRunner(
            system=MappingSystem(),
            metric_directions={"mae": "down"},
        ).run(_policy(), frame)


def test_temporal_system_runner_supports_no_metric_runs() -> None:
    frame = build_frame(size=10)
    result = TemporalSystemRunner(system=NoMetricSystem()).run(_policy(), frame)

    assert result.metric_names == []
    assert list(result.metric_trajectory().columns) == [
        "fold",
        "metric",
        "value",
        "direction",
        "updated",
    ]
    assert result.report_data()["metrics"] == []


def test_temporal_system_runner_validates_update_mode_arguments() -> None:
    with pytest.raises(ValueError, match="cannot be used together"):
        TemporalSystemRunner(
            system=MappingSystem(),
            update_interval=2,
            update_policy=PeriodicRetrain(2),
        )

    with pytest.raises(ValueError, match="update must be True, False"):
        TemporalSystemRunner(system=MappingSystem(), update="sometimes")

    with pytest.raises(ValueError, match="update_interval is required"):
        TemporalSystemRunner(system=MappingSystem(), update="periodic")

    assert type(TemporalSystemRunner(system=MappingSystem(), update="always").update_policy).__name__ == "AlwaysRetrain"
    assert type(TemporalSystemRunner(system=MappingSystem(), update="never").update_policy).__name__ == "NeverRetrain"
    assert type(TemporalSystemRunner(system=MappingSystem(), update=True, update_interval=2).update_policy).__name__ == "PeriodicRetrain"


def test_temporal_system_runner_can_reuse_state_without_updates() -> None:
    frame = build_frame(size=10)
    result = TemporalSystemRunner(system=MappingSystem(), update=False).run(_policy(), frame)

    assert result.update_policy == "NeverRetrain"
    assert result.to_frame()["updated"].tolist() == [True, False, False]


def test_temporal_system_runner_rejects_invalid_evaluation_outputs() -> None:
    frame = build_frame(size=10)
    with pytest.raises(TypeError, match="must return a mapping of metrics"):
        TemporalSystemRunner(system=InvalidEvaluationSystem()).run(_policy(), frame)


def test_temporal_system_runner_rejects_empty_workflows() -> None:
    frame = build_frame(size=10)

    class EmptyWorkflow:
        def as_splitter(self):
            class _Splitter:
                temporal_semantics = TemporalSemanticsSpec(timeline_col="timestamp")

                def iter_splits(self, X):
                    return iter(())

            return _Splitter()

    with pytest.raises(ValueError, match="did not produce any valid folds"):
        TemporalSystemRunner(system=MappingSystem()).run(EmptyWorkflow(), frame)


def test_temporal_system_runner_requires_train_and_test_segments() -> None:
    frame = build_frame(size=10)

    class MissingTrainTestWorkflow:
        def as_splitter(self):
            class _Splitter:
                temporal_semantics = TemporalSemanticsSpec(timeline_col="timestamp")

                def iter_splits(self, X):
                    yield TimeSplit(
                        fold=0,
                        segments={"validation": np.array([0, 1]), "test": np.array([2, 3])},
                        boundaries={
                            "validation": SegmentBoundaries(
                                pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")
                            ),
                            "test": SegmentBoundaries(
                                pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-05")
                            ),
                        },
                    )

            return _Splitter()

    with pytest.raises(ValueError, match="requires folds with 'train' and 'test' segments"):
        TemporalSystemRunner(system=MappingSystem()).run(MissingTrainTestWorkflow(), frame)
