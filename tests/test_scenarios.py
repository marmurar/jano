from __future__ import annotations

import numpy as np
import pytest

from conftest import MeanRegressor, build_frame, mae
from jano import (
    PredictionBandContext,
    PredictionBandScenarioResult,
    TemporalBacktestSplitter,
    TemporalPartitionSpec,
    TemporalSimulation,
    WalkForwardPolicy,
    estimate_prediction_band_by_fold,
)
from jano.splits import TimeSplit
from jano.types import SegmentBoundaries


class FixedWidthBand:
    def __init__(self, width: float = 1.5) -> None:
        self.width = width

    def estimate(self, context: PredictionBandContext) -> dict[str, object]:
        assert context.X_train.shape[0] >= 1
        assert context.X_test.shape[0] == len(context.predictions)
        lower = context.predictions - self.width
        upper = context.predictions + self.width
        return {
            "lower": lower,
            "upper": upper,
            "fold": {"custom_band_width": float(np.mean(upper - lower))},
            "predictions": {"band_center": context.predictions},
            "artifacts": {"method": "fixed_width"},
        }


def callable_band(context: PredictionBandContext) -> dict[str, object]:
    return {
        "lower": context.predictions - 1.0,
        "upper": context.predictions + 1.0,
    }


def test_prediction_band_scenario_delegates_band_logic_to_user_object() -> None:
    frame = build_frame(size=14)

    result = estimate_prediction_band_by_fold(
        frame,
        estimator=MeanRegressor(),
        band_estimator=FixedWidthBand(width=2.0),
        time_col="timestamp",
        target_col="target",
        feature_cols=["feature"],
        train_size=6,
        test_size=2,
        step=2,
        max_folds=3,
        metrics={"mae": mae},
    )

    assert isinstance(result, PredictionBandScenarioResult)
    assert result.to_frame()["fold"].tolist() == [0, 1, 2]
    assert result.to_frame()["mae"].notna().all()
    assert result.to_frame()["custom_band_width"].tolist() == [4.0, 4.0, 4.0]
    assert (result.to_frame()["prediction_band_width_mean"] == 4.0).all()
    assert len(result.predictions_frame()) == 6
    assert {
        "prediction",
        "prediction_lower",
        "prediction_upper",
        "prediction_band_width",
        "band_center",
    } <= set(result.predictions_frame().columns)
    assert result.artifacts_frame()["artifact"].tolist() == ["method", "method", "method"]
    assert result.summary()["folds"] == 3
    assert result.report_data(include_predictions=True, include_artifacts=True)["summary"]["metrics"] == ["mae"]
    assert result.to_dict(include_predictions=True, include_artifacts=True)["predictions"][0]["fold"] == 0


def test_prediction_band_scenario_accepts_callable_band_estimator() -> None:
    frame = build_frame(size=10)

    result = estimate_prediction_band_by_fold(
        frame,
        estimator=MeanRegressor(),
        band_estimator=callable_band,
        time_col="timestamp",
        target_col="target",
        feature_cols=["feature"],
        train_size=4,
        test_size=2,
        step=2,
    )

    assert len(result.to_frame()) == 3
    assert (result.predictions_frame()["prediction_band_width"] == 2.0).all()


def test_prediction_band_scenario_respects_metric_direction() -> None:
    frame = build_frame(size=12)

    def negative_mae(y_true, y_pred):
        return -mae(y_true, y_pred)

    result = estimate_prediction_band_by_fold(
        frame,
        estimator=MeanRegressor(),
        band_estimator=FixedWidthBand(),
        time_col="timestamp",
        target_col="target",
        feature_cols=["feature"],
        partition=TemporalPartitionSpec(layout="train_test", train_size=5, test_size=2),
        step=2,
        max_folds=2,
        metrics={"neg_mae": negative_mae},
        metric_directions={"neg_mae": "max"},
    )

    summary = result.summary()
    assert summary["neg_mae_best"] == result.to_frame()["neg_mae"].max()
    assert result.report_data()["metric_directions"] == {"neg_mae": "max"}


def test_prediction_band_scenario_accepts_existing_workflows() -> None:
    frame = build_frame(size=10)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
        step=2,
    )
    simulation = TemporalSimulation(
        "timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
        step=2,
        max_folds=2,
    )
    policy = WalkForwardPolicy(
        "timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
        step=2,
        max_folds=2,
    )

    class SimulationWrapper:
        def __init__(self, simulation):
            self.simulation = simulation

    class SplitterWrapper:
        def as_splitter(self):
            return splitter

    for workflow in [splitter, simulation, policy, SimulationWrapper(simulation), SplitterWrapper()]:
        result = estimate_prediction_band_by_fold(
            frame,
            estimator=MeanRegressor(),
            band_estimator=FixedWidthBand(),
            workflow=workflow,
            target_col="target",
            feature_cols=["feature"],
            prediction_column="y_hat",
        )
        assert "y_hat" in result.predictions_frame().columns
        assert len(result.to_frame()) >= 2


def test_prediction_band_scenario_validates_configuration() -> None:
    frame = build_frame(size=8)

    with pytest.raises(ValueError, match="time_col is required"):
        estimate_prediction_band_by_fold(
            frame,
            estimator=MeanRegressor(),
            band_estimator=FixedWidthBand(),
            target_col="target",
        )
    with pytest.raises(ValueError, match="estimator is required"):
        estimate_prediction_band_by_fold(
            frame,
            band_estimator=FixedWidthBand(),
            time_col="timestamp",
            target_col="target",
            train_size=4,
            test_size=2,
            step=2,
        )
    with pytest.raises(ValueError, match="workflow cannot be combined"):
        estimate_prediction_band_by_fold(
            frame,
            estimator=MeanRegressor(),
            band_estimator=FixedWidthBand(),
            workflow=WalkForwardPolicy(
                "timestamp",
                partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
                step=2,
            ),
            time_col="timestamp",
            target_col="target",
        )
    with pytest.raises(ValueError, match="train_size and test_size"):
        estimate_prediction_band_by_fold(
            frame,
            estimator=MeanRegressor(),
            band_estimator=FixedWidthBand(),
            time_col="timestamp",
            target_col="target",
            step=2,
        )
    with pytest.raises(ValueError, match="step is required"):
        estimate_prediction_band_by_fold(
            frame,
            estimator=MeanRegressor(),
            band_estimator=FixedWidthBand(),
            time_col="timestamp",
            target_col="target",
            train_size=4,
            test_size=2,
        )


def test_prediction_band_scenario_validates_band_estimator_output() -> None:
    frame = build_frame(size=8)
    kwargs = {
        "estimator": MeanRegressor(),
        "time_col": "timestamp",
        "target_col": "target",
        "feature_cols": ["feature"],
        "train_size": 4,
        "test_size": 2,
        "step": 2,
    }

    with pytest.raises(TypeError, match="band_estimator must be callable"):
        estimate_prediction_band_by_fold(frame, band_estimator=object(), **kwargs)
    with pytest.raises(TypeError, match="must return a mapping"):
        estimate_prediction_band_by_fold(frame, band_estimator=lambda context: None, **kwargs)
    with pytest.raises(ValueError, match="include lower and upper"):
        estimate_prediction_band_by_fold(frame, band_estimator=lambda context: {}, **kwargs)
    with pytest.raises(ValueError, match="lower must be"):
        estimate_prediction_band_by_fold(
            frame,
            band_estimator=lambda context: {"lower": [1.0], "upper": [2.0]},
            **kwargs,
        )
    with pytest.raises(ValueError, match="lower values"):
        estimate_prediction_band_by_fold(
            frame,
            band_estimator=lambda context: {
                "lower": context.predictions + 1.0,
                "upper": context.predictions - 1.0,
            },
            **kwargs,
        )
    with pytest.raises(TypeError, match="predictions payload"):
        estimate_prediction_band_by_fold(
            frame,
            band_estimator=lambda context: {
                "lower": context.predictions - 1.0,
                "upper": context.predictions + 1.0,
                "predictions": [],
            },
            **kwargs,
        )
    with pytest.raises(TypeError, match="fold and artifacts"):
        estimate_prediction_band_by_fold(
            frame,
            band_estimator=lambda context: {
                "lower": context.predictions - 1.0,
                "upper": context.predictions + 1.0,
                "fold": [],
            },
            **kwargs,
        )


def test_prediction_band_scenario_allows_sparse_optional_payloads() -> None:
    frame = build_frame(size=8)

    def sparse_band(context: PredictionBandContext) -> dict[str, object]:
        return {
            "lower": context.predictions - 1.0,
            "upper": context.predictions + 1.0,
            "fold": None,
            "predictions": None,
            "artifacts": None,
        }

    result = estimate_prediction_band_by_fold(
        frame,
        estimator=MeanRegressor(),
        band_estimator=sparse_band,
        time_col="timestamp",
        target_col="target",
        feature_cols=["feature"],
        train_size=4,
        test_size=2,
        step=2,
    )

    assert result.artifacts_frame().empty
    assert "prediction_lower" in result.predictions_frame()


def test_prediction_band_scenario_rejects_empty_or_invalid_workflow_outputs() -> None:
    frame = build_frame(size=8)
    splitter = TemporalBacktestSplitter(
        time_col="timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
        step=2,
    )

    class EmptySplitter:
        temporal_semantics = splitter.temporal_semantics

        def iter_splits(self, frame):
            return iter(())

    class EmptyWorkflow:
        def as_splitter(self):
            return EmptySplitter()

    class InvalidSplitter:
        temporal_semantics = splitter.temporal_semantics

        def iter_splits(self, frame):
            yield TimeSplit(
                fold=0,
                segments={"validation": np.array([0, 1, 2])},
                boundaries={
                    "validation": SegmentBoundaries(
                        start=frame["timestamp"].min(),
                        end=frame["timestamp"].max(),
                    )
                },
            )

    class InvalidWorkflow:
        def as_splitter(self):
            return InvalidSplitter()

    with pytest.raises(ValueError, match="did not produce"):
        estimate_prediction_band_by_fold(
            frame,
            estimator=MeanRegressor(),
            band_estimator=FixedWidthBand(),
            workflow=EmptyWorkflow(),
            target_col="target",
            feature_cols=["feature"],
        )
    with pytest.raises(ValueError, match="'train' and 'test'"):
        estimate_prediction_band_by_fold(
            frame,
            estimator=MeanRegressor(),
            band_estimator=FixedWidthBand(),
            workflow=InvalidWorkflow(),
            target_col="target",
            feature_cols=["feature"],
        )
    with pytest.raises(TypeError, match="workflow must be"):
        estimate_prediction_band_by_fold(
            frame,
            estimator=MeanRegressor(),
            band_estimator=FixedWidthBand(),
            workflow=object(),
            target_col="target",
            feature_cols=["feature"],
        )
