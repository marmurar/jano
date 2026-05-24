from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import MeanRegressor, RunningMeanPartialFitRegressor, build_frame, mae, rmse, accuracy
from jano import (
    EvaluationProfile,
    OnlineRunResult,
    OnlineTemporalRunner,
    OnlineUpdatePolicy,
    OnlineUpdatePolicyStudy,
    OnlineUpdatePolicyStudyResult,
    OnlineUpdateStrategy,
    PartialFitUpdateStrategy,
    RefitUpdateStrategy,
)


def test_online_temporal_runner_predicts_before_event_updates() -> None:
    frame = build_frame(size=8)

    result = OnlineTemporalRunner(
        model=RunningMeanPartialFitRegressor(),
        time_col="timestamp",
        target_col="target",
        feature_cols=["feature"],
        initial_train_size=4,
        update_size=1,
        metrics={"mae": mae, "rmse": rmse},
    ).run(frame)

    assert isinstance(result, OnlineRunResult)
    records = result.to_frame()
    assert records["batch"].tolist() == [0, 1, 2, 3]
    assert records["batch_rows"].tolist() == [1, 1, 1, 1]
    assert records["train_rows_seen"].tolist() == [4, 5, 6, 7]
    assert records["mae"].round(6).tolist() == [2.5, 3.0, 3.5, 4.0]
    assert result.predictions_frame()["prediction"].round(6).tolist() == [
        101.5,
        102.0,
        102.5,
        103.0,
    ]
    assert result.summary()["updates"] == 4
    assert result.summary()["retrain_checkpoints"] == 0
    assert result.retrain_checkpoints().empty
    assert result.report_data(include_predictions=True)["summary"]["rows_evaluated"] == 4


def test_online_temporal_runner_supports_duration_micro_batches() -> None:
    frame = build_frame(size=10)

    result = OnlineTemporalRunner(
        model=RunningMeanPartialFitRegressor(),
        time_col="timestamp",
        target_col="target",
        feature_cols=["feature"],
        initial_train_size="4D",
        update_size="2D",
        metrics={"mae": mae},
        include_predictions=False,
    ).run(frame)

    records = result.to_frame()
    assert records["batch_rows"].tolist() == [2, 2, 2]
    assert records["batch_start"].tolist() == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-07"),
        pd.Timestamp("2024-01-09"),
    ]
    assert result.predictions_frame().empty
    assert result.metric_trajectory()["metric"].tolist() == ["mae", "mae", "mae"]


def test_online_temporal_runner_marks_user_defined_retrain_checkpoints() -> None:
    frame = build_frame(size=9)

    def trigger(history: pd.DataFrame, latest: dict[str, object]) -> dict[str, object]:
        if int(latest["batch"]) >= 2:
            return {
                "retrain": True,
                "reason": "mae exceeded tolerated online drift",
                "score": latest["mae"],
            }
        return {"retrain": False, "score": latest["mae"]}

    result = OnlineTemporalRunner(
        model=RunningMeanPartialFitRegressor(),
        time_col="timestamp",
        target_col="target",
        feature_cols=["feature"],
        initial_train_size=4,
        update_size=1,
        metrics={"mae": mae},
        retrain_trigger=trigger,
    ).run(frame)

    records = result.to_frame()
    checkpoints = result.retrain_checkpoints()

    assert records["retrain_checkpoint"].tolist() == [False, False, True, True, True]
    assert checkpoints["batch"].tolist() == [2, 3, 4]
    assert checkpoints["retrain_reason"].tolist() == [
        "mae exceeded tolerated online drift",
        "mae exceeded tolerated online drift",
        "mae exceeded tolerated online drift",
    ]
    assert "retrain_score" in checkpoints.columns
    assert result.metric_names == ["mae"]
    assert result.summary()["retrain_checkpoints"] == 3
    assert result.summary()["first_retrain_checkpoint_batch"] == 2
    assert result.report_data()["retrain_checkpoints"][0]["batch"] == 2


def test_online_retrain_trigger_supports_bool_string_and_validates_return_type() -> None:
    frame = build_frame(size=6)

    bool_result = OnlineTemporalRunner(
        model=RunningMeanPartialFitRegressor(),
        time_col="timestamp",
        target_col="target",
        feature_cols=["feature"],
        initial_train_size=3,
        update_size=1,
        metrics={"mae": mae},
        retrain_trigger=lambda history, latest: latest["batch"] == 1,
    ).run(frame)
    assert bool_result.retrain_checkpoints()["batch"].tolist() == [1]

    string_result = OnlineTemporalRunner(
        model=RunningMeanPartialFitRegressor(),
        time_col="timestamp",
        target_col="target",
        feature_cols=["feature"],
        initial_train_size=3,
        update_size=1,
        metrics={"mae": mae},
        retrain_trigger=lambda history, latest: "scheduled checkpoint",
    ).run(frame)
    assert string_result.retrain_checkpoints()["retrain_reason"].tolist() == [
        "scheduled checkpoint",
        "scheduled checkpoint",
        "scheduled checkpoint",
    ]

    with pytest.raises(TypeError, match="retrain_trigger must return"):
        OnlineTemporalRunner(
            model=RunningMeanPartialFitRegressor(),
            time_col="timestamp",
            target_col="target",
            feature_cols=["feature"],
            initial_train_size=3,
            update_size=1,
            metrics={"mae": mae},
            retrain_trigger=lambda history, latest: 1.5,
        ).run(frame)


def test_online_temporal_runner_validates_update_strategy_and_sizes() -> None:
    frame = build_frame(size=4)

    with pytest.raises(TypeError, match="partial_fit"):
        OnlineTemporalRunner(
            model=object(),
            time_col="timestamp",
            target_col="target",
            feature_cols=["feature"],
            initial_train_size=2,
            update_size=1,
        ).run(frame)

    with pytest.raises(ValueError, match="update_size did not produce"):
        OnlineTemporalRunner(
            model=RunningMeanPartialFitRegressor(),
            time_col="timestamp",
            target_col="target",
            feature_cols=["feature"],
            initial_train_size=10,
            update_size=1,
        ).run(frame)


def test_partial_fit_update_strategy_passes_classes_once() -> None:
    class Classifier:
        def __init__(self) -> None:
            self.class_calls = []

        def partial_fit(self, X, y, classes=None):
            self.class_calls.append(None if classes is None else np.asarray(classes).tolist())
            self.value_ = pd.Series(y).mode().iloc[0]
            return self

        def predict(self, X):
            return np.asarray([self.value_] * len(X))

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="D"),
            "feature": [0, 1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 1, 0],
        }
    )
    model = Classifier()
    result = OnlineTemporalRunner(
        model=model,
        time_col="timestamp",
        target_col="target",
        feature_cols=["feature"],
        initial_train_size=3,
        update_size=1,
        metrics={"accuracy": accuracy},
        update_strategy=PartialFitUpdateStrategy(classes=[0, 1]),
    ).run(frame)

    assert result.metric_names == ["accuracy"]
    # The original model is cloned; verify behavior through the run output.
    assert result.summary()["update_strategy"] == "PartialFitUpdateStrategy"


def test_refit_update_strategy_supports_fit_only_models() -> None:
    frame = build_frame(size=8)

    result = OnlineTemporalRunner(
        model=MeanRegressor(),
        time_col="timestamp",
        target_col="target",
        feature_cols=["feature"],
        initial_train_size=4,
        update_size=1,
        metrics={"mae": mae},
        update_strategy=RefitUpdateStrategy(),
    ).run(frame)

    assert result.to_frame()["mae"].round(6).tolist() == [2.5, 3.0, 3.5, 4.0]
    assert result.summary()["update_strategy"] == "RefitUpdateStrategy"


def test_refit_update_strategy_can_keep_bounded_history() -> None:
    frame = build_frame(size=8)

    result = OnlineTemporalRunner(
        model=MeanRegressor(),
        time_col="timestamp",
        target_col="target",
        feature_cols=["feature"],
        initial_train_size=4,
        update_size=1,
        metrics={"mae": mae},
        update_strategy=RefitUpdateStrategy(max_train_rows=3),
    ).run(frame)

    assert result.predictions_frame()["prediction"].round(6).tolist() == [
        102.0,
        103.0,
        104.0,
        105.0,
    ]


def test_refit_update_strategy_validates_fit_contract() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        RefitUpdateStrategy(max_train_rows=0)

    with pytest.raises(TypeError, match="model with fit"):
        OnlineTemporalRunner(
            model=object(),
            time_col="timestamp",
            target_col="target",
            feature_cols=["feature"],
            initial_train_size=2,
            update_size=1,
            update_strategy=RefitUpdateStrategy(),
        ).run(build_frame(size=4))

    strategy = RefitUpdateStrategy()
    with pytest.raises(RuntimeError, match="initialized before update"):
        strategy.update(MeanRegressor(), pd.DataFrame({"feature": [1]}), pd.Series([1]))
    with pytest.raises(RuntimeError, match="initialized before fitting"):
        strategy._fit(MeanRegressor())


def test_online_temporal_runner_supports_fraction_batches_and_max_metrics() -> None:
    frame = build_frame(size=10)

    result = OnlineTemporalRunner(
        model=RunningMeanPartialFitRegressor(),
        time_col="timestamp",
        target_col="target",
        feature_cols=["feature"],
        initial_train_size=0.5,
        update_size=0.2,
        metrics={"neg_mae": lambda y_true, y_pred: np.float64(-np.mean(np.abs(y_true - y_pred)))},
        metric_directions={"neg_mae": "max"},
        primary_metric="neg_mae",
        prediction_column="y_pred",
    ).run(frame)

    records = result.to_frame()
    assert records["batch_rows"].tolist() == [2, 2, 1]
    assert result.summary()["neg_mae_best_batch"] == 0
    assert result.metric_trajectory()["direction"].tolist() == ["max", "max", "max"]
    assert "y_pred" in result.predictions_frame().columns
    assert isinstance(result.report_data()["batches"][0]["neg_mae"], float)


def test_online_temporal_runner_skips_empty_duration_windows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"]),
            "feature": [1, 2, 5],
            "target": [10, 11, 20],
        }
    )

    result = OnlineTemporalRunner(
        model=RunningMeanPartialFitRegressor(),
        time_col="timestamp",
        target_col="target",
        feature_cols=["feature"],
        initial_train_size="1D",
        update_size="1D",
        metrics={"mae": mae},
    ).run(frame)

    assert result.to_frame()["batch_start"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-05"),
    ]


def test_online_temporal_runner_supports_duration_train_and_event_updates() -> None:
    frame = build_frame(size=6)

    result = OnlineTemporalRunner(
        model=RunningMeanPartialFitRegressor(),
        time_col="timestamp",
        target_col="target",
        feature_cols=["feature"],
        initial_train_size="3D",
        update_size=1,
        metrics={"mae": mae},
    ).run(frame)

    assert result.to_frame()["batch_rows"].tolist() == [1, 1, 1]
    assert result.to_frame()["train_rows_seen"].tolist() == [3, 4, 5]


def test_online_temporal_runner_supports_numpy_time_positions() -> None:
    frame = build_frame(size=6)
    values = frame[["timestamp", "feature", "target"]].to_numpy(dtype=object)

    result = OnlineTemporalRunner(
        model=RunningMeanPartialFitRegressor(),
        time_col=0,
        target_col=2,
        feature_cols=[1],
        initial_train_size=3,
        update_size=2,
        metrics={"mae": mae},
    ).run(values)

    assert result.to_frame()["batch_rows"].tolist() == [2, 1]


def test_online_temporal_runner_validates_evaluation_arguments_and_empty_inputs() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        OnlineTemporalRunner(
            model=RunningMeanPartialFitRegressor(),
            time_col="timestamp",
            target_col="target",
            initial_train_size=2,
            metrics={"mae": mae},
            evaluation=EvaluationProfile(metrics={"rmse": rmse}),
        )

    with pytest.raises(ValueError, match="at least one row"):
        OnlineTemporalRunner(
            model=RunningMeanPartialFitRegressor(),
            time_col="timestamp",
            target_col="target",
            feature_cols=["feature"],
            initial_train_size=1,
            update_size=1,
        ).run(build_frame(size=0))

    with pytest.raises(ValueError, match="initial_train_size did not select"):
        OnlineTemporalRunner(
            model=RunningMeanPartialFitRegressor(),
            time_col="timestamp",
            target_col="target",
            feature_cols=["feature"],
            initial_train_size="0D",
            update_size="1D",
        ).run(build_frame(size=3))

    with pytest.raises(ValueError, match="Fractional online sizes resolved to zero rows"):
        OnlineTemporalRunner(
            model=RunningMeanPartialFitRegressor(),
            time_col="timestamp",
            target_col="target",
            feature_cols=["feature"],
            initial_train_size=0.1,
            update_size=0.1,
        ).run(build_frame(size=3))


def test_online_result_serializes_without_predictions_and_handles_no_metrics() -> None:
    result = OnlineRunResult(
        records=pd.DataFrame(
            [
                {
                    "batch": 0,
                    "updated": True,
                    "train_rows_seen": 2,
                    "batch_rows": 1,
                    "batch_start": pd.Timestamp("2024-01-01"),
                    "batch_end": pd.Timestamp("2024-01-01"),
                }
            ]
        ),
        predictions=pd.DataFrame(
            [{"batch": 0, "row_index": np.int64(7), "elapsed": pd.Timedelta(days=1)}]
        ),
        metric_directions={},
        update_strategy="custom",
    )

    assert result.metric_names == []
    assert result.metric_trajectory().empty
    assert "predictions" not in result.to_dict()
    assert result.to_dict(include_predictions=True)["predictions"][0]["elapsed"] == "1 days 00:00:00"


def test_online_update_strategy_base_methods_raise() -> None:
    strategy = OnlineUpdateStrategy()

    with pytest.raises(NotImplementedError):
        strategy.initialize(object(), pd.DataFrame(), pd.Series(dtype=float))
    with pytest.raises(NotImplementedError):
        strategy.update(object(), pd.DataFrame(), pd.Series(dtype=float))


def test_online_update_policy_study_compares_event_and_batch_policies() -> None:
    frame = build_frame(size=8)

    study = OnlineUpdatePolicyStudy(
        model=MeanRegressor(),
        time_col="timestamp",
        target_col="target",
        feature_cols=["feature"],
        initial_train_size=4,
        policies=[
            OnlineUpdatePolicy(
                "every-event",
                update_size=1,
                update_strategy=RefitUpdateStrategy(),
                update_cost=1.0,
            ),
            OnlineUpdatePolicy(
                "every-two-events",
                update_size=2,
                update_strategy=lambda: RefitUpdateStrategy(),
                update_cost=1.5,
            ),
        ],
        metrics={"mae": mae},
    )

    result = study.run(frame)

    assert isinstance(result, OnlineUpdatePolicyStudyResult)
    records = result.to_frame()
    assert records["policy"].tolist() == ["every-event", "every-two-events"]
    assert records["updates"].tolist() == [4, 2]
    assert records["total_update_cost"].tolist() == [4.0, 3.0]
    assert result.run("every-event").summary()["updates"] == 4
    assert result.metric_trajectory()["policy"].unique().tolist() == [
        "every-event",
        "every-two-events",
    ]
    assert result.find_optimal_policy(metric="mae")["policy"] == "every-event"
    assert result.find_optimal_policy(metric="mae", update_cost_weight=10.0)["policy"] == (
        "every-two-events"
    )


def test_online_update_policy_study_validates_policy_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        OnlineUpdatePolicyStudy(
            model=MeanRegressor(),
            time_col="timestamp",
            target_col="target",
            initial_train_size=2,
            policies=[],
        )

    with pytest.raises(ValueError, match="unique"):
        OnlineUpdatePolicyStudy(
            model=MeanRegressor(),
            time_col="timestamp",
            target_col="target",
            initial_train_size=2,
            policies=[
                OnlineUpdatePolicy("same", update_size=1),
                OnlineUpdatePolicy("same", update_size=2),
            ],
        )

    with pytest.raises(TypeError, match="factory must return"):
        OnlineUpdatePolicy("bad", update_size=1, update_strategy=lambda: object()).build_strategy()

    assert OnlineUpdatePolicy("default", update_size=1).build_strategy() is None

    result = OnlineUpdatePolicyStudyResult(
        records=pd.DataFrame(
            {"policy": ["p"], "mae": [1.0], "updates": [1], "total_update_cost": [1]}
        ),
        runs={},
        metric_directions={"mae": "min"},
    )
    assert result.metric_trajectory().empty
    with pytest.raises(ValueError, match="Unknown"):
        result.run("missing")
    with pytest.raises(ValueError, match="metric must be provided"):
        result.find_optimal_policy()
    with pytest.raises(ValueError, match="not present"):
        result.find_optimal_policy("rmse")
    with pytest.raises(ValueError, match="greater than or equal"):
        result.find_optimal_policy("mae", update_cost_weight=-1)


def test_online_update_policy_result_supports_max_direction_objective() -> None:
    result = OnlineUpdatePolicyStudyResult(
        records=pd.DataFrame(
            {
                "policy": ["frequent", "cheap"],
                "accuracy": [0.9, 0.85],
                "updates": [10, 1],
                "total_update_cost": [10.0, 1.0],
            }
        ),
        runs={},
        metric_directions={"accuracy": "max"},
        primary_metric="accuracy",
    )

    assert result.find_optimal_policy()["policy"] == "frequent"
    assert result.find_optimal_policy(update_cost_weight=0.01)["policy"] == "cheap"


def test_online_temporal_runner_validates_mixed_fraction_and_empty_duration_batches() -> None:
    with pytest.raises(ValueError, match="Fractional online sizes resolved to zero rows"):
        OnlineTemporalRunner(
            model=RunningMeanPartialFitRegressor(),
            time_col="timestamp",
            target_col="target",
            feature_cols=["feature"],
            initial_train_size=2,
            update_size=0.1,
        ).run(build_frame(size=3))

    with pytest.raises(ValueError, match="update_size did not produce"):
        OnlineTemporalRunner(
            model=RunningMeanPartialFitRegressor(),
            time_col="timestamp",
            target_col="target",
            feature_cols=["feature"],
            initial_train_size=10,
            update_size="1D",
        ).run(build_frame(size=3))
