from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import MeanRegressor, SimpleLinearRegressor, build_frame
from jano import (
    ClassificationProfile,
    DriftBasedRetrain,
    EvaluationProfile,
    FunctionRetrainPolicy,
    OrdinalClassificationProfile,
    RankingProfile,
    RegressionProfile,
    TemporalPartitionSpec,
    WalkForwardPolicy,
    WalkForwardRunner,
)


def _rolling_policy() -> WalkForwardPolicy:
    return WalkForwardPolicy(
        "timestamp",
        partition=TemporalPartitionSpec(layout="train_test", train_size=4, test_size=2),
        step=2,
        strategy="rolling",
    )


def test_evaluation_profile_drives_runner_metrics_and_primary_metric() -> None:
    frame = build_frame(size=10)

    def fit_score(y_true, y_pred) -> float:
        return float(1 / (1 + np.mean(np.abs(y_true - y_pred))))

    profile = EvaluationProfile(
        metrics={"fit_score": fit_score},
        metric_directions={"fit_score": "max"},
        primary_metric="fit_score",
    )
    result = WalkForwardRunner(
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
        evaluation=profile,
    ).run(_rolling_policy(), frame)

    assert result.primary_metric == "fit_score"
    assert result.metric_directions == {"fit_score": "max"}
    assert result.metric_trajectory()["direction"].unique().tolist() == ["max"]
    assert result.summary()["primary_metric"] == "fit_score"
    assert result.report_data()["primary_metric"] == "fit_score"


def test_drift_based_retrain_can_use_profile_primary_metric() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
            "feature": np.arange(10),
            "target": np.arange(10),
        }
    )

    def business_loss(y_true, y_pred) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    result = WalkForwardRunner(
        model=MeanRegressor(),
        target_col="target",
        feature_cols=["feature"],
        evaluation=EvaluationProfile(
            metrics={"business_loss": business_loss},
            primary_metric="business_loss",
        ),
        retrain_policy=DriftBasedRetrain(threshold=0.5, baseline="last_retrain"),
    ).run(_rolling_policy(), frame)

    assert result.to_frame()["retrained"].tolist() == [True, False, True]


def test_function_retrain_policy_receives_context_for_dynamic_rules() -> None:
    frame = build_frame(size=10)
    observed_contexts = []

    def dynamic_rule(context) -> bool:
        observed_contexts.append(context)
        if context.history.empty:
            return True
        latest = float(context.history[context.primary_metric].iloc[-1])
        return context.fold == 2 and latest >= 0

    result = WalkForwardRunner(
        model=SimpleLinearRegressor(),
        target_col="target",
        feature_cols=["feature"],
        metrics={"loss": lambda y_true, y_pred: float(np.mean(np.abs(y_true - y_pred)))},
        primary_metric="loss",
        retrain_policy=FunctionRetrainPolicy(dynamic_rule),
    ).run(_rolling_policy(), frame)

    assert result.retrain_policy == "FunctionRetrainPolicy"
    assert result.to_frame()["retrained"].tolist() == [True, False, True]
    assert [context.primary_metric for context in observed_contexts] == ["loss", "loss"]


def test_evaluation_profiles_validate_metric_metadata() -> None:
    assert RegressionProfile().resolve().primary_metric == "rmse"
    assert ClassificationProfile().resolve().primary_metric == "accuracy"
    assert (
        OrdinalClassificationProfile(
            {"ordinal_cost": lambda y_true, y_pred: float(np.mean(np.abs(y_true - y_pred)))}
        )
        .resolve()
        .primary_metric
        == "ordinal_cost"
    )
    assert (
        RankingProfile(
            {"ndcg": lambda y_true, y_pred: 1.0},
            metric_directions={"ndcg": "max"},
        )
        .resolve()
        .metric_directions["ndcg"]
        == "max"
    )

    with pytest.raises(ValueError, match="unknown metrics"):
        EvaluationProfile(metrics={"loss": lambda y_true, y_pred: 0.0}, metric_directions={"x": "min"}).resolve()
    with pytest.raises(ValueError, match="either 'min' or 'max'"):
        EvaluationProfile(
            metrics={"loss": lambda y_true, y_pred: 0.0},
            metric_directions={"loss": "lower"},
        ).resolve()
    with pytest.raises(ValueError, match="primary_metric"):
        EvaluationProfile(metrics={"loss": lambda y_true, y_pred: 0.0}, primary_metric="missing").resolve()


def test_runner_rejects_mixed_legacy_and_profile_evaluation_inputs() -> None:
    with pytest.raises(ValueError, match="evaluation cannot be combined"):
        WalkForwardRunner(
            model=SimpleLinearRegressor(),
            target_col="target",
            evaluation=RegressionProfile(),
            metrics="rmse",
        )


def test_drift_based_retrain_requires_metric_or_primary_metric_when_history_exists() -> None:
    context = type(
        "Context",
        (),
        {
            "history": pd.DataFrame({"fold": [0], "retrained": [True], "loss": [1.0]}),
            "primary_metric": None,
        },
    )()

    with pytest.raises(ValueError, match="requires metric or an evaluation primary_metric"):
        DriftBasedRetrain().should_retrain(context)
