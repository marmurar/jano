from __future__ import annotations

import pytest

from conftest import build_frame
from jano import BatchSimulationResult, SimulationCampaign, SimulationVariant, TemporalPartitionSpec, TemporalSimulation


def _daily_simulation() -> TemporalSimulation:
    return TemporalSimulation(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="8D",
            test_size="2D",
        ),
        step="2D",
        strategy="rolling",
    )


def _weekly_simulation() -> TemporalSimulation:
    return TemporalSimulation(
        time_col="timestamp",
        partition=TemporalPartitionSpec(
            layout="train_test",
            train_size="10D",
            test_size="3D",
        ),
        step="3D",
        strategy="rolling",
    )


def test_simulation_campaign_runs_variants_in_parallel() -> None:
    frame = build_frame(30)
    campaign = SimulationCampaign(
        [
            SimulationVariant(name="daily", simulation=_daily_simulation(), metadata={"clock": "daily"}),
            SimulationVariant(name="weekly", simulation=_weekly_simulation(), title="Weekly sweep"),
        ]
    )

    result = campaign.run(frame, max_workers=2)

    assert isinstance(result, BatchSimulationResult)
    assert list(result.to_frame()["variant"]) == ["daily", "weekly"]
    assert result.result_for("daily").total_folds > 0
    assert result.result_for("weekly").summary.title == "Weekly sweep"

    payload = result.to_dict()
    assert payload["summary"]["variant_count"] == 2
    assert payload["summary"]["total_folds"] >= 2
    assert len(payload["runs"]) == 2


def test_simulation_campaign_runs_sequentially_without_threads() -> None:
    frame = build_frame(30)
    campaign = SimulationCampaign(
        [
            SimulationVariant(name="daily", simulation=_daily_simulation()),
            SimulationVariant(name="weekly", simulation=_weekly_simulation()),
        ]
    )

    result_default = campaign.run(frame)
    result_single_worker = campaign.run(frame, max_workers=1)

    assert list(result_default.to_frame()["variant"]) == ["daily", "weekly"]
    assert list(result_single_worker.to_frame()["variant"]) == ["daily", "weekly"]
    assert result_default.summary.equals(result_single_worker.summary)


def test_simulation_campaign_rejects_duplicate_variant_names() -> None:
    with pytest.raises(ValueError, match="variant names must be unique"):
        SimulationCampaign(
            [
                SimulationVariant(name="duplicate", simulation=_daily_simulation()),
                SimulationVariant(name="duplicate", simulation=_weekly_simulation()),
            ]
        )


def test_simulation_campaign_sync_runs() -> None:
    frame = build_frame(30)
    campaign = SimulationCampaign(
        [
            SimulationVariant(name="daily", simulation=_daily_simulation()),
        ]
    )

    # Trigger len(self.variants) == 1
    result_single = campaign.run(frame)
    assert len(result_single.results) == 1

    campaign_multiple = SimulationCampaign(
        [
            SimulationVariant(name="daily", simulation=_daily_simulation()),
            SimulationVariant(name="weekly", simulation=_weekly_simulation()),
        ]
    )

    # Trigger max_workers = 1
    result_one_worker = campaign_multiple.run(frame, max_workers=1)
    assert len(result_one_worker.results) == 2


def test_simulation_campaign_invalid_arguments() -> None:
    frame = build_frame(30)
    campaign = SimulationCampaign([SimulationVariant(name="daily", simulation=_daily_simulation())])

    with pytest.raises(ValueError, match="max_workers must be greater than zero"):
        campaign.run(frame, max_workers=0)

    with pytest.raises(ValueError, match="requires at least one variant"):
        SimulationCampaign([])


def test_simulation_campaign_result_for_missing_key() -> None:
    frame = build_frame(30)
    campaign = SimulationCampaign([SimulationVariant(name="daily", simulation=_daily_simulation())])
    result = campaign.run(frame)

    with pytest.raises(KeyError, match="Unknown simulation variant: missing"):
        result.result_for("missing")

