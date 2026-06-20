from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from ._serialization import _json_ready_object
from .simulation import SimulationResult, TemporalSimulation


@dataclass(frozen=True)
class SimulationVariant:
    """One named simulation hypothesis inside a campaign."""

    name: str
    simulation: TemporalSimulation
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchSimulationResult:
    """Materialized result of running a batch of simulation variants."""

    variants: list[SimulationVariant]
    results: list[SimulationResult]
    summary: pd.DataFrame
    max_workers: int | None = None

    def to_frame(self) -> pd.DataFrame:
        """Return the per-variant comparison table."""
        return self.summary.copy()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready batch comparison payload."""
        return {
            "summary": {
                "variant_count": int(len(self.variants)),
                "total_folds": int(
                    sum(result.total_folds for result in self.results)
                ),
                "max_workers": self.max_workers,
            },
            "variants": [
                _json_ready_object(row)
                for row in self.summary.to_dict(orient="records")
            ],
            "runs": [
                {
                    "variant": variant.name,
                    "title": variant.title or variant.name,
                    "metadata": dict(variant.metadata),
                    "result": result.to_dict(),
                }
                for variant, result in zip(self.variants, self.results)
            ],
        }

    def result_for(self, name: str) -> SimulationResult:
        """Return the simulation result associated with ``name``."""
        for variant, result in zip(self.variants, self.results):
            if variant.name == name:
                return result
        raise KeyError(f"Unknown simulation variant: {name}")


class SimulationCampaign:
    """Run multiple simulation variants over the same dataset."""

    def __init__(self, variants: Sequence[SimulationVariant]) -> None:
        self.variants = list(variants)
        if not self.variants:
            raise ValueError("SimulationCampaign requires at least one variant")
        names = [variant.name for variant in self.variants]
        if len(set(names)) != len(names):
            raise ValueError("SimulationCampaign variant names must be unique")

    def run(
        self,
        frame: pd.DataFrame,
        *,
        max_workers: int | None = None,
    ) -> BatchSimulationResult:
        """Run all variants, optionally in parallel."""
        if max_workers is not None and max_workers <= 0:
            raise ValueError("max_workers must be greater than zero")

        if max_workers is None or max_workers == 1 or len(self.variants) == 1:
            results = [self._run_variant(variant, frame) for variant in self.variants]
        else:
            def run_variant(variant: SimulationVariant) -> SimulationResult:
                return self._run_variant(variant, frame)

            with ThreadPoolExecutor(max_workers=min(max_workers, len(self.variants))) as pool:
                results = list(pool.map(run_variant, self.variants))

        summary = pd.DataFrame(
            [
                self._summarize_variant(variant, result)
                for variant, result in zip(self.variants, results)
            ]
        )
        return BatchSimulationResult(
            variants=self.variants,
            results=results,
            summary=summary,
            max_workers=max_workers,
        )

    def _run_variant(self, variant: SimulationVariant, frame: pd.DataFrame) -> SimulationResult:
        title = variant.title or variant.name
        return variant.simulation.run(frame, title=title)

    def _summarize_variant(
        self,
        variant: SimulationVariant,
        result: SimulationResult,
    ) -> dict[str, object]:
        summary = result.summary.to_dict()
        row: dict[str, object] = {
            "variant": variant.name,
            "title": variant.title or summary["title"],
            "total_folds": int(result.total_folds),
            "total_rows": int(summary["total_rows"]),
            "dataset_start": summary["dataset_start"],
            "dataset_end": summary["dataset_end"],
            "time_col": summary["time_col"],
            "strategy": summary["strategy"],
            "size_kind": summary["size_kind"],
            "segment_order": list(summary["segment_order"]),
            "engine": result.engine_metadata.to_dict(),
        }
        if variant.metadata:
            row["metadata"] = dict(variant.metadata)
        return row
