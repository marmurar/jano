from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from .types import FeatureLookbackSpec, SegmentBoundaries


@dataclass(frozen=True)
class TimeSplit:
    """A single temporal partition with named segments and metadata.

    Attributes:
        fold: Zero-based fold number.
        segments: Mapping from segment name to positional NumPy indices.
        boundaries: Mapping from segment name to temporal boundaries.
        metadata: Additional metadata such as strategy or size kind.
    """

    fold: int
    segments: Dict[str, np.ndarray]
    boundaries: Dict[str, SegmentBoundaries]
    metadata: Dict[str, object] = field(default_factory=dict)

    def slice(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Slice a DataFrame into segment-specific DataFrames."""
        return {name: X.iloc[index] for name, index in self.segments.items()}

    def slice_xy(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, pd.DataFrame | pd.Series]:
        """Slice features and target into segment-specific objects."""
        sliced: Dict[str, pd.DataFrame | pd.Series] = {}
        for name, index in self.segments.items():
            sliced[f"X_{name}"] = X.iloc[index]
            sliced[f"y_{name}"] = y.iloc[index]
        return sliced

    def summary(self) -> Dict[str, object]:
        """Return a serializable summary of the fold and its segments."""
        return {
            "fold": self.fold,
            "segments": {
                name: {
                    "start": boundary.start,
                    "end": boundary.end,
                    "rows": int(len(self.segments[name])),
                }
                for name, boundary in self.boundaries.items()
            },
            **self.metadata,
        }

    def feature_history_bounds(
        self,
        lookbacks: FeatureLookbackSpec,
        *,
        segment_name: str = "train",
    ) -> Dict[str, SegmentBoundaries]:
        """Return per-group historical windows needed to build feature groups.

        The returned windows end at the start of ``segment_name`` and extend backward
        according to each configured feature-group lookback.
        """
        if segment_name not in self.boundaries:
            raise ValueError(f"Unknown segment '{segment_name}'")

        anchor = self.boundaries[segment_name].start
        bounds: Dict[str, SegmentBoundaries] = {}
        for group_name, spec in lookbacks.normalized_group_lookbacks().items():
            bounds[group_name] = SegmentBoundaries(start=anchor - spec.value, end=anchor)

        default_spec = lookbacks.normalized_default_lookback()
        if default_spec is not None:
            bounds["__default__"] = SegmentBoundaries(
                start=anchor - default_spec.value,
                end=anchor,
            )
        return bounds

    def slice_feature_history(
        self,
        X: pd.DataFrame,
        lookbacks: FeatureLookbackSpec,
        *,
        time_col: str,
        segment_name: str = "train",
    ) -> Dict[str, pd.DataFrame]:
        """Slice historical context windows needed by feature groups.

        This helper is useful when the fold itself is fixed, but different feature
        groups need different amounts of past data to be engineered.
        """
        timestamps = pd.to_datetime(X[time_col])
        sliced: Dict[str, pd.DataFrame] = {}
        for group_name, boundary in self.feature_history_bounds(
            lookbacks,
            segment_name=segment_name,
        ).items():
            mask = (timestamps >= boundary.start) & (timestamps < boundary.end)
            sliced[group_name] = X.loc[mask].copy()
        return sliced
