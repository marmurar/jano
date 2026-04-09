from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from .types import SegmentBoundaries


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
