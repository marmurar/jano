from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:  # pragma: no cover - exercised when polars is not installed
    pl = None

from .io import coerce_tabular_input
from .types import ColumnRef


@dataclass(frozen=True)
class PartitionEngineMetadata:
    """Execution metadata for the internal partition engine.

    Attributes:
        engine: Internal engine selected to compute temporal boundaries and indices.
        input_backend: Backend detected from the user-provided input.
        converted: Whether the full dataset was converted before planning.
    """

    engine: str
    input_backend: str
    converted: bool

    def to_dict(self) -> dict[str, object]:
        """Return metadata as a serializable dictionary."""
        return {
            "engine": self.engine,
            "input_backend": self.input_backend,
            "converted": self.converted,
        }


class PartitionEngine:
    """Thin internal adapter used by the splitter to avoid unnecessary conversions."""

    def __init__(
        self,
        data: Any,
        *,
        engine: str,
        input_backend: str,
        converted: bool = False,
    ) -> None:
        self.data = data
        self.metadata = PartitionEngineMetadata(
            engine=engine,
            input_backend=input_backend,
            converted=converted,
        )
        self.columns = self._resolve_columns()
        self.total_rows = self._resolve_total_rows()

    @classmethod
    def from_input(cls, X: Any, prefer: str = "auto") -> "PartitionEngine":
        """Select the safest available internal engine for ``X``."""
        if prefer not in {"auto", "pandas", "polars", "numpy"}:
            raise ValueError("engine must be one of 'auto', 'pandas', 'polars' or 'numpy'")

        input_backend = detect_backend(X)
        selected = input_backend if prefer == "auto" else prefer

        if selected == "pandas":
            data = coerce_tabular_input(X)
            return cls(
                data,
                engine="pandas",
                input_backend=input_backend,
                converted=input_backend != "pandas",
            )

        if selected == "polars":
            if pl is None:
                raise ImportError("Polars engine requires the optional 'polars' dependency")
            if input_backend == "polars":
                return cls(X, engine="polars", input_backend=input_backend)
            if input_backend == "pandas":
                return cls(
                    pl.from_pandas(X),
                    engine="polars",
                    input_backend=input_backend,
                    converted=True,
                )
            raise ValueError("Polars engine can only be forced for pandas or polars inputs")

        if selected == "numpy":
            if input_backend == "numpy":
                return cls(X, engine="numpy", input_backend=input_backend)
            if input_backend == "pandas":
                return cls(
                    X.to_numpy(),
                    engine="numpy",
                    input_backend=input_backend,
                    converted=True,
                )
            raise ValueError("NumPy engine can only be forced for pandas or numpy inputs")

        # Unknown tabular-like objects are normalized through the stable pandas path.
        data = coerce_tabular_input(X)
        return cls(
            data,
            engine="pandas",
            input_backend=input_backend,
            converted=input_backend != "pandas",
        )

    @property
    def empty(self) -> bool:
        return self.total_rows == 0

    def column_values(self, ref: ColumnRef) -> np.ndarray:
        """Return one column as a NumPy array without converting the whole dataset."""
        if self.metadata.engine == "pandas":
            return self.data[self._resolve_column_ref(ref)].to_numpy()
        if self.metadata.engine == "polars":
            return self.data[self._resolve_column_ref(ref)].to_numpy()
        if self.metadata.engine == "numpy":
            if self.data.dtype.names is not None:
                return self.data[self._resolve_column_ref(ref)]
            return self.data[:, self._resolve_column_ref(ref)]
        raise RuntimeError(f"Unsupported partition engine '{self.metadata.engine}'")

    def to_pandas(self) -> pd.DataFrame:
        """Materialize the full dataset as pandas for reporting or user-facing slices."""
        return coerce_tabular_input(self.data)

    def _resolve_columns(self) -> list[object]:
        if isinstance(self.data, pd.DataFrame):
            return list(self.data.columns)
        if pl is not None and isinstance(self.data, pl.DataFrame):
            return list(self.data.columns)
        if isinstance(self.data, np.ndarray):
            if self.data.dtype.names is not None:
                return list(self.data.dtype.names)
            if self.data.ndim == 1:
                return [0]
            return list(range(self.data.shape[1]))
        return list(coerce_tabular_input(self.data).columns)

    def _resolve_total_rows(self) -> int:
        if isinstance(self.data, pd.DataFrame):
            return len(self.data)
        if pl is not None and isinstance(self.data, pl.DataFrame):
            return self.data.height
        if isinstance(self.data, np.ndarray):
            if self.data.ndim == 0:
                raise TypeError("NumPy scalar inputs are not supported; provide a tabular array")
            return int(self.data.shape[0])
        return len(coerce_tabular_input(self.data))

    def _resolve_column_ref(self, ref: ColumnRef) -> object:
        if isinstance(ref, int):
            if ref < 0 or ref >= len(self.columns):
                raise ValueError(f"Column position {ref} is out of bounds")
            if self.metadata.engine == "numpy" and self.data.dtype.names is None:
                return ref
            return self.columns[ref]
        if ref not in self.columns:
            raise ValueError(f"Column '{ref}' was not found in the dataset")
        return ref


def detect_backend(X: Any) -> str:
    """Return the input backend name used for engine selection metadata."""
    if isinstance(X, pd.DataFrame):
        return "pandas"
    if isinstance(X, np.ndarray):
        return "numpy"
    if pl is not None and isinstance(X, pl.DataFrame):
        return "polars"
    module_name = getattr(type(X), "__module__", "")
    if module_name.startswith("polars"):
        raise ImportError(
            "Polars input support requires the optional 'polars' dependency to be installed"
        )
    return "pandas"


def missing_columns(columns: Iterable[ColumnRef], available: Iterable[object]) -> list[object]:
    """Return missing named columns while treating integer refs as positional."""
    available_list = list(available)
    missing: list[object] = []
    for column in columns:
        if isinstance(column, int):
            if column < 0 or column >= len(available_list):
                missing.append(column)
        elif column not in available_list:
            missing.append(column)
    return missing
