from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:  # pragma: no cover - exercised when polars is not installed
    pl = None


def coerce_tabular_input(X: Any) -> pd.DataFrame:
    """Normalize supported tabular inputs into a pandas DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X

    if isinstance(X, np.ndarray):
        if X.ndim == 0:
            raise TypeError("NumPy scalar inputs are not supported; provide a tabular array")
        if X.dtype.names is not None:
            return pd.DataFrame.from_records(X)
        return pd.DataFrame(X)

    if pl is not None and isinstance(X, pl.DataFrame):
        return pd.DataFrame(X.to_dict(as_series=False))

    module_name = getattr(type(X), "__module__", "")
    if module_name.startswith("polars"):
        raise ImportError(
            "Polars input support requires the optional 'polars' dependency to be installed"
        )

    raise TypeError(
        "TemporalBacktestSplitter expects a pandas DataFrame, NumPy ndarray or polars DataFrame"
    )
