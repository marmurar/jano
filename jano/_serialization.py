from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def _frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {str(key): _json_ready(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _json_ready_object(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready_object(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready_object(item) for item in value]
    return _json_ready(value)


def _json_ready(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value
