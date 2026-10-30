"""Aggregation units for event evaluation (spec 4.3).

`point` judges every (window, lead time) value, `window` judges each test
window by its forecast-horizon maximum, and `day` judges, like `window`, only
the windows the 24-hour cut keeps. All values and thresholds here are in the
stored space (log10 or physical, as the run's transform left them)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

GRANULARITIES = ("point", "window", "day")
DAY_HOURS = 24

_HOUR_NS = 3600 * 10**9


def as_ns_int64(times) -> np.ndarray:
    """Window times as int64 nanoseconds since the epoch (UTC).

    datetime64 arrays of any unit, pandas datetime indexes and series (naive or
    tz-aware) are converted to nanoseconds; integers are taken to be nanoseconds
    already. Anything else, floats included, is a TypeError: under pandas >= 3
    times often come as datetime64[us], and reading those (or a float) as a
    nanosecond count would silently shift every window and break the 24-hour
    cut (spec 4.4)."""
    if not isinstance(times, np.ndarray) and hasattr(times, "dtype"):
        import pandas as pd
        if isinstance(times, (pd.Series, pd.Index)) and pd.api.types.is_datetime64_any_dtype(times.dtype):
            return np.asarray(pd.DatetimeIndex(times).as_unit("ns").asi8, dtype=np.int64)
    a = np.asarray(times)
    if a.dtype.kind == "M":
        return a.astype("datetime64[ns]").view(np.int64)
    if a.dtype.kind in "iu":
        return a.astype(np.int64, copy=False)
    raise TypeError(f"forecast_start must be datetime64 values or integer nanoseconds "
                    f"since the epoch; got dtype {a.dtype}")


def day_indices(forecast_start_ns: np.ndarray, hours: int = DAY_HOURS) -> np.ndarray:
    """Indices of the windows the 24-hour cut keeps, in time order.

    The windows are sorted by forecast start (stable, so equal times keep
    their order); the first is kept, then repeatedly the first window at least
    `hours` after the last kept one. After a gap in the data the next window is
    kept at once. Among selections whose spacing is at least `hours`, this one
    keeps the most windows. Returned indices refer to the input array."""
    if hours <= 0:
        raise ValueError(f"hours must be positive; got {hours}")
    fs = as_ns_int64(forecast_start_ns)
    order = np.argsort(fs, kind="stable")
    times = fs[order]
    gap = int(hours * _HOUR_NS)
    kept = []
    k = 0
    while k < times.size:
        kept.append(k)
        k = int(np.searchsorted(times, times[k] + gap, side="left"))
    return order[np.asarray(kept, dtype=np.intp)]


def event_masks(pred_c: np.ndarray, true_c: np.ndarray, thr: float, granularity: str,
                day_idx: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean (predicted, observed) event masks of one channel's (N, P) values
    for one aggregation unit; an event is a value `>= thr` (inclusive).

    point: (N, P) element-wise; window: (N,) on each row's maximum; day: the
    window judgement on the rows `day_idx` only (from `day_indices`)."""
    if granularity == "point":
        return pred_c >= thr, true_c >= thr
    if granularity == "window":
        return pred_c.max(axis=1) >= thr, true_c.max(axis=1) >= thr
    if granularity == "day":
        if day_idx is None:
            raise ValueError("granularity 'day' needs day_idx (see day_indices)")
        return pred_c[day_idx].max(axis=1) >= thr, true_c[day_idx].max(axis=1) >= thr
    raise ValueError(f"unknown granularity {granularity!r}; expected one of {GRANULARITIES}")
