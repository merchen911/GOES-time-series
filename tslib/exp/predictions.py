"""Self-contained prediction files: one npz per (run, model) holding the test
predictions, the matching observations, each window's forecast start time and
the metadata needed to re-score without the original run (spec 4.6)."""
from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np

from .granularity import as_ns_int64

PRED_FORMAT = 1


def prediction_path(run_root: str, model_name: str) -> str:
    return os.path.join(run_root, "pred", f"{model_name}.npz")


def _json_default(o):
    """Meta values often come straight from numpy (thresholds, lengths)."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"meta value of type {type(o).__name__} is not JSON serializable")


def save_predictions(path: str, pred: np.ndarray, true: np.ndarray,
                     forecast_start: np.ndarray, meta: dict) -> None:
    """Atomically write `pred`/`true` float32 (N, P, T), `forecast_start` int64
    (N,) ns UTC and `meta` (JSON, plus `format`) to `path`. `forecast_start` may
    be datetime64 of any unit (converted) or integer nanoseconds; other types
    raise TypeError (see `granularity.as_ns_int64`).

    Written to `path + ".tmp"` through a file handle (np.savez would otherwise
    append ".npz" to the name) and renamed into place, so a crash can never
    leave a half-written file; on any failure the tmp file is removed and the
    previous `path`, if any, is kept."""
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    forecast_start = as_ns_int64(forecast_start)
    if pred.ndim != 3:
        raise ValueError(f"pred must be (N, P, T); got shape {pred.shape}")
    if pred.shape != true.shape:
        raise ValueError(f"pred {pred.shape} and true {true.shape} shapes differ")
    if forecast_start.shape != (pred.shape[0],):
        raise ValueError(f"forecast_start {forecast_start.shape} must be "
                         f"({pred.shape[0]},), one per window")
    meta_json = json.dumps({**meta, "format": PRED_FORMAT}, default=_json_default)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            np.savez(f, pred=pred, true=true, forecast_start=forecast_start,
                     meta=meta_json)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_predictions(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return `(pred, true, forecast_start, meta)` as written by
    `save_predictions`; `meta` includes the `format` key."""
    with np.load(path, allow_pickle=False) as z:
        return (z["pred"], z["true"], z["forecast_start"],
                json.loads(str(z["meta"])))
