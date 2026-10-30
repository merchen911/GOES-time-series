"""Turn test predictions into per-channel regression rows and per-channel x
per-granularity event rows (spec 4.2-4.5). The training path and the
re-scoring tool both call `evaluate`, so a number has one implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .granularity import GRANULARITIES, as_ns_int64, day_indices, event_masks
from .metrics import (
    EVENT_SCORES,
    REGRESSION_METRICS,
    MetricContext,
    metric_kind,
    to_stored_space,
)

_NAN = float("nan")

# rmse/mae keep the v1 convention and use the stored values (physical units
# under transform "none"); every other regression metric works on log10 values.
_STORED_SPACE_METRICS = frozenset({"rmse", "mae"})


@dataclass
class Evaluation:
    regression: List[dict]
    event: List[dict]


def evaluate(pred: np.ndarray, true: np.ndarray, ctx: MetricContext,
             metric_names: Optional[Sequence[str]] = None) -> Evaluation:
    """Evaluate `pred`/`true` (N, P, T) in the stored space described by `ctx`.

    One regression row per channel; one event row per channel and granularity
    when `ctx.thresholds` is set (no `day` rows without `ctx.forecast_start`).
    `metric_names=None` selects every registered metric; the contingency counts
    are always written. Undefined values are nan; nothing raises on data.

    Values are evaluated at float32 precision, the precision of the model output
    and of the prediction files, so a run scored in-line and a re-score of its
    saved file give the same numbers whatever dtype the caller passed."""
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    if pred.ndim != 3 or pred.shape != true.shape:
        raise ValueError(f"pred {pred.shape} and true {true.shape} must both be (N, P, T)")
    n_windows, _, n_channels = pred.shape
    reg_names, event_names = _split_names(metric_names)

    thresholds = [] if ctx.thresholds is None else [float(x) for x in ctx.thresholds]
    if thresholds and len(thresholds) != n_channels:
        raise ValueError(f"{len(thresholds)} event thresholds for {n_channels} target channels")
    day_idx = None
    if thresholds and ctx.forecast_start is not None:
        fs = as_ns_int64(ctx.forecast_start)
        if fs.shape != (n_windows,):
            raise ValueError(f"forecast_start {fs.shape} must be ({n_windows},), one per window")
        day_idx = day_indices(fs)

    regression, event = [], []
    for i in range(n_channels):
        channel = ctx.target_cols[i] if i < len(ctx.target_cols) else f"c{i}"
        p, t = pred[..., i], true[..., i]
        regression.append({"channel": channel,
                           **_regression_values(p, t, ctx.transform, reg_names)})
        if thresholds:
            event.extend(_event_rows(channel, p, t, thresholds[i], ctx.transform,
                                     day_idx, event_names))
    return Evaluation(regression=regression, event=event)


def _split_names(metric_names: Optional[Sequence[str]]) -> Tuple[List[str], List[str]]:
    if metric_names is None:
        return list(REGRESSION_METRICS), list(EVENT_SCORES)
    names = list(dict.fromkeys(metric_names))   # metric_kind raises KeyError on unknown
    return ([n for n in names if metric_kind(n) == "regression"],
            [n for n in names if metric_kind(n) == "event"])


def _log10_values(p: np.ndarray, t: np.ndarray, transform: str):
    """(log10 pred, log10 true), or None when log10 is undefined (a value <= 0)."""
    if transform == "log10":
        return p, t
    if transform == "none":
        if (p <= 0).any() or (t <= 0).any():
            return None
        return np.log10(p), np.log10(t)
    raise ValueError(f"unknown transform: {transform}")


def _regression_values(p: np.ndarray, t: np.ndarray, transform: str,
                       names: List[str]) -> dict:
    out = {"n": int(p.size)}
    if not names:
        return out
    # float64 once per channel; the metric functions would each cast otherwise
    p = np.asarray(p, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    if not np.isfinite(p).all():
        out.update({name: _NAN for name in names})
        return out
    logs = None
    if any(name not in _STORED_SPACE_METRICS for name in names):
        logs = _log10_values(p, t, transform)
    for name in names:
        if name in _STORED_SPACE_METRICS:
            out[name] = float(REGRESSION_METRICS[name](p, t))
        elif logs is None:
            out[name] = _NAN
        else:
            out[name] = float(REGRESSION_METRICS[name](*logs))
    return out


def _event_rows(channel: str, p: np.ndarray, t: np.ndarray, threshold: float,
                transform: str, day_idx: Optional[np.ndarray],
                names: List[str]) -> List[dict]:
    """Rows for one channel. `p`, `t` are (N, P) stored-space values and
    `threshold` is in physical units. One non-finite prediction anywhere in the
    channel makes every count and score of the channel nan (spec 4.2), so a
    unit whose own samples happen to be finite is not scored either;
    `n_nonfinite` counts the unit's own non-finite samples (rows for
    window/day)."""
    # a Python float threshold is compared in the arrays' dtype, float32 (see
    # evaluate), the same comparison v1 made
    thr = to_stored_space(threshold, transform)
    finite = np.isfinite(p)
    row_finite = finite.all(axis=1)
    channel_finite = bool(row_finite.all())

    rows = []
    for g in GRANULARITIES:
        if g == "point":
            n, n_bad = finite.size, finite.size - np.count_nonzero(finite)
        elif g == "window":
            n, n_bad = row_finite.size, row_finite.size - np.count_nonzero(row_finite)
        elif day_idx is None:
            continue
        else:
            n, n_bad = day_idx.size, day_idx.size - np.count_nonzero(row_finite[day_idx])
        row = {"channel": channel, "granularity": g, "threshold": threshold, "n": int(n)}
        if channel_finite:
            pm, tm = event_masks(p, t, thr, g, day_idx)
            tp = int(np.count_nonzero(pm & tm))
            fp = int(np.count_nonzero(pm)) - tp
            fn = int(np.count_nonzero(tm)) - tp
            tn = int(n) - tp - fp - fn
            row.update(n_events=tp + fn, tp=tp, fp=fp, fn=fn, tn=tn, n_nonfinite=int(n_bad))
            row.update({name: float(EVENT_SCORES[name](tp, fp, fn, tn)) for name in names})
        else:
            row.update(n_events=_NAN, tp=_NAN, fp=_NAN, fn=_NAN, tn=_NAN,
                       n_nonfinite=int(n_bad))
            row.update({name: _NAN for name in names})
        rows.append(row)
    return rows
