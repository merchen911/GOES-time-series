from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class MetricContext:
    thresholds: Optional[List[float]] = None   # physical units, aligned to target_cols
    transform: str = "none"
    target_cols: List[str] = field(default_factory=list)
    # UTC time of each window's first forecast step: int64 nanoseconds, or
    # datetime64 of any unit (converted); floats are rejected (granularity.as_ns_int64)
    forecast_start: Optional[np.ndarray] = None


def to_stored_space(threshold, transform):
    if transform == "log10":
        return float(np.log10(threshold))
    if transform == "none":
        return float(threshold)
    raise ValueError(f"unknown transform: {transform}")


# ---------------------------------------------------------------------------
# Formula registries. Plain functions of counts / 1-D arrays; the caller
# (evaluate.py) does the spatial transform, the aggregation unit and non-finite
# handling. Every function returns a float and never raises: an undefined
# value is nan (spec 4.2).
# ---------------------------------------------------------------------------

_NAN = float("nan")
_LN10 = math.log(10.0)

# (tp, fp, fn, tn) -> score
EVENT_SCORES: Dict[str, Callable[[int, int, int, int], float]] = {}
# (pred_1d, true_1d) -> value
REGRESSION_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {}


def _registrar(registry, name):
    if "_" in name:
        # the leaderboard reads the token before the first "_" as the metric name
        raise ValueError(f"metric name must not contain '_': {name!r}")

    def deco(fn):
        registry[name] = fn
        return fn
    return deco


def register_event_score(name: str):
    return _registrar(EVENT_SCORES, name)


def register_regression_metric(name: str):
    return _registrar(REGRESSION_METRICS, name)


def metric_kind(name: str) -> str:
    if name in EVENT_SCORES:
        return "event"
    if name in REGRESSION_METRICS:
        return "regression"
    raise KeyError(f"unknown metric: {name!r}")


# ---- event scores: a=tp, b=fp, c=fn, d=tn, H=a/(a+c), F=b/(b+d) ----

@register_event_score("tss")
def _score_tss(tp, fp, fn, tn):
    if tp + fn == 0 or fp + tn == 0:
        return _NAN
    return tp / (tp + fn) - fp / (fp + tn)


@register_event_score("hss")
def _score_hss(tp, fp, fn, tn):
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    if denom == 0:
        return _NAN
    return 2 * (tp * tn - fp * fn) / denom


@register_event_score("pod")
def _score_pod(tp, fp, fn, tn):
    return tp / (tp + fn) if tp + fn else _NAN


@register_event_score("far")
def _score_far(tp, fp, fn, tn):
    return fp / (tp + fp) if tp + fp else _NAN


@register_event_score("csi")
def _score_csi(tp, fp, fn, tn):
    return tp / (tp + fp + fn) if tp + fp + fn else _NAN


@register_event_score("f1")
def _score_f1(tp, fp, fn, tn):
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else _NAN


@register_event_score("seds")
def _score_seds(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    if tp == 0 or tp == n:
        return _NAN
    return (math.log((tp + fp) / n) + math.log((tp + fn) / n)) / math.log(tp / n) - 1.0


@register_event_score("sedi")
def _score_sedi(tp, fp, fn, tn):
    # H or F equal to 0 or 1 (or either undefined) is exactly one of the four
    # counts being 0; with all four positive the denominator is a sum of
    # negative logs, so it cannot vanish.
    if min(tp, fp, fn, tn) == 0:
        return _NAN
    h, nh = tp / (tp + fn), fn / (tp + fn)     # H and 1-H, without cancellation
    f, nf = fp / (fp + tn), tn / (fp + tn)     # F and 1-F
    lh, lnh, lf, lnf = math.log(h), math.log(nh), math.log(f), math.log(nf)
    return (lf - lh - lnf + lnh) / (lf + lh + lnf + lnh)


# ---- regression metrics on 1-D arrays; delta = pred - true (log10 space) ----

def _delta(pred, true):
    return np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)


@register_regression_metric("rmse")
def _reg_rmse(pred, true):
    d = _delta(pred, true)
    return float(np.sqrt(np.mean(d * d))) if d.size else _NAN


@register_regression_metric("mae")
def _reg_mae(pred, true):
    d = _delta(pred, true)
    return float(np.mean(np.abs(d))) if d.size else _NAN


def _percent_factor(m):
    # 100 * (10**m - 1) via expm1: exact for small m, inf (not an error) when huge
    with np.errstate(over="ignore"):
        return 100.0 * float(np.expm1(m * _LN10))


@register_regression_metric("msa")
def _reg_msa(pred, true):
    d = _delta(pred, true)
    return _percent_factor(float(np.median(np.abs(d)))) if d.size else _NAN


@register_regression_metric("sspb")
def _reg_sspb(pred, true):
    d = _delta(pred, true)
    if not d.size:
        return _NAN
    m = float(np.median(d))
    return math.copysign(_percent_factor(abs(m)), m)


@register_regression_metric("logr")
def _reg_logr(pred, true):
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    # a constant series has no correlation; compare min/max rather than a std
    # because the mean of a constant is not always exactly that constant
    if p.size == 0 or p.min() == p.max() or t.min() == t.max():
        return _NAN
    pc, tc = p - p.mean(), t - t.mean()
    return float(np.dot(pc, tc) / (math.sqrt(np.dot(pc, pc)) * math.sqrt(np.dot(tc, tc))))
