from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class MetricContext:
    thresholds: Optional[List[float]] = None   # physical units, aligned to target_cols
    transform: str = "none"
    target_cols: List[str] = field(default_factory=list)


@dataclass
class Metric:
    fn: Callable
    kind: str          # "regression" | "event"


METRIC_REGISTRY: Dict[str, Metric] = {}


def register_metric(name, kind="regression"):
    def deco(fn):
        METRIC_REGISTRY[name] = Metric(fn=fn, kind=kind)
        return fn
    return deco


@register_metric("mse", "regression")
def _mse(pred, true, ctx):
    return float(np.mean((pred - true) ** 2))


@register_metric("mae", "regression")
def _mae(pred, true, ctx):
    return float(np.mean(np.abs(pred - true)))


@register_metric("rmse", "regression")
def _rmse(pred, true, ctx):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _to_stored_space(threshold, transform):
    if transform == "log10":
        return float(np.log10(threshold))
    if transform == "none":
        return float(threshold)
    raise ValueError(f"unknown transform: {transform}")


def _confusion(pred_ch, true_ch, thr):
    p = pred_ch >= thr
    t = true_ch >= thr
    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t))
    tn = int(np.sum(~p & ~t))
    return tp, fp, fn, tn


def _event_metric(score_fn):
    def fn(pred, true, ctx):
        if not ctx.thresholds:
            raise ValueError("event metric requires ctx.thresholds (physical units)")
        out = {}
        n_targets = pred.shape[-1]
        for i in range(n_targets):
            thr = _to_stored_space(ctx.thresholds[i], ctx.transform)
            tp, fp, fn_, tn = _confusion(pred[..., i], true[..., i], thr)
            name = ctx.target_cols[i] if i < len(ctx.target_cols) else f"c{i}"
            out[name] = score_fn(tp, fp, fn_, tn)
        return out
    return fn


def _tss(tp, fp, fn, tn):
    pod = tp / (tp + fn) if (tp + fn) else 0.0
    pofd = fp / (fp + tn) if (fp + tn) else 0.0
    return pod - pofd


def _hss(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    exp = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / n
    denom = n - exp
    return (tp + tn - exp) / denom if denom else 0.0


def _pod(tp, fp, fn, tn):
    return tp / (tp + fn) if (tp + fn) else 0.0


def _far(tp, fp, fn, tn):
    return fp / (tp + fp) if (tp + fp) else 0.0


register_metric("tss", "event")(_event_metric(_tss))
register_metric("hss", "event")(_event_metric(_hss))
register_metric("pod", "event")(_event_metric(_pod))
register_metric("far", "event")(_event_metric(_far))
