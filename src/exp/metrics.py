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
