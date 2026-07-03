from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F

LOSS_REGISTRY: Dict[str, Callable] = {}


def register_loss(name):
    def deco(fn):
        LOSS_REGISTRY[name] = fn
        return fn
    return deco


@register_loss("mse")
def _mse(pred, true):
    return F.mse_loss(pred, true)


@register_loss("mae")
def _mae(pred, true):
    return F.l1_loss(pred, true)


@register_loss("huber")
def _huber(pred, true):
    return F.huber_loss(pred, true)


@register_loss("weighted_mse")
def _weighted_mse(pred, true):
    # Up-weight high-magnitude (rare/energetic) targets. Data is log10-flux, so
    # larger `true` = larger flux; weight grows with positive log10 magnitude.
    # Simple, tunable heuristic (not learned).
    w = 1.0 + true.clamp(min=0.0)
    return (w * (pred - true) ** 2).mean()


def build_loss(config):
    name = getattr(config, "loss", "mse")
    if name not in LOSS_REGISTRY:
        raise ValueError(f"unknown loss '{name}'; registered: {sorted(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name]
