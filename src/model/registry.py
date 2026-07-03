from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class ModelSpec:
    ctor: Callable          # config -> base nn.Module
    adapter: str = "standard"


MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register_model(name, *, adapter="standard"):
    def deco(ctor):
        MODEL_REGISTRY[name] = ModelSpec(ctor=ctor, adapter=adapter)
        return ctor
    return deco


# Central table: register existing THU-ML-style backbones WITHOUT editing their
# files. The ctor lazily imports the module so registering does not import torch
# models until one is actually built.
_LEGACY = ["lstm", "timesnet", "patchtst", "itransformer", "micn",
           "nonstationary_transformer", "patchmixer", "scinet", "segrnn",
           "segrnn_thuml", "timemixer", "etsformer", "xpatch"]


def _legacy_ctor(name):
    return lambda cfg, _n=name: importlib.import_module(f"model.{_n}").Model(cfg)


for _n in _LEGACY:
    register_model(_n)(_legacy_ctor(_n))
