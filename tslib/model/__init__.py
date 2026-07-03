from __future__ import annotations

import importlib

from torch import nn


class StandardForecastAdapter(nn.Module):
    """표준 모델(legacy-style 4-argument forward) 어댑터."""

    def __init__(self, base_model: nn.Module, config, target_indices) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        # list of channel indices to forecast (>=1); supports multi-target output
        self.target_indices = list(target_indices)

    @staticmethod
    def _build_time_mark(batch: int, length: int, device) -> "torch.Tensor":
        import torch

        return torch.zeros((batch, length, 5), dtype=torch.long, device=device)

    def forward(self, x):
        import torch

        b = x.shape[0]
        dev = x.device
        x_mark_enc = self._build_time_mark(b, self.config.seq_len, dev)
        x_dec = torch.zeros((b, self.config.label_len + self.config.pred_len, x.shape[-1]), device=dev)
        x_mark_dec = self._build_time_mark(b, self.config.label_len + self.config.pred_len, dev)

        pred = self.base_model(x, x_mark_enc, x_dec, x_mark_dec)
        # base model emits all enc_in channels; select the target channel(s).
        # -> [B, pred_len, len(target_indices)] to match y = (B, pred_len, T).
        if pred.ndim == 3 and pred.shape[-1] > 1:
            pred = pred[..., self.target_indices]
        return pred


def build_model(model_name: str, config, input_size: int, target_indices):
    from tslib.model.registry import MODEL_REGISTRY
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model '{model_name}'; registered: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_name]
    # legacy-style configs
    config.model = model_name
    config.enc_in = input_size
    config.dec_in = input_size
    config.c_out = input_size
    base_model = spec.ctor(config)
    if spec.adapter == "standard":
        return StandardForecastAdapter(base_model=base_model, config=config,
                                       target_indices=target_indices)
    raise ValueError(f"unknown adapter '{spec.adapter}'")