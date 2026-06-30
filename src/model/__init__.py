from __future__ import annotations

import importlib

from torch import nn


class StandardForecastAdapter(nn.Module):
    """표준 모델(legacy-style 4-argument forward) 어댑터."""

    def __init__(self, base_model: nn.Module, config, target_index: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.target_index = target_index

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
        if pred.ndim == 3 and pred.shape[-1] > 1:
            pred = pred[..., self.target_index : self.target_index + 1]
        return pred


def build_model(model_name: str, config, input_size: int, target_index: int):
    # legacy-style configs
    config.model = model_name
    config.enc_in = input_size
    config.dec_in = input_size
    config.c_out = input_size

    module = importlib.import_module(f"{__name__}.{model_name}")
    base_model = getattr(module, "Model")(config)
    return StandardForecastAdapter(base_model=base_model, config=config, target_index=target_index)