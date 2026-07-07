from __future__ import annotations

import copy

from torch import nn


# THUML's freq -> calendar-feature-count convention (see data_provider/
# timefeatures.py / models/TiDE.py). Floored at 4 so TemporalEmbedding's
# fixed month/day/weekday/hour indices (0-3) are always present regardless
# of freq.
_TIME_FEAT_DIM = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}


def _build_time_mark(batch: int, length: int, device, n_features: int = 4):
    import torch
    # float32 (not long): TemporalEmbedding-style consumers cast internally
    # via `.long()`, but some models (e.g. TiDE) feed the mark straight into
    # an nn.Linear and require a float dtype.
    return torch.zeros((batch, length, n_features), dtype=torch.float32, device=device)


def _call_base(base_model, config, x, dec_len: int):
    """Build zero time-marks + zero decoder input (length label_len + dec_len)
    and call a legacy-style 4-arg base model. Returns its raw output."""
    import torch
    b, dev = x.shape[0], x.device
    n_feat = max(_TIME_FEAT_DIM.get(getattr(config, "freq", "h"), 4), 4)
    x_mark_enc = _build_time_mark(b, config.seq_len, dev, n_feat)
    x_dec = torch.zeros((b, config.label_len + dec_len, x.shape[-1]), device=dev)
    x_mark_dec = _build_time_mark(b, config.label_len + dec_len, dev, n_feat)
    return base_model(x, x_mark_enc, x_dec, x_mark_dec)


class StandardForecastAdapter(nn.Module):
    """Direct many-to-many: base emits full pred_len; select target channel(s)."""

    def __init__(self, base_model: nn.Module, config, target_indices) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.target_indices = list(target_indices)

    def forward(self, x):
        pred = _call_base(self.base_model, self.config, x, self.config.pred_len)
        if pred.ndim == 3 and pred.shape[-1] > 1:
            pred = pred[..., self.target_indices]
        return pred


class RecursiveForecastAdapter(nn.Module):
    """Train: emit one step. Eval: roll the one-step prediction forward
    rollout_len times, feeding the full predicted frame back into the window."""

    def __init__(self, base_model: nn.Module, config, target_indices,
                 rollout_len: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.target_indices = list(target_indices)
        self.rollout_len = int(rollout_len)

    def forward(self, x):
        import torch
        if self.training:
            step = _call_base(self.base_model, self.config, x, 1)  # (B,1,C)
            return step[..., self.target_indices]
        window, outs = x, []
        for _ in range(self.rollout_len):
            step = _call_base(self.base_model, self.config, window, 1)  # (B,1,C)
            outs.append(step)
            window = torch.cat([window[:, 1:, :], step], dim=1)
        pred = torch.cat(outs, dim=1)               # (B, rollout_len, C)
        return pred[..., self.target_indices]        # (B, rollout_len, T)


def build_model(model_name: str, config, input_size: int, target_indices,
                strategy: str = "direct"):
    from tslib.model.registry import MODEL_REGISTRY
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model '{model_name}'; registered: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_name]
    config.model = model_name
    config.enc_in = input_size
    config.dec_in = input_size
    config.c_out = input_size
    if spec.adapter != "standard":
        raise ValueError(f"unknown adapter '{spec.adapter}'")
    if strategy == "recursive":
        base_cfg = copy.copy(config)
        base_cfg.pred_len = 1                      # 1-step head
        base_model = spec.ctor(base_cfg)
        return RecursiveForecastAdapter(base_model, config, target_indices,
                                        rollout_len=config.pred_len)
    base_model = spec.ctor(config)
    return StandardForecastAdapter(base_model=base_model, config=config,
                                   target_indices=target_indices)
