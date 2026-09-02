import torch
import torch.nn as nn

from .layers.RevIN import RevIN


class Model(nn.Module):
    """RLinear: RevIN + a single linear map over the time axis.

    Paper: "Revisiting Long-term Time Series Forecasting: An Investigation on
    Linear Mapping" (Li et al., 2023, arXiv:2305.10721).

    Ported to this framework's THU-ML-style base-model convention: takes the
    4-arg forecast signature (the zero time-marks / decoder input built by
    tslib.model._call_base are ignored) and returns the prediction only; the
    LightningModule owns the loss. Config fields are mapped to the shared
    exp_parser names — channel->enc_in, drop->dropout, rev->revin — and RevIN
    affine/subtract_last follow the same flags used by patchmixer/xpatch. The
    per-channel `individual` head is opt-in via an optional config attribute
    (default False), matching dlinear's shared-by-default behaviour.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = bool(getattr(configs, "individual", False))

        if self.individual:
            self.Linear = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len)
                for _ in range(self.channels)
            ])
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

        self.dropout = nn.Dropout(configs.dropout)
        self.rev = (RevIN(self.channels, affine=bool(configs.affine),
                          subtract_last=bool(configs.subtract_last))
                    if configs.revin else None)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None):
        # x_enc: [B, seq_len, C] -> pred: [B, pred_len, C]
        x = self.rev(x_enc, 'norm') if self.rev else x_enc
        x = self.dropout(x)
        if self.individual:
            pred = torch.zeros(x.size(0), self.pred_len, self.channels,
                               dtype=x.dtype, device=x.device)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm') if self.rev else pred
        return pred
