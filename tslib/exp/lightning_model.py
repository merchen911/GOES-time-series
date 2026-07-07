from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import pytorch_lightning as pl

from tslib.exp.losses import build_loss
from tslib.exp.metrics import run_metrics


@dataclass
class TrainResult:
    model_name: str
    best_val_loss: float
    metrics: Dict[str, float]
    ckpt_path: str
    strategy: str = "direct"
    skipped: bool = False
    est_train_hours: float = float("nan")


class ForecastModule(pl.LightningModule):
    """Wraps a forecast adapter/model and drives it under PyTorch Lightning.

    The wrapped model's own train/eval branch (e.g. RecursiveForecastAdapter's
    1-step train vs full rollout eval) is honored automatically because
    Lightning toggles module train/eval mode around the step hooks.
    """

    def __init__(self, model, config, metric_ctx, strategy: str = "direct") -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.metric_ctx = metric_ctx
        self.strategy = strategy
        self.criterion = build_loss(config)
        self._test_pred: List[np.ndarray] = []
        self._test_true: List[np.ndarray] = []
        self.test_metrics: Dict[str, float] = {}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y[:, :pred.shape[1], :])
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y[:, :pred.shape[1], :])
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        self._test_pred.append(pred.detach().cpu().numpy())
        self._test_true.append(y.detach().cpu().numpy())

    def on_test_epoch_end(self):
        pred = np.concatenate(self._test_pred, axis=0)
        true = np.concatenate(self._test_true, axis=0)
        self.test_metrics = run_metrics(pred, true, self.metric_ctx,
                                        self.config.metrics)
        self._test_pred.clear()
        self._test_true.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr,
                                weight_decay=self.config.weight_decay)
