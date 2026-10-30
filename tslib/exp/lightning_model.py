from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import pytorch_lightning as pl

from tslib.exp.evaluate import Evaluation, evaluate
from tslib.exp.losses import build_loss


@dataclass
class TrainResult:
    model_name: str
    best_val_loss: float
    evaluation: Evaluation
    ckpt_path: str
    strategy: str = "direct"


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
        # set by on_test_epoch_end: the whole test set (N, pred_len, T) and its
        # evaluation; the caller saves the arrays and then drops them
        self.test_pred: Optional[np.ndarray] = None
        self.test_true: Optional[np.ndarray] = None
        self.evaluation: Optional[Evaluation] = None

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
        self.test_pred = np.concatenate(self._test_pred, axis=0)
        self.test_true = np.concatenate(self._test_true, axis=0)
        self._test_pred.clear()
        self._test_true.clear()
        # metrics=None evaluates every registered metric
        self.evaluation = evaluate(self.test_pred, self.test_true, self.metric_ctx,
                                   getattr(self.config, "metrics", None))

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr,
                                weight_decay=self.config.weight_decay)
