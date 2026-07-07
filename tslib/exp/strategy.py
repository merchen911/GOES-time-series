from __future__ import annotations

import os

import numpy as np

from tslib.model import build_model
from tslib.exp.lightning_model import ForecastModule, TrainResult
from tslib.exp.metrics import MetricContext, run_metrics


class StatisticalRunner:
    def __init__(self, model_name, config):
        from tslib.model.statistical import STAT_REGISTRY
        self.builder = STAT_REGISTRY[model_name]
        self.model_name = model_name
        self.config = config

    def _forecast(self, history):
        steps = self.config.pred_len
        try:
            fc = self.builder(self.config).fit(history).forecast(steps)
            fc = np.asarray(fc, dtype=float)
            if fc.shape != (steps,) or not np.all(np.isfinite(fc)):
                raise ValueError("bad forecast")
            return fc
        except Exception:
            return np.full(steps, history[-1], dtype=float)  # persistence

    def fit_and_test(self, data_bundle, model_name, ckpt_path) -> TrainResult:
        tgt = 0
        ti = getattr(data_bundle, "target_indices", None)
        if ti:
            tgt = ti[0]
        preds, trues = [], []
        for x, y in data_bundle.test_loader:
            xb = np.asarray(x, dtype=float)   # (B, seq_len, C)
            yb = np.asarray(y, dtype=float)   # (B, pred_len, 1)
            for i in range(xb.shape[0]):
                preds.append(self._forecast(xb[i, :, tgt]).reshape(-1, 1))
                trues.append(yb[i])
        pred = np.stack(preds, axis=0)        # (N, pred_len, 1)
        true = np.stack(trues, axis=0)
        ctx = MetricContext(
            thresholds=getattr(self.config, "event_threshold", None),
            transform=getattr(self.config, "transform", "none"),
            target_cols=list(getattr(data_bundle, "target_cols", []) or []))
        metrics = run_metrics(pred, true, ctx, self.config.metrics)
        return TrainResult(model_name=model_name, best_val_loss=float("nan"),
                           metrics=metrics, ckpt_path=ckpt_path,
                           strategy="statistic")


def _run_neural(strategy, model_name, data_bundle, config, ckpt_path) -> TrainResult:
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    model = build_model(model_name, config, data_bundle.input_size,
                        data_bundle.target_indices, strategy=strategy)
    ctx = MetricContext(
        thresholds=getattr(config, "event_threshold", None),
        transform=getattr(config, "transform", "none"),
        target_cols=list(getattr(data_bundle, "target_cols", []) or []))
    module = ForecastModule(model, config, ctx, strategy=strategy)

    ckpt_dir = os.path.dirname(ckpt_path) or "."
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    ckpt_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                              dirpath=ckpt_dir, filename=ckpt_name)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=config.epochs, accelerator=accelerator, devices=1,
        callbacks=[ckpt_cb],
        logger=CSVLogger(save_dir=ckpt_dir, name="lightning"),
        enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(module, data_bundle.train_loader, data_bundle.val_loader)

    trainer.test(module, data_bundle.test_loader, ckpt_path="best")
    best = (float(ckpt_cb.best_model_score)
            if ckpt_cb.best_model_score is not None else float("nan"))
    return TrainResult(model_name=model_name, best_val_loss=best,
                       metrics=module.test_metrics, ckpt_path=ckpt_path,
                       strategy=strategy)


def run_strategy(strategy, model_name, data_bundle, config, ckpt_path) -> TrainResult:
    if strategy in ("direct", "recursive"):
        return _run_neural(strategy, model_name, data_bundle, config, ckpt_path)
    if strategy == "statistic":
        return StatisticalRunner(model_name, config).fit_and_test(
            data_bundle, model_name=model_name, ckpt_path=ckpt_path)
    raise ValueError(f"unknown forecast_strategy '{strategy}'")
