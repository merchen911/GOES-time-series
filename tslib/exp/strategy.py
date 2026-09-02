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


def _metric_ctx(config, data_bundle):
    return MetricContext(
        thresholds=getattr(config, "event_threshold", None),
        transform=getattr(config, "transform", "none"),
        target_cols=list(getattr(data_bundle, "target_cols", []) or []))


def _best_val_from_ckpt(ckpt: dict) -> float:
    """Recover the monitored best val_loss from a Lightning checkpoint's
    stored callback state (ModelCheckpoint.best_model_score), or NaN."""
    for v in (ckpt.get("callbacks") or {}).values():
        if isinstance(v, dict):
            score = v.get("best_model_score")
            if score is not None:
                try:
                    return float(score)
                except (TypeError, ValueError):
                    pass
    return float("nan")


def test_only_neural(strategy, model_name, data_bundle, config, ckpt_path,
                     ckpt_file) -> TrainResult:
    """Recompute a model's test metrics from an EXISTING trained checkpoint,
    with no training. Used to resume a cell that died after a model finished
    training (checkpoint on disk) but before its result row was logged — so we
    pay only an inference pass, not another full fit."""
    import torch
    import pytorch_lightning as pl

    model = build_model(model_name, config, data_bundle.input_size,
                        data_bundle.target_indices, strategy=strategy)
    module = ForecastModule(model, config, _metric_ctx(config, data_bundle),
                            strategy=strategy)
    ckpt = torch.load(ckpt_file, map_location="cpu")
    module.load_state_dict(ckpt["state_dict"])
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(accelerator=accelerator, devices=1, logger=False,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.test(module, data_bundle.test_loader)
    return TrainResult(model_name=model_name,
                       best_val_loss=_best_val_from_ckpt(ckpt),
                       metrics=module.test_metrics, ckpt_path=ckpt_path,
                       strategy=strategy)


def _run_neural(strategy, model_name, data_bundle, config, ckpt_path) -> TrainResult:
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import CSVLogger

    model = build_model(model_name, config, data_bundle.input_size,
                        data_bundle.target_indices, strategy=strategy)
    module = ForecastModule(model, config, _metric_ctx(config, data_bundle),
                            strategy=strategy)

    n_train_batches = len(data_bundle.train_loader)
    if n_train_batches < 200:
        print(f"[warn] {model_name}: {n_train_batches} train steps/epoch "
              f"(below the 200 floor)")

    ckpt_dir = os.path.dirname(ckpt_path) or "."
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    ckpt_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                              dirpath=ckpt_dir, filename=ckpt_name)
    # min_delta guards against a pathological non-halt: on some folds a smooth
    # linear model (e.g. dlinear) keeps improving val_loss by float-noise
    # amounts (~1e-8/epoch), so with the default min_delta=0 EarlyStopping never
    # fires and training runs to max_epochs (observed: fold2 dlinear stuck at
    # 2696 epochs / ~34h). 1e-6 sits between real improvement and that noise:
    # simulated over recorded trajectories it halts the runaway at ~13 epochs
    # while leaving every normally-finished cell's best val_loss unchanged.
    early_cb = EarlyStopping(monitor="val_loss", mode="min",
                             patience=config.early_stop_patience,
                             min_delta=1e-6)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=config.epochs, accelerator=accelerator, devices=1,
        callbacks=[ckpt_cb, early_cb],
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
