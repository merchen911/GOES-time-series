from __future__ import annotations

import numpy as np

from tslib.model import build_model
from tslib.exp.lightning_model import pl_model, TrainResult
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


def run_strategy(strategy, model_name, data_bundle, config, ckpt_path) -> TrainResult:
    if strategy in ("direct", "recursive"):
        model = build_model(model_name, config, data_bundle.input_size,
                            data_bundle.target_indices, strategy=strategy)
        result = pl_model(model, config).fit_and_test(
            data_bundle, model_name=model_name, ckpt_path=ckpt_path)
        result.strategy = strategy
        return result
    if strategy == "statistic":
        return StatisticalRunner(model_name, config).fit_and_test(
            data_bundle, model_name=model_name, ckpt_path=ckpt_path)
    raise ValueError(f"unknown forecast_strategy '{strategy}'")
