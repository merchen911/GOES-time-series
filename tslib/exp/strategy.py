from __future__ import annotations

import functools
import importlib.metadata
import os
import subprocess
from pathlib import Path

import numpy as np

import tslib

from tslib.model import build_model
from tslib.exp.evaluate import evaluate
from tslib.exp.lightning_model import ForecastModule, TrainResult
from tslib.exp.metrics import MetricContext
from tslib.exp.predictions import save_predictions


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

    def fit_and_test(self, data_bundle, model_name, ckpt_path,
                     pred_path=None) -> TrainResult:
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
        evaluation = evaluate(pred, true, _metric_ctx(self.config, data_bundle),
                              getattr(self.config, "metrics", None))
        _save_test_predictions(pred_path, pred, true, data_bundle, self.config,
                               model_name, "statistic")
        return TrainResult(model_name=model_name, best_val_loss=float("nan"),
                           evaluation=evaluation, ckpt_path=ckpt_path,
                           strategy="statistic")


def _metric_ctx(config, data_bundle):
    return MetricContext(
        thresholds=getattr(config, "event_threshold", None),
        transform=getattr(config, "transform", "none"),
        target_cols=list(getattr(data_bundle, "target_cols", []) or []),
        forecast_start=getattr(data_bundle, "test_forecast_start", None))


# Run settings a prediction file carries so it can be re-scored on its own.
_META_CONFIG_KEYS = ("run_name", "seq_len", "pred_len", "fold_numb", "transform",
                     "cadence_min", "event_threshold")


@functools.lru_cache(maxsize=1)
def _tslib_version() -> str:
    """The code version written into prediction files: the installed package
    version, or `tslib.__version__` when the package is not installed (a
    checkout run in place, as for the archive dump, which recorded "unknown"),
    plus `+g<commit>` (and `.dirty` with uncommitted changes) when the code
    runs from a git checkout."""
    try:
        version = importlib.metadata.version("tslib")
    except importlib.metadata.PackageNotFoundError:
        version = tslib.__version__
    commit = _git_commit()
    return f"{version}+g{commit}" if commit else version


def _git_commit():
    """Short commit of the checkout this code runs from, or None (not a
    checkout, no git, or git fails)."""
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    git = ["git", "-C", str(root)]
    try:
        head = subprocess.run([*git, "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, timeout=30)
        if head.returncode != 0 or not head.stdout.strip():
            return None
        status = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    dirty = status.returncode == 0 and bool(status.stdout.strip())
    return head.stdout.strip() + (".dirty" if dirty else "")


def _save_test_predictions(pred_path, pred, true, data_bundle, config,
                           model_name, strategy) -> None:
    """Write the test predictions to `pred_path` when one is given and
    `config.save_predictions` is on.

    Never raises: by now the run's evaluation exists, and on the neural paths
    an exception here would leave the checkpoint without its completion marker,
    so the next run would retrain the model (or, on recovery, fall back to a
    retrain). A missing file can be dumped again from the checkpoint."""
    if not pred_path or not getattr(config, "save_predictions", True):
        return
    forecast_start = getattr(data_bundle, "test_forecast_start", None)
    if forecast_start is None:
        print(f"[warn] {model_name}: predictions not saved: the test data "
              f"carries no forecast start times", flush=True)
        return
    try:
        meta = {"model": model_name, "strategy": strategy,
                **{k: getattr(config, k, None) for k in _META_CONFIG_KEYS},
                "target_cols": list(getattr(data_bundle, "target_cols", []) or []),
                "tslib_version": _tslib_version()}
        save_predictions(pred_path, pred, true, forecast_start, meta)
    except Exception as e:  # noqa: BLE001 - the result must survive a failed save
        print(f"[warn] {model_name}: predictions not saved to {pred_path}: "
              f"{e!r}", flush=True)


def _save_and_release(module, pred_path, data_bundle, config, model_name,
                      strategy) -> None:
    """Save the module's test arrays, then drop them: the whole test set can be
    close to a GB per array, and only the evaluation is kept."""
    _save_test_predictions(pred_path, module.test_pred, module.test_true,
                           data_bundle, config, model_name, strategy)
    module.test_pred = module.test_true = None


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
                     ckpt_file, pred_path=None) -> TrainResult:
    """Recompute a model's test evaluation (and, with `pred_path`, its
    prediction file) from an EXISTING trained checkpoint, with no training.
    Used to resume a cell that died after a model finished training (checkpoint
    on disk) but before its result row was logged — so we pay only an inference
    pass, not another full fit."""
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
    _save_and_release(module, pred_path, data_bundle, config, model_name, strategy)
    return TrainResult(model_name=model_name,
                       best_val_loss=_best_val_from_ckpt(ckpt),
                       evaluation=module.evaluation, ckpt_path=ckpt_path,
                       strategy=strategy)


def _run_neural(strategy, model_name, data_bundle, config, ckpt_path,
                pred_path=None) -> TrainResult:
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
        # version=model_name, not Lightning's auto-incrementing version_N:
        # v004 runs several models of one cell concurrently on different GPUs
        # and they share ckpt_dir, so auto-versioning races (two processes pick
        # the same version_N and interleave rows into one metrics.csv). Keying
        # the directory on the model is race-free and makes the training curve
        # attributable to a model instead of to an anonymous counter.
        logger=CSVLogger(save_dir=ckpt_dir, name="lightning",
                         version=model_name),
        enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(module, data_bundle.train_loader, data_bundle.val_loader)

    trainer.test(module, data_bundle.test_loader, ckpt_path="best")
    _save_and_release(module, pred_path, data_bundle, config, model_name, strategy)
    best = (float(ckpt_cb.best_model_score)
            if ckpt_cb.best_model_score is not None else float("nan"))
    return TrainResult(model_name=model_name, best_val_loss=best,
                       evaluation=module.evaluation, ckpt_path=ckpt_path,
                       strategy=strategy)


def run_strategy(strategy, model_name, data_bundle, config, ckpt_path,
                 pred_path: str | None = None) -> TrainResult:
    """Train (or fit) and test one model; with `pred_path` the test predictions
    are also written there (unless `config.save_predictions` is off)."""
    if strategy in ("direct", "recursive"):
        return _run_neural(strategy, model_name, data_bundle, config, ckpt_path,
                           pred_path=pred_path)
    if strategy == "statistic":
        return StatisticalRunner(model_name, config).fit_and_test(
            data_bundle, model_name=model_name, ckpt_path=ckpt_path,
            pred_path=pred_path)
    raise ValueError(f"unknown forecast_strategy '{strategy}'")
