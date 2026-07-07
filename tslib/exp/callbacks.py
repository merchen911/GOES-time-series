from __future__ import annotations

import statistics
import time

import torch
import pytorch_lightning as pl


def estimate_train_hours(sec_per_batch: float, n_batches: float,
                         epochs: int) -> float:
    """Wall-clock hours to train `epochs` full epochs at `sec_per_batch`."""
    return float(sec_per_batch) * float(n_batches) * float(epochs) / 3600.0


class TimingGateCallback(pl.Callback):
    """Times the first `probe_batches` training batches, estimates full
    training time, and applies the `on_slow` policy if it exceeds the limit.

    median over the probed batches absorbs the first-batch CUDA/cuDNN warmup.
    """

    def __init__(self, config) -> None:
        self.max_train_hours = float(config.max_train_hours)
        self.on_slow = config.on_slow
        self.probe_batches = int(config.probe_batches)
        self._durations: list[float] = []
        self._t0 = None
        self._done = False

    def _sync(self, device) -> None:
        if device is not None and str(device).startswith("cuda"):
            torch.cuda.synchronize()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._done:
            return
        self._sync(pl_module.device)
        self._t0 = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._done or self._t0 is None:
            return
        self._sync(pl_module.device)
        self._durations.append(time.perf_counter() - self._t0)
        if len(self._durations) < self.probe_batches:
            return
        self._done = True
        sec = statistics.median(self._durations)
        est = estimate_train_hours(sec, trainer.num_training_batches,
                                   trainer.max_epochs)
        pl_module._est_train_hours = est
        name = type(getattr(pl_module, "model", pl_module)).__name__
        print(f"[gate] model={name} s/batch={sec:.3f} "
              f"est_train~{est:.2f}h (limit {self.max_train_hours}h)")
        if est <= self.max_train_hours:
            return
        if self.on_slow == "abort":
            raise RuntimeError(
                f"train-time gate: est {est:.2f}h > limit "
                f"{self.max_train_hours}h for {name}")
        if self.on_slow == "skip":
            pl_module._gate_skipped = True
            trainer.should_stop = True
        # "proceed": log-only, continue training
