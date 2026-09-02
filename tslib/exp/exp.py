from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd
import torch

from tslib.data.loader import DataModule
from .strategy import run_strategy, test_only_neural


def _result_row(r) -> dict:
    return {"strategy": r.strategy, "model": r.model_name,
            "best_val_loss": r.best_val_loss,
            "skipped": r.skipped, "est_train_hours": r.est_train_hours,
            **r.metrics}


# Skill scores where a HIGHER value is a better model (sort descending so the
# best sits on top). Everything else — best_val_loss, rmse, mae, mse, far — is
# an error/false-alarm measure where LOWER is better (sort ascending).
_HIGHER_IS_BETTER = {"tss", "hss", "pod", "csi", "gss", "ets",
                     "acc", "f1", "precision", "recall", "auc"}


def _higher_is_better(metric: str) -> bool:
    # metric may be bare ("tss") or per-target ("tss_p_gt10") — key on the
    # leading family token before the first underscore.
    return metric.split("_", 1)[0] in _HIGHER_IS_BETTER


def _sort_frame(df, sort_metric):
    """Sort so the best model is always on top: descending for higher-is-better
    skill scores, ascending otherwise. Failed/NaN rows stay at the bottom."""
    key = sort_metric if sort_metric in df.columns else "best_val_loss"
    return df.sort_values(by=key, ascending=not _higher_is_better(key),
                          na_position="last").reset_index(drop=True)


def _rows_to_frame(rows, sort_metric):
    return _sort_frame(pd.DataFrame(rows), sort_metric)


def build_comparison(results, sort_metric):
    return _rows_to_frame([_result_row(r) for r in results], sort_metric)


def _row_path(dirs, model_name: str) -> str:
    return os.path.join(dirs["score"], "rows", f"{model_name}.json")


def _write_row(path: str, row: dict) -> None:
    """Atomically persist one model's result row (write-tmp-then-rename), so a
    crash can never leave a half-written row file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(row, f)
    os.replace(tmp, path)


def _write_csv_atomic(df, path: str) -> None:
    """Write a comparison table atomically (tmp + rename). The monitor and the
    benchmark driver (rebuild_master / cell_is_complete) read comparison.csv
    concurrently with this process; a plain in-place to_csv can be observed
    half-written (truncated row / missing columns), so mirror _write_row's
    tmp-then-replace discipline here too."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _done_marker(ckpt_dir: str, model_name: str) -> str:
    return os.path.join(ckpt_dir, f"{model_name}.done")


def _touch(path: str) -> None:
    """Create an empty marker file; its mere presence is the signal."""
    with open(path, "w"):
        pass


def _clear_ckpts(ckpt_dir: str, model_name: str) -> None:
    """Remove a model's checkpoint(s) and completion marker before a fresh
    (re)train: the canonical '{name}.ckpt', Lightning's versioned
    '{name}-vN.ckpt' siblings, and '{name}.done'. This guarantees the new best
    lands on the canonical name (Lightning otherwise versions around an existing
    file) and that no stale orphan can later be mistaken for a finished model.

    Matches only '{name}.ckpt' and '{name}-v*.ckpt' — NOT a bare '{name}*'
    glob, which would wrongly sweep e.g. 'segrnn_thuml.ckpt' when clearing
    'segrnn'."""
    import glob
    targets = [os.path.join(ckpt_dir, f"{model_name}.ckpt"),
               _done_marker(ckpt_dir, model_name)]
    targets += glob.glob(os.path.join(ckpt_dir, f"{model_name}-v*.ckpt"))
    for p in targets:
        if os.path.exists(p):
            os.remove(p)


def _train_fresh(strategy, model_name, data_bundle, config, dirs, neural):
    """(Re)train a model from scratch. For neural strategies, first clear any
    stale/orphan checkpoint so Lightning writes to the canonical name, then drop
    a completion marker once run_strategy returns so a later resume can tell the
    training finished cleanly (vs. a mid-training crash)."""
    ckpt_dir = dirs["ckpt"]
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pt")
    if neural:
        _clear_ckpts(ckpt_dir, model_name)
    result = run_strategy(strategy, model_name, data_bundle, config, ckpt_path)
    if neural:
        _touch(_done_marker(ckpt_dir, model_name))
    return result


def merge_comparisons(frames, sort_metric="best_val_loss"):
    return _sort_frame(pd.concat(list(frames), ignore_index=True), sort_metric)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_dir_name(config, makedir: bool = False):
    run_root = os.path.join("runs", config.run_name)
    dir_types = {
        "root": run_root,
        "ckpt": os.path.join(run_root, "ckpt"),
        "score": os.path.join(run_root, "score"),
    }
    if makedir:
        for p in dir_types.values():
            os.makedirs(p, exist_ok=True)
    return dir_types


def run_experiment(config):
    _set_seed(config.seed)
    dirs = save_dir_name(config, makedir=True)

    data_module = DataModule(config)
    data_bundle = data_module.setup()

    strategy = getattr(config, "forecast_strategy", "direct")
    neural = strategy in ("direct", "recursive")
    comparison_path = os.path.join(dirs["score"], "comparison.csv")
    n = len(config.models)
    rows = []
    for i, model_name in enumerate(config.models, 1):
        row_path = _row_path(dirs, model_name)
        ckpt_path = os.path.join(dirs["ckpt"], f"{model_name}.pt")
        ckpt_file = os.path.join(dirs["ckpt"], f"{model_name}.ckpt")
        done_file = _done_marker(dirs["ckpt"], model_name)
        if os.path.exists(row_path):
            # Result already logged on a prior (possibly interrupted) run —
            # skip entirely, no reload/retrain.
            with open(row_path) as f:
                row = json.load(f)
            print(f"[run_experiment] [{i}/{n}] SKIP {model_name}: "
                  f"result already logged", flush=True)
        else:
            recoverable = (neural and os.path.exists(ckpt_file)
                           and os.path.exists(done_file))
            if recoverable:
                # A checkpoint WITH its completion marker: training finished but
                # the result row was never logged (crash between the two) —
                # recover metrics with a test-only pass, not a full retrain.
                # Degrade to retraining if recovery fails, so a bad checkpoint
                # never kills the cell.
                try:
                    print(f"[run_experiment] [{i}/{n}] RECOVER {model_name}: "
                          f"test-only from {ckpt_file}", flush=True)
                    result = test_only_neural(strategy, model_name, data_bundle,
                                              config, ckpt_path, ckpt_file)
                except Exception as e:  # noqa: BLE001 - never let it kill the cell
                    print(f"[run_experiment] [{i}/{n}] recover failed ({e!r}); "
                          f"retraining {model_name}", flush=True)
                    result = _train_fresh(strategy, model_name, data_bundle,
                                          config, dirs, neural)
            else:
                # No checkpoint, or a checkpoint with NO completion marker (an
                # interrupted mid-training orphan whose best-so-far weights are
                # under-trained). Never recover the latter as if final — retrain
                # from scratch. _train_fresh() first clears any orphan so it
                # cannot be mis-recovered later and the new best lands on the
                # canonical checkpoint name.
                if neural and os.path.exists(ckpt_file):
                    print(f"[run_experiment] [{i}/{n}] DISCARD orphan "
                          f"{model_name}: checkpoint has no completion marker "
                          f"(interrupted mid-training); retraining", flush=True)
                result = _train_fresh(strategy, model_name, data_bundle,
                                      config, dirs, neural)
            row = _result_row(result)
            _write_row(row_path, row)  # persist the instant this model is done
        rows.append(row)
        # Rebuild the comparison table after EACH model, so a partial cell
        # already has a usable (growing) table on disk and the monitor can
        # accumulate results one model at a time. Written atomically because
        # the monitor / driver read this file concurrently.
        comparison = _rows_to_frame(rows, config.sort_metric)
        _write_csv_atomic(comparison, comparison_path)
        print(f"[run_experiment] [{i}/{n}] logged (+{model_name}); "
              f"{comparison_path} updated", flush=True)
        print(comparison.to_string(index=False), flush=True)

    comparison = _rows_to_frame(rows, config.sort_metric)
    _write_csv_atomic(comparison, comparison_path)
    return comparison, comparison_path
