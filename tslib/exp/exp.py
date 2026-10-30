from __future__ import annotations

import json
import os
import random
import uuid

import numpy as np
import pandas as pd
import torch

from tslib.data.loader import DataModule
from .predictions import prediction_path
from .records import (event_frame, is_current_record, regression_frame, result_record,
                      stale_record_note)
from .strategy import run_strategy, test_only_neural


# Skill scores where a HIGHER value is a better model (sort descending so the
# best sits on top). Everything else — best_val_loss, rmse, mae, msa, far — is
# an error/false-alarm measure where LOWER is better (sort ascending). sspb is
# signed with 0 best, so it cannot rank models (config rejects it).
_HIGHER_IS_BETTER = {"tss", "hss", "pod", "csi", "gss", "ets",
                     "acc", "f1", "precision", "recall", "auc",
                     "seds", "sedi", "logr"}


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


def _comparison_tables(records, sort_metric):
    """(regression, event) tables of one run directory's records: regression
    rows best-first by `sort_metric`; event rows grouped by channel and
    granularity, best HSS first within each group (NaN last)."""
    regression = _sort_frame(regression_frame(records), sort_metric)
    event = event_frame(records).sort_values(
        by=["channel", "granularity", "hss"], ascending=[True, True, False],
        na_position="last", kind="stable").reset_index(drop=True)
    return regression, event


def build_comparison(results, sort_metric):
    """(regression, event) comparison tables of in-memory `TrainResult`s."""
    return _comparison_tables([result_record(r) for r in results], sort_metric)


def comparison_paths(dirs) -> tuple[str, str]:
    """(regression, event) comparison table paths of a run directory."""
    return (os.path.join(dirs["score"], "comparison_regression.csv"),
            os.path.join(dirs["score"], "comparison_event.csv"))


def _row_path(dirs, model_name: str) -> str:
    return os.path.join(dirs["score"], "rows", f"{model_name}.json")


def _rows_on_disk(dirs) -> list:
    """Every result record persisted for this run directory, whoever wrote it.

    The comparison tables are derived from THIS, not from the models of the
    calling process: the v004 sweep runs one model per process across several
    GPUs, so two processes can share a run directory and each would otherwise
    overwrite the tables with only its own rows. Row files are one-writer-per-file, so
    re-reading them makes every write a merge instead of a clobber. A row being
    replaced right now (tmp+rename) is skipped and picked up on the next
    rebuild. A row of another format (a v1.0.0 row file) is skipped with a
    warning, so it cannot stop the run that is about to replace it."""
    import glob
    rows = []
    for path in sorted(glob.glob(os.path.join(dirs["score"], "rows", "*.json"))):
        try:
            with open(path) as f:
                rec = json.load(f)
        except (OSError, ValueError):
            continue
        if not is_current_record(rec):
            print(f"[warn] {stale_record_note(path, rec)}", flush=True)
            continue
        rows.append(rec)
    return rows


def _logged_record(path: str):
    """The result record already logged at `path`, or None when the model has
    none to skip on: no row file, an unreadable one, or a row of another format
    (a v1.0.0 row file, which the model's re-test or retrain replaces)."""
    try:
        with open(path) as f:
            rec = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        print(f"[warn] {path}: unreadable result row ({e!r}); running the model again",
              flush=True)
        return None
    return rec if is_current_record(rec) else None


def _tmp_path(path: str) -> str:
    """A temp name next to `path` that no other writer uses: the scheduler and
    the main.py processes of one cell rewrite the same files, and a shared
    `path + ".tmp"` let one writer rename another's half-written file away
    (FileNotFoundError) or interleave two writes in one file. Ends in `.tmp`,
    so the `*.json` / `*.csv` readers never pick it up."""
    return f"{path}.{os.getpid()}.{uuid.uuid4().hex[:12]}.tmp"


def _replace_atomically(path: str, write) -> None:
    """Run `write(tmp)` on a fresh temp file, then rename it onto `path`; on any
    failure the temp file is removed and the previous `path` is kept."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = _tmp_path(path)
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _write_row(path: str, row: dict) -> None:
    """Atomically persist one model's result row (write-tmp-then-rename), so a
    crash can never leave a half-written row file."""
    def write(tmp):
        with open(tmp, "x") as f:
            json.dump(row, f)
    _replace_atomically(path, write)


def _write_csv_atomic(df, path: str) -> None:
    """Write a comparison table atomically (tmp + rename). The monitor and the
    benchmark driver read the comparison tables concurrently with this process;
    a plain in-place to_csv can be observed half-written (truncated row /
    missing columns), so mirror _write_row's tmp-then-replace discipline here
    too, with a temp name of its own for every writer."""
    _replace_atomically(path, lambda tmp: df.to_csv(tmp, index=False))


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


def _train_fresh(strategy, model_name, data_bundle, config, dirs, neural,
                 pred_path):
    """(Re)train a model from scratch. For neural strategies, first clear any
    stale/orphan checkpoint so Lightning writes to the canonical name, then drop
    a completion marker once run_strategy returns so a later resume can tell the
    training finished cleanly (vs. a mid-training crash)."""
    ckpt_dir = dirs["ckpt"]
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pt")
    if neural:
        _clear_ckpts(ckpt_dir, model_name)
    result = run_strategy(strategy, model_name, data_bundle, config, ckpt_path,
                          pred_path=pred_path)
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
    run_root = os.path.join(getattr(config, "runs_root", None) or "runs",
                            config.run_name)
    dir_types = {
        "root": run_root,
        "ckpt": os.path.join(run_root, "ckpt"),
        "score": os.path.join(run_root, "score"),
    }
    if makedir:
        for p in dir_types.values():
            os.makedirs(p, exist_ok=True)
    return dir_types


def _write_comparisons(dirs, sort_metric):
    """Rebuild both comparison tables from the records on disk and write them
    atomically; returns (regression, event)."""
    regression, event = _comparison_tables(_rows_on_disk(dirs), sort_metric)
    reg_path, event_path = comparison_paths(dirs)
    _write_csv_atomic(regression, reg_path)
    _write_csv_atomic(event, event_path)
    return regression, event


def run_experiment(config):
    """Train and test every model of `config` in one run directory; returns the
    (regression, event) comparison tables, also written under `score/`."""
    _set_seed(config.seed)
    dirs = save_dir_name(config, makedir=True)

    data_module = DataModule(config)
    data_bundle = data_module.setup()

    strategy = getattr(config, "forecast_strategy", "direct")
    neural = strategy in ("direct", "recursive")
    save = getattr(config, "save_predictions", True)
    # a prediction file needs each window's forecast start (CSV input without
    # --time_col has none), so only then can a missing one be rebuilt
    can_save = save and getattr(data_bundle, "test_forecast_start", None) is not None
    n = len(config.models)
    for i, model_name in enumerate(config.models, 1):
        row_path = _row_path(dirs, model_name)
        ckpt_path = os.path.join(dirs["ckpt"], f"{model_name}.pt")
        ckpt_file = os.path.join(dirs["ckpt"], f"{model_name}.ckpt")
        done_file = _done_marker(dirs["ckpt"], model_name)
        pred_path = prediction_path(dirs["root"], model_name)
        logged = _logged_record(row_path) is not None
        if logged and can_save and not os.path.exists(pred_path):
            # Logged, but the prediction file is missing (its save failed): the
            # row could never be re-scored, so it is not a finished model.
            # Rebuild the file (and the row, from the same pass) from the
            # finished checkpoint; never retrain for it.
            if neural and os.path.exists(ckpt_file) and os.path.exists(done_file):
                print(f"[run_experiment] [{i}/{n}] REPREDICT {model_name}: "
                      f"no {pred_path}; test-only from {ckpt_file}", flush=True)
                try:
                    result = test_only_neural(strategy, model_name, data_bundle,
                                              config, ckpt_path, ckpt_file,
                                              pred_path=pred_path)
                    _write_row(row_path, result_record(result))
                except Exception as e:  # noqa: BLE001 - the logged row stays usable
                    print(f"[warn] {model_name}: re-testing for the missing "
                          f"prediction file failed ({e!r}); the row is kept", flush=True)
            else:
                print(f"[warn] {model_name}: result logged but {pred_path} is missing "
                      f"and there is no checkpoint with a .done marker to re-test; the "
                      f"row is kept but cannot be re-scored. Delete {row_path} to run "
                      f"the model again", flush=True)
        elif logged:
            # Result already logged on a prior (possibly interrupted) run —
            # skip entirely, no reload/retrain.
            print(f"[run_experiment] [{i}/{n}] SKIP {model_name}: "
                  f"result already logged", flush=True)
        else:
            if os.path.exists(row_path):
                # a v1.0.0 row file (or an unreadable one): not a result of
                # this version, so the model is re-tested or retrained below
                print(f"[run_experiment] [{i}/{n}] UPGRADE {model_name}: "
                      f"{row_path} is not a v2 result record; replacing it", flush=True)
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
                                              config, ckpt_path, ckpt_file,
                                              pred_path=pred_path)
                except Exception as e:  # noqa: BLE001 - never let it kill the cell
                    print(f"[run_experiment] [{i}/{n}] recover failed ({e!r}); "
                          f"retraining {model_name}", flush=True)
                    result = _train_fresh(strategy, model_name, data_bundle,
                                          config, dirs, neural, pred_path)
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
                                      config, dirs, neural, pred_path)
            # persist the instant this model is done
            _write_row(row_path, result_record(result))
        # Rebuild the comparison tables after EACH model, so a partial cell
        # already has usable (growing) tables on disk and the monitor can
        # accumulate results one model at a time. Written atomically because
        # the monitor / driver read these files concurrently.
        regression, event = _write_comparisons(dirs, config.sort_metric)
        print(f"[run_experiment] [{i}/{n}] logged (+{model_name}); "
              f"{dirs['score']} comparison tables updated", flush=True)
        print(regression.to_string(index=False), flush=True)
        print(event.to_string(index=False), flush=True)

    if save:
        missing = [m for m in config.models
                   if not os.path.exists(prediction_path(dirs["root"], m))]
        if missing:
            print(f"[warn] no prediction file for {missing} under "
                  f"{os.path.join(dirs['root'], 'pred')}: these rows cannot be "
                  f"re-scored (scripts/rescore.py refuses the cell); see the "
                  f"warnings above", flush=True)
    return _write_comparisons(dirs, config.sort_metric)
