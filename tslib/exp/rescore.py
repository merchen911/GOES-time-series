"""Re-score saved predictions and merge master tables (spec 4.7).

`rescore_cell` rebuilds a run directory's result rows and comparison tables from
its prediction files alone. It calls the same `evaluate` as the training path, so
a new metric or a changed rule needs neither a GPU nor another inference pass.
`merge_masters` joins master tables built from separate run trees."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence, Tuple

import pandas as pd

from .evaluate import evaluate
from .exp import _row_path, _write_comparisons, _write_csv_atomic, _write_row, save_dir_name
from .lightning_model import TrainResult
from .metrics import MetricContext
from .predictions import load_predictions
from .records import (RUN_KEYS, event_columns, regression_columns, result_record,
                      with_count_dtypes)

# the comparison tables of a run directory are ranked like a training run's
_SORT_METRIC = "best_val_loss"


def _required(meta: dict, key: str, path: Path):
    if meta.get(key) is None:
        raise ValueError(f"{path}: the prediction file's meta has no {key!r}; "
                         f"it cannot be re-scored on its own")
    return meta[key]


def _previous_best_val_loss(path: str) -> float:
    """`best_val_loss` of the row file about to be replaced. It comes from
    training and cannot be recomputed from predictions; nan when there is no
    (readable) previous row."""
    try:
        with open(path) as f:
            return float(json.load(f)["best_val_loss"])
    except (OSError, ValueError, KeyError, TypeError):
        return float("nan")


def rescore_cell(cell_dir: Path, metric_names: Optional[Sequence[str]] = None) -> int:
    """Re-score every `<cell_dir>/pred/<model>.npz`: rewrite
    `score/rows/<model>.json` from the saved predictions (keeping the old row's
    `best_val_loss`) and rebuild both comparison tables. Returns the number of
    models re-scored; a cell without prediction files is left untouched.

    `<model>.npz.tmp` files are an interrupted save and are never read.
    `metric_names=None` computes every registered metric.

    A result row without a prediction file (its save failed, or the run had
    saving off) is a ValueError, raised before anything is rewritten: re-scoring
    around it would leave that run on the old rule in the tables with nothing
    to show it. Re-run such a model (main.py re-tests a checkpoint that has its
    .done marker and writes the file) or delete its row."""
    cell_dir = Path(cell_dir)
    files = sorted((cell_dir / "pred").glob("*.npz"))
    scored = {path.stem for path in files}
    unscorable = sorted(p.stem for p in (cell_dir / "score" / "rows").glob("*.json")
                        if p.stem not in scored)
    if unscorable:
        raise ValueError(
            f"{cell_dir}: result rows without a prediction file: {unscorable}; "
            f"they cannot be re-scored and would keep the old numbers. Re-run those "
            f"models (a checkpoint with .done is re-tested, not retrained) or delete "
            f"their rows")
    if not files:
        return 0
    dirs = save_dir_name(SimpleNamespace(runs_root=str(cell_dir.parent),
                                         run_name=cell_dir.name))
    for path in files:
        model = path.stem
        pred, true, forecast_start, meta = load_predictions(str(path))
        ctx = MetricContext(thresholds=meta.get("event_threshold"),
                            transform=_required(meta, "transform", path),
                            target_cols=list(meta.get("target_cols") or []),
                            forecast_start=forecast_start)
        row_path = _row_path(dirs, model)
        result = TrainResult(model_name=model,
                             best_val_loss=_previous_best_val_loss(row_path),
                             evaluation=evaluate(pred, true, ctx, metric_names),
                             ckpt_path=os.path.join(dirs["ckpt"], f"{model}.pt"),
                             strategy=_required(meta, "strategy", path))
        _write_row(row_path, result_record(result))
    _write_comparisons(dirs, _SORT_METRIC)
    return len(files)


def _merged_table(dirs: Sequence[Path], name: str, columns: list,
                  row_key: list) -> pd.DataFrame:
    frames = []
    for d in dirs:
        # round_trip: the default float parser can change the last digit
        frame = pd.read_csv(d / name, float_precision="round_trip")
        if set(frame.columns) != set(columns):
            missing = [c for c in columns if c not in frame.columns]
            extra = [c for c in frame.columns if c not in columns]
            raise ValueError(f"{d / name} is not a master table of this layout "
                             f"(missing {missing}, unexpected {extra}); "
                             f"re-score its runs instead")
        if len(frame):     # concat of empty frames drops dtypes
            # counts as integers, also from tables that wrote them as `20.0`
            frames.append(with_count_dtypes(frame[columns]))
    merged = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=columns))
    repeated = merged.duplicated(subset=row_key, keep=False)
    if repeated.any():
        runs = merged.loc[repeated, RUN_KEYS].drop_duplicates()
        raise ValueError(
            f"{name}: {len(runs)} run(s) occur more than once across "
            f"{[str(d) for d in dirs]}, e.g. {runs.iloc[0].to_dict()}")
    # the order a single rebuild over all the runs would give, whatever the dir order
    return merged.sort_values(RUN_KEYS, kind="stable").reset_index(drop=True)


def merge_masters(dirs: Sequence[Path], out_dir: Path) -> Tuple[int, int]:
    """Join the master tables of `dirs` (each holding `results_regression.csv`
    and `results_event.csv`) into `out_dir`, rows ordered by run key. A run that
    appears twice, or a table that is not of the `regression_columns()` /
    `event_columns()` layout, is a ValueError and nothing is written. Returns
    (regression rows, event rows)."""
    dirs = [Path(d) for d in dirs]
    if not dirs:
        raise ValueError("no master directories to merge")
    regression = _merged_table(dirs, "results_regression.csv", regression_columns(),
                               RUN_KEYS + ["channel"])
    event = _merged_table(dirs, "results_event.csv", event_columns(),
                          RUN_KEYS + ["channel", "granularity"])
    out_dir = Path(out_dir)
    _write_csv_atomic(regression, str(out_dir / "results_regression.csv"))
    _write_csv_atomic(event, str(out_dir / "results_event.csv"))
    return len(regression), len(event)
