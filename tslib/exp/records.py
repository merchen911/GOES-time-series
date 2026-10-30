"""Result records and the two result tables (spec 4.5).

A record is the JSON-ready form of one model's result: run identity plus the
per-channel regression rows and per-channel x per-granularity event rows of its
`Evaluation`. It is what a row file holds, and `regression_frame` /
`event_frame` turn a list of records into the two tables (cell comparison and
master tables, re-scoring, archive dump), so the column layout lives here only."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from .metrics import EVENT_SCORES, REGRESSION_METRICS

if TYPE_CHECKING:   # lightning_model pulls in torch; records stay light to import
    from .lightning_model import TrainResult

RECORD_FORMAT = 2

# What identifies one run in a master table; the first four come from the
# benchmark cell, the last two from the record.
CELL_KEYS = ["track", "seq_len", "pred_len", "fold"]
RUN_KEYS = CELL_KEYS + ["strategy", "model"]
# The columns before the metric columns. The metric columns are not listed
# here: they are the registered metrics, in registration order (see
# `regression_columns` / `event_columns`), so registering a metric is all it
# takes to get its column in every table (spec 4.5).
REGRESSION_KEY_COLUMNS = RUN_KEYS + ["best_val_loss", "channel", "n"]
EVENT_KEY_COLUMNS = RUN_KEYS + ["channel", "granularity", "threshold", "n", "n_events",
                                "tp", "fp", "fn", "tn", "n_nonfinite"]
# Integer columns. A voided channel has NaN counts, which would make the whole
# column float64 and print every count as `20.0`; they are kept as the nullable
# integer dtype instead (a NaN count is written as an empty field).
COUNT_COLUMNS = ["n", "n_events", "tp", "fp", "fn", "tn", "n_nonfinite"]


def regression_columns() -> List[str]:
    """Columns of the regression tables: `REGRESSION_KEY_COLUMNS`, then one per
    metric registered with `register_regression_metric` (built-ins: rmse mae
    msa sspb logr). Read at call time, so a metric registered later is included."""
    return list(dict.fromkeys([*REGRESSION_KEY_COLUMNS, *REGRESSION_METRICS]))


def event_columns() -> List[str]:
    """Columns of the event tables: `EVENT_KEY_COLUMNS`, then one per score
    registered with `register_event_score` (built-ins: tss hss pod far csi f1
    seds sedi). Read at call time, so a score registered later is included."""
    return list(dict.fromkeys([*EVENT_KEY_COLUMNS, *EVENT_SCORES]))


def with_count_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """`frame` with its `COUNT_COLUMNS` as pandas "Int64" (integers, NaN kept as NA)."""
    counts = {c: "Int64" for c in COUNT_COLUMNS if c in frame.columns}
    return frame.astype(counts) if counts else frame


def is_current_record(rec) -> bool:
    """True for a record of this format; False for a v1.0.0 row file (no
    `format` key) or anything else."""
    return isinstance(rec, dict) and rec.get("format") == RECORD_FORMAT


def stale_record_note(path, rec) -> str:
    """Why the row file at `path` is left out, and what to do about it."""
    fmt = rec.get("format") if isinstance(rec, dict) else None
    return (f"{path} is not a format-{RECORD_FORMAT} result record (format {fmt!r}; "
            f"a v1.0.0 row?) and is left out of the tables; re-run its model (one with "
            f"a checkpoint and .done is re-tested, not retrained) or delete the file")


def result_record(result: "TrainResult") -> dict:
    """The row-file form of `result` (JSON-serialisable; NaN is kept as NaN)."""
    ev = result.evaluation
    if ev is None:
        raise ValueError(f"result of {result.model_name!r} has no evaluation to record")
    return {"format": RECORD_FORMAT,
            "strategy": result.strategy,
            "model": result.model_name,
            "best_val_loss": float(result.best_val_loss),
            "regression": [dict(r) for r in ev.regression],
            "event": [dict(r) for r in ev.event]}


def regression_frame(records: List[dict], cell: Optional[dict] = None) -> pd.DataFrame:
    """One row per record and channel, columns `regression_columns()` (without
    the cell keys when `cell` is None). Metrics a run did not select are NaN."""
    return _frame(records, cell, "regression", regression_columns())


def event_frame(records: List[dict], cell: Optional[dict] = None) -> pd.DataFrame:
    """One row per record, channel and granularity, columns `event_columns()`
    (without the cell keys when `cell` is None). Unselected scores are NaN."""
    return _frame(records, cell, "event", event_columns())


def _frame(records: List[dict], cell: Optional[dict], key: str,
           columns: List[str]) -> pd.DataFrame:
    for rec in records:
        if not is_current_record(rec):
            model = rec.get("model", "<model>") if isinstance(rec, dict) else "<model>"
            raise ValueError(
                f"v1 result rows are not supported: expected a record with format "
                f"{RECORD_FORMAT}, got {rec.get('format')!r} (model {model!r}); "
                f"delete score/rows/{model}.json and run the model again: a model "
                f"with a checkpoint and .done is re-tested, not retrained")
    if cell is None:
        columns = [c for c in columns if c not in CELL_KEYS]
        cell_part = {}
    else:
        missing = [k for k in CELL_KEYS if k not in cell]
        if missing:
            raise ValueError(f"cell is missing {missing}; needs {CELL_KEYS}")
        cell_part = {k: cell[k] for k in CELL_KEYS}
    rows = []
    for rec in records:
        run = {**cell_part, "strategy": rec["strategy"], "model": rec["model"],
               "best_val_loss": rec["best_val_loss"]}
        rows.extend({**run, **row} for row in rec[key])
    return with_count_dtypes(pd.DataFrame(rows, columns=columns))
