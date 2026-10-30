"""Check the dumped prediction files against the v1 results (spec 4.8).

The archive dump re-runs every v003/v004 checkpoint. Before its predictions
replace the original test pass, the v1 numbers are recomputed from them with
the v1 rules (per-point events, an undefined score is 0.0, a NaN prediction is
no event, rmse/mae over every channel at once) and compared with the v1 master
table. rmse/mae must agree to a relative 1e-4. GPU non-determinism can flip a
few predictions that sit on a threshold, so an event score passes when at most
5 flipped points explain the difference. The v1 rules live here only, for this
check; they are never written out as metrics.

Two checks against the paper's fold tables (``event_skill{,_p288}_fold<k>.csv``)
come with it: this package's formulas on the paper's own counts (a pass
condition, spec 6) and the paper's counts against the dump's (reported only:
the paper re-scored with another inference path)."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from tslib.exp.granularity import event_masks
from tslib.exp.metrics import EVENT_SCORES, to_stored_space
from tslib.exp.predictions import load_predictions, prediction_path
from tslib.exp.records import RUN_KEYS

from .archive import cell_name
from .driver import parse_cell_name

_NAN = float("nan")

FLIP_BOUND = 5               # threshold points that may flip between two test passes
REGRESSION_REL_TOL = 1e-4    # rmse/mae, relative to the v1 value
_BOUND_TOL = 1e-9            # slack on the flip-bound interval

VERIFY_COLUMNS = RUN_KEYS + ["metric", "v1", "archive", "ok"]
COUNT_DIFFS = ["dTP", "dFP", "dFN", "dTN"]
PAPER_COUNT_COLUMNS = RUN_KEYS + ["channel", "granularity"] + COUNT_DIFFS
FORMULA_COLUMNS = ["file", "track", "model", "gran", "fold", "metric", "paper", "ours", "ok"]


# ---- the v1 rules (v003 tslib/exp/metrics.py): an undefined score is 0.0 ----

def _v1_tss(tp, fp, fn, tn):
    pod = tp / (tp + fn) if (tp + fn) else 0.0
    pofd = fp / (fp + tn) if (fp + tn) else 0.0
    return pod - pofd


def _v1_hss(tp, fp, fn, tn):
    # v1's form of HSS (equal to spec 4.2's where both are defined); kept as v1
    # wrote it so the recomputed values are the same floats
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    exp = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / n
    denom = n - exp
    return (tp + tn - exp) / denom if denom else 0.0


def _v1_pod(tp, fp, fn, tn):
    return tp / (tp + fn) if (tp + fn) else 0.0


def _v1_far(tp, fp, fn, tn):
    return fp / (tp + fp) if (tp + fp) else 0.0


V1_EVENT_SCORES = {"tss": _v1_tss, "hss": _v1_hss, "pod": _v1_pod, "far": _v1_far}
_V1_REGRESSION = ("rmse", "mae")


def _counts(pm: np.ndarray, tm: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.count_nonzero(pm & tm))
    fp = int(np.count_nonzero(pm)) - tp
    fn = int(np.count_nonzero(tm)) - tp
    return tp, fp, fn, int(pm.size) - tp - fp - fn


def legacy_v1_scores(pred: np.ndarray, true: np.ndarray, thresholds_stored: Sequence[float],
                     target_cols: Sequence[str]) -> Tuple[dict, dict]:
    """The v1 scores of `pred`/`true` (N, P, T, stored space), named as the v1
    master table names them: ``rmse``, ``mae`` (every channel and point at once)
    and ``tss_/hss_/pod_/far_<channel>`` (per point, inclusive threshold, a NaN
    prediction is no event, an undefined score is 0.0). Also returns each
    channel's ``(tp, fp, fn, tn)``. Channels without a name are ``c<i>``."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.ndim != 3 or pred.shape != true.shape:
        raise ValueError(f"pred {pred.shape} and true {true.shape} must both be (N, P, T)")
    n_channels = pred.shape[-1]
    if len(thresholds_stored) != n_channels:
        raise ValueError(f"{len(thresholds_stored)} event thresholds for "
                         f"{n_channels} target channels")
    squares = absolutes = 0.0
    events, counts = {}, {}
    for i in range(n_channels):
        name = target_cols[i] if i < len(target_cols) else f"c{i}"
        p, t = pred[..., i], true[..., i]
        # float64 sums (v1 summed in float32; the two differ by ~1e-7 relative)
        d = np.subtract(p, t, dtype=np.float64).ravel()
        squares += float(np.dot(d, d))
        absolutes += float(np.abs(d, out=d).sum())
        del d
        c = _counts(*event_masks(p, t, float(thresholds_stored[i]), "point"))
        counts[name] = c
        events.update({f"{m}_{name}": float(fn(*c)) for m, fn in V1_EVENT_SCORES.items()})
    n = pred.size
    scores = {"rmse": math.sqrt(squares / n) if n else _NAN,
              "mae": absolutes / n if n else _NAN, **events}
    return scores, counts


def within_flip_bound(metric: str, counts: Tuple[int, int, int, int], v1_value: float,
                      k: int = FLIP_BOUND) -> bool:
    """Whether flipping at most `k` points of `counts` explains `v1_value` of the
    v1 event score `metric` (tss, hss, pod or far).

    The observations are the same in both passes, so a flip moves a point
    between TP and FN (i) or between FP and TN (j). True when `v1_value` lies
    (within 1e-9) in [min, max] of the score over every (i, j) with
    |i| + |j| <= k that leaves no count negative. A nan `v1_value` passes only
    when the score of `counts` is nan too, which the v1 rules never give."""
    score = V1_EVENT_SCORES[metric]
    tp, fp, fn, tn = (int(c) for c in counts)
    if math.isnan(v1_value):
        return math.isnan(score(tp, fp, fn, tn))
    values = []
    for i in range(-k, k + 1):
        rest = k - abs(i)
        for j in range(-rest, rest + 1):
            flipped = (tp + i, fp + j, fn - i, tn - j)
            if min(flipped) >= 0:
                values.append(score(*flipped))
    return min(values) - _BOUND_TOL <= v1_value <= max(values) + _BOUND_TOL


def _regression_ok(v1: float, archive: float) -> bool:
    if math.isnan(v1) or math.isnan(archive):
        return math.isnan(v1) and math.isnan(archive)
    if math.isinf(v1) or math.isinf(archive):
        return v1 == archive
    return abs(archive - v1) <= REGRESSION_REL_TOL * abs(v1)


def _run_key(row) -> tuple:
    return (str(row["track"]), int(row["seq_len"]), int(row["pred_len"]),
            int(row["fold"]), str(row["strategy"]), str(row["model"]))


def _v1_runs(v1_master_csv: Path) -> Dict[tuple, dict]:
    frame = pd.read_csv(v1_master_csv, float_precision="round_trip")
    missing = [k for k in RUN_KEYS if k not in frame.columns]
    if missing:
        raise ValueError(f"{v1_master_csv} has no {missing} columns; not a v1 master table")
    runs = {}
    for row in frame.to_dict("records"):
        key = _run_key(row)
        if key in runs:
            raise ValueError(f"{v1_master_csv}: run {key} appears twice")
        runs[key] = row
    return runs


def _archive_runs(archive_bench: Path):
    """``(cell, model, npz path)`` of every ``<cell>/pred/<model>.npz``; folders
    whose name ``parse_cell_name`` rejects are not cells."""
    for path in sorted(Path(archive_bench).glob("*/pred/*.npz")):
        try:
            cell = parse_cell_name(path.parents[1].name)
        except ValueError:
            continue
        yield cell, path.stem, path


def _float(value) -> float:
    return _NAN if value is None else float(value)


def verify(archive_bench: Path, v1_master_csv: Path) -> pd.DataFrame:
    """One row per dumped run and v1 metric: run keys, ``metric`` (the v1
    column name), ``v1``, ``archive`` (v1 rules on the dumped predictions) and
    ``ok``. A run found on one side only is one row with metric ``run``,
    ``v1``/``archive`` 1.0 where the run is present and 0.0 where it is not,
    and ``ok`` False. Rows are in run-key order."""
    v1 = _v1_runs(v1_master_csv)
    rows, seen = [], set()
    for cell, model, path in _archive_runs(archive_bench):
        run = {**cell, "model": model}
        key = _run_key(run)
        seen.add(key)
        if key not in v1:
            rows.append({**run, "metric": "run", "v1": 0.0, "archive": 1.0, "ok": False})
            continue
        pred, true, _, meta = load_predictions(str(path))
        thresholds = [to_stored_space(x, meta["transform"]) for x in meta["event_threshold"]]
        scores, counts = legacy_v1_scores(pred, true, thresholds, meta["target_cols"])
        del pred, true
        for metric, value in scores.items():
            expected = _float(v1[key].get(metric))
            if metric in _V1_REGRESSION:
                ok = _regression_ok(expected, value)
            else:
                score, channel = metric.split("_", 1)
                ok = within_flip_bound(score, counts[channel], expected)
            rows.append({**run, "metric": metric, "v1": expected, "archive": value,
                         "ok": bool(ok)})
    for key in v1.keys() - seen:
        rows.append({**dict(zip(RUN_KEYS, key)), "metric": "run", "v1": 1.0,
                     "archive": 0.0, "ok": False})
    report = pd.DataFrame(rows, columns=VERIFY_COLUMNS)
    return report.sort_values(RUN_KEYS, kind="stable").reset_index(drop=True)


# ---- the paper's fold tables ----

# paper track -> (track, channel); the X-class track has no counterpart here
PAPER_TRACKS = {
    "Proton SPE": ("uni_a", "p_gt10"),
    "Proton SPE (multi)": ("multi", "p_gt10"),
    "XRS M-class": ("uni_b", "xrs_long"),
    "XRS M-class (multi)": ("multi", "xrs_long"),
}
PAPER_MODELS = {"segrnn": "segrnn_thuml"}
# per-day is left out: the paper picked its days differently (spec 4.3)
PAPER_GRANULARITIES = {"per-point": "point", "per-window": "window"}
PAPER_BASELINE = "naive (persistence)"
PAPER_SEQ_LEN = 288
PAPER_STRATEGY = "direct"
FORMULA_METRICS = ("tss", "hss", "csi", "sedi")
# The paper rounded to 3 decimals with round(): at most 5e-4 off. An exact half
# (csi 0.0625 -> 0.062) is 5e-4 off on paper but a few 1e-19 more in floats.
FORMULA_TOL = 5e-4 + 1e-12

_PAPER_FILE = re.compile(r"event_skill(_p288)?_fold(\d+)\.csv")


def _paper_tables(paper_folds_dir: Path) -> List[Tuple[str, int, pd.DataFrame]]:
    """``(file name, pred_len, table)`` of every event-skill fold table:
    ``event_skill_fold<k>.csv`` is pred 144, ``event_skill_p288_fold<k>.csv``
    pred 288."""
    tables = []
    for path in sorted(Path(paper_folds_dir).glob("event_skill*_fold*.csv")):
        m = _PAPER_FILE.fullmatch(path.name)
        if m:
            tables.append((path.name, 288 if m.group(1) else 144, pd.read_csv(path)))
    if not tables:
        raise ValueError(f"no event_skill{{,_p288}}_fold<k>.csv tables in {paper_folds_dir}")
    return tables


def compare_paper_counts(archive_bench: Path, paper_folds_dir: Path) -> pd.DataFrame:
    """For every model row of the paper's M-class and SPE tracks at per-point
    and per-window: run keys of the matching dumped run (seq 288, direct),
    ``channel``, ``granularity`` and the count differences dump - paper
    (``dTP dFP dFN dTN``; empty when the run has no prediction file). The
    window judgement is the horizon maximum, as in the paper. For reference:
    not a pass condition."""
    wanted: Dict[Path, list] = {}
    for _, pred_len, table in _paper_tables(paper_folds_dir):
        keep = table[table.track.isin(list(PAPER_TRACKS))
                     & table.gran.isin(list(PAPER_GRANULARITIES))
                     & (table.model != PAPER_BASELINE)]
        for r in keep.itertuples(index=False):
            track, channel = PAPER_TRACKS[r.track]
            cell = {"track": track, "seq_len": PAPER_SEQ_LEN, "pred_len": pred_len,
                    "fold": int(r.fold), "strategy": PAPER_STRATEGY}
            model = PAPER_MODELS.get(r.model, r.model)
            path = Path(prediction_path(str(Path(archive_bench) / cell_name(cell)), model))
            wanted.setdefault(path, []).append(
                ({**cell, "model": model, "channel": channel,
                  "granularity": PAPER_GRANULARITIES[r.gran]},
                 (int(r.TP), int(r.FP), int(r.FN), int(r.TN))))
    rows = []
    for path, items in wanted.items():     # each file is loaded once
        loaded = load_predictions(str(path)) if path.exists() else None
        for row, paper in items:
            if loaded is None:
                diffs = [pd.NA] * 4
            else:
                pred, true, _, meta = loaded
                i = meta["target_cols"].index(row["channel"])
                thr = to_stored_space(meta["event_threshold"][i], meta["transform"])
                ours = _counts(*event_masks(pred[..., i], true[..., i], thr,
                                            row["granularity"]))
                diffs = [o - p for o, p in zip(ours, paper)]
            rows.append({**row, **dict(zip(COUNT_DIFFS, diffs))})
        loaded = None
    out = pd.DataFrame(rows, columns=PAPER_COUNT_COLUMNS)
    out = out.astype({d: "Int64" for d in COUNT_DIFFS})   # integers, empty when missing
    return out.sort_values(RUN_KEYS + ["channel", "granularity"],
                           kind="stable").reset_index(drop=True)


def _formula_ok(paper: float, ours: float) -> bool:
    if math.isnan(paper) or math.isnan(ours):
        return math.isnan(paper) and math.isnan(ours)
    return abs(ours - paper) <= FORMULA_TOL


def check_formulas_against_paper(paper_folds_dir: Path) -> pd.DataFrame:
    """``tss hss csi sedi`` of ``EVENT_SCORES`` on the TP/FP/FN/TN of every row
    of the paper's fold tables (baseline included) against the paper's
    ``TSS HSS CSI SEDI``: one row per table row and metric with ``file``,
    ``track``, ``model``, ``gran``, ``fold``, ``metric``, ``paper``, ``ours``
    and ``ok`` (within 5e-4, the paper's rounding, or both undefined)."""
    rows = []
    for name, _, table in _paper_tables(paper_folds_dir):
        for r in table.itertuples(index=False):
            counts = (int(r.TP), int(r.FP), int(r.FN), int(r.TN))
            for metric in FORMULA_METRICS:
                paper = float(getattr(r, metric.upper()))
                ours = float(EVENT_SCORES[metric](*counts))
                rows.append({"file": name, "track": r.track, "model": r.model,
                             "gran": r.gran, "fold": int(r.fold), "metric": metric,
                             "paper": paper, "ours": ours, "ok": _formula_ok(paper, ours)})
    return pd.DataFrame(rows, columns=FORMULA_COLUMNS)
