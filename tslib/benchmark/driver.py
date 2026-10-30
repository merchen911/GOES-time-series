"""Sequential benchmark driver: the sweep one cell at a time.

Orchestrates the sweep from ``docs/benchmark-conditions.md``. Each (track,
seq_len, pred_len, fold, strategy) cell is run
as an **isolated** ``main.py`` subprocess (so one run's OOM/crash cannot kill the
whole sweep, and GPU memory is released between runs). After each cell, the
result records of every cell that has any are folded — with track/seq_len/
pred_len/fold columns attached — into the two master tables
(``results_regression.csv``: one row per run and channel; ``results_event.csv``:
one row per run, channel and aggregation unit), so results accumulate one cell
at a time.

The models of a cell are ``models_for_cell``: the track's model set minus
``MODEL_EXCLUSIONS`` (xPatch at the 7-day input), the same grid the multi-GPU
``tslib.benchmark.scheduler`` runs one model per job: 1,740 runs for the four
forecast lengths and five folds. A model counts as done when its result row
and its prediction file exist.

Defaults: direct strategy, fold 0 (``--folds 0 1 2 3 4`` for the full sweep).
``timesnet`` is excluded from every model set (too slow — see the benchmark
conditions doc); it stays in the model registry and can be re-added to a
model set explicitly if desired.

Run the full sweep:

    python3.12 -m tslib.benchmark.driver --folds 0 1 2 3 4

Preview without executing:

    python3.12 -m tslib.benchmark.driver --dry-run
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = os.environ.get(
    "SW_DATA_DIR",
    str(REPO / "data" / "goes_data" / "processed"))
PARTICLE = f"{DATA_DIR}/kswrc_particle_5m.parquet"
XRAY = f"{DATA_DIR}/kswrc_xray_1m.parquet"

# Direct model sets — fast tier only. Selected by a per-epoch speed probe
# (2026-07-07, direct mode, all seq_len/pred_len combos): every model here stays
# under ~10 min/epoch in EVERY sweep cell. Excluded as too slow (>=10 min/epoch
# in at least one combo): timemixer, etsformer, micn, scinet, timesnet, and
# nonstationary_transformer (also OOMs at seq_len>=864). segrnn omitted as
# redundant with segrnn_thuml.
_FAST = ["dlinear", "segrnn_thuml", "tsmixer", "patchmixer", "tide",
         "xpatch", "patchtst", "frets", "itransformer", "rlinear"]
# lstm runs recursive in every track (its natural mode); the fast tier are the
# direct backbones. Same direct set for univariate and multivariable tracks.
DIRECT_UNI = list(_FAST)
DIRECT_MULTI = list(_FAST)

SEQ_LENS = [288, 864, 2016]   # 1 d / 3 d / 7 d — light first, so the first/cheapest cells run first
PRED_LENS = [36, 72, 144, 288]  # 3 h / 6 h / 12 h / 24 h (v003 ran 144/288, v004 added 36/72)

# SegRNN's segment length. FIXED at 24 (2 h on the 5-min grid) for every
# horizon in the sweep, including pred_len 36, which 24 does not divide: the
# model decodes up to the next whole segment (48 steps) and trims to pred_len,
# so the horizon never perturbs the hyperparameter. Making seg_len a function
# of pred_len instead would confound the axis this sweep measures. seg_len must
# still divide seq_len — 24 divides 288/864/2016 — and segrnn_thuml raises with
# a named error if a future seq_len breaks that. See test_horizons.py.
SEG_LEN = 24

# Fixed conditions common to every run (docs/benchmark-conditions.md).
FIXED = [
    "--n_fold", "5",
    "--transform", "log10",
    "--batch_size", "64",
    "--lr", "1e-3",
    "--d_model", "128",
    "--num_layers", "2",
    "--dropout", "0.1",
    "--early_stop_patience", "10",
    "--seg_len", str(SEG_LEN),
    # 8, not v003's 12: v004 runs three jobs at once (one per GPU), so the
    # worker pools are additive — 3 x 8 = 24 loaders on a 48-core box that is
    # shared with other users.
    "--num_workers", "8",
]

# Per-track definitions. The multivariable track forecasts exactly its 2 input
# channels (p_gt10 + xrs_long), so targets == channels and recursive applies to
# it too (recursive lstm rolls the full 2-channel frame forward). p_gt100 /
# xrs_short are intentionally not modelled.
TRACKS = {
    "uni_a": {
        "data_path": PARTICLE, "target_col": "p_gt10",
        "channels": None, "target_cols": None,
        "event_threshold": ["10"],
        "direct_models": DIRECT_UNI, "recursive_models": ["lstm"],
    },
    "uni_b": {
        "data_path": XRAY, "target_col": "xrs_long",
        "channels": None, "target_cols": None,
        "event_threshold": ["1e-5"],
        "direct_models": DIRECT_UNI, "recursive_models": ["lstm"],
    },
    "multi": {
        "data_path": PARTICLE, "target_col": "p_gt10",  # required placeholder; --channels drives loading
        "channels": [f"{PARTICLE}:p_gt10", f"{XRAY}:xrs_long"],
        "target_cols": ["p_gt10", "xrs_long"],
        "event_threshold": ["10", "1e-5"],
        "direct_models": DIRECT_MULTI, "recursive_models": ["lstm"],
    },
}


# (model, seq_len) pairs dropped from the sweep. xpatch's training diverges at
# the 7-day lookback (seen across every v003 fold), so at seq_len 2016 it is
# excluded rather than left to contribute a garbage row. v003 encoded this by
# hand-editing the manifest; keeping it in code means the grid the scheduler
# enumerates is already the grid that should run.
MODEL_EXCLUSIONS = {("xpatch", 2016)}


def models_for(track_key: str, strategy: str) -> list[str]:
    t = TRACKS[track_key]
    return t["direct_models"] if strategy == "direct" else t["recursive_models"]


def models_for_cell(cell: dict) -> list[str]:
    """Models that actually run in this cell — the track/strategy set minus any
    (model, seq_len) exclusion."""
    return [m for m in models_for(cell["track"], cell["strategy"])
            if (m, cell["seq_len"]) not in MODEL_EXCLUSIONS]


def run_name_for(track_key: str, seq: int, pred: int, fold: int,
                 strategy: str) -> str:
    return f"bench/{track_key}_seq{seq}_pred{pred}_f{fold}_{strategy}"


def enumerate_cells(tracks, seq_lens, pred_lens, folds, strategies):
    """Every applicable cell as a dict. Cells whose (track, strategy) has no
    models (e.g. multi + recursive) are dropped."""
    cells = []
    for track_key, seq, pred, fold, strategy in itertools.product(
            tracks, seq_lens, pred_lens, folds, strategies):
        if not models_for(track_key, strategy):
            continue
        cells.append({"track": track_key, "seq_len": seq, "pred_len": pred,
                      "fold": fold, "strategy": strategy})
    return cells


def cell_argv(cell, epochs: int, models: list[str], runs_root=None) -> list[str]:
    """main.py flag list for one cell with an explicit model list."""
    t = TRACKS[cell["track"]]
    run_name = run_name_for(cell["track"], cell["seq_len"], cell["pred_len"],
                            cell["fold"], cell["strategy"])
    argv = ["--data_path", t["data_path"],
            "--target_col", t["target_col"],
            "--seq_len", str(cell["seq_len"]),
            "--pred_len", str(cell["pred_len"]),
            "--fold_numb", str(cell["fold"]),
            "--forecast_strategy", cell["strategy"],
            "--run_name", run_name,
            "--epochs", str(epochs),
            "--event_threshold", *t["event_threshold"],
            "--models", *models,
            *FIXED]
    if t["channels"]:
        argv += ["--channels", *t["channels"],
                 "--target_cols", *t["target_cols"]]
    if runs_root is not None:
        # The subprocess resolves its own output paths, so a caller that
        # redirects runs_root MUST tell the subprocess too — otherwise the run
        # writes to <cwd>/runs while the caller looks for it elsewhere, and a
        # successful job reads as a failure.
        argv += ["--runs_root", str(runs_root)]
    return argv


def build_cmd(cell, epochs: int, models=None, runs_root=None) -> list[str]:
    """The main.py command of one cell; `models` defaults to the cell's sweep
    models (``models_for_cell``, exclusions applied)."""
    if models is None:
        models = models_for_cell(cell)
    return [sys.executable, str(REPO / "main.py"),
            *cell_argv(cell, epochs, models, runs_root)]


def cells_from_manifest(manifest: dict):
    """Group manifest['approved'] entries into [(cell, [models...]), ...],
    preserving first-seen order of cells and of models within a cell."""
    order, grouped = [], {}
    for e in manifest.get("approved", []):
        cell = {"track": e["track"], "seq_len": e["seq_len"],
                "pred_len": e["pred_len"], "fold": e["fold"],
                "strategy": e["strategy"]}
        key = (e["track"], e["seq_len"], e["pred_len"], e["fold"], e["strategy"])
        if key not in grouped:
            grouped[key] = {"cell": cell, "models": []}
            order.append(key)
        grouped[key]["models"].append(e["model"])
    return [(grouped[k]["cell"], grouped[k]["models"]) for k in order]


def score_dir(cell, runs_root: Path) -> Path:
    run_name = run_name_for(cell["track"], cell["seq_len"], cell["pred_len"],
                            cell["fold"], cell["strategy"])
    return runs_root / run_name / "score"


def comparison_paths(cell, runs_root: Path) -> tuple[Path, Path]:
    """(regression, event) comparison tables of one cell, as run_experiment
    writes them (their file names belong to ``tslib.exp.exp.comparison_paths``)."""
    from tslib.exp.exp import comparison_paths as exp_comparison_paths
    reg, event = exp_comparison_paths({"score": str(score_dir(cell, runs_root))})
    return Path(reg), Path(event)


def row_path(cell, model: str, runs_root: Path) -> Path:
    """Per-model result row written by run_experiment the instant that model
    finishes. This (with the prediction file), not the comparison tables, is the
    completion source of truth: the scheduler runs one model per process
    across several GPUs, so many processes share a cell directory and each
    rewrites the comparison tables, while the row files are one-writer-per-file
    and therefore race-free."""
    return score_dir(cell, runs_root) / "rows" / f"{model}.json"


def _read_rows(rows_dir: Path) -> list[dict]:
    """Every result record in one ``score/rows`` folder, in file-name order. A
    row of another format (a v1.0.0 row file) is left out with a warning: it
    must not stop the master rebuild, and the model re-runs anyway because it
    has no prediction file."""
    from tslib.exp.records import is_current_record, stale_record_note
    rows = []
    for f in sorted(rows_dir.glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError):
            continue  # a row being rewritten right now — next rebuild picks it up
        if not is_current_record(rec):
            print(f"[warn] {stale_record_note(f, rec)}", flush=True)
            continue
        rows.append(rec)
    return rows


def read_rows(cell, runs_root: Path) -> list[dict]:
    """Every result record already on disk for this cell."""
    return _read_rows(score_dir(cell, runs_root) / "rows")


def prediction_file(cell, model: str, runs_root: Path) -> Path:
    """The model's saved test predictions, ``<cell>/pred/<model>.npz``."""
    run_name = run_name_for(cell["track"], cell["seq_len"], cell["pred_len"],
                            cell["fold"], cell["strategy"])
    return runs_root / run_name / "pred" / f"{model}.npz"


def model_is_complete(cell, model: str, runs_root: Path,
                      require_predictions: bool = True) -> bool:
    """True when the model's result row exists and (by default; every sweep
    run saves its predictions) its prediction file too. A row whose npz save
    failed is not done: it could never be re-scored, so it would keep the
    numbers of an older rule in the tables. Re-running such a model re-tests
    its checkpoint instead of retraining it (see run_experiment)."""
    if not row_path(cell, model, runs_root).exists():
        return False
    return not require_predictions or prediction_file(cell, model, runs_root).exists()


def cell_is_complete(cell, models, runs_root) -> bool:
    """True iff every model in `models` is complete (result row and prediction
    file), so re-running would add nothing."""
    return all(model_is_complete(cell, m, runs_root) for m in models)


# A cell's run directory under ``runs/bench/`` — what ``run_name_for`` names,
# minus the ``bench/`` prefix.
_CELL_NAME = re.compile(
    r"^(uni_a|uni_b|multi)_seq(\d+)_pred(\d+)_f(\d+)_(direct|recursive|statistic)$")


def parse_cell_name(name: str) -> dict:
    """Inverse of ``run_name_for``: the cell of a run directory name (with or
    without the leading ``bench/``). ValueError for any other name."""
    m = _CELL_NAME.fullmatch(name.removeprefix("bench/"))
    if m is None:
        raise ValueError(f"not a benchmark cell name: {name!r}")
    track, seq, pred, fold, strategy = m.groups()
    return {"track": track, "seq_len": int(seq), "pred_len": int(pred),
            "fold": int(fold), "strategy": strategy}


def _concat(frames, columns) -> pd.DataFrame:
    frames = [f for f in frames if len(f)]   # concat of empty frames drops dtypes
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=columns))


def _write_master(entries, master_dir: Path) -> int:
    """Fold ``(cell, records)`` pairs into ``master_dir/results_regression.csv``
    and ``results_event.csv``, rows in cell order (track, seq_len, pred_len,
    fold, strategy). Cells without records are skipped, and with none at all
    nothing is written. Both tables are built before either is written, so a
    bad record (a v1 row) leaves the existing tables alone. Returns the number
    of cells folded in."""
    from tslib.exp.exp import _write_csv_atomic
    from tslib.exp.records import (event_columns, event_frame, regression_columns,
                                   regression_frame)
    entries = sorted(((c, r) for c, r in entries if r),
                     key=lambda e: (e[0]["track"], e[0]["seq_len"], e[0]["pred_len"],
                                    e[0]["fold"], e[0]["strategy"]))
    if not entries:
        return 0
    regression = _concat([regression_frame(r, c) for c, r in entries],
                         regression_columns())
    event = _concat([event_frame(r, c) for c, r in entries], event_columns())
    master_dir = Path(master_dir)
    _write_csv_atomic(regression, str(master_dir / "results_regression.csv"))
    _write_csv_atomic(event, str(master_dir / "results_event.csv"))
    return len(entries)


def rebuild_master(cells, runs_root: Path, master_dir: Path) -> int:
    """Fold the result records of every given cell into the two master tables
    in ``master_dir``. Idempotent: rewrites both from scratch each call, so
    re-runs never duplicate rows. Returns the number of cells folded in."""
    return _write_master(((c, read_rows(c, runs_root)) for c in cells), master_dir)


def rebuild_master_from_dirs(bench_root: Path, master_dir: Path) -> int:
    """``rebuild_master`` for a tree whose cells are not listed up front (an
    archive, a re-scored copy): every ``bench_root/<cell>/score/rows`` folder
    whose name ``parse_cell_name`` accepts is a cell; logs, pilot runs and
    stray files are ignored. Returns the number of cells folded in."""
    entries = []
    for rows_dir in Path(bench_root).glob("*/score/rows"):
        try:
            cell = parse_cell_name(rows_dir.parents[1].name)
        except ValueError:
            continue
        entries.append((cell, _read_rows(rows_dir)))
    return _write_master(entries, master_dir)


def run_benchmark(runs, epochs, runs_root, master_dir, dry_run=False):
    total = len(runs)
    env = os.environ.copy()
    cells = [c for c, _ in runs]
    for i, (cell, models) in enumerate(runs, 1):
        run_name = run_name_for(cell["track"], cell["seq_len"],
                                cell["pred_len"], cell["fold"], cell["strategy"])
        cmd = build_cmd(cell, epochs, models, runs_root)
        if dry_run:
            print(f"[{i}/{total}] {run_name} models={models}")
            print("    " + " ".join(cmd))
            continue
        if cell_is_complete(cell, models, runs_root):
            print(f"[{i}/{total}] SKIP (done): {run_name}")
            continue
        print(f"[{i}/{total}] RUN: {run_name}\n    {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO), env=env)
        if proc.returncode != 0:
            print(f"[{i}/{total}] FAILED (exit {proc.returncode}): {run_name} "
                  f"— continuing", flush=True)
            continue
        n = rebuild_master(cells, runs_root, master_dir)
        print(f"[{i}/{total}] DONE: {run_name} — master now has {n} cells "
              f"in {master_dir}", flush=True)
        for table in comparison_paths(cell, runs_root):
            if table.exists():
                print(pd.read_csv(table).to_string(index=False), flush=True)
    if not dry_run:
        n = rebuild_master(cells, runs_root, master_dir)
        print(f"\nBenchmark sweep complete: {n}/{total} cells in {master_dir}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Sequential benchmark sweep driver (one cell at a time).")
    p.add_argument("--tracks", nargs="+", default=list(TRACKS),
                   choices=list(TRACKS))
    p.add_argument("--seq-lens", nargs="+", type=int, default=SEQ_LENS)
    p.add_argument("--pred-lens", nargs="+", type=int, default=PRED_LENS)
    p.add_argument("--folds", nargs="+", type=int, default=[0])
    p.add_argument("--strategies", nargs="+", default=["direct"],
                   choices=["direct", "recursive"])
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--manifest", type=str, default=None,
                   help="run only (cell,model) entries approved in this manifest")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    runs_root = REPO / "runs"
    master_dir = runs_root / "bench"
    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)
        runs = cells_from_manifest(manifest)
        print(f"Manifest {args.manifest}: {len(runs)} approved cells")
    else:
        cells = enumerate_cells(args.tracks, args.seq_lens, args.pred_lens,
                                args.folds, args.strategies)
        runs = [(c, models_for_cell(c)) for c in cells]
        runs = [(c, models) for c, models in runs if models]
        print(f"Planned cells: {len(runs)} (tracks={args.tracks} "
              f"seq={args.seq_lens} pred={args.pred_lens} folds={args.folds} "
              f"strategies={args.strategies} epochs={args.epochs})")
    run_benchmark(runs, args.epochs, runs_root, master_dir,
                  dry_run=args.dry_run)


if __name__ == "__main__":
    main()
