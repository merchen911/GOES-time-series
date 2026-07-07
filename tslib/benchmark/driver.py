"""Official v003 benchmark driver.

Orchestrates the sweep from ``docs/benchmark-conditions.md`` on the v003
Lightning pipeline. Each (track, seq_len, pred_len, fold, strategy) cell is run
as an **isolated** ``main.py`` subprocess (so one run's OOM/crash cannot kill the
whole sweep, and GPU memory is released between runs). After each cell, every
cell's ``comparison.csv`` that exists is concatenated — with track/seq_len/
pred_len/fold columns attached — into one master table so results accumulate
one cell at a time.

Scope defaults reflect the current phase: **direct strategy only, fold 0**
(recursive and folds 1-4 are added later via ``--strategies`` / ``--folds``).
``timesnet`` is excluded from every model set (too slow — see the benchmark
conditions doc); it stays in the model registry and can still be run by hand
with ``--on_slow proceed``.

Run it (GPU 1):

    CUDA_VISIBLE_DEVICES=1 python3.12 -m tslib.benchmark.driver --folds 0 --strategies direct

Preview without executing:

    python3.12 -m tslib.benchmark.driver --dry-run
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = os.environ.get(
    "SW_DATA_DIR",
    "/NAS/ioGuard3/vol3/spaceai/SW_framework/data/goes_data/processed")
PARTICLE = f"{DATA_DIR}/kasi_swpc_particle_5m.parquet"
XRAY = f"{DATA_DIR}/kasi_swpc_xray_1m.parquet"

# Direct model sets (timesnet excluded per benchmark spec).
DIRECT_UNI = ["patchtst", "itransformer", "timemixer"]
DIRECT_MULTI = ["lstm", "patchtst", "itransformer", "timemixer"]

SEQ_LENS = [288, 864, 2016]   # 1 d / 3 d / 7 d — light first, so the first cell (and the gate's sec/batch estimate) lands fast
PRED_LENS = [144, 288]        # 0.5 d / 1 d

# Fixed conditions common to every run (docs/benchmark-conditions.md).
FIXED = [
    "--n_fold", "5",
    "--transform", "log10",
    "--metrics", "rmse", "mae", "tss", "hss", "pod", "far",
    "--batch_size", "64",
    "--lr", "1e-3",
    "--d_model", "128",
    "--num_layers", "2",
    "--dropout", "0.1",
    "--max_train_hours", "6.0",
    "--on_slow", "skip",
    "--probe_batches", "3",
]

# Per-track definitions. ``recursive_models`` is empty where recursive is not
# applicable (multivariable multi-target: targets != all channels).
TRACKS = {
    "uni_a": {
        "data_path": PARTICLE, "target_col": "p_gt10",
        "channels": None, "target_cols": None,
        "event_threshold": ["10"], "sort_metric": "tss_p_gt10",
        "direct_models": DIRECT_UNI, "recursive_models": ["lstm"],
    },
    "uni_b": {
        "data_path": XRAY, "target_col": "xrs_long",
        "channels": None, "target_cols": None,
        "event_threshold": ["1e-5"], "sort_metric": "tss_xrs_long",
        "direct_models": DIRECT_UNI, "recursive_models": ["lstm"],
    },
    "multi": {
        "data_path": PARTICLE, "target_col": "p_gt10",  # required placeholder; --channels drives loading
        "channels": [f"{PARTICLE}:p_gt10", f"{PARTICLE}:p_gt100",
                     f"{XRAY}:xrs_long", f"{XRAY}:xrs_short"],
        "target_cols": ["p_gt10", "xrs_long"],
        "event_threshold": ["10", "1e-5"], "sort_metric": "tss_p_gt10",
        "direct_models": DIRECT_MULTI, "recursive_models": [],
    },
}


def models_for(track_key: str, strategy: str) -> list[str]:
    t = TRACKS[track_key]
    return t["direct_models"] if strategy == "direct" else t["recursive_models"]


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


def build_cmd(cell, epochs: int) -> list[str]:
    """The exact ``main.py`` argv for one cell."""
    t = TRACKS[cell["track"]]
    models = models_for(cell["track"], cell["strategy"])
    run_name = run_name_for(cell["track"], cell["seq_len"], cell["pred_len"],
                            cell["fold"], cell["strategy"])
    cmd = [sys.executable, str(REPO / "main.py"),
           "--data_path", t["data_path"],
           "--target_col", t["target_col"],
           "--seq_len", str(cell["seq_len"]),
           "--pred_len", str(cell["pred_len"]),
           "--fold_numb", str(cell["fold"]),
           "--forecast_strategy", cell["strategy"],
           "--run_name", run_name,
           "--sort_metric", t["sort_metric"],
           "--epochs", str(epochs),
           "--event_threshold", *t["event_threshold"],
           "--models", *models,
           *FIXED]
    if t["channels"]:
        cmd += ["--channels", *t["channels"],
                "--target_cols", *t["target_cols"]]
    return cmd


def comparison_path(cell, runs_root: Path) -> Path:
    run_name = run_name_for(cell["track"], cell["seq_len"], cell["pred_len"],
                            cell["fold"], cell["strategy"])
    return runs_root / run_name / "score" / "comparison.csv"


def rebuild_master(cells, runs_root: Path, master_path: Path) -> int:
    """Concatenate every cell's comparison.csv that exists into the master
    table, prefixing track/seq_len/pred_len/fold columns. Idempotent: rewrites
    the master from scratch each call, so re-runs never duplicate rows.
    Returns the number of cells folded in."""
    frames = []
    for cell in cells:
        cp = comparison_path(cell, runs_root)
        if not cp.exists():
            continue
        df = pd.read_csv(cp)
        df.insert(0, "track", cell["track"])
        df.insert(1, "seq_len", cell["seq_len"])
        df.insert(2, "pred_len", cell["pred_len"])
        df.insert(3, "fold", cell["fold"])
        frames.append(df)
    if frames:
        master_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(master_path, index=False)
    return len(frames)


def run_benchmark(cells, epochs: int, runs_root: Path, master_path: Path,
                  dry_run: bool = False) -> None:
    total = len(cells)
    env = os.environ.copy()
    for i, cell in enumerate(cells, 1):
        run_name = run_name_for(cell["track"], cell["seq_len"],
                                cell["pred_len"], cell["fold"], cell["strategy"])
        cmd = build_cmd(cell, epochs)
        cp = comparison_path(cell, runs_root)
        if dry_run:
            print(f"[{i}/{total}] {run_name}")
            print("    " + " ".join(cmd))
            continue
        if cp.exists():
            print(f"[{i}/{total}] SKIP (done): {run_name}")
            continue
        print(f"[{i}/{total}] RUN: {run_name}\n    {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO), env=env)
        if proc.returncode != 0:
            print(f"[{i}/{total}] FAILED (exit {proc.returncode}): {run_name} "
                  f"— continuing", flush=True)
            continue
        n = rebuild_master(cells, runs_root, master_path)
        print(f"[{i}/{total}] DONE: {run_name} — master now has {n} cells "
              f"at {master_path}", flush=True)
        if cp.exists():
            print(pd.read_csv(cp).to_string(index=False), flush=True)
    if not dry_run:
        n = rebuild_master(cells, runs_root, master_path)
        print(f"\nBenchmark sweep complete: {n}/{total} cells in {master_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="v003 benchmark sweep driver")
    p.add_argument("--tracks", nargs="+", default=list(TRACKS),
                   choices=list(TRACKS))
    p.add_argument("--seq-lens", nargs="+", type=int, default=SEQ_LENS)
    p.add_argument("--pred-lens", nargs="+", type=int, default=PRED_LENS)
    p.add_argument("--folds", nargs="+", type=int, default=[0])
    p.add_argument("--strategies", nargs="+", default=["direct"],
                   choices=["direct", "recursive"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cells = enumerate_cells(args.tracks, args.seq_lens, args.pred_lens,
                            args.folds, args.strategies)
    runs_root = REPO / "runs"
    master_path = runs_root / "bench" / "results_master.csv"
    print(f"Planned cells: {len(cells)} "
          f"(tracks={args.tracks} seq={args.seq_lens} pred={args.pred_lens} "
          f"folds={args.folds} strategies={args.strategies} epochs={args.epochs})")
    run_benchmark(cells, args.epochs, runs_root, master_path,
                  dry_run=args.dry_run)


if __name__ == "__main__":
    main()
