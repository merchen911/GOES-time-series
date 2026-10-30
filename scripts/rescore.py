#!/usr/bin/env python3.12
"""Re-score saved prediction files, and merge master tables. No GPU needed.

Re-score every cell under a bench tree from its pred/*.npz files, then rebuild
the two master tables (results_regression.csv, results_event.csv):

    python3.12 scripts/rescore.py --runs runs/archive/v003/bench \\
        --master-dir runs/archive/v003 --workers 8

Join master tables built from separate trees:

    python3.12 scripts/rescore.py --merge-masters runs/archive/v003 runs/archive/v004 \\
        --out results
"""
import argparse
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:    # run as a plain script from any folder
    sys.path.insert(0, str(REPO))

from tslib.exp.metrics import EVENT_SCORES, REGRESSION_METRICS  # noqa: E402 (light: numpy only)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Re-score saved predictions, or merge master tables.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--runs", metavar="BENCH_ROOT",
                      help="folder of cell folders (<cell>/pred/*.npz) to re-score")
    mode.add_argument("--merge-masters", nargs="+", metavar="DIR",
                      help="master-table folders to join (needs --out)")
    p.add_argument("--master-dir", metavar="DIR",
                   help="where --runs writes the master tables (default: the --runs folder)")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                   help="cells re-scored in parallel processes (default: %(default)s)")
    p.add_argument("--metrics", nargs="+", metavar="NAME",
                   choices=sorted(REGRESSION_METRICS) + sorted(EVENT_SCORES),
                   help="re-score only these metrics (the others become nan; "
                        "default: every registered metric)")
    p.add_argument("--out", metavar="DIR", help="where --merge-masters writes")
    args = p.parse_args(argv)
    if args.merge_masters and not args.out:
        p.error("--merge-masters needs --out")
    if args.out and not args.merge_masters:
        p.error("--out is only for --merge-masters")
    if args.workers < 1:
        p.error("--workers must be at least 1")
    return args


def discover_cells(bench_root: Path) -> list:
    """Cell folders of `bench_root` that hold a pred/ folder or result rows, in
    name order. A folder is a cell when `parse_cell_name` accepts its name. A
    cell with rows but no prediction files is listed too, so that it fails (its
    rows cannot be re-scored) instead of entering the master tables unchanged."""
    from tslib.benchmark.driver import parse_cell_name
    cells = []
    for d in sorted(Path(bench_root).iterdir()):
        if not ((d / "pred").is_dir() or (d / "score" / "rows").is_dir()):
            continue
        try:
            parse_cell_name(d.name)
        except ValueError:
            continue
        cells.append(d)
    return cells


def _outcomes(cells, workers, metric_names):
    """Yield `(cell, models re-scored | the exception it failed with)` as each
    cell finishes."""
    from tslib.exp.rescore import rescore_cell
    if workers <= 1:
        for cell in cells:
            try:
                outcome = rescore_cell(cell, metric_names)
            except Exception as e:  # noqa: BLE001 - one bad cell must not stop the others
                outcome = e
            yield cell, outcome
        return
    # spawn: the parent has torch loaded, and forking it into workers is unsafe
    with ProcessPoolExecutor(max_workers=min(workers, len(cells)),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(rescore_cell, cell, metric_names): cell for cell in cells}
        for future in as_completed(futures):
            try:
                outcome = future.result()
            except Exception as e:  # noqa: BLE001
                outcome = e
            yield futures[future], outcome


def _rescore(args) -> int:
    from tslib.benchmark.driver import rebuild_master_from_dirs
    bench = Path(args.runs)
    if not bench.is_dir():
        print(f"error: {bench} is not a folder", file=sys.stderr)
        return 1
    cells = discover_cells(bench)
    if not cells:
        print(f"error: no cells with prediction files or result rows under {bench}",
              file=sys.stderr)
        return 1
    n_models, failed = 0, []
    for i, (cell, outcome) in enumerate(_outcomes(cells, args.workers, args.metrics), 1):
        if isinstance(outcome, Exception):
            failed.append(cell.name)
            print(f"[{i}/{len(cells)}] FAILED {cell.name}: {outcome!r}",
                  file=sys.stderr, flush=True)
        else:
            n_models += outcome
            print(f"[{i}/{len(cells)}] {cell.name}: {outcome} models", flush=True)
    if failed:
        # a master built now would silently lack (or mix old rows for) these cells
        print(f"error: {len(failed)} of {len(cells)} cells failed; master tables "
              f"not rebuilt", file=sys.stderr)
        return 1
    master_dir = Path(args.master_dir) if args.master_dir else bench
    try:
        n_cells = rebuild_master_from_dirs(bench, master_dir)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Re-scored {n_models} models in {len(cells)} cells; master tables of "
          f"{n_cells} cells in {master_dir}")
    return 0


def _merge(args) -> int:
    from tslib.exp.rescore import merge_masters
    try:
        n_reg, n_event = merge_masters([Path(d) for d in args.merge_masters],
                                       Path(args.out))
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Merged {len(args.merge_masters)} master folders into {args.out}: "
          f"{n_reg} regression rows, {n_event} event rows")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    return _merge(args) if args.merge_masters else _rescore(args)


if __name__ == "__main__":
    sys.exit(main())
