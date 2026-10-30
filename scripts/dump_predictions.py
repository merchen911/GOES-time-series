#!/usr/bin/env python3.12
"""Dump test predictions from the checkpoints of an old (v003/v004) run tree.

For every <src>/<cell>/ckpt/<model>.ckpt this runs a test-only inference pass
with the cell's sweep settings and writes <out>/<cell>/pred/<model>.npz and
<out>/<cell>/score/rows/<model>.json; scripts/rescore.py then works from those
files. The source tree is only read. Models already dumped are skipped, so an
interrupted dump resumes by running the same command again.

Pick the GPU with CUDA_VISIBLE_DEVICES and split the cells over GPUs with
--shard (one process per GPU):

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python3.12 scripts/dump_predictions.py \\
          --src ../sw-framework-v003/runs/bench --out runs/archive/v003/bench \\
          --shard $i/4 > runs/archive/dump_v003_$i.log 2>&1 &
    done

--only and --models restrict the work to some cells or models (a smoke test,
a retry).
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:    # run as a plain script from any folder
    sys.path.insert(0, str(REPO))

from tslib.benchmark import archive  # noqa: E402


def _shard(text):
    try:
        return archive.parse_shard(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Dump test predictions from the checkpoints of an old run tree.")
    p.add_argument("--src", required=True, metavar="BENCH",
                   help="old run tree to read (<cell>/ckpt/<model>.ckpt)")
    p.add_argument("--out", required=True, metavar="BENCH",
                   help="where to write <cell>/pred and <cell>/score; the last "
                        "folder must be named bench (e.g. runs/archive/v003/bench)")
    p.add_argument("--shard", type=_shard, default=(0, 1), metavar="I/N",
                   help="dump only the cells k with k %% N == I, in sorted cell "
                        "order (default: 0/1, every cell)")
    p.add_argument("--only", nargs="+", metavar="CELL",
                   help="dump only these cells (folder names)")
    p.add_argument("--models", nargs="+", metavar="MODEL",
                   help="dump only these models")
    return p.parse_args(argv)


def _unknown(wanted, known):
    return sorted(set(wanted or ()) - set(known))


def main(argv=None) -> int:
    args = parse_args(argv)
    src, out = Path(args.src), Path(args.out)
    archive.check_out_bench(out)
    if src.resolve() in (out.resolve(), *out.resolve().parents):
        raise ValueError(f"--out {out} is inside the source tree {src}; the source "
                         f"runs are read-only, write the dump elsewhere")
    if not src.is_dir():
        print(f"error: {src} is not a folder", file=sys.stderr)
        return 1
    cells = archive.group_cells(archive.discover_checkpoints(src))
    if not cells:
        print(f"error: no <cell>/ckpt/<model>.ckpt under {src}", file=sys.stderr)
        return 1
    bad_cells = _unknown(args.only, (archive.cell_name(c) for c, _ in cells))
    bad_models = _unknown(args.models, (m for _, items in cells for m, _ in items))
    if bad_cells or bad_models:
        print(f"error: not in {src}: cells {bad_cells}, models {bad_models}",
              file=sys.stderr)
        return 1

    cells = archive.select_shard(cells, *args.shard)
    if args.only:
        cells = [(c, items) for c, items in cells if archive.cell_name(c) in args.only]
    if args.models:
        cells = [(c, [(m, f) for m, f in items if m in args.models])
                 for c, items in cells]
        cells = [(c, items) for c, items in cells if items]
    n_ckpt = sum(len(items) for _, items in cells)
    print(f"Shard {args.shard[0]}/{args.shard[1]}: {len(cells)} cells, {n_ckpt} "
          f"checkpoints from {src} -> {out}", flush=True)

    started, n_dumped, failed = time.monotonic(), 0, []
    for k, (cell, items) in enumerate(cells, 1):
        name, t0 = archive.cell_name(cell), time.monotonic()
        try:
            dumped = archive.dump_cell(cell, items, out)
        except Exception as e:  # noqa: BLE001 - one bad cell must not stop the others
            failed.append(name)
            print(f"[{k}/{len(cells)}] FAILED {name}: {e}", file=sys.stderr, flush=True)
            continue
        n_dumped += len(dumped)
        print(f"[{k}/{len(cells)}] {name}: {len(dumped)} dumped, "
              f"{len(items) - len(dumped)} already done "
              f"({time.monotonic() - t0:.0f} s)", flush=True)
    elapsed = time.monotonic() - started
    if failed:
        print(f"error: {len(failed)} of {len(cells)} cells failed (run the same "
              f"command again to retry): {failed}", file=sys.stderr)
        return 1
    print(f"Dumped {n_dumped} models in {len(cells)} cells in {elapsed:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
