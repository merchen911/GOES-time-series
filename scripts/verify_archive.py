#!/usr/bin/env python3.12
"""Check dumped prediction files against the v1 results (spec 4.8). No GPU needed.

    python3.12 scripts/verify_archive.py --archive runs/archive/v003/bench \\
        --v1 results/v1/results_master.csv --out runs/archive/verify_v003.csv \\
        --paper-folds ../../paper/v001/analysis/folds

--out gets one row per dumped run and v1 metric. rmse/mae must match to a
relative 1e-4; an event score passes when at most 5 flipped threshold points
explain the difference. A run on one side only fails as a "run" row.

With --paper-folds two more files are written next to --out:
<out>_paper_formulas.csv checks this package's tss/hss/csi/sedi on the paper's
own counts (a pass condition), and <out>_paper_counts.csv lists the count
differences dump - paper (reference only).

The exit code is 1 when any row fails, 0 otherwise.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:    # run as a plain script from any folder
    sys.path.insert(0, str(REPO))

from tslib.benchmark import verify_archive as va  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Check dumped prediction files against the v1 results.")
    p.add_argument("--archive", required=True, metavar="BENCH",
                   help="dumped run tree (<cell>/pred/<model>.npz)")
    p.add_argument("--v1", required=True, metavar="CSV",
                   help="v1 master table of the same runs (results_master.csv)")
    p.add_argument("--out", required=True, metavar="CSV", help="report to write")
    p.add_argument("--paper-folds", metavar="DIR",
                   help="the paper's event_skill{,_p288}_fold<k>.csv tables: also "
                        "check the formulas on them and compare their counts")
    return p.parse_args(argv)


def _write(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _failures(frame, by: str) -> int:
    """Print the failing rows per value of `by`; return their number."""
    failed = frame.loc[~frame.ok.astype(bool), by]
    for value, n in failed.value_counts(sort=False).sort_index().items():
        print(f"  failing {by} {value}: {n}")
    return len(failed)


def _paper(args, out: Path) -> int:
    formulas = va.check_formulas_against_paper(Path(args.paper_folds))
    formulas_path = out.with_name(f"{out.stem}_paper_formulas.csv")
    _write(formulas, formulas_path)
    n_failed = int((~formulas.ok).sum())
    print(f"paper formulas: rows: {len(formulas)}, failing rows: {n_failed} "
          f"-> {formulas_path}")
    _failures(formulas, "metric")

    counts = va.compare_paper_counts(Path(args.archive), Path(args.paper_folds))
    counts_path = out.with_name(f"{out.stem}_paper_counts.csv")
    _write(counts, counts_path)
    largest = counts[va.COUNT_DIFFS].abs().max(axis=1)
    spread = ", ".join(f"{int(d)}: {n}" for d, n in
                       largest.dropna().value_counts().sort_index().items())
    print(f"paper counts (reference only, not a pass condition): rows: {len(counts)}, "
          f"without a prediction file: {int(largest.isna().sum())} -> {counts_path}")
    print(f"  rows by largest |count difference|: {spread or 'none'}")
    return n_failed


def main(argv=None) -> int:
    args = parse_args(argv)
    archive, v1, out = Path(args.archive), Path(args.v1), Path(args.out)
    if not archive.is_dir():
        print(f"error: --archive {archive} is not a folder", file=sys.stderr)
        return 1
    if not v1.is_file():
        print(f"error: --v1 {v1} is not a file", file=sys.stderr)
        return 1
    if args.paper_folds and not Path(args.paper_folds).is_dir():
        print(f"error: --paper-folds {args.paper_folds} is not a folder", file=sys.stderr)
        return 1
    try:
        report = va.verify(archive, v1)
        _write(report, out)
        n_runs = len(report[va.RUN_KEYS].drop_duplicates())
        n_failed = int((~report.ok).sum())
        print(f"runs: {n_runs}, rows: {len(report)}, failing rows: {n_failed} -> {out}")
        _failures(report, "metric")
        if args.paper_folds:
            n_failed += _paper(args, out)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
