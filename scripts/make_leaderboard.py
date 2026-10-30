#!/usr/bin/env python3.12
"""Write the per-horizon leaderboard into README.md, between the LEADERBOARD
markers, from the two v2 result tables.

Run from anywhere:  python3.12 scripts/make_leaderboard.py
Other inputs:       --regression CSV  --event CSV  --readme MD
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:    # run as a plain script from any folder
    sys.path.insert(0, str(REPO))

from tslib.benchmark.leaderboard import leaderboard_markdown  # noqa: E402

START, END = "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->"
INTRO = (
    "_Each cell is the mean of a model's runs at that forecast length, over all "
    "folds and input lengths unless a \u2020 footnote says otherwise; runs with an "
    "undefined value are left out of the mean. `k/n` marks a cell where only k of "
    "the n runs of that forecast length (folds x input lengths) have a value, and "
    "\u2020 marks a model with fewer runs than that, whose rows are not directly "
    "comparable with the others (see the note under the section). RMSE (log10 flux) and "
    "MSA (median symmetric accuracy, %) are better lower; logR (correlation of the "
    "log10 values), HSS and TSS are better higher. HSS and TSS are per-day scores "
    "(one forecast per 24 h) at the M-class (SXR) and 10 pfu (proton) thresholds. "
    "Rows are ordered by the first channel's RMSE. Per-run results: "
    "[`results/results_regression.csv`](results/results_regression.csv), "
    "[`results/results_event.csv`](results/results_event.csv)._")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Write the per-horizon leaderboard into README.md.")
    p.add_argument("--regression", type=Path, default=REPO / "results" / "results_regression.csv",
                   metavar="CSV", help="regression result table (default: %(default)s)")
    p.add_argument("--event", type=Path, default=REPO / "results" / "results_event.csv",
                   metavar="CSV", help="event result table (default: %(default)s)")
    p.add_argument("--readme", type=Path, default=REPO / "README.md",
                   metavar="MD", help="file with the LEADERBOARD markers (default: %(default)s)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    for path in (args.regression, args.event, args.readme):
        if not path.is_file():
            print(f"error: {path} is not a file", file=sys.stderr)
            return 1
    text = args.readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print(f"error: {args.readme} has no {START} ... {END} block", file=sys.stderr)
        return 1
    body = leaderboard_markdown(pd.read_csv(args.regression), pd.read_csv(args.event))
    pre, rest = text.split(START, 1)
    _, post = rest.split(END, 1)
    args.readme.write_text(f"{pre}{START}\n\n{INTRO}\n\n{body}\n\n{END}{post}",
                           encoding="utf-8")
    print(f"leaderboard written to {args.readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
