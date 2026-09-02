#!/usr/bin/env python3.12
"""Build manifest/designation_table.csv (proton rows) — spec Phase 1."""
from __future__ import annotations

import argparse
import os
import sys

from designation_table import assemble_rows, validate_rows, write_csv
from designation_sources import (
    GLOBAL_COVERAGE_START, historical_cited_intervals, read_current_designation,
)

DEFAULT_ROOT = os.environ.get("SW_DATA_ROOT", "data/goes_data")


def build(root: str) -> list:
    swpc_dir = os.path.join(root, "raw", "proton", "swpc")
    current = read_current_designation(swpc_dir)
    cited = historical_cited_intervals()
    rows = assemble_rows(GLOBAL_COVERAGE_START, current, cited)
    validate_rows(rows)
    manifest_dir = os.path.join(root, "manifest")
    os.makedirs(manifest_dir, exist_ok=True)
    out = os.path.join(manifest_dir, "designation_table.csv")
    write_csv(rows, out)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="goes_data root")
    args = ap.parse_args(argv)
    rows = build(args.root)
    n_unknown = sum(1 for r in rows if r.primary_sat == "unknown")
    print(f"designation_table.csv: {len(rows)} proton rows "
          f"({n_unknown} unknown), current primary={rows[-1].primary_sat} "
          f"secondary={rows[-1].secondary_sat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
