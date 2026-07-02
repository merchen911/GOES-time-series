"""Count trainable sliding-window samples per fold / split, for adjustable
history (seq) and forecast (pred) lengths.

5-min cadence => 288 steps/day (legacy hourly was 24/day; factor x12).
A trainable sample = one sliding window of length L = (hist+pred)*288 that
(1) lies entirely within a single half-year term (strict split, no boundary
crossing), (2) sits on a regular 5-min grid, and (3) has NO missing value in
the whole window for the target (legacy nonvalue2nan rule, option 1).

Efficiency: each dataset's per-term grid-validity boolean is built ONCE, then
windows are counted for every (hist, pred) combo by run-length arithmetic.

Usage:
    python3.12 count_fold_samples.py                       # sweep + per-fold
    python3.12 count_fold_samples.py --hist-days 7 5 3 2 1 --pred-days 1 3
    python3.12 count_fold_samples.py --hist-days 3 --pred-days 1   # single config
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import term_split as ts

STEP = pd.Timedelta("5min")
DAY = 24 * 12  # 288 five-minute steps per day

PROC = "../../data/goes_data/processed"
PART = f"{PROC}/kasi_swpc_particle_5m_v02.parquet"
XRAY = f"{PROC}/kasi_swpc_xray_1m_v02.parquet"

# Each config is a list of (parquet, target) channels. Multivariate configs
# require ALL channels present (AND) across the whole window on a common 5-min
# grid. xray (native 1-min) is aligned to the 5-min grid by exact-mark reindex.
CONFIGS = {
    "particle (univar p_gt10)":      [(PART, "p_gt10")],
    "xray (univar xrs_long)":        [(XRAY, "xrs_long")],
    "multivar (p_gt10 + xrs_long)":  [(PART, "p_gt10"), (XRAY, "xrs_long")],
}


def windows_in_run(valid: np.ndarray, L: int) -> int:
    """# of length-L all-valid windows: sum over True-runs of max(0, run-L+1)."""
    if valid.size == 0:
        return 0
    padded = np.concatenate([[0], valid.astype(np.int8), [0]])
    diff = np.diff(padded)
    runs = np.where(diff == -1)[0] - np.where(diff == 1)[0]
    return int(np.maximum(0, runs - L + 1).sum())


def _channel_series(parquet: str, target: str, role: str) -> pd.Series:
    """Primary-role target as a Series indexed by tz-aware time_utc."""
    df = pq.read_table(parquet, columns=["time_utc", "role", target]).to_pandas()
    df = df.loc[df["role"] == role, ["time_utc", target]]
    df["time_utc"] = pd.to_datetime(df["time_utc"])
    df = df.dropna(subset=["time_utc"]).drop_duplicates("time_utc", keep="last")
    return df.set_index("time_utc")[target].sort_index()


def load_validity(channels: list[tuple[str, str]], role: str = "primary"
                  ) -> dict[str, np.ndarray]:
    """term -> bool array: ALL channels present on the common 5-min grid.

    For 1 channel this is the univariate case. For >1, channels are outer-joined
    on their timestamps; reindexing each term to a regular 5-min grid aligns the
    native-1-min xray onto the 5-min marks, and validity is the AND across
    channels (a multivariate window needs every channel non-missing).
    """
    cols = [f"c{i}" for i in range(len(channels))]
    series = [_channel_series(p, t, role).rename(c)
              for c, (p, t) in zip(cols, channels)]
    joined = pd.concat(series, axis=1, join="outer").sort_index()
    joined["_term"] = ts.add_term(joined.reset_index(), time_col="time_utc",
                                  split_type="year_half")["_term"].to_numpy()
    out: dict[str, np.ndarray] = {}
    for term, g in joined.groupby("_term"):
        grid = pd.date_range(g.index.min(), g.index.max(), freq=STEP)
        r = g[cols].reindex(grid)
        out[term] = r.notna().all(axis=1).to_numpy()
    return dict(sorted(out.items()))


def fold_split_counts(term_valid: dict[str, np.ndarray], L: int, n_fold: int = 5
                      ) -> pd.DataFrame:
    """Per-fold train/val/test trainable-window totals."""
    terms = list(term_valid)
    win = {t: windows_in_run(v, L) for t, v in term_valid.items()}
    rows = []
    for k in range(n_fold):
        fold = ts.make_fold_indices(len(terms), n_fold, k)
        row = {"fold": k}
        for split, idxs in fold.items():
            row[split] = int(sum(win[terms[i]] for i in idxs.tolist()))
        rows.append(row)
    t = pd.DataFrame(rows)
    t["total"] = t[["train", "val", "test"]].sum(axis=1)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist-days", type=float, nargs="+", default=[7, 5, 3, 2, 1])
    ap.add_argument("--pred-days", type=float, nargs="+", default=[1, 3])
    args = ap.parse_args()

    print(f"5-min cadence: 1 day = {DAY} steps "
          f"(legacy hourly 24/day x12)\nrule: option 1 (full window no-gap)\n")
    for name, channels in CONFIGS.items():
        tv = load_validity(channels)
        vf = np.concatenate(list(tv.values()))
        print(f"{'='*70}\n{name}  v02  role=primary  terms={len(tv)}  "
              f"common-grid valid={vf.mean()*100:.1f}%\n{'='*70}")

        # sweep table: total trainable windows by (hist, pred)
        print("total trainable windows  (rows=hist days, cols=pred days):")
        hdr = "  hist\\pred " + "".join(f"{p:>14g}d" for p in args.pred_days)
        print(hdr)
        for h in args.hist_days:
            cells = []
            for p in args.pred_days:
                L = int(round((h + p) * DAY))
                cells.append(sum(windows_in_run(v, L) for v in tv.values()))
            print(f"  {h:>8g}d  " + "".join(f"{c:>15,}" for c in cells))

        # per-fold detail for each pred at the SHORTEST and 7d hist for comparison
        focus = sorted({min(args.hist_days), 3.0, 7.0} & set(args.hist_days))
        for h in focus:
            for p in args.pred_days:
                L = int(round((h + p) * DAY))
                ft = fold_split_counts(tv, L)
                print(f"\n  -- hist {h:g}d (={int(h*DAY)}) + pred {p:g}d  "
                      f"=> L={L} ({L/DAY:g}d no-gap)")
                print("   " + ft.to_string(index=False).replace("\n", "\n   "))
        print()


if __name__ == "__main__":
    main()
