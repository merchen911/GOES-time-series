"""Half-year (term) tagging + strict time-series train/val/test split.

The KASI SWPC parquet datasets (goes_data/processed/kasi_swpc_*.parquet) are
NOT rewritten. Instead we read the `time_utc` column and derive a period tag
("_term") in memory at load time -- cheap on I/O, no duplicated 100+ MB files.

Splitting is done at the **term** granularity, never per row: a whole half-year
block goes entirely to train, val, or test. Because no half-year period is ever
shared across splits, any windowing performed later within a (series, term)
block cannot leak across the train/val/test boundary.

term granularity:
  "year"       -> "2003"               (one tag per calendar year)
  "year_half"  -> "2003-H1"/"2003-H2"  (Jan-Jun = H1, Jul-Dec = H2)  [default]

The fold assignment reproduces the legacy rotating k-fold exactly
(legacy/2026/src/utils/data_prepare.py::FoldSplitter, and the v001 reimpl
src/data/loader.py::_fold_indices): terms are sorted chronologically, then
interleaved with stride = n_fold so each split spans the full timeline
(all solar-cycle phases) rather than one contiguous era.

    fold i:
        val  = terms[(n_fold-2+i) % n_fold :: n_fold]
        test = terms[(n_fold-1+i) % n_fold :: n_fold]
        train = the rest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIME_COL = "time_utc"
TERM_COL = "_term"
SPLIT_COL = "_split"


def add_term(df: pd.DataFrame, *, time_col: str = TIME_COL,
             split_type: str = "year_half") -> pd.DataFrame:
    """Return a copy of `df` with a `_term` column derived from `time_col`.

    Mirrors legacy add_halfyear_group / _add_term_column. `time_col` may be
    tz-aware (the KASI parquet is UTC) -- only year/month are used.
    """
    out = df.copy()
    t = pd.to_datetime(out[time_col])
    year = t.dt.year.astype("Int64").astype(str)
    if split_type == "year":
        out[TERM_COL] = year.astype("string")
    elif split_type == "year_half":
        half = np.where(t.dt.month <= 6, "H1", "H2")
        out[TERM_COL] = (year + "-" + half).astype("string")
    else:
        raise ValueError(f"unsupported split_type: {split_type!r} "
                         "(use 'year' or 'year_half')")
    return out


def make_fold_indices(n_term: int, n_fold: int, fold_numb: int) -> dict[str, np.ndarray]:
    """Rotating k-fold term-index assignment (identical math to legacy)."""
    if n_term < n_fold:
        raise ValueError(f"n_term={n_term} < n_fold={n_fold}: not enough terms")
    if not 0 <= fold_numb < n_fold:
        raise ValueError(f"fold_numb must be in [0, {n_fold}); got {fold_numb}")
    rng = np.arange(n_term)
    vl = rng[(n_fold - 2 + fold_numb) % n_fold:: n_fold]
    ts = rng[(n_fold - 1 + fold_numb) % n_fold:: n_fold]
    tr = np.setdiff1d(rng, np.concatenate([vl, ts]))
    return {"train": tr, "val": vl, "test": ts}


def assign_split(df: pd.DataFrame, *, n_fold: int = 5, fold_numb: int = 0,
                 split_type: str = "year_half", time_col: str = TIME_COL,
                 keep_term: bool = True) -> pd.DataFrame:
    """Tag every row with `_split` in {train, val, test} by its half-year term.

    Whole terms are assigned to a split, so the boundary is leakage-free.
    Returns a copy with `_split` (and `_term` if keep_term) added.
    """
    out = add_term(df, time_col=time_col, split_type=split_type)
    terms = sorted(out[TERM_COL].dropna().unique().tolist())
    fold = make_fold_indices(len(terms), n_fold, fold_numb)
    term2split: dict[str, str] = {}
    for split_name, idxs in fold.items():
        for i in idxs.tolist():
            term2split[terms[i]] = split_name
    out[SPLIT_COL] = out[TERM_COL].map(term2split).astype("string")
    if not keep_term:
        out = out.drop(columns=[TERM_COL])
    return out


def split_frames(df: pd.DataFrame, **kwargs) -> dict[str, pd.DataFrame]:
    """Convenience: return {'train','val','test': sub-DataFrame} for one fold."""
    tagged = assign_split(df, **kwargs)
    return {name: tagged.loc[tagged[SPLIT_COL] == name].copy()
            for name in ("train", "val", "test")}


def term_distribution(parquet_path: str, *, split_type: str = "year_half",
                      time_col: str = TIME_COL) -> pd.DataFrame:
    """I/O-light: read ONLY `time_utc` and return per-term row counts."""
    import pyarrow.parquet as pq
    t = pq.read_table(parquet_path, columns=[time_col]).to_pandas()
    t = add_term(t, time_col=time_col, split_type=split_type)
    return (t.groupby(TERM_COL).size().rename("rows")
            .reset_index().sort_values(TERM_COL).reset_index(drop=True))


def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet")
    ap.add_argument("--split-type", default="year_half", choices=["year", "year_half"])
    ap.add_argument("--n-fold", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    dist = term_distribution(args.parquet, split_type=args.split_type)
    terms = dist[TERM_COL].tolist()
    fold = make_fold_indices(len(terms), args.n_fold, args.fold)
    term2split = {terms[i]: name for name, idxs in fold.items() for i in idxs.tolist()}
    dist[SPLIT_COL] = dist[TERM_COL].map(term2split)

    print(f"# {args.parquet}")
    print(f"# split_type={args.split_type} n_fold={args.n_fold} fold={args.fold}")
    print(f"# terms={len(terms)}  rows={int(dist['rows'].sum()):,}")
    by = dist.groupby(SPLIT_COL).agg(terms=("rows", "size"), rows=("rows", "sum"))
    tot = int(dist["rows"].sum())
    for name in ("train", "val", "test"):
        if name in by.index:
            r = int(by.loc[name, "rows"])
            print(f"  {name:5s}: {int(by.loc[name,'terms']):3d} terms  "
                  f"{r:>12,} rows  ({r/tot*100:5.1f}%)")
    print("\n# term -> split")
    for _, row in dist.iterrows():
        print(f"  {row[TERM_COL]}  {int(row['rows']):>10,}  {row[SPLIT_COL]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
