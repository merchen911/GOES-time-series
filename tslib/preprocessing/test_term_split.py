"""Tests for term_split: tagging correctness + strict (leakage-free) split."""
import unittest

import numpy as np
import pandas as pd

from tslib.preprocessing import term_split as ts


def _toy(start="1998-01-01", periods=57 * 100, freq="3D"):
    # ~57 half-years of coverage so n_fold=5 is satisfiable
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame({"time_utc": idx, "v": np.arange(periods)})


class TestAddTerm(unittest.TestCase):
    def test_year_half_boundary(self):
        df = pd.DataFrame({"time_utc": pd.to_datetime(
            ["2003-06-30T23:59", "2003-07-01T00:00", "2020-01-01T00:00"], utc=True)})
        out = ts.add_term(df, split_type="year_half")
        self.assertEqual(list(out["_term"]), ["2003-H1", "2003-H2", "2020-H1"])

    def test_year(self):
        df = pd.DataFrame({"time_utc": pd.to_datetime(["2003-07-01T00:00"], utc=True)})
        self.assertEqual(ts.add_term(df, split_type="year")["_term"].iloc[0], "2003")

    def test_bad_split_type(self):
        with self.assertRaises(ValueError):
            ts.add_term(_toy(), split_type="quarter")


class TestFolds(unittest.TestCase):
    def test_partition_no_overlap(self):
        f = ts.make_fold_indices(57, 5, 0)
        allidx = np.concatenate([f["train"], f["val"], f["test"]])
        self.assertEqual(sorted(allidx.tolist()), list(range(57)))  # covers all, no dup

    def test_too_few_terms(self):
        with self.assertRaises(ValueError):
            ts.make_fold_indices(3, 5, 0)

    def test_every_term_tested_across_folds(self):
        n = 57
        tested = set()
        for k in range(5):
            tested |= set(ts.make_fold_indices(n, 5, k)["test"].tolist())
        self.assertEqual(tested, set(range(n)))  # rotation covers all terms


class TestStrictSplit(unittest.TestCase):
    def setUp(self):
        self.df = ts.assign_split(_toy(), n_fold=5, fold_numb=0)

    def test_no_term_shared_across_splits(self):
        # the core guarantee: each half-year term lives in exactly one split
        g = self.df.groupby("_term")["_split"].nunique()
        self.assertTrue((g == 1).all(), "a term leaked across splits")

    def test_every_row_assigned(self):
        self.assertFalse(self.df["_split"].isna().any())
        self.assertEqual(set(self.df["_split"].unique()), {"train", "val", "test"})

    def test_split_frames_disjoint_and_complete(self):
        parts = ts.split_frames(_toy(), n_fold=5, fold_numb=0)
        total = sum(len(v) for v in parts.values())
        self.assertEqual(total, len(_toy()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
