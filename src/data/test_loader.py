import os, tempfile, unittest
import numpy as np
import pandas as pd
import torch
from data.loader import _read_table, _valid_starts, _term_labels, _prepare_series, _grid_and_starts, WindowDataset

PROC = "/NAS/ioGuard3/vol3/spaceai/SW_framework/data/goes_data/processed"
PARTICLE = os.path.join(PROC, "kasi_swpc_particle_5m_v02.parquet")


class TestValidStarts(unittest.TestCase):
    def test_all_valid(self):
        v = np.ones(10, dtype=bool)
        np.testing.assert_array_equal(_valid_starts(v, 3), np.arange(8))

    def test_gap_excludes_spanning_windows(self):
        v = np.ones(10, dtype=bool); v[5] = False
        # windows of len 3 not covering index 5: starts 0,1,2 and 6,7
        np.testing.assert_array_equal(_valid_starts(v, 3), np.array([0, 1, 2, 6, 7]))

    def test_too_short(self):
        self.assertEqual(len(_valid_starts(np.ones(2, dtype=bool), 3)), 0)


class TestReadTable(unittest.TestCase):
    def test_parquet_columns(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.parquet")
            pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}).to_parquet(p)
            df = _read_table(p, columns=["a", "c"])
            self.assertEqual(list(df.columns), ["a", "c"])


class TestTermLabels(unittest.TestCase):
    def test_year_half(self):
        idx = pd.to_datetime(["2003-06-30T23:59", "2003-07-01T00:00", "2020-01-01"], utc=True, format='mixed')
        np.testing.assert_array_equal(
            _term_labels(idx, "year_half"), np.array(["2003-H1", "2003-H2", "2020-H1"]))

    def test_year(self):
        idx = pd.to_datetime(["2003-07-01"], utc=True)
        np.testing.assert_array_equal(_term_labels(idx, "year"), np.array(["2003"]))


class TestPrepareSeries(unittest.TestCase):
    def _df(self):
        return pd.DataFrame({
            "time_utc": pd.to_datetime(
                ["2020-01-01T00:00", "2020-01-01T00:00", "2020-01-01T00:05"], utc=True),
            "role": ["primary", "secondary", "primary"],
            "p_gt10": [10.0, 999.0, 100.0],
        })

    def test_role_filter_dedup_and_log10(self):
        s = _prepare_series(self._df(), "time_utc", "p_gt10", "primary", "log10")
        self.assertEqual(len(s), 2)               # secondary dropped
        self.assertAlmostEqual(float(s.iloc[0]), 1.0)   # log10(10)
        self.assertAlmostEqual(float(s.iloc[1]), 2.0)   # log10(100)

    def test_transform_none_identity(self):
        s = _prepare_series(self._df(), "time_utc", "p_gt10", "primary", "none")
        self.assertAlmostEqual(float(s.iloc[0]), 10.0)


class TestGridAndStarts(unittest.TestCase):
    def _series(self, times, vals):
        return pd.Series(vals, index=pd.to_datetime(times, utc=True), dtype="float64")

    def test_gap_breaks_windows(self):
        # 6 points at 5-min, but one missing -> grid has a NaN hole
        t = ["2020-01-01T00:00", "2020-01-01T00:05", "2020-01-01T00:15",
             "2020-01-01T00:20", "2020-01-01T00:25", "2020-01-01T00:30"]
        s = self._series(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 00:10 missing
        # grid 00:00..00:30 = 7 slots; slot idx2 (00:10) is NaN
        values, starts = _grid_and_starts(s, ["2020-H1"], 5, 2, 1, "year_half")
        self.assertEqual(len(values), 7)
        self.assertTrue(np.isnan(values[2]))
        # L=3 windows avoiding idx2: starts 0? no (covers 0,1,2). valid: 3,4 (->len? n=5)
        np.testing.assert_array_equal(starts, np.array([3, 4]))

    def test_no_cross_term_boundary(self):
        # two terms; each fully dense; window must not span the join
        t1 = pd.date_range("2020-06-29", periods=4, freq="5min", tz="UTC")   # H1
        t2 = pd.date_range("2020-07-01", periods=4, freq="5min", tz="UTC")   # H2
        s = pd.Series(np.arange(8.0),
                      index=t1.append(t2), dtype="float64")
        values, starts = _grid_and_starts(s, ["2020-H1", "2020-H2"], 5, 2, 1, "year_half")
        L = 3
        # term grids are length 4 each, concatenated -> offsets 0 and 4
        # valid starts per term: [0,1], second term +4 -> [4,5]
        np.testing.assert_array_equal(starts, np.array([0, 1, 4, 5]))


class TestWindowDataset(unittest.TestCase):
    def test_shapes_and_values(self):
        values = np.arange(10.0)
        ds = WindowDataset(values, np.array([0, 2]), seq_len=2, pred_len=1)
        self.assertEqual(len(ds), 2)
        x, y = ds[1]                      # start=2, L=3 -> [2,3,4]
        self.assertEqual(tuple(x.shape), (2, 1))
        self.assertEqual(tuple(y.shape), (1, 1))
        self.assertEqual(x.dtype, torch.float32)
        np.testing.assert_array_equal(x.squeeze(-1).numpy(), np.array([2.0, 3.0]))
        np.testing.assert_array_equal(y.squeeze(-1).numpy(), np.array([4.0]))


from types import SimpleNamespace
from data.loader import DataModule


def _cfg(**kw):
    base = dict(data_path=PARTICLE, target_col="p_gt10", time_col=None,
                role="primary", transform="log10", cadence_min=5,
                split_type="year_half", n_fold=5, fold_numb=0,
                seq_len=288, pred_len=12, batch_size=32, shuffle_train=False,
                num_workers=0, feature_cols=None,
                train_ratio=0.7, val_ratio=0.15)
    base.update(kw)
    return SimpleNamespace(**base)


@unittest.skipUnless(os.path.exists(PARTICLE), "real parquet not present")
class TestSetupParquet(unittest.TestCase):
    def test_bundle_shapes_and_nonempty(self):
        b = DataModule(_cfg()).setup()
        self.assertEqual(b.input_size, 1)
        self.assertEqual(b.target_index, 0)
        for loader in (b.train_loader, b.val_loader, b.test_loader):
            self.assertGreater(len(loader.dataset), 0)
        xb, yb = next(iter(b.train_loader))
        self.assertEqual(tuple(xb.shape[1:]), (288, 1))
        self.assertEqual(tuple(yb.shape[1:]), (12, 1))

    def test_leakage_free_terms(self):
        from data.loader import _read_table, _prepare_series, _term_labels, _fold_indices
        c = _cfg()
        s = _prepare_series(_read_table(c.data_path, [ "time_utc", "role", c.target_col]),
                            "time_utc", c.target_col, c.role, c.transform)
        terms = sorted(pd.unique(_term_labels(s.index, c.split_type)).tolist())
        fold = _fold_indices(len(terms), c.n_fold, c.fold_numb)
        sets = [set(idx.tolist()) for idx in fold.values()]
        self.assertEqual(sum(len(x) for x in sets), len(terms))   # partition
        self.assertEqual(len(sets[0] & sets[1]), 0)
        self.assertEqual(len(sets[0] & sets[2]), 0)
        self.assertEqual(len(sets[1] & sets[2]), 0)
