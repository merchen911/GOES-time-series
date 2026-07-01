import os, tempfile, unittest
import numpy as np
import pandas as pd
import torch
from data.loader import _read_table, _valid_starts, _term_labels, _prepare_series, _grid_and_starts, WindowDataset
from types import SimpleNamespace as _NS
from data.loader import _parse_channels, _resolve_channels

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
    base = dict(data_path=PARTICLE, target_col="p_gt10", channels=None,
                target_cols=None, time_col=None,
                role="primary", transform="log10", cadence_min=5,
                min_bin_count=1,
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


class TestParseChannels(unittest.TestCase):
    def test_parse_ok(self):
        self.assertEqual(
            _parse_channels(["/a/b.parquet:p_gt10", "/c.parquet:xrs_long"]),
            [("/a/b.parquet", "p_gt10"), ("/c.parquet", "xrs_long")])

    def test_parse_no_colon_errors(self):
        with self.assertRaises(ValueError):
            _parse_channels(["/a/b.parquet"])

    def test_resolve_defaults_from_data_path(self):
        cfg = _NS(channels=None, target_cols=None,
                  data_path="p.parquet", target_col="p_gt10")
        ch, tgt = _resolve_channels(cfg)
        self.assertEqual(ch, [("p.parquet", "p_gt10")])
        self.assertEqual(tgt, ["p_gt10"])

    def test_resolve_explicit_and_target_subset(self):
        cfg = _NS(channels=["p.parquet:p_gt10", "x.parquet:xrs_long"],
                  target_cols=["xrs_long"], data_path=None, target_col=None)
        ch, tgt = _resolve_channels(cfg)
        self.assertEqual([c for _, c in ch], ["p_gt10", "xrs_long"])
        self.assertEqual(tgt, ["xrs_long"])

    def test_resolve_bad_target_errors(self):
        cfg = _NS(channels=["p.parquet:p_gt10"], target_cols=["nope"],
                  data_path=None, target_col=None)
        with self.assertRaises(ValueError):
            _resolve_channels(cfg)


class TestWindowDatasetMV(unittest.TestCase):
    def test_2d_values_and_target_idx(self):
        values = np.arange(20.0).reshape(10, 2)   # (G=10, C=2)
        ds = WindowDataset(values, np.array([0, 2]), seq_len=2, pred_len=1,
                           target_idx=[1])
        x, y = ds[1]                               # start=2, L=3 rows 2,3,4
        self.assertEqual(tuple(x.shape), (2, 2))   # (seq, C)
        self.assertEqual(tuple(y.shape), (1, 1))   # (pred, T=1)
        np.testing.assert_array_equal(x.numpy(), np.array([[4., 5.], [6., 7.]]))
        np.testing.assert_array_equal(y.squeeze(-1).numpy(), np.array([9.]))  # col 1 of row 4

    def test_1d_values_still_work(self):
        ds = WindowDataset(np.arange(10.0), np.array([0]), seq_len=2, pred_len=1)
        x, y = ds[0]
        self.assertEqual(tuple(x.shape), (2, 1))
        self.assertEqual(tuple(y.shape), (1, 1))


class TestDataBundleOutputSize(unittest.TestCase):
    def test_default_output_size(self):
        from data.loader import DataBundle
        b = DataBundle(train_loader=None, val_loader=None, test_loader=None,
                       input_size=2, target_index=0)
        self.assertEqual(b.output_size, 1)


class TestGridResampleMean(unittest.TestCase):
    def _frame(self, times, **cols):
        idx = pd.to_datetime(times, utc=True)
        return pd.DataFrame(cols, index=idx)

    def test_mean_then_log10(self):
        # two 1-min samples in one 5-min bin: mean(10,1000)=505 -> log10(505)
        f = self._frame(["2020-01-01T00:00", "2020-01-01T00:01"], a=[10.0, 1000.0])
        values, starts = _grid_and_starts(f, ["2020-H1"], 5, 1, 0, "year_half",
                                          transform="log10", min_bin_count=1)
        # L=1 window on a single valid bin -> one row, value log10(505)
        self.assertEqual(values.shape, (1, 1))
        self.assertAlmostEqual(float(values[0, 0]), np.log10(505.0), places=6)

    def test_min_bin_count_masks_sparse_bin(self):
        # 4 one-min samples in a 5-min bin; require >=5 -> bin becomes NaN
        f = self._frame([f"2020-01-01T00:0{m}" for m in range(4)], a=[1.0, 2.0, 3.0, 4.0])
        values, starts = _grid_and_starts(f, ["2020-H1"], 5, 1, 0, "year_half",
                                          transform="none", min_bin_count=5)
        self.assertTrue(np.isnan(values[0, 0]))
        self.assertEqual(len(starts), 0)

    def test_multichannel_and_validity(self):
        # channel b missing its second bin -> that row invalid for AND rule
        idx = pd.date_range("2020-01-01T00:00", periods=3, freq="5min", tz="UTC")
        f = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, np.nan, 3.0]}, index=idx)
        values, starts = _grid_and_starts(f, ["2020-H1"], 5, 1, 0, "year_half",
                                          transform="none", min_bin_count=1)
        self.assertEqual(values.shape, (3, 2))
        np.testing.assert_array_equal(starts, np.array([0, 2]))  # row 1 invalid


XRAY = os.path.join(PROC, "kasi_swpc_xray_1m_v02.parquet")


def _cfg_mv(**kw):
    base = dict(data_path=PARTICLE, target_col="p_gt10", channels=None,
                target_cols=None, time_col=None, role="primary",
                transform="log10", cadence_min=5, min_bin_count=1,
                split_type="year_half", n_fold=5, fold_numb=0,
                seq_len=288, pred_len=12, batch_size=32, shuffle_train=False,
                num_workers=0, feature_cols=None, train_ratio=0.7, val_ratio=0.15)
    base.update(kw)
    return SimpleNamespace(**base)


@unittest.skipUnless(os.path.exists(PARTICLE), "real parquet not present")
class TestSetupUnivariateRegression(unittest.TestCase):
    def test_counts_match_recorded(self):
        b = DataModule(_cfg_mv()).setup()
        self.assertEqual(b.input_size, 1)
        self.assertEqual(b.output_size, 1)
        self.assertEqual(len(b.train_loader.dataset), 1_076_731)
        self.assertEqual(len(b.val_loader.dataset), 370_516)
        self.assertEqual(len(b.test_loader.dataset), 350_429)


@unittest.skipUnless(os.path.exists(PARTICLE) and os.path.exists(XRAY),
                     "real parquet not present")
class TestSetupMultivar(unittest.TestCase):
    def test_two_channels_one_target(self):
        cfg = _cfg_mv(channels=[f"{PARTICLE}:p_gt10", f"{XRAY}:xrs_long"],
                      target_cols=["p_gt10"])
        b = DataModule(cfg).setup()
        self.assertEqual(b.input_size, 2)
        self.assertEqual(b.output_size, 1)
        for loader in (b.train_loader, b.val_loader, b.test_loader):
            self.assertGreater(len(loader.dataset), 0)
        xb, yb = next(iter(b.train_loader))
        self.assertEqual(tuple(xb.shape[1:]), (288, 2))
        self.assertEqual(tuple(yb.shape[1:]), (12, 1))
