import os, tempfile, unittest
import numpy as np
import pandas as pd
from data.loader import _read_table, _valid_starts

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
