import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tslib.benchmark import driver


class TestEnumerateCells(unittest.TestCase):
    def test_direct_fold0_full_sweep_count(self):
        cells = driver.enumerate_cells(
            tracks=list(driver.TRACKS), seq_lens=driver.SEQ_LENS,
            pred_lens=driver.PRED_LENS, folds=[0], strategies=["direct"])
        # 3 tracks x 3 seq x 2 pred x 1 fold x 1 strategy = 18 (all direct-applicable)
        self.assertEqual(len(cells), 18)
        self.assertTrue(all(c["strategy"] == "direct" for c in cells))

    def test_recursive_excludes_multi(self):
        cells = driver.enumerate_cells(
            tracks=list(driver.TRACKS), seq_lens=[288], pred_lens=[144],
            folds=[0], strategies=["recursive"])
        # only uni_a and uni_b have a recursive model; multi is dropped
        tracks = {c["track"] for c in cells}
        self.assertEqual(tracks, {"uni_a", "uni_b"})
        self.assertEqual(len(cells), 2)


class TestBuildCmd(unittest.TestCase):
    def test_uni_a_direct_flags(self):
        cell = {"track": "uni_a", "seq_len": 288, "pred_len": 144,
                "fold": 0, "strategy": "direct"}
        cmd = driver.build_cmd(cell, epochs=30)
        self.assertIn("--forecast_strategy", cmd)
        self.assertEqual(cmd[cmd.index("--forecast_strategy") + 1], "direct")
        # direct univariate model set (timesnet excluded)
        mi = cmd.index("--models")
        self.assertEqual(cmd[mi + 1:mi + 4],
                         ["patchtst", "itransformer", "timemixer"])
        self.assertEqual(cmd[cmd.index("--event_threshold") + 1], "10")
        self.assertEqual(cmd[cmd.index("--seq_len") + 1], "288")
        self.assertEqual(cmd[cmd.index("--pred_len") + 1], "144")
        self.assertEqual(cmd[cmd.index("--fold_numb") + 1], "0")
        self.assertEqual(cmd[cmd.index("--epochs") + 1], "30")
        self.assertEqual(cmd[cmd.index("--run_name") + 1],
                         "bench/uni_a_seq288_pred144_f0_direct")
        # gate flags removed; early-stop present
        self.assertNotIn("--max_train_hours", cmd)
        self.assertEqual(cmd[cmd.index("--early_stop_patience") + 1], "10")
        self.assertEqual(cmd[cmd.index("--epochs") + 1], "30")
        # univariate has no channels
        self.assertNotIn("--channels", cmd)

    def test_multi_direct_has_channels_and_two_thresholds(self):
        cell = {"track": "multi", "seq_len": 864, "pred_len": 288,
                "fold": 0, "strategy": "direct"}
        cmd = driver.build_cmd(cell, epochs=30)
        self.assertIn("--channels", cmd)
        ti = cmd.index("--target_cols")
        self.assertEqual(cmd[ti + 1:ti + 3], ["p_gt10", "xrs_long"])
        ei = cmd.index("--event_threshold")
        self.assertEqual(cmd[ei + 1:ei + 3], ["10", "1e-5"])
        mi = cmd.index("--models")
        self.assertEqual(cmd[mi + 1:mi + 5],
                         ["lstm", "patchtst", "itransformer", "timemixer"])


class TestRebuildMaster(unittest.TestCase):
    def test_attaches_columns_and_is_idempotent(self):
        cell = {"track": "uni_a", "seq_len": 288, "pred_len": 144,
                "fold": 0, "strategy": "direct"}
        with tempfile.TemporaryDirectory() as d:
            runs_root = Path(d) / "runs"
            cp = driver.comparison_path(cell, runs_root)
            cp.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"strategy": "direct", "model": "patchtst",
                           "best_val_loss": 0.2, "mse": 0.1}]).to_csv(
                cp, index=False)
            master = runs_root / "bench" / "results_master.csv"
            n1 = driver.rebuild_master([cell], runs_root, master)
            n2 = driver.rebuild_master([cell], runs_root, master)  # idempotent
            self.assertEqual((n1, n2), (1, 1))
            df = pd.read_csv(master)
            self.assertEqual(len(df), 1)  # no duplicate on second rebuild
            for col in ("track", "seq_len", "pred_len", "fold"):
                self.assertIn(col, df.columns)
            self.assertEqual(df.loc[0, "track"], "uni_a")
            self.assertEqual(df.loc[0, "model"], "patchtst")

    def test_skips_missing_comparison(self):
        cell = {"track": "uni_b", "seq_len": 288, "pred_len": 144,
                "fold": 0, "strategy": "direct"}
        with tempfile.TemporaryDirectory() as d:
            runs_root = Path(d) / "runs"
            master = runs_root / "bench" / "results_master.csv"
            n = driver.rebuild_master([cell], runs_root, master)
            self.assertEqual(n, 0)
            self.assertFalse(master.exists())


class TestManifest(unittest.TestCase):
    def test_cells_from_manifest_groups_models(self):
        manifest = {"approved": [
            {"track": "uni_a", "seq_len": 288, "pred_len": 144, "fold": 0,
             "strategy": "direct", "model": "patchtst"},
            {"track": "uni_a", "seq_len": 288, "pred_len": 144, "fold": 0,
             "strategy": "direct", "model": "timemixer"},
            {"track": "uni_b", "seq_len": 288, "pred_len": 144, "fold": 0,
             "strategy": "direct", "model": "patchtst"},
        ]}
        out = driver.cells_from_manifest(manifest)
        self.assertEqual(len(out), 2)  # two distinct cells
        cell0, models0 = out[0]
        self.assertEqual(cell0["track"], "uni_a")
        self.assertEqual(models0, ["patchtst", "timemixer"])

    def test_build_cmd_honours_explicit_models(self):
        cell = {"track": "uni_a", "seq_len": 288, "pred_len": 144,
                "fold": 0, "strategy": "direct"}
        cmd = driver.build_cmd(cell, epochs=30, models=["timemixer"])
        mi = cmd.index("--models")
        self.assertEqual(cmd[mi + 1], "timemixer")
        self.assertNotIn("patchtst", cmd)


class TestCellComplete(unittest.TestCase):
    def test_complete_only_when_all_models_present(self):
        cell = {"track": "uni_a", "seq_len": 288, "pred_len": 144,
                "fold": 0, "strategy": "direct"}
        with tempfile.TemporaryDirectory() as d:
            runs_root = Path(d) / "runs"
            self.assertFalse(driver.cell_is_complete(cell, ["patchtst"], runs_root))
            cp = driver.comparison_path(cell, runs_root)
            cp.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"model": "patchtst"}, {"model": "timemixer"}]).to_csv(cp, index=False)
            self.assertTrue(driver.cell_is_complete(cell, ["patchtst"], runs_root))
            self.assertTrue(driver.cell_is_complete(cell, ["patchtst", "timemixer"], runs_root))
            self.assertFalse(driver.cell_is_complete(cell, ["itransformer"], runs_root))


if __name__ == "__main__":
    unittest.main()
