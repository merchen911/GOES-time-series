import json
import tempfile
import unittest
from pathlib import Path

from tslib.benchmark import pretest


class TestEstimate(unittest.TestCase):
    def test_formula(self):
        self.assertAlmostEqual(pretest.estimate_train_hours(0.5, 100, 50),
                               0.5 * 100 * 50 / 3600.0)


class TestDecide(unittest.TestCase):
    def test_under_threshold_auto_approves_without_prompt(self):
        calls = []
        ok = pretest.decide(3.0, 12.0, lambda h: calls.append(h) or False)
        self.assertTrue(ok)
        self.assertEqual(calls, [])  # never prompted

    def test_over_threshold_prompts_and_honours_no(self):
        self.assertFalse(pretest.decide(20.0, 12.0, lambda h: False))

    def test_over_threshold_prompts_and_honours_yes(self):
        self.assertTrue(pretest.decide(20.0, 12.0, lambda h: True))


class TestPretestGrid(unittest.TestCase):
    def test_manifest_never_approves_an_excluded_model(self):
        # the manifest feeds `driver --manifest`, so it must hold the sweep's
        # grid: xPatch is not probed (or approved) at the 7-day input
        import contextlib
        import io
        from types import SimpleNamespace
        from unittest import mock
        written = {}
        bundle = SimpleNamespace(train_loader=[0] * 10, input_size=1, target_indices=[0])
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(pretest, "DataModule") as DM, \
                mock.patch.object(pretest, "build_model"), \
                mock.patch.object(pretest, "probe_sec_per_batch", return_value=0.01), \
                mock.patch.object(pretest, "write_manifest",
                                  lambda approved, params, out: written.update(a=approved)), \
                contextlib.redirect_stdout(io.StringIO()):
            DM.return_value.setup.return_value = bundle
            pretest.main(["--tracks", "uni_a", "--seq-lens", "864", "2016",
                          "--pred-lens", "36", "--out", str(Path(d) / "m.json")])
        approved = {(e["seq_len"], e["model"]) for e in written["a"]}
        self.assertIn((864, "xpatch"), approved)
        self.assertNotIn((2016, "xpatch"), approved)
        self.assertIn((2016, "dlinear"), approved)


class TestManifestRoundTrip(unittest.TestCase):
    def test_write_then_group(self):
        from tslib.benchmark import driver
        approved = [
            {"track": "uni_a", "seq_len": 288, "pred_len": 144, "fold": 0,
             "strategy": "direct", "model": "patchtst", "projected_hours": 2.1},
            {"track": "uni_a", "seq_len": 288, "pred_len": 144, "fold": 0,
             "strategy": "direct", "model": "timemixer", "projected_hours": 3.4},
        ]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "manifest.json"
            pretest.write_manifest(approved, {"nominal_epochs": 50}, out)
            with open(out) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["params"]["nominal_epochs"], 50)
            runs = driver.cells_from_manifest(loaded)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0][1], ["patchtst", "timemixer"])
