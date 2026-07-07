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
