# Benchmark Pre-test + Early Stopping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the inline auto-skip training gate with (a) long training + `EarlyStopping`, and (b) a separate interactive pre-test tool that projects per-model runtime, double-checks slow models, and writes an approved manifest the benchmark driver consumes.

**Architecture:** `_run_neural` drops `TimingGateCallback` and adds `EarlyStopping(monitor="val_loss")` with `--epochs` defaulting to 10000. A new `tslib/benchmark/pretest.py` probes each cell's models (few real training steps), projects `per_epoch_time × nominal_epochs`, prompts for models over a threshold, and writes `runs/bench/manifest.json`. The driver gains `--manifest` to run only approved (cell, model) entries.

**Tech Stack:** Python 3.12, PyTorch 2.4.1, pytorch_lightning 2.6.5, einops, unittest.

## Global Constraints

- **Tests:** unittest only (no pytest). Discover: `python3.12 -m unittest discover -t . -s tslib -p 'test_*.py'`; single module e.g. `python3.12 -m unittest tslib.configs.test_config -v`. Always `python3.12` (never `python3`).
- **Lightning:** `import pytorch_lightning as pl` (classic, 2.6.5). `EarlyStopping`/`ModelCheckpoint` from `pytorch_lightning.callbacks`.
- **Estimate basis (fixed):** pre-test projects `projected_hours = per_epoch_time × nominal_epochs`, `nominal_epochs` default **50**; `threshold_hours` default **12.0**. Actual training uses `max_epochs=config.epochs (10000)` + EarlyStopping.
- **Early stop:** `EarlyStopping(monitor="val_loss", mode="min", patience=config.early_stop_patience)`, `early_stop_patience` default **10**.
- **Steps/epoch floor:** log a warning if `len(train_loader) < 200` (benchmark data has ~13,500, so the floor is informational).
- **`--probe_batches`** (default 3) is retained in config for the pre-test; it is NOT passed to `main.py` by the driver.
- `TrainResult.skipped` / `est_train_hours` fields and `build_comparison`'s columns are **kept unchanged** (no gate sets `skipped` anymore, so it is always `False`; this avoids churn in `exp.py`/`test_exp.py`).
- GPU 1 for real runs via `CUDA_VISIBLE_DEVICES=1`; `accelerator="gpu" if torch.cuda.is_available() else "cpu"`, `devices=1`.
- Never commit datasets (gitignored).

---

## File Structure

- `tslib/configs/config.py` — remove `--max_train_hours`/`--on_slow`; keep `--probe_batches`; `--epochs` default → 10000; add `--early_stop_patience`.
- `tslib/exp/strategy.py` — `_run_neural`: drop the gate + skip-branch, add `EarlyStopping`, add the steps floor warning.
- `tslib/exp/lightning_model.py` — remove `ForecastModule._gate_skipped`/`_est_train_hours` attrs.
- `tslib/exp/callbacks.py`, `tslib/exp/test_callbacks.py` — **deleted**.
- `tslib/benchmark/driver.py` — extract `cell_argv(cell, epochs, models)`; `build_cmd` uses it; `FIXED` drops gate flags and adds `--early_stop_patience`; add `--manifest` consumption + `cells_from_manifest`.
- `tslib/benchmark/pretest.py` — **new**: `estimate_train_hours`, `decide`, `probe_sec_per_batch`, manifest writer, interactive `main`.
- Tests: `tslib/configs/test_config.py`, `tslib/exp/test_strategy.py`, `tslib/benchmark/test_driver.py` updated; `tslib/benchmark/test_pretest.py` new; `tslib/exp/test_callbacks.py` deleted.
- Docs: `docs/lightning-migration.md`, `docs/benchmark-conditions.md`.

---

### Task 1: Config — drop gate flags, add early-stop, long epochs

**Files:**
- Modify: `tslib/configs/config.py`
- Test: `tslib/configs/test_config.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `exp_parser()` no longer has `--max_train_hours`/`--on_slow`; still has `--probe_batches` (int, default 3); `--epochs` default is `10000`; new `--early_stop_patience` (int, default 10). `config_postprocess` raises `ValueError` if `probe_batches < 1` or `early_stop_patience < 1`.

- [ ] **Step 1: Update the failing tests**

In `tslib/configs/test_config.py`, replace the entire `TestGateFlags` class with:
```python
class TestPretestAndEarlyStopFlags(unittest.TestCase):
    def _args(self, *extra):
        from tslib.configs.config import exp_parser
        return exp_parser().parse_args(
            ["--data_path", "x.parquet", "--target_col", "p_gt10", *extra])

    def test_defaults(self):
        cfg = self._args()
        self.assertEqual(cfg.epochs, 10000)
        self.assertEqual(cfg.early_stop_patience, 10)
        self.assertEqual(cfg.probe_batches, 3)
        self.assertFalse(hasattr(cfg, "max_train_hours"))
        self.assertFalse(hasattr(cfg, "on_slow"))

    def test_rejects_probe_batches_below_one(self):
        from tslib.configs.config import config_postprocess
        with self.assertRaises(ValueError):
            config_postprocess(self._args("--probe_batches", "0"))

    def test_rejects_patience_below_one(self):
        from tslib.configs.config import config_postprocess
        with self.assertRaises(ValueError):
            config_postprocess(self._args("--early_stop_patience", "0"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.12 -m unittest tslib.configs.test_config.TestPretestAndEarlyStopFlags -v`
Expected: FAIL (`epochs` default is still 20; `early_stop_patience` missing).

- [ ] **Step 3: Edit `config.py`**

Change the epochs line (currently `parser.add_argument("--epochs", type=int, default=20)`) to:
```python
    parser.add_argument("--epochs", type=int, default=10000,
                        help="max epochs; training stops earlier via EarlyStopping")
```

Replace the three gate-flag lines (`--max_train_hours`, `--on_slow`, `--probe_batches`) with:
```python
    parser.add_argument("--probe_batches", type=int, default=3,
                        help="training batches timed by the pre-test tool")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="EarlyStopping patience in epochs (monitors val_loss)")
```

In `config_postprocess`, replace the gate validation block:
```python
    if getattr(config, "max_train_hours", 1.0) <= 0:
        raise ValueError("max_train_hours must be > 0.")
    if getattr(config, "probe_batches", 1) < 1:
        raise ValueError("probe_batches must be >= 1.")
```
with:
```python
    if getattr(config, "probe_batches", 1) < 1:
        raise ValueError("probe_batches must be >= 1.")
    if getattr(config, "early_stop_patience", 1) < 1:
        raise ValueError("early_stop_patience must be >= 1.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.12 -m unittest tslib.configs.test_config -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tslib/configs/config.py tslib/configs/test_config.py
git commit -m "feat(config): drop gate flags; --epochs default 10000 + --early_stop_patience"
```

---

### Task 2: Strategy — remove gate, add EarlyStopping

**Files:**
- Modify: `tslib/exp/strategy.py`, `tslib/exp/lightning_model.py`
- Delete: `tslib/exp/callbacks.py`, `tslib/exp/test_callbacks.py`
- Test: `tslib/exp/test_strategy.py`

**Interfaces:**
- Consumes: `config.epochs`, `config.early_stop_patience` (Task 1); `ForecastModule`, `TrainResult`, `build_model`, `MetricContext`.
- Produces: `_run_neural` builds a `pl.Trainer` with callbacks `[ModelCheckpoint(monitor="val_loss"), EarlyStopping(monitor="val_loss", mode="min", patience=config.early_stop_patience)]`, no gate, always runs `trainer.test(ckpt_path="best")`. `TimingGateCallback` and `estimate_train_hours` are removed from `tslib/exp/` (a fresh copy of `estimate_train_hours` lives in the pre-test — Task 4).

- [ ] **Step 1: Update `test_strategy.py`**

In `tslib/exp/test_strategy.py`: **delete** the entire `TestGateSkipInStrategy` class (it forced a skip via `max_train_hours=1e-12`, which no longer exists). In `TestNeuralStrategyLightning.test_direct_routes_through_trainer`, change the `cfg = SimpleNamespace(...)` to drop the gate attrs and add `early_stop_patience`:
```python
            cfg = SimpleNamespace(lr=1e-3, weight_decay=0.0, epochs=1, loss="mse",
                                  metrics=["mse"], transform="log10",
                                  event_threshold=None, seq_len=4, pred_len=2,
                                  label_len=0, early_stop_patience=10)
```

- [ ] **Step 2: Run to verify failure**

Run: `python3.12 -m unittest tslib.exp.test_strategy -v`
Expected: FAIL/ERROR — `_run_neural` still imports the (to-be-deleted) `TimingGateCallback` and reads `module._gate_skipped`; the removed `TestGateSkipInStrategy` reference and cfg change surface the gap.

- [ ] **Step 3: Rewrite `_run_neural` in `strategy.py`**

Replace the entire `_run_neural` function with:
```python
def _run_neural(strategy, model_name, data_bundle, config, ckpt_path) -> TrainResult:
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import CSVLogger

    model = build_model(model_name, config, data_bundle.input_size,
                        data_bundle.target_indices, strategy=strategy)
    ctx = MetricContext(
        thresholds=getattr(config, "event_threshold", None),
        transform=getattr(config, "transform", "none"),
        target_cols=list(getattr(data_bundle, "target_cols", []) or []))
    module = ForecastModule(model, config, ctx, strategy=strategy)

    n_train_batches = len(data_bundle.train_loader)
    if n_train_batches < 200:
        print(f"[warn] {model_name}: {n_train_batches} train steps/epoch "
              f"(below the 200 floor)")

    ckpt_dir = os.path.dirname(ckpt_path) or "."
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    ckpt_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                              dirpath=ckpt_dir, filename=ckpt_name)
    early_cb = EarlyStopping(monitor="val_loss", mode="min",
                             patience=config.early_stop_patience)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=config.epochs, accelerator=accelerator, devices=1,
        callbacks=[ckpt_cb, early_cb],
        logger=CSVLogger(save_dir=ckpt_dir, name="lightning"),
        enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(module, data_bundle.train_loader, data_bundle.val_loader)

    trainer.test(module, data_bundle.test_loader, ckpt_path="best")
    best = (float(ckpt_cb.best_model_score)
            if ckpt_cb.best_model_score is not None else float("nan"))
    return TrainResult(model_name=model_name, best_val_loss=best,
                       metrics=module.test_metrics, ckpt_path=ckpt_path,
                       strategy=strategy)
```
(The `from tslib.exp.callbacks import TimingGateCallback` line at the top of `_run_neural` is removed as part of this replacement.)

- [ ] **Step 4: Remove the gate attrs from `ForecastModule`**

In `tslib/exp/lightning_model.py`, delete these two lines from `ForecastModule.__init__`:
```python
        # gate hooks (set by TimingGateCallback)
        self._gate_skipped: bool = False
        self._est_train_hours: float = float("nan")
```
(Keep the `TrainResult.skipped`/`est_train_hours` dataclass fields — they are unchanged and still defaulted.)

- [ ] **Step 5: Delete the gate files**

```bash
git rm tslib/exp/callbacks.py tslib/exp/test_callbacks.py
```

- [ ] **Step 6: Run tests**

Run: `python3.12 -m unittest tslib.exp.test_strategy tslib.exp.test_lightning_model tslib.exp.test_exp -v`
Expected: PASS. Then confirm no dangling imports: `grep -rn "callbacks import\|TimingGateCallback\|_gate_skipped" tslib` returns nothing.

- [ ] **Step 7: Commit**

```bash
git add -A tslib/exp
git commit -m "feat(exp): remove inline gate; add EarlyStopping(val_loss) to _run_neural"
```

---

### Task 3: Driver — cell_argv, manifest, FIXED update

**Files:**
- Modify: `tslib/benchmark/driver.py`
- Test: `tslib/benchmark/test_driver.py`

**Interfaces:**
- Consumes: `--epochs`/`--early_stop_patience` semantics (Task 1).
- Produces:
  - `cell_argv(cell, epochs, models) -> list[str]` — the `main.py` flag list for a cell with an explicit model list.
  - `build_cmd(cell, epochs, models=None)` — `[python, main.py, *cell_argv(...)]`; `models` defaults to `models_for(track, strategy)`.
  - `cells_from_manifest(manifest: dict) -> list[tuple[dict, list[str]]]` — approved (cell, models) grouped from `manifest["approved"]`.
  - `FIXED` no longer contains `--max_train_hours`/`--on_slow`; contains `--early_stop_patience 10`; does NOT contain `--probe_batches`. `--epochs` default (driver arg) is `10000`.
  - `main` accepts `--manifest <path>`; when given, the run list comes from the manifest (models filtered to approved), else from the enumerated flags.

- [ ] **Step 1: Update `test_driver.py`**

Replace `TestBuildCmd.test_uni_a_direct_flags`'s gate-flag assertions and add manifest tests. Specifically, in `test_uni_a_direct_flags` replace the two lines asserting `--max_train_hours`/`--on_slow` with:
```python
        self.assertNotIn("--max_train_hours", cmd)
        self.assertEqual(cmd[cmd.index("--early_stop_patience") + 1], "10")
        self.assertEqual(cmd[cmd.index("--epochs") + 1], "30")
```
Append a new class:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3.12 -m unittest tslib.benchmark.test_driver -v`
Expected: FAIL — `cells_from_manifest` missing; `build_cmd` has no `models` param; `--early_stop_patience` not in cmd.

- [ ] **Step 3: Edit `driver.py`**

Replace the `FIXED` list with (gate flags removed, early-stop added; no `--probe_batches`, no `--epochs` here since epochs is passed per call):
```python
FIXED = [
    "--n_fold", "5",
    "--transform", "log10",
    "--metrics", "rmse", "mae", "tss", "hss", "pod", "far",
    "--batch_size", "64",
    "--lr", "1e-3",
    "--d_model", "128",
    "--num_layers", "2",
    "--dropout", "0.1",
    "--early_stop_patience", "10",
]
```

Replace `build_cmd` with `cell_argv` + a thin `build_cmd`:
```python
def cell_argv(cell, epochs: int, models: list[str]) -> list[str]:
    """main.py flag list for one cell with an explicit model list."""
    t = TRACKS[cell["track"]]
    run_name = run_name_for(cell["track"], cell["seq_len"], cell["pred_len"],
                            cell["fold"], cell["strategy"])
    argv = ["--data_path", t["data_path"],
            "--target_col", t["target_col"],
            "--seq_len", str(cell["seq_len"]),
            "--pred_len", str(cell["pred_len"]),
            "--fold_numb", str(cell["fold"]),
            "--forecast_strategy", cell["strategy"],
            "--run_name", run_name,
            "--sort_metric", t["sort_metric"],
            "--epochs", str(epochs),
            "--event_threshold", *t["event_threshold"],
            "--models", *models,
            *FIXED]
    if t["channels"]:
        argv += ["--channels", *t["channels"],
                 "--target_cols", *t["target_cols"]]
    return argv


def build_cmd(cell, epochs: int, models=None) -> list[str]:
    if models is None:
        models = models_for(cell["track"], cell["strategy"])
    return [sys.executable, str(REPO / "main.py"), *cell_argv(cell, epochs, models)]


def cells_from_manifest(manifest: dict):
    """Group manifest['approved'] entries into [(cell, [models...]), ...],
    preserving first-seen order of cells and of models within a cell."""
    order, grouped = [], {}
    for e in manifest.get("approved", []):
        cell = {"track": e["track"], "seq_len": e["seq_len"],
                "pred_len": e["pred_len"], "fold": e["fold"],
                "strategy": e["strategy"]}
        key = (e["track"], e["seq_len"], e["pred_len"], e["fold"], e["strategy"])
        if key not in grouped:
            grouped[key] = {"cell": cell, "models": []}
            order.append(key)
        grouped[key]["models"].append(e["model"])
    return [(grouped[k]["cell"], grouped[k]["models"]) for k in order]
```

Update `run_benchmark` to take an optional per-cell model list. Change its signature and the two `build_cmd`/`comparison_path` call sites to accept `runs` as a list of `(cell, models)` pairs:
```python
def run_benchmark(runs, epochs, runs_root, master_path, dry_run=False):
    total = len(runs)
    env = os.environ.copy()
    cells = [c for c, _ in runs]
    for i, (cell, models) in enumerate(runs, 1):
        run_name = run_name_for(cell["track"], cell["seq_len"],
                                cell["pred_len"], cell["fold"], cell["strategy"])
        cmd = build_cmd(cell, epochs, models)
        cp = comparison_path(cell, runs_root)
        if dry_run:
            print(f"[{i}/{total}] {run_name} models={models}")
            print("    " + " ".join(cmd))
            continue
        if cp.exists():
            print(f"[{i}/{total}] SKIP (done): {run_name}")
            continue
        print(f"[{i}/{total}] RUN: {run_name}\n    {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO), env=env)
        if proc.returncode != 0:
            print(f"[{i}/{total}] FAILED (exit {proc.returncode}): {run_name} "
                  f"— continuing", flush=True)
            continue
        n = rebuild_master(cells, runs_root, master_path)
        print(f"[{i}/{total}] DONE: {run_name} — master now has {n} cells "
              f"at {master_path}", flush=True)
        if cp.exists():
            print(pd.read_csv(cp).to_string(index=False), flush=True)
    if not dry_run:
        n = rebuild_master(cells, runs_root, master_path)
        print(f"\nBenchmark sweep complete: {n}/{total} cells in {master_path}")
```

Update `main` to build `runs` from either the manifest or the enumeration, and add `--manifest`:
```python
def main(argv=None):
    args = parse_args(argv)
    runs_root = REPO / "runs"
    master_path = runs_root / "bench" / "results_master.csv"
    if args.manifest:
        import json
        with open(args.manifest) as f:
            manifest = json.load(f)
        runs = cells_from_manifest(manifest)
        print(f"Manifest {args.manifest}: {len(runs)} approved cells")
    else:
        cells = enumerate_cells(args.tracks, args.seq_lens, args.pred_lens,
                                args.folds, args.strategies)
        runs = [(c, models_for(c["track"], c["strategy"])) for c in cells]
        print(f"Planned cells: {len(runs)} (tracks={args.tracks} "
              f"seq={args.seq_lens} pred={args.pred_lens} folds={args.folds} "
              f"strategies={args.strategies} epochs={args.epochs})")
    run_benchmark(runs, args.epochs, runs_root, master_path,
                  dry_run=args.dry_run)
```

In `parse_args`, add:
```python
    p.add_argument("--manifest", type=str, default=None,
                   help="run only (cell,model) entries approved in this manifest")
```
and change the epochs default:
```python
    p.add_argument("--epochs", type=int, default=10000)
```

- [ ] **Step 4: Run tests**

Run: `python3.12 -m unittest tslib.benchmark.test_driver -v`
Expected: PASS (enumerate, build_cmd with explicit models, manifest grouping, no gate flags).

- [ ] **Step 5: Commit**

```bash
git add tslib/benchmark/driver.py tslib/benchmark/test_driver.py
git commit -m "feat(benchmark): driver cell_argv + --manifest consumption; FIXED early-stop, no gate"
```

---

### Task 4: Pre-test tool

**Files:**
- Create: `tslib/benchmark/pretest.py`
- Test: `tslib/benchmark/test_pretest.py`

**Interfaces:**
- Consumes: `driver.enumerate_cells`, `driver.cell_argv`, `driver.TRACKS`, `driver.models_for`; `exp_parser`/`config_postprocess`; `DataModule`; `build_model`; `build_loss`.
- Produces:
  - `estimate_train_hours(sec_per_batch, n_batches, epochs) -> float`.
  - `decide(projected_hours, threshold_hours, prompt_fn) -> bool` — auto-approve at/under threshold; else `prompt_fn(projected_hours)`.
  - `probe_sec_per_batch(model, train_loader, config, probe_batches) -> float`.
  - `write_manifest(approved, params, out_path)` / manifest schema `{"params": {...}, "approved": [ {track,seq_len,pred_len,fold,strategy,model,projected_hours}, ... ]}` matching `driver.cells_from_manifest`.
  - `main(argv=None)` — interactive; probes each cell's models, prompts over threshold, writes the manifest.

- [ ] **Step 1: Write the failing tests**

Create `tslib/benchmark/test_pretest.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3.12 -m unittest tslib.benchmark.test_pretest -v`
Expected: FAIL — `No module named 'tslib.benchmark.pretest'`.

- [ ] **Step 3: Create `pretest.py`**

```python
"""Interactive pre-test: probe each benchmark cell's models, project runtime,
double-check models over a threshold, and write an approved manifest for the
driver (``tslib.benchmark.driver --manifest``).

Projection basis: ``per_epoch_time * nominal_epochs`` (actual training uses
max_epochs + EarlyStopping, so this is a yardstick, not the real duration).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tslib.benchmark import driver
from tslib.configs.config import exp_parser, config_postprocess
from tslib.data.loader import DataModule
from tslib.model import build_model
from tslib.exp.losses import build_loss


def estimate_train_hours(sec_per_batch: float, n_batches: float,
                         epochs: int) -> float:
    return float(sec_per_batch) * float(n_batches) * float(epochs) / 3600.0


def decide(projected_hours: float, threshold_hours: float, prompt_fn) -> bool:
    """Auto-approve at/under threshold; otherwise defer to prompt_fn(hours)."""
    if projected_hours <= threshold_hours:
        return True
    return bool(prompt_fn(projected_hours))


def _default_prompt(cell, model):
    def ask(hours):
        ans = input(f"[{cell['track']} seq{cell['seq_len']} "
                    f"pred{cell['pred_len']} f{cell['fold']} {cell['strategy']}] "
                    f"{model} ~{hours:.1f}h projected — proceed? [y/N] ")
        return ans.strip().lower() in ("y", "yes")
    return ask


def probe_sec_per_batch(model, train_loader, config, probe_batches: int) -> float:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    model.train()
    crit = build_loss(config)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr,
                           weight_decay=config.weight_decay)
    durations = []
    it = iter(train_loader)
    for i in range(probe_batches + 1):  # +1 warmup, dropped
        try:
            x, y = next(it)
        except StopIteration:
            break
        x, y = x.to(dev), y.to(dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = model(x)
        loss = crit(pred, y[:, :pred.shape[1], :])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        if i > 0:
            durations.append(time.perf_counter() - t0)
    return statistics.median(durations) if durations else float("inf")


def write_manifest(approved, params, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"params": params, "approved": approved}, f, indent=2)


def _config_for(cell, model, nominal_epochs):
    argv = driver.cell_argv(cell, epochs=nominal_epochs, models=[model])
    return config_postprocess(exp_parser().parse_args(argv))


def main(argv=None):
    p = argparse.ArgumentParser(description="benchmark runtime pre-test")
    p.add_argument("--tracks", nargs="+", default=list(driver.TRACKS),
                   choices=list(driver.TRACKS))
    p.add_argument("--seq-lens", nargs="+", type=int, default=driver.SEQ_LENS)
    p.add_argument("--pred-lens", nargs="+", type=int, default=driver.PRED_LENS)
    p.add_argument("--folds", nargs="+", type=int, default=[0])
    p.add_argument("--strategies", nargs="+", default=["direct"],
                   choices=["direct", "recursive"])
    p.add_argument("--probe-batches", type=int, default=3)
    p.add_argument("--nominal-epochs", type=int, default=50)
    p.add_argument("--threshold-hours", type=float, default=12.0)
    p.add_argument("--out", type=str,
                   default=str(driver.REPO / "runs" / "bench" / "manifest.json"))
    args = p.parse_args(argv)

    cells = driver.enumerate_cells(args.tracks, args.seq_lens, args.pred_lens,
                                   args.folds, args.strategies)
    approved, rejected = [], 0
    for cell in cells:
        models = driver.models_for(cell["track"], cell["strategy"])
        cfg = _config_for(cell, models[0], args.nominal_epochs)
        bundle = DataModule(cfg).setup()  # build_model sets enc_in/dec_in/c_out later
        steps = len(bundle.train_loader)
        for model in models:
            try:
                mcfg = _config_for(cell, model, args.nominal_epochs)
                net = build_model(model, mcfg, bundle.input_size,
                                  bundle.target_indices, strategy=cell["strategy"])
                sec = probe_sec_per_batch(net, bundle.train_loader, mcfg,
                                          args.probe_batches)
                proj = estimate_train_hours(sec, steps, args.nominal_epochs)
                print(f"  {cell['track']} seq{cell['seq_len']} "
                      f"pred{cell['pred_len']} {model}: "
                      f"{sec:.3f}s/batch, ~{proj:.1f}h @ {args.nominal_epochs}ep")
                if decide(proj, args.threshold_hours,
                          _default_prompt(cell, model)):
                    approved.append({**cell, "model": model,
                                     "projected_hours": round(proj, 2)})
                else:
                    rejected += 1
            except Exception as e:  # bad model/probe: reject, keep going
                print(f"  {cell['track']} {model}: probe failed ({e}) — rejected")
                rejected += 1

    params = {"nominal_epochs": args.nominal_epochs,
              "threshold_hours": args.threshold_hours,
              "probe_batches": args.probe_batches,
              "seq_lens": args.seq_lens, "pred_lens": args.pred_lens,
              "folds": args.folds, "strategies": args.strategies}
    write_manifest(approved, params, args.out)
    print(f"\nPre-test done: {len(approved)} approved, {rejected} rejected. "
          f"Manifest → {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `python3.12 -m unittest tslib.benchmark.test_pretest -v`
Expected: PASS (estimate, decide branches, manifest round-trip through `driver.cells_from_manifest`).

- [ ] **Step 5: Commit**

```bash
git add tslib/benchmark/pretest.py tslib/benchmark/test_pretest.py
git commit -m "feat(benchmark): interactive runtime pre-test writing an approved manifest"
```

---

### Task 5: Documentation

**Files:**
- Modify: `docs/lightning-migration.md`, `docs/benchmark-conditions.md`

**Interfaces:**
- Consumes: shipped behavior from Tasks 1-4.
- Produces: docs describing early stopping + the pre-test/manifest workflow; no references to the removed gate flags.

- [ ] **Step 1: Update `docs/lightning-migration.md`**

Replace the "Train-time gate" section with an "Early stopping + long epochs" section: training runs `--epochs` (default 10000) with `EarlyStopping(monitor="val_loss", mode="min", patience=--early_stop_patience [default 10])`; `ModelCheckpoint` still saves best-val and `trainer.test(ckpt_path="best")` reloads it. Add a short "Runtime pre-test" note: `python3.12 -m tslib.benchmark.pretest` probes each cell's models, projects `per_epoch × nominal_epochs` (default 50), and for models over `--threshold-hours` (default 12) asks interactively; approved models are written to `runs/bench/manifest.json`, consumed by `driver --manifest`. Remove any `--max_train_hours`/`--on_slow`/`SKIPPED (too slow)` wording.

- [ ] **Step 2: Update `docs/benchmark-conditions.md`**

Remove the "Train-time gate" subsection. Add a "Pre-test + early stopping" subsection documenting the two-step workflow (`pretest` → manifest → `driver --manifest`), `--epochs 10000`, `--early_stop_patience 10`, and the `--nominal-epochs`/`--threshold-hours` double-check. Update the "Fixed conditions" table: replace the `--epochs 30` reference with `--epochs 10000` + `--early_stop_patience 10`; drop the gate-flag mentions.

- [ ] **Step 3: Verify**

Run:
```bash
cd /NAS/ioGuard3/vol3/spaceai/SW_framework/workdir/sw-framework-v003
grep -n "max_train_hours\|on_slow\|SKIPPED (too slow)" docs/*.md || echo "no stale gate refs"
grep -n "early_stop_patience\|pretest\|manifest" docs/lightning-migration.md docs/benchmark-conditions.md
```
Expected: no stale gate refs; early-stop/pretest/manifest present.

- [ ] **Step 4: Commit**

```bash
git add docs/lightning-migration.md docs/benchmark-conditions.md
git commit -m "docs: early stopping + pre-test/manifest workflow (replaces gate docs)"
```

---

## Final steps (after all tasks)

- [ ] Full suite: `python3.12 -m unittest discover -t . -s tslib -p 'test_*.py'` — expect all green.
- [ ] Confirm no stale gate references: `grep -rn "TimingGateCallback\|max_train_hours\|on_slow\|_gate_skipped" tslib docs` returns nothing.
- [ ] Whole-branch review, then superpowers:finishing-a-development-branch (this work is on `main`; the branch decision is the user's).
