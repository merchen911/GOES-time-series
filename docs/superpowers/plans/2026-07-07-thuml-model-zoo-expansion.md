# THUML Model-Zoo Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every registered forecasting model construct and run — enable the config-missing ones, repair `etsformer`, import lightweight/recent THUML models (DLinear, TSMixer, TiDE, FreTS) — and fix the pre-test GPU memory leak.

**Architecture:** THUML `Model(configs)` + `forward(x_enc, x_mark_enc, x_dec, x_mark_dec)` matches this repo's 4-arg `_call_base` adapter, so models plug into `StandardForecastAdapter` unchanged. Missing config flags are added to `exp_parser` with source-confirmed defaults; missing layer modules and new model files are ported from THUML with imports adapted to `tslib.model.layers.*`. A single all-registry construct+forward test certifies the whole zoo.

**Tech Stack:** Python 3.12, PyTorch 2.4.1, pytorch_lightning 2.6.5, einops, unittest.

## Global Constraints

- **Tests:** unittest only (no pytest). `python3.12 -m unittest tslib.model.test_registry -v`; full: `python3.12 -m unittest discover -t . -s tslib -p 'test_*.py'`. Always `python3.12`.
- **THUML reference:** fetch source from `https://raw.githubusercontent.com/thuml/Time-Series-Library/main/models/<Name>.py` and `.../main/layers/<Name>.py`. Adapt `from layers.X import ...` → `from .layers.X import ...`.
- **`enc_in`/`dec_in`/`c_out` are NOT config gaps** — `build_model` sets them (`config.enc_in = config.dec_in = config.c_out = input_size`). Never add CLI flags for these.
- **Acceptance for every model:** `build_model(name, cfg, input_size=1, target_indices=[0], strategy="direct")` from a real `config_postprocess(exp_parser().parse_args(...))` runs a forward on `(2, seq_len, 1)` and returns `(2, pred_len, 1)`. Use `seq_len=96, pred_len=24` in tests (large enough for patching/downsampling models).
- **Defaults:** use the model's upstream defaults (THUML `run.py` for THUML models; the model file's own paper/repo values for non-THUML ones like patchmixer/xpatch/segrnn). Confirm by reading the model source; the acceptance test is the final arbiter.
- **A model that cannot be made to run** after reasonable effort is removed from `registry._LEGACY` with a logged reason (the all-registry test must stay green) — report it rather than leave it half-wired.
- Device-aware toy usage; forward on CPU in tests. Never commit datasets.

---

## File Structure

- `tslib/configs/config.py` — add missing model config flags (Tasks 1-2, 4-5).
- `tslib/model/layers/Embed.py`, `tslib/model/layers/ETSformer_EncDec.py` — new, ported (Task 3).
- `tslib/model/dlinear.py`, `frets.py`, `tsmixer.py`, `tide.py` — new model files (Tasks 4-5). (lowercase filenames to match repo convention, e.g. `patchtst.py`.)
- `tslib/model/registry.py` — add new names to `_LEGACY` (Tasks 4-5).
- `tslib/model/test_registry.py` — per-task construct tests, then the all-registry sweep (Task 6).
- `tslib/benchmark/pretest.py` — memory-leak fix (Task 6).

**Test helper (used by every task).** Add once at the top of `tslib/model/test_registry.py` (Task 1 Step 1 introduces it):
```python
def _real_cfg(models, seq_len=96, pred_len=24):
    from tslib.configs.config import exp_parser, config_postprocess
    argv = ["--data_path", "/tmp/x.parquet", "--target_col", "p_gt10",
            "--seq_len", str(seq_len), "--pred_len", str(pred_len),
            "--fold_numb", "0", "--forecast_strategy", "direct",
            "--event_threshold", "10",
            "--metrics", "rmse", "mae", "tss", "hss", "pod", "far",
            "--models", *models]
    return config_postprocess(exp_parser().parse_args(argv))


def _assert_builds(testcase, names, channels=1):
    import torch
    from tslib.model import build_model
    cfg = _real_cfg(names, channels=channels)
    x = torch.zeros(2, cfg.seq_len, channels)
    tgt = list(range(channels)) if channels > 1 else [0]
    for name in names:
        net = build_model(name, cfg, channels, tgt, strategy="direct")
        net.eval()
        with torch.no_grad():
            out = net(x)
        testcase.assertEqual(tuple(out.shape), (2, cfg.pred_len, len(tgt)), msg=name)
```

---

### Task 1: Enable micn, nonstationary_transformer, scinet, segrnn_thuml

**Files:** Modify `tslib/configs/config.py`; Test `tslib/model/test_registry.py`.

**Interfaces:**
- Consumes: existing model files (already in repo).
- Produces: these 4 models construct+forward from the real config. New `exp_parser` flags (defaults confirmed against the in-repo model source + THUML run.py).

Exact missing flags (from `configs.X` usage; `enc_in`/`c_out`/`dec_in` excluded):
- `micn`: `--conv_kernel` (nargs, ints; THUML default `[12, 16]`)
- `nonstationary_transformer`: `--p_hidden_dims` (nargs ints, default `[128, 128]`), `--p_hidden_layers` (int, default `2`)
- `scinet`: `--current_level` (int, default `3`), `--causal_block_level`? — read `scinet.py`; add whatever it reads with the file's implied default
- `segrnn_thuml`: `--seg_len` (int, default `48`)

- [ ] **Step 1: Add the test helper + a failing test for these 4**

In `tslib/model/test_registry.py`, add the `_real_cfg` and `_assert_builds` helpers (shown in File Structure above), then add:
```python
class TestEnableGroup1(unittest.TestCase):
    def test_builds(self):
        _assert_builds(self, ["micn", "nonstationary_transformer",
                              "scinet", "segrnn_thuml"])
```

- [ ] **Step 2: Run — verify it fails**

Run: `python3.12 -m unittest tslib.model.test_registry.TestEnableGroup1 -v`
Expected: FAIL — `AttributeError` on the first missing attr (e.g. `conv_kernel`).

- [ ] **Step 3: Read each model's config usage and add flags**

For each of the 4 models: `grep -oE "configs\.[a-zA-Z_]+" tslib/model/<m>.py | sort -u`, and for every attr not already in `exp_parser` (and not `enc_in`/`dec_in`/`c_out`), add an argument to `exp_parser` in `tslib/configs/config.py`. Cross-check the default against THUML `run.py` (`https://raw.githubusercontent.com/thuml/Time-Series-Library/main/run.py`) when the model is a THUML model; otherwise use the value the model file implies. Add them under a new comment block `# extra backbone flags (THUML defaults)`. Concretely, at minimum:
```python
    parser.add_argument("--conv_kernel", nargs="+", type=int, default=[12, 16])
    parser.add_argument("--p_hidden_dims", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--p_hidden_layers", type=int, default=2)
    parser.add_argument("--current_level", type=int, default=3)
    parser.add_argument("--seg_len", type=int, default=48)
```
Add any additional attr the grep reveals (e.g. `causal_block_level` for SCINet) with the source-confirmed default. Iterate: re-run the test, read the next `AttributeError`, add that flag, until it passes.

- [ ] **Step 4: Run — verify pass**

Run: `python3.12 -m unittest tslib.model.test_registry -v`
Expected: PASS (existing tests + `TestEnableGroup1`).

- [ ] **Step 5: Commit**

```bash
git add tslib/configs/config.py tslib/model/test_registry.py
git commit -m "feat(model): enable micn/nonstationary/scinet/segrnn_thuml (config flags)"
```

---

### Task 2: Enable patchmixer, segrnn, xpatch

**Files:** Modify `tslib/configs/config.py`; Test `tslib/model/test_registry.py`.

**Interfaces:**
- Produces: these 3 (non-THUML-core) models construct+forward. Defaults come from each model file's own logic / upstream paper (PatchMixer, SegRNN, xPatch repos).

Exact missing flags:
- `patchmixer`: `--a` (int), `--affine` (int 0/1), `--head_dropout` (float), `--mixer_kernel_size` (int), `--revin` (int 0/1), `--subtract_last` (int 0/1)
- `segrnn`: `--channel_id` (int), `--dec_way` (str), `--revin` (int), `--rnn_type` (str), `--seg_len` (shared with Task 1)
- `xpatch`: `--alpha` (float), `--beta` (float), `--ma_type` (str), `--padding_patch` (str), `--revin` (shared)

- [ ] **Step 1: Add failing test**

```python
class TestEnableGroup2(unittest.TestCase):
    def test_builds(self):
        _assert_builds(self, ["patchmixer", "segrnn", "xpatch"])
```

- [ ] **Step 2: Run — verify fail**

Run: `python3.12 -m unittest tslib.model.test_registry.TestEnableGroup2 -v`
Expected: FAIL — missing attr.

- [ ] **Step 3: Determine defaults from source and add flags**

For each model, read the `.py` to see how each attr is used and pick the upstream default (e.g. `revin=1`, `affine=0`, `subtract_last=0` are the standard RevIN settings; `padding_patch="end"`; `ma_type="ema"`; `alpha`/`beta` the file's decomposition weights; `rnn_type="gru"`, `dec_way="pmf"`, `channel_id=1`, `seg_len` already added; `mixer_kernel_size=8`, `head_dropout=0.0`, `a`=the file's kernel/param). Add to `exp_parser`:
```python
    parser.add_argument("--revin", type=int, default=1)
    parser.add_argument("--affine", type=int, default=0)
    parser.add_argument("--subtract_last", type=int, default=0)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--mixer_kernel_size", type=int, default=8)
    parser.add_argument("--a", type=int, default=2)
    parser.add_argument("--rnn_type", type=str, default="gru")
    parser.add_argument("--dec_way", type=str, default="pmf")
    parser.add_argument("--channel_id", type=int, default=1)
    parser.add_argument("--padding_patch", type=str, default="end")
    parser.add_argument("--ma_type", type=str, default="ema")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.3)
```
Adjust each default if the model source requires a specific value to run (the test enforces runnability). Iterate until the test passes. If a model genuinely cannot run (e.g. a shape assertion incompatible with `seq_len=96/pred_len=24`), try `seq_len` divisible parameters or, as a last resort, remove it from `_LEGACY` in `tslib/model/registry.py` with a code comment stating why, and drop it from this test — report it in the task report.

- [ ] **Step 4: Run — verify pass**

Run: `python3.12 -m unittest tslib.model.test_registry -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tslib/configs/config.py tslib/model/test_registry.py tslib/model/registry.py
git commit -m "feat(model): enable patchmixer/segrnn/xpatch (config flags)"
```

---

### Task 3: Repair etsformer (port missing layers)

**Files:** Create `tslib/model/layers/Embed.py`, `tslib/model/layers/ETSformer_EncDec.py`; possibly modify `tslib/configs/config.py`; Test `tslib/model/test_registry.py`.

**Interfaces:**
- Consumes: `tslib/model/etsformer.py` (imports `from .layers.Embed import DataEmbedding` and `from .layers.ETSformer_EncDec import EncoderLayer, Encoder, DecoderLayer, Decoder, Transform`).
- Produces: `etsformer` constructs+forward.

- [ ] **Step 1: Add failing test**

```python
class TestEtsformer(unittest.TestCase):
    def test_builds(self):
        _assert_builds(self, ["etsformer"])
```

- [ ] **Step 2: Run — verify fail**

Run: `python3.12 -m unittest tslib.model.test_registry.TestEtsformer -v`
Expected: FAIL — `ModuleNotFoundError: tslib.model.layers.Embed`.

- [ ] **Step 3: Port the two layer files from THUML**

Fetch `https://raw.githubusercontent.com/thuml/Time-Series-Library/main/layers/Embed.py` → save as `tslib/model/layers/Embed.py`, adapting any `from layers.X import ...` to `from .X import ...`. Fetch `https://raw.githubusercontent.com/thuml/Time-Series-Library/main/layers/ETSformer_EncDec.py` → `tslib/model/layers/ETSformer_EncDec.py`, same import adaptation. If `Embed.py` imports a helper not present in `tslib/model/layers/`, port that too. Add any config attr `etsformer.py` reads that is unwired (grep `configs.` in `etsformer.py`; note it asserts `e_layers == d_layers` — the defaults are `e_layers=2, d_layers=1`, so etsformer needs `d_layers` set equal; the test's config must satisfy this — if the model can't run with defaults, add an etsformer-specific note and, if unresolvable, drop it from `_LEGACY` with a reason).

- [ ] **Step 4: Run — verify pass**

Run: `python3.12 -m unittest tslib.model.test_registry -v`
Expected: PASS (or, if etsformer is unportable, it's removed from `_LEGACY` and the test list, with a documented reason — report it).

- [ ] **Step 5: Commit**

```bash
git add tslib/model/layers/ tslib/model/test_registry.py tslib/configs/config.py
git commit -m "fix(model): port Embed + ETSformer_EncDec layers to repair etsformer"
```

---

### Task 4: Import DLinear + FreTS

**Files:** Create `tslib/model/dlinear.py`, `tslib/model/frets.py`, any missing `tslib/model/layers/*.py`; Modify `tslib/model/registry.py`, `tslib/configs/config.py`; Test `tslib/model/test_registry.py`.

**Interfaces:**
- Produces: `dlinear`, `frets` registered and construct+forward.
- DLinear reads `configs.task_name/seq_len/pred_len/moving_avg/enc_in` (moving_avg already exists) and imports `series_decomp` from `layers.Autoformer_EncDec` (present here as `tslib/model/layers/Autoformer_EncDec.py`). FreTS reads `configs.seq_len/pred_len/enc_in` and typically `embed_size`/`hidden_size` (add if missing).

- [ ] **Step 1: Add failing test**

```python
class TestNewLightweight1(unittest.TestCase):
    def test_builds(self):
        _assert_builds(self, ["dlinear", "frets"])
    def test_builds_multivar(self):
        _assert_builds(self, ["dlinear", "frets"], channels=3)
```

- [ ] **Step 2: Run — verify fail**

Run: `python3.12 -m unittest tslib.model.test_registry.TestNewLightweight1 -v`
Expected: FAIL — models not registered / files absent.

- [ ] **Step 3: Port the two models**

- Fetch `https://raw.githubusercontent.com/thuml/Time-Series-Library/main/models/DLinear.py` → `tslib/model/dlinear.py`; change `from layers.Autoformer_EncDec import series_decomp` to `from .layers.Autoformer_EncDec import series_decomp`. Verify `series_decomp` exists in this repo's `Autoformer_EncDec.py` (`grep -n "class series_decomp" tslib/model/layers/Autoformer_EncDec.py`); if absent, port it from THUML's `layers/Autoformer_EncDec.py`.
- Fetch `https://raw.githubusercontent.com/thuml/Time-Series-Library/main/models/FreTS.py` → `tslib/model/frets.py`; adapt imports. Add any config attr it reads that is unwired (grep `configs.` in the new file; e.g. `--embed_size` (int, default `128`) if used) to `exp_parser`.
- Register both: in `tslib/model/registry.py`, add `"dlinear"` and `"frets"` to the `_LEGACY` list. (Registry maps name → `tslib.model.<name>` module's `Model`; filenames must be lowercase matching the name.)

- [ ] **Step 4: Run — verify pass**

Run: `python3.12 -m unittest tslib.model.test_registry -v`
Expected: PASS (univariate and 3-channel).

- [ ] **Step 5: Commit**

```bash
git add tslib/model/dlinear.py tslib/model/frets.py tslib/model/layers/ tslib/model/registry.py tslib/configs/config.py tslib/model/test_registry.py
git commit -m "feat(model): import DLinear + FreTS from THUML"
```

---

### Task 5: Import TSMixer + TiDE

**Files:** Create `tslib/model/tsmixer.py`, `tslib/model/tide.py`, any missing layers; Modify `tslib/model/registry.py`, `tslib/configs/config.py`; Test `tslib/model/test_registry.py`.

**Interfaces:**
- Produces: `tsmixer`, `tide` registered and construct+forward (univariate + multivariable).
- TSMixer reads `configs.seq_len/pred_len/enc_in/e_layers/d_model/dropout` (mostly present). TiDE reads `configs.seq_len/pred_len/d_model/e_layers/d_layers/dropout` and TiDE-specific dims (add if missing, e.g. `--bias`, feature dims — confirm from source).

- [ ] **Step 1: Add failing test**

```python
class TestNewLightweight2(unittest.TestCase):
    def test_builds(self):
        _assert_builds(self, ["tsmixer", "tide"])
    def test_builds_multivar(self):
        _assert_builds(self, ["tsmixer", "tide"], channels=3)
```

- [ ] **Step 2: Run — verify fail**

Run: `python3.12 -m unittest tslib.model.test_registry.TestNewLightweight2 -v`
Expected: FAIL — not registered.

- [ ] **Step 3: Port the two models**

- Fetch `https://raw.githubusercontent.com/thuml/Time-Series-Library/main/models/TSMixer.py` → `tslib/model/tsmixer.py`; adapt imports; add any unwired config attr (grep `configs.` in the new file) to `exp_parser` with the THUML `run.py` default.
- Fetch `https://raw.githubusercontent.com/thuml/Time-Series-Library/main/models/TiDE.py` → `tslib/model/tide.py`; adapt imports; add unwired config attrs (e.g. `--bias` (store_true or int), `--feature_encode_dim` — confirm names/defaults from the fetched source).
- Register `"tsmixer"` and `"tide"` in `registry._LEGACY`.

- [ ] **Step 4: Run — verify pass**

Run: `python3.12 -m unittest tslib.model.test_registry -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tslib/model/tsmixer.py tslib/model/tide.py tslib/model/layers/ tslib/model/registry.py tslib/configs/config.py tslib/model/test_registry.py
git commit -m "feat(model): import TSMixer + TiDE from THUML"
```

---

### Task 6: All-registry sweep test + pre-test memory-leak fix

**Files:** Modify `tslib/model/test_registry.py`, `tslib/benchmark/pretest.py`; Test `tslib/benchmark/test_pretest.py` (unchanged, must stay green).

**Interfaces:**
- Produces: one comprehensive test iterating the full `MODEL_REGISTRY`; pretest releases GPU memory between probes.

- [ ] **Step 1: Add the all-registry sweep test**

In `tslib/model/test_registry.py` add:
```python
class TestAllRegistryModelsBuild(unittest.TestCase):
    def test_every_registered_model_builds_and_forwards(self):
        from tslib.model.registry import MODEL_REGISTRY
        _assert_builds(self, sorted(MODEL_REGISTRY))
```

- [ ] **Step 2: Run — verify pass**

Run: `python3.12 -m unittest tslib.model.test_registry.TestAllRegistryModelsBuild -v`
Expected: PASS — every registered model builds and forwards. (If any fails here, that model must be fixed or removed from `_LEGACY` with a documented reason before this task completes.)

- [ ] **Step 3: Fix the pre-test memory leak**

In `tslib/benchmark/pretest.py` `main()`, inside the per-model loop, after computing the projection (whether approved or rejected), release the model and free CUDA cache. Change the per-model `try` body so that after `sec = probe_sec_per_batch(...)` and the decide/append, it does:
```python
                import torch
                del net
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
```
Place this in a `finally` on the per-model `try/except` so it runs whether the probe succeeded or raised (guard with `if "net" in locals()` before `del net`).

- [ ] **Step 4: Run — verify pretest tests + full suite**

Run: `python3.12 -m unittest tslib.benchmark.test_pretest -v`
Expected: PASS (5/5 — the leak fix doesn't change the pure-function tests).
Run: `python3.12 -m unittest discover -t . -s tslib -p 'test_*.py'`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add tslib/model/test_registry.py tslib/benchmark/pretest.py
git commit -m "test(model): all-registry construct+forward sweep; fix(pretest): free GPU per probe"
```

---

## Final steps (after all tasks)

- [ ] Full suite: `python3.12 -m unittest discover -t . -s tslib -p 'test_*.py'` — expect green.
- [ ] Confirm registry coverage: `python3.12 -c "from tslib.model.registry import MODEL_REGISTRY; print(sorted(MODEL_REGISTRY))"` — includes dlinear, frets, tsmixer, tide.
- [ ] Whole-branch review, then superpowers:finishing-a-development-branch.
- [ ] (Follow-up, not this plan) Re-run `pretest` + `driver` to fold the new fast models into the benchmark.
