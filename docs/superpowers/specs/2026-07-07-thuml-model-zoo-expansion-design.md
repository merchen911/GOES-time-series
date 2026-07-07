# THUML Model-Zoo Expansion — Design

**Date:** 2026-07-07

## Goal

Broaden the set of runnable forecasting backbones so the benchmark has fast,
recent alternatives to the budget-hogging `timemixer`/`timesnet`. Referencing
the THUML `Time-Series-Library` source, (A) enable the registered-but-inactive
models whose config flags were never wired, (B) repair `etsformer`'s missing
layer modules, and (C) import a set of lightweight/recent THUML models not
present in this repo. Guarantee every registered model constructs and runs a
forward pass from the real `exp_parser` config. Also fix the pre-test's GPU
memory accumulation.

## Reference method

Fetch THUML source per file from `https://raw.githubusercontent.com/thuml/
Time-Series-Library/main/{models,layers}/<name>.py` (network verified). THUML
model signature is `Model(configs)` with `forward(x_enc, x_mark_enc, x_dec,
x_mark_dec)`, which matches this repo's existing 4-arg `_call_base` adapter — so
ported models plug into `StandardForecastAdapter` unchanged. When porting, adapt
imports from `from layers.X import ...` to `from .layers.X import ...` (or
`from tslib.model.layers.X import ...`) to match this repo's package layout.

## Current state (verified)

- Registry (`_LEGACY` in `tslib/model/registry.py`) has 13 models; the only
  unregistered `.py` files are `_xpatch_rev`/`scinet_old` (old variants — not
  registered, out of scope).
- 5 models construct today: `lstm`, `patchtst`, `itransformer`, `timemixer`,
  `timesnet`. The other 8 fail:
  - `micn` (needs `conv_kernel`), `nonstationary_transformer` (`p_hidden_dims`,
    `p_hidden_layers`), `patchmixer` (`mixer_kernel_size`), `scinet`
    (`current_level`, …), `segrnn` (`rnn_type`, `seg_len`, …), `segrnn_thuml`
    (`seg_len`, …), `xpatch` (`padding_patch`, …) — all missing config flags.
  - `etsformer` — imports `from .layers.Embed import DataEmbedding` and
    `from .layers.ETSformer_EncDec import ...`; neither file exists (repo has
    lowercase `embed.py` but no `Embed.py`, no `ETSformer_EncDec.py`).
- Exact missing flags/defaults per model are confirmed during implementation by
  reading each model's `configs.X` usage against THUML's `run.py` argparse.

## Workstreams

### A. Enable the 7 config-missing registered models

**Files:** `tslib/configs/config.py` (add flags with THUML defaults).

For each of `micn`, `nonstationary_transformer`, `patchmixer`, `scinet`,
`segrnn`, `segrnn_thuml`, `xpatch`: read the model's `configs.X` references,
cross-check the default in THUML `run.py`, and add any missing argument to
`exp_parser` with that default. No model source changes expected (they already
live in the repo); only config wiring. Representative flags (exact set finalized
per-model during implementation): `conv_kernel` (MICN), `p_hidden_dims` +
`p_hidden_layers` (Nonstationary), `mixer_kernel_size` (PatchMixer),
`current_level`/`num_stacks`/etc. (SCINet), `rnn_type`/`seg_len`/`dec_way`/etc.
(SegRNN family), `padding_patch`/`ma_type`/`alpha`/`beta` (xPatch).

### B. Repair etsformer

**Files:** `tslib/model/layers/Embed.py` (new), `tslib/model/layers/
ETSformer_EncDec.py` (new); possibly a small `config.py` flag if etsformer reads
an unwired attr.

Port `Embed.py` (with `DataEmbedding` and its dependencies) and
`ETSformer_EncDec.py` from THUML, adapting internal imports to this repo's
layers package. Reuse existing layer files where THUML's `Embed.py` depends on
them. etsformer requires `e_layers == d_layers` (asserted in its `__init__`) —
the config defaults already satisfy this (both default paths resolve equal); if
not, the construct test will catch it and the flags are adjusted.

### C. Import lightweight/recent THUML models

**Files:** `tslib/model/<name>.py` (new per model), any missing
`tslib/model/layers/*.py` deps (new), `tslib/model/registry.py` (register),
`tslib/configs/config.py` (flags).

Import set: **DLinear, TSMixer, TiDE, FreTS**. For each:
1. Fetch THUML `models/<Name>.py`; save as `tslib/model/<name>.py` with imports
   adapted to `tslib.model.layers.*`.
2. Fetch and port any layer dependency it imports that is absent here.
3. Add its config flags (with THUML defaults) to `exp_parser`.
4. Add its name to `registry._LEGACY`.

(Classic transformers — Autoformer/Informer/FEDformer — are explicitly out of
scope. Koopa/TimeXer may be added later if wanted.)

### D. Fix the pre-test GPU memory leak

**Files:** `tslib/benchmark/pretest.py`.

In `main()`, after probing each model, release it: `del net`, and when CUDA is
available `torch.cuda.empty_cache()` (also drop the timing optimizer/model refs
inside `probe_sec_per_batch` or after it). This prevents the accumulation that
OOM'd the largest model at the end of the 54-probe run.

## Verification (the central guard)

Replace/extend `tslib/model/test_registry.py` with a test that iterates the
**entire** `MODEL_REGISTRY`, builds each via `build_model(name, cfg, input_size=1,
target_indices=[0], strategy="direct")` from a real `config_postprocess(
exp_parser().parse_args(...))`, runs a forward on a tiny `(2, seq_len, 1)` input,
and asserts the output is `(2, pred_len, 1)`. Any model missing a flag or layer
makes this test fail — so it certifies the whole zoo constructs and runs. A
multivariable variant (input_size>1) is included for models whose behavior
differs across channel counts (e.g. DLinear individual mode, channel-mixing
models).

Standard constraints: unittest only, `python3.12`, device-aware toy usage,
forward on CPU in tests.

## Error handling

- A model that cannot be made to construct after wiring (e.g. an irreconcilable
  THUML dependency) is reported, left unregistered (or its registration removed),
  and noted — it must not break the all-models test. The goal is "every
  *registered* model runs," so a genuinely unportable model is dropped from the
  registry with a logged reason rather than left half-wired.
- Pre-test per-cell/per-model failures already reject-and-continue (existing).

## Testing

- All-registry construct+forward test (above) — the primary gate.
- Per-workstream: after A, the 7 previously-failing models pass the all-registry
  test; after B, `etsformer` passes; after C, each new model is registered and
  passes; after D, `test_pretest` still green and the leak fix does not change
  probe results.
- Full suite stays green.

## Out of scope (YAGNI)

- Classic transformer imports (Autoformer/Informer/FEDformer/Crossformer/…).
- Registering old variants (`_xpatch_rev`, `scinet_old`).
- Hyperparameter tuning of the new models (THUML defaults only).
- Re-running the full benchmark (a separate step; the pre-test/manifest workflow
  already exists and will pick up the new models).
