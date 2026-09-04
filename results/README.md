# Benchmark results

## `results_master.csv`

One row per completed run: `track` × `seq_len` × `pred_len` × `fold` × `model`
(876 rows; folds 0–4 of the rotating term-based split). Columns cover the
regression scores (`rmse`, `mae`) and the benchmark's logged per-window event
skill at the operational thresholds (`tss_*`, `hss_*`, `pod_*`, `far_*` for
`p_gt10` and `xrs_long`). The static leaderboard in the top-level `README.md`
is the mean of these rows over the five folds and all input-length/horizon
cells (`scripts/make_leaderboard.py`).

## `event_skill_5fold_ci.csv` / `event_skill_5fold_ci_24h.csv`

The manuscript's event-skill analysis, **re-scored at the reference cell**
(`seq_len=288` → `pred_len=144` for the 12 h file, `pred_len=288` for the 24 h
file) as the **5-fold mean ± 95% confidence-interval half-width** (Student-t,
df = 4). One row per `track` × `model` × `gran`, with `<metric>_mean` and
`<metric>_ci` for TSS, HSS, CSI, MCC, SEDI, precision and FB.

- `track` — `Proton SPE`, `XRS M-class`, `XRS X-class`, and the `… (multi)`
  variants scored from joint multivariate input.
- `gran` — `per-point` (every window × timestep), `per-window` (window-max ≥
  threshold), `per-day` (1-day-stride subsample, then window-max).
- `model` — the ten deep architectures plus `naive (persistence)`, the
  flat-value persistence floor.

Thresholds: M-class ≥ 10⁻⁵ W m⁻², X-class ≥ 10⁻⁴ W m⁻², proton SPE ≥ 10 pfu.
Blank cells are undefined metrics (e.g. SEDI when POD or POFD is 0 or 1;
precision when no event is predicted). These tables back the manuscript's
event-skill tables and figures.
