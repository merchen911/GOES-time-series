import argparse


def exp_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--runs_root", type=str, default="runs",
                        help="directory that holds run directories. Relative "
                             "to the working directory by default; give an "
                             "absolute path to write a sweep somewhere else "
                             "(e.g. a scratch root for a short-epoch pilot, so "
                             "it cannot mark real sweep jobs as finished).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10000,
                        help="max epochs; training stops earlier via EarlyStopping")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)

    # data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    parser.add_argument("--feature_cols", nargs="*", default=None)
    parser.add_argument("--time_col", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--split_type", type=str, default="year_half", choices=["year_half", "year", "ratio"])
    parser.add_argument("--n_fold", type=int, default=5)
    parser.add_argument("--fold_numb", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--role", type=str, default="primary",
                        help="parquet: which 'role' rows to use (primary/secondary)")
    parser.add_argument("--transform", type=str, default="log10",
                        choices=["none", "log10"], help="target transform")
    parser.add_argument("--cadence_min", type=int, default=5,
                        help="regular-grid step in minutes for parquet windowing")
    parser.add_argument("--channels", nargs="*", default=None,
                        help="multivariate input channels as PATH:COL (>=1). "
                             "If omitted, uses --data_path:--target_col.")
    parser.add_argument("--target_cols", nargs="*", default=None,
                        help="prediction target columns (subset of channels). "
                             "Default: first channel's column.")
    parser.add_argument("--min_bin_count", type=int, default=1,
                        help="min native samples per resample bin to keep it")

    # model comparison
    parser.add_argument("--models", nargs="+", default=["lstm", "patchtst"])
    parser.add_argument("--forecast_strategy", type=str, default="direct",
                        choices=["direct", "recursive", "statistic"],
                        help="how the forecast horizon is produced")
    parser.add_argument("--arima_order", nargs=3, type=int, default=[1, 0, 0],
                        help="ARIMA (p,d,q) for statistic strategy")
    parser.add_argument("--ar_lags", type=int, default=1,
                        help="AR lag count for statistic strategy")
    parser.add_argument("--hidden_size", type=int, default=64)

    # pre-test (probe) + early stopping
    parser.add_argument("--probe_batches", type=int, default=3,
                        help="training batches timed by the pre-test tool")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="EarlyStopping patience in epochs (monitors val_loss)")

    # 표준 모델 공통 기본값
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--embed", type=str, default="fixed")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--num_class", type=int, default=1)
    # iTransformer
    parser.add_argument("--output_attention", action="store_true",
                        help="iTransformer: return attention weights")
    # TimeMixer (THUML Time-Series-Library defaults)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--decomp_method", type=str, default="moving_avg",
                        choices=["moving_avg", "dft_decomp"])
    parser.add_argument("--channel_independence", type=int, default=1)
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--down_sampling_layers", type=int, default=3)
    parser.add_argument("--down_sampling_window", type=int, default=2)
    parser.add_argument("--down_sampling_method", type=str, default="avg",
                        choices=["avg", "max", "conv"])

    # extra backbone flags (THUML defaults)
    parser.add_argument("--conv_kernel", nargs="+", type=int, default=[12, 16],
                        help="MICN: downsampling/upsampling conv kernel sizes")
    parser.add_argument("--p_hidden_dims", nargs="+", type=int, default=[128, 128],
                        help="Non-stationary Transformer: projector hidden dims")
    parser.add_argument("--p_hidden_layers", type=int, default=2,
                        help="Non-stationary Transformer: projector hidden layers")
    parser.add_argument("--current_level", type=int, default=3,
                        help="SCINet: tree recursion depth")
    parser.add_argument("--causal_block_level", type=int, default=4,
                        help="SCINet: causal conv blocks per SCIBlock")
    parser.add_argument("--seg_len", type=int, default=24,
                        help="SegRNN: segment length for segment-wise iteration "
                             "(must evenly divide both seq_len and pred_len; "
                             "48 does not divide pred_len=24 and produces an "
                             "invalid reshape in SegRNN's decoder)")
    # PatchMixer / SegRNN / xPatch (upstream repo defaults)
    parser.add_argument("--revin", type=int, default=1,
                        help="PatchMixer/SegRNN/xPatch: apply RevIN normalization")
    parser.add_argument("--affine", type=int, default=0,
                        help="PatchMixer: RevIN affine transform")
    parser.add_argument("--subtract_last", type=int, default=0,
                        help="PatchMixer: RevIN subtract-last instead of mean")
    parser.add_argument("--head_dropout", type=float, default=0.0,
                        help="PatchMixer: dropout in the prediction heads")
    parser.add_argument("--mixer_kernel_size", type=int, default=8,
                        help="PatchMixer: depthwise conv kernel size")
    parser.add_argument("--rnn_type", type=str, default="gru",
                        choices=["rnn", "gru", "lstm"],
                        help="SegRNN: recurrent cell type")
    parser.add_argument("--dec_way", type=str, default="pmf",
                        choices=["rmf", "pmf"],
                        help="SegRNN: decoding scheme")
    parser.add_argument("--channel_id", type=int, default=1,
                        help="SegRNN: use per-channel positional embedding")
    parser.add_argument("--padding_patch", type=str, default="end",
                        help="xPatch: patch padding mode")
    parser.add_argument("--ma_type", type=str, default="ema",
                        choices=["ema", "dema", "reg"],
                        help="xPatch: moving-average decomposition type")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="xPatch: EMA/DEMA smoothing factor")
    parser.add_argument("--beta", type=float, default=0.3,
                        help="xPatch: DEMA secondary smoothing factor")

    # modeling: pluggable loss / metrics
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="metrics to compute (default: every registered "
                             "one); contingency counts are always recorded")
    parser.add_argument("--event_threshold", nargs="*", type=float, default=None,
                        help="event threshold per target, physical units; "
                             "without it no event rows are written")
    parser.add_argument("--sort_metric", type=str, default="best_val_loss",
                        help="ranks the regression comparison table: "
                             "best_val_loss or a regression metric except sspb")
    parser.add_argument("--save_predictions", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="write <runs_root>/<run_name>/pred/<model>.npz "
                             "after testing")
    # the underscore spelling, like every other flag (BooleanOptionalAction
    # only makes --no-save_predictions)
    parser.add_argument("--no_save_predictions", dest="save_predictions",
                        action="store_false", help=argparse.SUPPRESS)

    return parser


def config_postprocess(config):
    if not 0 < config.train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= config.val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if config.train_ratio + config.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")
    if "micn" in getattr(config, "models", []):
        # MICN's decoder rebuilds its input from the full seq_len (ignoring
        # label_len) but is fed x_mark_dec of length label_len+pred_len, so
        # label_len must equal seq_len for the shapes to line up (matches
        # THUML's own MICN scripts, which always set label_len == seq_len).
        # This must hold unconditionally, even if the user passed an explicit
        # --label_len, since any other value breaks MICN's shapes.
        config.label_len = config.seq_len
    elif config.label_len <= 0:
        config.label_len = max((config.seq_len + config.pred_len) // 4, 1)
    if "etsformer" in getattr(config, "models", []):
        # etsformer asserts e_layers == d_layers internally; auto-reconcile so
        # it is runnable from defaults (e_layers=2, d_layers=1) without
        # requiring the caller to pass --d_layers explicitly.
        config.d_layers = config.e_layers
    if config.n_fold < 3:
        raise ValueError("n_fold must be >= 3.")
    if not 0 <= config.fold_numb < config.n_fold:
        raise ValueError("fold_numb must be within [0, n_fold).")
    if (config.split_type in {"year", "year_half"} and not config.time_col
            and not str(config.data_path).endswith(".parquet")):
        raise ValueError("time-based k-fold requires --time_col.")

    from tslib.exp.metrics import EVENT_SCORES, REGRESSION_METRICS
    registered = [*REGRESSION_METRICS, *EVENT_SCORES]
    for m in config.metrics or []:
        if m not in registered:
            raise ValueError(f"unknown metric '{m}'; registered: {registered}")
    event = [m for m in config.metrics or [] if m in EVENT_SCORES]
    if event and not config.event_threshold:
        raise ValueError(f"event metrics {event} require --event_threshold")
    if config.event_threshold:
        n_targets = (len(config.target_cols)
                     if getattr(config, "target_cols", None) else 1)
        if len(config.event_threshold) != n_targets:
            raise ValueError(
                f"--event_threshold length {len(config.event_threshold)} "
                f"!= number of targets {n_targets}")
    # sspb is signed with 0 best, so neither sort direction ranks models
    sortable = ["best_val_loss", *(m for m in REGRESSION_METRICS if m != "sspb")]
    if config.sort_metric not in sortable:
        raise ValueError(
            f"--sort_metric '{config.sort_metric}' cannot rank the regression "
            f"table; choose one of {sortable}")
    if (config.metrics is not None and config.sort_metric != "best_val_loss"
            and config.sort_metric not in config.metrics):
        raise ValueError(
            f"--sort_metric '{config.sort_metric}' is not computed with "
            f"--metrics {' '.join(config.metrics)}; add it to --metrics or sort "
            f"by best_val_loss")

    strategy = getattr(config, "forecast_strategy", "direct")
    if strategy in ("direct", "recursive"):
        from tslib.model.registry import MODEL_REGISTRY
        for m in config.models:
            if m not in MODEL_REGISTRY:
                raise ValueError(
                    f"unknown model '{m}' for strategy '{strategy}'; "
                    f"registered: {sorted(MODEL_REGISTRY)}")
    if strategy == "recursive":
        n_ch = len(config.channels) if getattr(config, "channels", None) else 1
        n_tgt = len(config.target_cols) if getattr(config, "target_cols", None) else 1
        if n_tgt != n_ch:
            raise ValueError(
                "recursive strategy must forecast all input channels: "
                f"target count {n_tgt} != channel count {n_ch}")
    if strategy == "statistic":
        from tslib.model.statistical import STAT_REGISTRY
        for m in config.models:
            if m not in STAT_REGISTRY:
                raise ValueError(
                    f"unknown statistical model '{m}'; "
                    f"registered: {sorted(STAT_REGISTRY)}")
        n_tgt = len(config.target_cols) if getattr(config, "target_cols", None) else 1
        if n_tgt != 1:
            raise ValueError(
                f"statistic strategy is univariate; got {n_tgt} targets")

    if getattr(config, "probe_batches", 1) < 1:
        raise ValueError("probe_batches must be >= 1.")
    if getattr(config, "early_stop_patience", 1) < 1:
        raise ValueError("early_stop_patience must be >= 1.")

    return config
