import argparse


def exp_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
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
    parser.add_argument("--models", nargs="+", default=["lstm", "timesnet"])
    parser.add_argument("--forecast_strategy", type=str, default="direct",
                        choices=["direct", "recursive", "statistic"],
                        help="how the forecast horizon is produced")
    parser.add_argument("--hidden_size", type=int, default=64)

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

    # modeling: pluggable loss / metrics
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--metrics", nargs="+", default=["mse", "mae"])
    parser.add_argument("--event_threshold", nargs="*", type=float, default=None)
    parser.add_argument("--sort_metric", type=str, default="best_val_loss")

    return parser


def config_postprocess(config):
    if not 0 < config.train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= config.val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if config.train_ratio + config.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")
    if config.label_len <= 0:
        config.label_len = max((config.seq_len + config.pred_len) // 4, 1)
    if config.n_fold < 3:
        raise ValueError("n_fold must be >= 3.")
    if not 0 <= config.fold_numb < config.n_fold:
        raise ValueError("fold_numb must be within [0, n_fold).")
    if (config.split_type in {"year", "year_half"} and not config.time_col
            and not str(config.data_path).endswith(".parquet")):
        raise ValueError("time-based k-fold requires --time_col.")

    from tslib.exp.metrics import METRIC_REGISTRY
    for m in config.metrics:
        if m not in METRIC_REGISTRY:
            raise ValueError(
                f"unknown metric '{m}'; registered: {sorted(METRIC_REGISTRY)}")
    event = [m for m in config.metrics
             if METRIC_REGISTRY[m].kind == "event"]
    if event:
        n_targets = (len(config.target_cols)
                     if getattr(config, "target_cols", None) else 1)
        if not config.event_threshold:
            raise ValueError(f"event metrics {event} require --event_threshold")
        if len(config.event_threshold) != n_targets:
            raise ValueError(
                f"--event_threshold length {len(config.event_threshold)} "
                f"!= number of targets {n_targets}")

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

    return config
