from __future__ import annotations

from tslib.model import build_model
from tslib.exp.lightning_model import pl_model, TrainResult


def run_strategy(strategy, model_name, data_bundle, config, ckpt_path) -> TrainResult:
    if strategy in ("direct", "recursive"):
        model = build_model(model_name, config, data_bundle.input_size,
                            data_bundle.target_indices, strategy=strategy)
        result = pl_model(model, config).fit_and_test(
            data_bundle, model_name=model_name, ckpt_path=ckpt_path)
        result.strategy = strategy
        return result
    if strategy == "statistic":
        raise ValueError("statistic strategy is not available until Phase 2")
    raise ValueError(f"unknown forecast_strategy '{strategy}'")
