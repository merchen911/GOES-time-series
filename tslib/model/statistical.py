from __future__ import annotations

from typing import Dict

import numpy as np

STAT_REGISTRY: Dict[str, type] = {}


def register_stat(name):
    def deco(cls):
        STAT_REGISTRY[name] = cls
        return cls
    return deco


@register_stat("arima")
class ArimaModel:
    def __init__(self, config):
        self.order = tuple(getattr(config, "arima_order", [1, 0, 0]))

    def fit(self, history):
        from statsmodels.tsa.arima.model import ARIMA
        self._res = ARIMA(np.asarray(history, dtype=float),
                          order=self.order).fit()
        return self

    def forecast(self, steps):
        return np.asarray(self._res.forecast(steps), dtype=float)


@register_stat("ar")
class ArModel:
    def __init__(self, config):
        self.lags = int(getattr(config, "ar_lags", 1))

    def fit(self, history):
        from statsmodels.tsa.ar_model import AutoReg
        self._hist = np.asarray(history, dtype=float)
        self._res = AutoReg(self._hist, lags=self.lags, old_names=False).fit()
        return self

    def forecast(self, steps):
        n = len(self._hist)
        return np.asarray(self._res.predict(start=n, end=n + steps - 1),
                          dtype=float)


@register_stat("theta")
class ThetaModelWrap:
    def __init__(self, config):
        pass

    def fit(self, history):
        from statsmodels.tsa.forecasting.theta import ThetaModel
        self._res = ThetaModel(np.asarray(history, dtype=float),
                               period=1).fit()
        return self

    def forecast(self, steps):
        return np.asarray(self._res.forecast(steps), dtype=float)
