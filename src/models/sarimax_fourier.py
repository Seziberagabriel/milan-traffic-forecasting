"""Model 1: regression on Fourier terms with ARMA(p, q) errors (SARIMAX with exogenous terms).

Why Fourier terms instead of seasonal ARIMA: the season lengths are 144 (daily) and
1008 (weekly) steps. A seasonal ARMA with s=1008 needs a huge state vector and is not
practical on a laptop; Fourier terms capture both cycles with a few coefficients
(Hyndman & Athanasopoulos, FPP3, dynamic harmonic regression).
"""
import time
import warnings

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .. import config as C
from ..features import calendar_features

DAY = 144


def fourier_terms(n, period, K, skip_multiples_of=None):
    t = np.arange(n)
    cols = []
    for k in range(1, K + 1):
        if skip_multiples_of and k % skip_multiples_of == 0:
            continue  # weekly harmonic k=7 equals the daily fundamental: avoid collinearity
        cols += [np.sin(2 * np.pi * k * t / period), np.cos(2 * np.pi * k * t / period)]
    return np.column_stack(cols) if cols else np.empty((n, 0))


def build_exog(index, p):
    n = len(index)
    parts = [fourier_terms(n, DAY, p["daily_k"]),
             fourier_terms(n, C.WEEK, p["weekly_k"], skip_multiples_of=7)]
    if p["holiday"]:
        parts.append(calendar_features(index)[:, [4]])
    return np.column_stack(parts)


def fit_predict(y, index, splits, eval_start, eval_end, p):
    """Fit on the training split; one-step-ahead forecasts for positions [eval_start, eval_end).

    One-step-ahead: the fitted parameters are re-applied with the Kalman filter to the
    series up to eval_end, and each prediction uses only observations before it
    (dynamic=False). No refitting on validation/test data.
    """
    target = np.log1p(y) if p["log"] else y.astype(np.float64)
    exog = build_exog(index, p)
    order = (p["p"], 0, p["q"])
    n_fit = splits["train_end"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        res = SARIMAX(target[:n_fit], exog=exog[:n_fit], order=order, trend="c").fit(
            disp=False, maxiter=p["maxiter"])
        train_seconds = time.perf_counter() - t0

    def predict():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = SARIMAX(target[:eval_end], exog=exog[:eval_end], order=order, trend="c")
            out = full.filter(res.params).get_prediction(
                start=eval_start, end=eval_end - 1, dynamic=False).predicted_mean
        out = np.asarray(out, dtype=np.float64)
        return np.expm1(out) if p["log"] else out

    info = {"train_seconds": train_seconds, "aic": float(res.aic), "bic": float(res.bic),
            "converged": bool(res.mle_retvals.get("converged", True)) if res.mle_retvals else None,
            "n_params": int(len(res.params))}
    return predict, info
