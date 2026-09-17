"""Splits, calendar features, scaling, windowing and metrics shared by all models."""
import numpy as np
import pandas as pd

from . import config as C


# ----------------------------------------------------------------- splits
def split_positions(index: pd.DatetimeIndex) -> dict:
    """Integer positions of the chronological splits (end positions are exclusive)."""
    def pos(ts):
        i = int(index.get_indexer([pd.Timestamp(ts, tz=C.TZ)])[0])
        if i < 0:
            raise ValueError(f"{ts} not found in index")
        return i
    return {
        "train_end": pos(C.TRAIN_END) + 1,
        "val_start": pos(C.VAL_START), "val_end": pos(C.VAL_END) + 1,
        "test_start": pos(C.TEST_START), "test_end": pos(C.TEST_END) + 1,
    }


# ----------------------------------------------------------------- features
def calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    """[sin/cos time-of-day, sin/cos time-of-week, holiday flag] per time step.
    Cyclical encoding keeps 23:50 close to 00:00 and Sunday close to Monday."""
    minutes = (index.hour * 60 + index.minute).to_numpy()
    tod = 2 * np.pi * minutes / 1440
    tow = 2 * np.pi * (index.dayofweek.to_numpy() + minutes / 1440) / 7
    holidays = set(pd.to_datetime(C.HOLIDAYS).date)
    hol = pd.Series(index.date).isin(holidays).to_numpy()
    return np.column_stack([np.sin(tod), np.cos(tod), np.sin(tow), np.cos(tow),
                            hol]).astype(np.float32)


class LogStandardScaler:
    """log1p compresses the heavy right tail (spikes up to 5x the mean); z-scoring
    gives the networks inputs near 0 with unit variance. Fitted on TRAIN only."""

    def fit(self, y):
        z = np.log1p(y)
        self.mu, self.sd = float(z.mean()), float(z.std())
        return self

    def transform(self, y):
        return ((np.log1p(y) - self.mu) / self.sd).astype(np.float32)

    def inverse(self, z):
        return np.expm1(np.asarray(z, dtype=np.float64) * self.sd + self.mu)


def build_nn_inputs(y, index, splits, weekly_lag: bool):
    """Per-step feature matrix X[t] and scaled target z[t].

    Channels: scaled traffic, 5 calendar features, and optionally the scaled traffic
    one week before the NEXT step (z[t+1-1008]); the last row of an input window
    then tells the model what happened at the target time one week earlier.
    """
    scaler = LogStandardScaler().fit(y[:splits["train_end"]])
    z = scaler.transform(y)
    cols = [z[:, None], calendar_features(index)]
    if weekly_lag:
        lag = np.zeros_like(z)
        lag[C.WEEK - 1:] = z[:len(z) - (C.WEEK - 1)]
        cols.append(lag[:, None])
    return np.concatenate(cols, axis=1).astype(np.float32), z, scaler


def gather_windows(X, targets, window):
    """Input windows X[t-window : t] for each target position t -> (n, window, features).
    Only the requested batch is copied, keeping memory use small."""
    idx = np.asarray(targets)[:, None] + np.arange(-window, 0)[None, :]
    return X[idx]


# ----------------------------------------------------------------- metrics
def compute_metrics(actual, pred) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = pred - actual
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(np.mean(np.abs(err) / np.abs(actual)) * 100),
    }
