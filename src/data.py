"""Memory-efficient loading of the Milan Telecom 'SMS, Call, Internet - MI' dataset.

Strategy
--------
1. Process the dataset one daily file at a time (never hold all raw rows in RAM).
2. Read only the 3 needed columns (usecols) with compact dtypes
   (int16 square id, int64 timestamp, float32 internet).
3. Aggregate away the country_code dimension immediately (groupby-sum).
4. Write into a pre-allocated dense float32 matrix [time slots x squares]
   (8928 x 10000 x 4 bytes ~ 357 MB), saved as .npy so it can be memory-mapped later.
"""
import gc
import os
import time

import numpy as np
import pandas as pd
import psutil

from . import config as C


def rss_mb() -> float:
    """Resident memory of the current process in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def list_raw_files(raw_dir=C.RAW_DIR):
    files = sorted(list(raw_dir.glob("*.txt")) + list(raw_dir.glob("*.zip")) +
                   list(raw_dir.glob("*.csv")))
    if not files:
        raise FileNotFoundError(f"No raw files found in {raw_dir}")
    return files


def read_naive(path) -> pd.DataFrame:
    """Baseline: default pandas read, all columns, default (64-bit) dtypes."""
    return pd.read_csv(path, sep="\t", header=None, names=C.RAW_COLUMNS)


def read_optimised(path) -> pd.DataFrame:
    """Only needed columns, compact dtypes, country codes aggregated immediately."""
    df = pd.read_csv(
        path, sep="\t", header=None, names=C.RAW_COLUMNS,
        usecols=["square_id", "time_interval", "internet"],
        dtype={"square_id": "int16", "time_interval": "int64", "internet": "float32"},
    )
    # sum() skips NaN: a (square, slot) with no internet records becomes 0
    out = df.groupby(["square_id", "time_interval"], sort=False)["internet"].sum()
    return out.reset_index()


def benchmark_single_file(path) -> pd.DataFrame:
    """Measure DataFrame size, process RSS growth and time for both loaders."""
    rows = []
    for name, loader in [("naive", read_naive), ("optimised", read_optimised)]:
        gc.collect()
        before = rss_mb()
        t0 = time.perf_counter()
        df = loader(path)
        elapsed = time.perf_counter() - t0
        rows.append({
            "loader": name,
            "rows": len(df),
            "columns": df.shape[1],
            "dataframe_mb": df.memory_usage(deep=True).sum() / 1024 ** 2,
            "rss_increase_mb": rss_mb() - before,
            "load_seconds": elapsed,
        })
        del df
        gc.collect()
    return pd.DataFrame(rows)


def build_internet_matrix(files):
    """Stream daily files into a dense float32 matrix [N_SLOTS, N_SQUARES]."""
    origin_ms = int(pd.Timestamp(C.START, tz=C.TZ).timestamp() * 1000)
    slot_ms = C.SLOT_MINUTES * 60 * 1000
    M = np.zeros((C.N_SLOTS, C.N_SQUARES), dtype=np.float32)
    dropped = 0
    peak = rss_mb()
    for i, f in enumerate(files, 1):
        day = read_optimised(f)
        slot = (day["time_interval"].to_numpy() - origin_ms) // slot_ms
        sq = day["square_id"].to_numpy().astype(np.int64) - 1
        ok = (slot >= 0) & (slot < C.N_SLOTS) & (sq >= 0) & (sq < C.N_SQUARES)
        dropped += int((~ok).sum())
        np.add.at(M, (slot[ok], sq[ok]), day["internet"].to_numpy()[ok])
        del day, slot, sq, ok
        gc.collect()
        peak = max(peak, rss_mb())
        print(f"[{i:02d}/{len(files)}] {f.name}  RSS={rss_mb():.0f} MB")
    index = pd.date_range(C.START, periods=C.N_SLOTS,
                          freq=f"{C.SLOT_MINUTES}min", tz=C.TZ)
    return M, index, dropped, peak


def fill_missing_slots(M):
    """Slots where the whole city reports zero traffic are data gaps, not real zeros.
    They are marked and linearly interpolated per square."""
    missing = M.sum(axis=1) == 0
    if missing.any():
        df = pd.DataFrame(M)
        df.loc[missing] = np.nan
        M = df.interpolate(limit_direction="both").to_numpy(dtype=np.float32)
    return M, missing
