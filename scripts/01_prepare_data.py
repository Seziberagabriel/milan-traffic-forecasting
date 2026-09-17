"""Stage 1: memory benchmark + build the processed internet-traffic matrix.

Run from the repo root:  python -m scripts.01_prepare_data
"""
import json
import time

import numpy as np
import pandas as pd

from src import config as C
from src.data import (benchmark_single_file, build_internet_matrix,
                      fill_missing_slots, list_raw_files, rss_mb)


def main():
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = list_raw_files()
    print(f"Found {len(files)} raw files")

    # 1) Before/after memory evidence on one daily file
    bench = benchmark_single_file(files[0])
    bench["extrapolated_full_dataset_gb"] = bench["dataframe_mb"] * len(files) / 1024
    bench.to_csv(C.RESULTS_DIR / "memory_benchmark.csv", index=False)
    print("\n=== Memory benchmark (one daily file) ===")
    print(bench.round(2).to_string(index=False))

    # 2) Stream everything into a float32 matrix
    t0 = time.perf_counter()
    M, index, dropped, peak = build_internet_matrix(files)
    M, missing = fill_missing_slots(M)
    build_s = time.perf_counter() - t0

    np.save(C.PROCESSED_DIR / "internet_matrix.npy", M)
    pd.Series(index.astype(str)).to_csv(C.PROCESSED_DIR / "timestamps.csv", index=False)
    np.save(C.PROCESSED_DIR / "missing_mask.npy", missing)

    totals = M.sum(axis=0)
    top3 = (np.argsort(totals)[::-1][:3] + 1).tolist()
    hourly = pd.Series(M.sum(axis=1), index=index).groupby(index.hour).mean()

    summary = {
        "n_files": len(files),
        "matrix_shape": list(M.shape),
        "matrix_mb": round(M.nbytes / 1024 ** 2, 1),
        "peak_rss_mb_during_build": round(peak, 1),
        "final_rss_mb": round(rss_mb(), 1),
        "build_seconds": round(build_s, 1),
        "rows_outside_time_range_dropped": dropped,
        "missing_slots_interpolated": int(missing.sum()),
        "top3_squares": top3,
        "top3_totals": [float(totals[s - 1]) for s in top3],
        "quietest_local_hour": int(hourly.idxmin()),  # sanity check: expect ~3-5 am
        "busiest_local_hour": int(hourly.idxmax()),
    }
    with open(C.RESULTS_DIR / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
