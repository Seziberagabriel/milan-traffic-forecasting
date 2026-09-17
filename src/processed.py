"""Helpers to load the processed data produced by scripts/01_prepare_data.py."""
import json

import numpy as np
import pandas as pd

from . import config as C

GRID_FILE = C.ROOT / "data" / "milano-grid.geojson"


def load_matrix(mmap: bool = True):
    """Memory-map the float32 matrix (no full copy in RAM) and load timestamps."""
    M = np.load(C.PROCESSED_DIR / "internet_matrix.npy",
                mmap_mode="r" if mmap else None)
    ts = pd.read_csv(C.PROCESSED_DIR / "timestamps.csv").iloc[:, 0]
    index = pd.DatetimeIndex(pd.to_datetime(ts, utc=True).dt.tz_convert(C.TZ))
    return M, index


def square_series(M, index, square_id: int) -> pd.Series:
    """Traffic time series for one square (ids are 1-based)."""
    return pd.Series(np.asarray(M[:, square_id - 1], dtype=np.float64),
                     index=index, name=f"square_{square_id}")


def load_top_squares(k: int = 3):
    with open(C.RESULTS_DIR / "data_summary.json") as f:
        return json.load(f)["top3_squares"][:k]


def load_grid_centroids():
    """Return DataFrame [square_id, lon, lat] from the Milano grid GeoJSON, or None."""
    if not GRID_FILE.exists():
        return None
    with open(GRID_FILE) as f:
        gj = json.load(f)
    rows = []
    for feat in gj["features"]:
        props = feat.get("properties", {})
        cid = props.get("cellId", props.get("cellid", props.get("id")))
        ring = np.asarray(feat["geometry"]["coordinates"][0], dtype=float)
        rows.append((int(cid), ring[:, 0].mean(), ring[:, 1].mean()))
    return pd.DataFrame(rows, columns=["square_id", "lon", "lat"]).sort_values("square_id")
