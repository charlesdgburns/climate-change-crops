"""
load_data.py
------------
Load the prepared FutureCrop dataset from the .npy caches written by
prepare_data.py (never reads the raw parquet files).

Model input contract (returned to the searched model()):
    x is a dict with two float32 arrays, aligned on the first axis (rows):
        x['climate'] : (T, 240, 5) climate channels
                       [tasmax, tasmin, pr (mm/day), rsds, cumulative rsds]
        x['meta']    : (T, 21) fixed metadata columns, see prepare_data.py / prep.json
                       0 cell_id, 1 lon, 2 lat, 3 year, 4 co2, 5 nitrogen,
                       6..18 texture one-hot, 19 crop (0=wheat,1=maize),
                       20 yield_train_cell_mean
    y is a (T,) float array of simulated crop yield.

Split is temporal: train = years 381-409, validation = years 410-419.
"""

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data" / "prepared"
VAL_YEAR_MIN = 410


def load(dataset_name: str = "futurecrop"):
    """Return (x_train, y_train, x_val, y_val). x_* are dicts (see module doc)."""
    del dataset_name  # single prepared dataset; kept for FunSearch interface
    if not (DATA_DIR / "climate_train.npy").exists():
        raise FileNotFoundError(
            f"Prepared data not found in {DATA_DIR}. Run prepare_data.py first."
        )

    def x(clim, meta):
        return {"climate": np.load(clim, mmap_mode=None),
                "meta": np.load(meta, mmap_mode=None)}

    x_train = x(DATA_DIR / "climate_train.npy", DATA_DIR / "meta_train.npy")
    x_val = x(DATA_DIR / "climate_val.npy", DATA_DIR / "meta_val.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")
    return x_train, y_train, x_val, y_val


def cell_index(meta: np.ndarray) -> np.ndarray:
    """Return 1D int array mapping each row to its cell (see meta col 0)."""
    return meta[:, 0].astype(np.int64)


def info() -> dict:
    p = DATA_DIR / "prep.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


if __name__ == "__main__":
    info = info()
    print("prep info:", info)
    xt, yt, xv, yv = load()
    print(f"train: climate {xt['climate'].shape}  meta {xt['meta'].shape}  y {yt.shape}")
    print(f"val  : climate {xv['climate'].shape}  meta {xv['meta'].shape}  y {yv.shape}")
    yr = xt['meta'][:, 3]
    print(f"train year range [{yr.min():.0f}, {yr.max():.0f}]  "
          f"val year range [{xv['meta'][:,3].min():.0f}, {xv['meta'][:,3].max():.0f}]")