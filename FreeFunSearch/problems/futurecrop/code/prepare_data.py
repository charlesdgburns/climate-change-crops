"""
prepare_data.py
---------------
Transform the FutureCrop parquet files into the per-location format used by
FunSearch. One-off step: run this once per cell subsample, then everything
else loads from the cached .npy files (reading parquet takes ~tens of minutes).

Output
------
For each split (train / validation) writes to problems/futurecrop/data/prepared/:
    climate_train.npy / climate_val.npy : float32 (T, 240, 5) climate matrix
    meta_train.npy    / meta_val.npy    : float32 (T, 21)     metadata matrix
    y_train.npy       / y_val.npy       : float32 (T,)        yield target
    cell_info.npy                        : int64   (C, 3)     (cell_id, crop, lon, lat)
    prep.json                            : column index map + shapes

Split is TEMPORAL, not random: train = years 381-409, validation = 410-419.
This mirrors the competition (test set = future years with higher CO2 than
anything in training), so the LLM must generalise across time, not just interpolate.

Climate channels (240 days after sowing), fixed column order:
    0 tasmax (degC), 1 tasmin (degC), 2 pr (mm/day), 3 rsds (W/m2),
    4 cumulative rsds (running sum of channel 3)

Meta columns (fixed index order, used by the searched model() functions):
    0  cell_id                (int, unique per (crop, lon, lat))
    1  lon
    2  lat
    3  year                   (381-419)
    4  co2                    (annual atmospheric CO2, ppm)
    5  nitrogen               (kg/ha)
    6  ..18 texture one-hot   (13 classes, class values 1-13)
    19 crop                   (0=wheat, 1=maize)
    20 yield_train_cell_mean  (per-cell mean yield computed from TRAIN rows only)

Usage
-----
    python prepare_data.py --data-dir ../../../../data --n-cells 3000

--n-cells  : max cells to sample per crop (None = all cells; large memory).
             Cells are chosen deterministically (seeded shuffle) so repeated
             runs with the same n-cells give identical caches.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).parent
PROBLEM_DIR = CODE_DIR.parent
DEFAULT_DATA_DIR = (PROBLEM_DIR / ".." / ".." / ".." / "data").resolve()
OUT_DIR = PROBLEM_DIR / "data" / "prepared"

CROPS = ["wheat", "maize"]
CROP_CODE = {"wheat": 0, "maize": 1}
CLIMATE_FEATURES = ["tasmax", "tasmin", "pr", "rsds"]
N_DAYS = 240
N_TEXTURE = 13  # texture classes 1..13
VAL_YEAR_MIN = 410  # validation = years >= this (410-419); train = 381-409

PREP_JSON = {
    "climate_channels": ["tasmax", "tasmin", "pr", "rsds", "cumulative_rsds"],
    "meta_indices": {
        "cell_id": 0, "lon": 1, "lat": 2, "year": 3, "co2": 4, "nitrogen": 5,
        "texture_start": 6, "texture_n": N_TEXTURE, "crop": 19,
        "yield_train_cell_mean": 20,
    },
    "train_years": [381, 409],
    "val_years": [VAL_YEAR_MIN, 419],
    "n_days": N_DAYS,
    "crop_codes": CROP_CODE,
}


def _read_climate(crop: str, feature: str, data_dir: Path) -> np.ndarray:
    """Return (T, 240) float32 day-series for one feature parquet, in row order."""
    f = data_dir / f"{feature}_{crop}_train.parquet"
    df = pd.read_parquet(f, columns=[str(i) for i in range(N_DAYS)])
    return df.to_numpy(dtype=np.float32)


def _read_meta(crop: str, data_dir: Path) -> pd.DataFrame:
    """Metadata table indexed like the climate frames: year, lon, lat, soil."""
    soil = pd.read_parquet(data_dir / f"soil_co2_{crop}_train.parquet")
    return soil[["year", "lon", "lat", "texture_class", "co2", "nitrogen"]]


def _read_yield(crop: str, data_dir: Path) -> np.ndarray:
    y = pd.read_parquet(data_dir / f"train_solutions_{crop}.parquet")
    return y["yield"].to_numpy(dtype=np.float64)


def prepare_crop(crop: str, data_dir: Path, n_cells: int | None):
    """Build climate/meta/y arrays for one crop, optionally cell-subsampled."""
    yield_ = _read_yield(crop, data_dir)
    meta_df = _read_meta(crop, data_dir)
    n = len(yield_)
    assert len(meta_df) == n, f"{crop}: yield ({n}) vs soil ({len(meta_df)}) mismatch"

    # Read climate features; verify consistent row ordering across files.
    tasmax = _read_climate(crop, "tasmax", data_dir)
    tasmin = _read_climate(crop, "tasmin", data_dir)
    pr = _read_climate(crop, "pr", data_dir)
    rsds = _read_climate(crop, "rsds", data_dir)
    pr_mm = pr * 1000.0  # stored in ~m/day; convert to mm/day
    cum_rsds = np.cumsum(rsds, axis=1).astype(np.float32)
    climate = np.stack([tasmax, tasmin, pr_mm, rsds, cum_rsds], axis=-1)

    # --- Build the per-location index: unique (lon, lat) per crop ---
    loc = meta_df.groupby(["lon", "lat"], sort=False).ngroup()
    loc = loc.to_numpy()
    n_locs = loc.max() + 1

    if n_cells is not None and n_cells < n_locs:
        rng = np.random.default_rng(0)
        keep = sorted(rng.choice(n_locs, size=n_cells, replace=False))
        mask = np.isin(loc, keep)
        # Renumber kept cells 0..n_cells-1 for smaller cell ids
        remap = np.full(n_locs, -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        loc = remap[loc[mask]]
        climate = climate[mask]
        meta_df = meta_df.iloc[mask]
        yield_ = yield_[mask]

    cell_id = loc.astype(np.int64)

    # --- Split temporally: train years 381-409, validation 410-419 ---
    year = meta_df["year"].to_numpy()
    train_mask = year < VAL_YEAR_MIN
    val_mask = ~train_mask

    # --- Assemble meta matrix (order: cell_id, lon, lat, year, co2, nitrogen,
    #     texture one-hot, crop, yield_train_cell_mean) ---
    texture = meta_df["texture_class"].to_numpy().astype(np.int64) - 1
    texture_oh = np.zeros((len(texture), N_TEXTURE), dtype=np.float32)
    texture_oh[np.arange(len(texture)), texture] = 1.0

    # per-cell train mean yield (train rows only; broadcast to val rows too)
    cell_mean = np.zeros(cell_id.max() + 1, dtype=np.float64)
    np.add.at(cell_mean, cell_id, train_mask * yield_)
    cell_count = np.zeros_like(cell_mean)
    np.add.at(cell_count, cell_id, train_mask)
    with np.errstate(invalid="ignore"):
        cell_mean = np.where(cell_count > 0, cell_mean / np.maximum(cell_count, 1), np.nan)
    cell_mean_yield = cell_mean[cell_id]

    meta = np.column_stack([
        cell_id, meta_df["lon"], meta_df["lat"], year,
        meta_df["co2"], meta_df["nitrogen"], texture_oh,
        np.full(len(cell_id), CROP_CODE[crop]), cell_mean_yield,
    ]).astype(np.float32)

    return climate, meta, yield_.astype(np.float32), cell_id, train_mask, val_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--n-cells", type=int, default=3000,
                    help="max cells to sample per crop (None = all; large memory)")
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    if not (data_dir / "tasmax_wheat_train.parquet").exists():
        sys.exit(f"data dir not found: {data_dir}. Pass --data-dir.")

    print(f"Reading parquet from {data_dir}  (this is the slow step, ~minutes)")
    all_climate, all_meta, all_y = [], [], []
    id_offset = 0
    for crop in CROPS:
        print(f"  [{crop}] loading + stacking ...")
        clim, meta, y, cell_id, tr, va = prepare_crop(crop, data_dir, args.n_cells)
        # Unique cell ids across crops: offset each crop's ids so no two crops collide.
        meta[:, 0] = meta[:, 0] + id_offset
        id_offset += cell_id.max() + 1
        n_tr, n_va = int(tr.sum()), int(va.sum())
        print(f"    rows={len(y)}  train={n_tr}  val={n_va}  cells={cell_id.max() + 1}")
        all_climate.append(clim)
        all_meta.append(meta)
        all_y.append(y)

    climate = np.concatenate(all_climate, axis=0)
    meta = np.concatenate(all_meta, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Rebuild split masks on the concatenated rows; temporal split is per-crop
    # so recompute by year (cheap, exact).
    year = meta[:, PREP_JSON["meta_indices"]["year"]]
    train_mask = year < VAL_YEAR_MIN
    val_mask = ~train_mask

    # Recompute per-cell train-mean yield on concatenated data for safety
    cell_mean = np.zeros(meta[:, 0].max().astype(int) + 1, dtype=np.float64)
    ci = meta[:, 0].astype(np.int64)
    np.add.at(cell_mean, ci, train_mask * y)
    cnt = np.zeros_like(cell_mean)
    np.add.at(cnt, ci, train_mask)
    with np.errstate(invalid="ignore"):
        cell_mean = np.where(cnt > 0, cell_mean / np.maximum(cnt, 1), np.nan)
    # Cells with no training rows at all: fall back to the crop-level train mean.
    missing_cells = np.isnan(cell_mean)
    if np.any(missing_cells):
        for name, code in CROP_CODE.items():
            crop_rows = meta[:, 19] == code
            tr_rows = crop_rows & train_mask
            if not np.any(tr_rows):
                continue
            crop_mean = float(np.mean(y[tr_rows]))
            crop_cells = np.unique(ci[crop_rows])
            fill = missing_cells & np.isin(np.arange(len(cell_mean)), crop_cells)
            cell_mean[fill] = crop_mean
    meta[:, PREP_JSON["meta_indices"]["yield_train_cell_mean"]] = cell_mean[ci]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "climate_train.npy", climate[train_mask])
    np.save(OUT_DIR / "meta_train.npy", meta[train_mask])
    np.save(OUT_DIR / "y_train.npy", y[train_mask])
    np.save(OUT_DIR / "climate_val.npy", climate[val_mask])
    np.save(OUT_DIR / "meta_val.npy", meta[val_mask])
    np.save(OUT_DIR / "y_val.npy", y[val_mask])

    loc = meta[:, [0, 19, 1, 2]]  # cell_id, crop, lon, lat
    _, idx = np.unique(loc[:, 0], return_index=True)
    cell_info = loc[idx].astype(np.float64)
    cell_info[:, 0] = cell_info[:, 0].astype(np.int64)
    np.save(OUT_DIR / "cell_info.npy", cell_info)

    prep = dict(PREP_JSON)
    prep.update({
        "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()),
        "n_cells": len(cell_info),
        "climate_shape": list(climate.shape),
        "meta_shape": list(meta[train_mask].shape),
    })
    (OUT_DIR / "prep.json").write_text(json.dumps(prep, indent=2))

    print(f"\nWrote prepared data to {OUT_DIR}")
    print(json.dumps(prep, indent=2))


if __name__ == "__main__":
    main()