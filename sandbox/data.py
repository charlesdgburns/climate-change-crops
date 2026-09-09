"""
data.py
-------
Loads the prepared FutureCrop caches (built by
FreeFunSearch/problems/futurecrop/code/prepare_data.py) and splits them into
per-location train/test slices for the validation harness.

Design principle: each candidate model is fitted to a SINGLE cell (location).
Within one location only two inputs carry year-to-year signal: the 240-day
climate series and annual co2. Everything else in meta (nitrogen, texture,
lon, lat) is constant per location and therefore unusable within a location's
fit, so the candidate interface exposes only:
    tasmax, tasmin, pr, rsds, cumrsds   (240, n_years)   day axis first
    co2                                 (n_years,)
The constant intercept (params[0]) absorbs the per-location mean yield, so no
cell_mean / cell_id are needed.

meta columns (prep.json): 0 cell_id, 1 lon, 2 lat, 3 year, 4 co2, 5 nitrogen,
    6..18 texture one-hot, 19 crop (0=wheat, 1=maize), 20 yield_train_cell_mean
climate channels: 0 tasmax, 1 tasmin, 2 pr (mm/day), 3 rsds, 4 cumulative_rsds
"""

from pathlib import Path

import numpy as np

PREPARED_DIR = (
    Path(__file__).resolve().parent.parent
    / "FreeFunSearch" / "problems" / "futurecrop" / "data" / "prepared"
)

CROPS = {"wheat": 0.0, "maize": 1.0}

DEFAULT_TRAIN_YEARS = (381, 411)   # first 31 years
DEFAULT_TEST_YEARS = (412, 419)    # last 8 years
MIN_TRAIN_YEARS = 20               # cells with fewer train rows are not fittable
MIN_TEST_YEARS = 3
N_FEATURES = ["tasmax", "tasmin", "pr", "rsds", "cumrsds", "co2"]


class Dataset:
    """One crop's rows (all cells) + light index helpers."""

    def __init__(self, climate: np.ndarray, meta: np.ndarray, y: np.ndarray, crop: str):
        self.climate = climate
        self.meta = meta
        self.y = y
        self.crop = crop
        self.cell_ids = meta[:, 0].astype(np.int64)
        self.cell_id_set = np.unique(self.cell_ids)

    def cell_rows(self, cells) -> np.ndarray:
        """Bool mask of rows belonging to the given cells."""
        return np.isin(self.cell_ids, np.asarray(cells))

    def pick_cells(self, n_cells: int, seed: int = 0,
                   train_years=DEFAULT_TRAIN_YEARS,
                   min_train_years: int = MIN_TRAIN_YEARS) -> list[int]:
        """Sample n_cells locations that have >= min_train_years rows inside
        the train window (so every cell is fittable)."""
        tr = self.year_rows(train_years)
        idx = np.searchsorted(self.cell_id_set, self.cell_ids[tr])
        cnt = np.bincount(idx, minlength=len(self.cell_id_set))
        eligible = self.cell_id_set[cnt >= min_train_years]
        if eligible.size == 0:
            raise RuntimeError(f"no {self.crop} cell has >= {min_train_years} "
                               f"train-year rows in {train_years}")
        rng = np.random.default_rng(seed)
        k = min(n_cells, len(eligible))
        return sorted(rng.choice(eligible, size=k, replace=False).tolist())

    def year_rows(self, years) -> np.ndarray:
        """Bool mask of rows whose year is inside [years[0], years[1]]."""
        return (self.meta[:, 3] >= years[0]) & (self.meta[:, 3] <= years[1])

    @staticmethod
    def to_features(climate: np.ndarray, meta: np.ndarray) -> dict[str, np.ndarray]:
        """
        Named features for the candidate interface. Day axis first:
        tasmax..cumrsds -> (Tdays, N); co2 -> (N,).
        """
        clim = climate.astype(np.float64)
        m = meta.astype(np.float64)
        return {
            "tasmax": clim[:, :, 0].T,      # (T, N) degC
            "tasmin": clim[:, :, 1].T,      # (T, N) degC
            "pr": clim[:, :, 2].T,          # (T, N) mm/day
            "rsds": clim[:, :, 3].T,        # (T, N) W/m2
            "cumrsds": clim[:, :, 4].T,     # (T, N) W/m2
            "co2": m[:, 4],                 # (N,) ppm
        }


def load_crop(crop: str) -> Dataset:
    """Load one crop's rows from the prepared cache (train + val, years 381-419)."""
    if crop not in CROPS:
        raise ValueError(f"crop must be one of {list(CROPS)}; got {crop!r}")

    def _load(name):
        return np.load(PREPARED_DIR / name)

    climate = np.concatenate([_load("climate_train.npy"), _load("climate_val.npy")])
    meta = np.concatenate([_load("meta_train.npy"), _load("meta_val.npy")])
    y = np.concatenate([_load("y_train.npy"), _load("y_val.npy")])

    mask = meta[:, 19] == CROPS[crop]
    return Dataset(climate[mask], meta[mask], y[mask].astype(np.float64), crop)


def _as_dataset(crop_or_ds) -> Dataset:
    return load_crop(crop_or_ds) if isinstance(crop_or_ds, str) else crop_or_ds


def pick_cells(crop_or_ds, n_cells: int, seed: int = 0,
               train_years=DEFAULT_TRAIN_YEARS,
               min_train_years: int = MIN_TRAIN_YEARS) -> list[int]:
    return _as_dataset(crop_or_ds).pick_cells(n_cells, seed, train_years,
                                              min_train_years)


def per_cell_splits(crop_or_ds, cells,
                    train_years=DEFAULT_TRAIN_YEARS,
                    test_years=DEFAULT_TEST_YEARS) -> list[dict]:
    """
    Return one dict per location with that cell's train & test slices:
        {cell, n_train, n_test, train_mean,
         train: {tasmax.., co2, y}, test: {tasmax.., co2, y}}
    Cells that cannot be fitted (too few train rows) or scored (too few test
    rows) for the requested windows are skipped.
    """
    ds = _as_dataset(crop_or_ds)
    splits = []
    for c in cells:
        rows = ds.cell_rows([int(c)])
        clim, meta, yy = ds.climate[rows], ds.meta[rows], ds.y[rows]
        year = meta[:, 3]
        tr = (year >= train_years[0]) & (year <= train_years[1])
        te = (year >= test_years[0]) & (year <= test_years[1])
        ntr, nte = int(tr.sum()), int(te.sum())
        if ntr < MIN_TRAIN_YEARS or nte < MIN_TEST_YEARS:
            continue
        fit = ds.to_features(clim[tr], meta[tr])
        fit["y"] = yy[tr]
        test = ds.to_features(clim[te], meta[te])
        test["y"] = yy[te]
        splits.append({
            "cell": int(c),
            "n_train": ntr,
            "n_test": nte,
            "train_mean": float(np.mean(yy[tr])),
            "train": fit,
            "test": test,
        })
    return splits


if __name__ == "__main__":
    ds = load_crop("wheat")
    cells = ds.pick_cells(5, seed=0)
    splits = per_cell_splits(ds, cells)
    print(f"wheat: picked {len(cells)} cells -> {len(splits)} fittable splits")
    s = splits[0]
    print("first cell:", s["cell"], "train rows:", s["n_train"],
          "test rows:", s["n_test"], "train_mean:", round(s["train_mean"], 3))
    print("train tasmax shape:", s["train"]["tasmax"].shape,
          " co2 shape:", s["train"]["co2"].shape)