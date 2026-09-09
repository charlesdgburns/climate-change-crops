"""
co2_response.py
---------------
CO2-response surface diagnostic for the literature-CO2 candidates (24-27).

The per-cell held-out R2 on the val window (412-419) cannot exercise the
400 -> 1108 ppm extrapolation (val CO2 only reaches ~440 ppm), so the CO2
component is checked here against domain-science targets instead. For each
candidate the per-location fit is recomputed (same protocol as validate.py),
then the fitted model is evaluated on that cell's *climatological* weather
(each 240-day series averaged over train years) at a sweep of CO2 levels. The
reported number is the median across cells of the implied % yield change
relative to C = 400 ppm. A couple of model-free reference rows are included:
  - 06/13 : the data-driven CO2 response from the incumbent models
            (06 fits f*log(co2/380); 13 has no CO2 term at all -> ~0)
  - EPIC-linear : the linear EPIC multiplier f = 1 + beta*(C-350)/350 with
            beta = 0.77 (C3) / 0.11 (C4) as a warning: linear extrapolation
            to 1108 ppm is unbounded and inconsistent with the saturating
            mechanism; included purely as a reference row.

Target bands (literature, % yield change vs ~400 ppm, + = gain):
  wheat (C3):   @550  +10..+19   (Kimball 2016 +19% avg over 353->550;
                                 Ainsworth & Long 2020 optimality +11.7%)
                @1108 +20..+35   (asymptotic; NBER/OCO-2 upper bound)
  maize (C4):   @550   ~0        (no direct FACE response; only via WUE under
                                 drought) so the drought-interaction channel
                                 (25) may show a *bounded* positive at the
                                 wet/dry mix actually seen in climatology.

Usage:
    python3 co2_response.py --n-cells 200 --models 24 25 26 27
    python3 co2_response.py --crops wheat maize --n-cells 300 --n-workers 12
Writes sandbox/results/co2_response.md (and prints the table).
"""

import argparse
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data import (DEFAULT_TEST_YEARS, DEFAULT_TRAIN_YEARS, load_crop,
                  per_cell_splits, pick_cells)
from validate import _kw, fit_score_cell, load_program

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CO2_SWEEP = [380.0, 400.0, 550.0, 700.0, 1108.0]
CREF = 400.0

EPIC_LINEAR_BETA = {"wheat": 0.77, "maize": 0.11}
TARGETS = {
    "wheat": {550.0: (10.0, 19.0), 1108.0: (20.0, 35.0)},
    "maize": {550.0: (-5.0, 5.0), 1108.0: (-5.0, 15.0)},
}


def climatology_inputs(splits: list[dict]) -> list[dict]:
    """Per-cell climatological 240-day series (mean over train years)."""
    out = []
    for s in splits:
        tr = s["train"]
        shape = tr["tasmax"].shape  # (240, T)
        cl = {
            "tasmax": tr["tasmax"].mean(axis=1, keepdims=True),
            "tasmin": tr["tasmin"].mean(axis=1, keepdims=True),
            "pr": tr["pr"].mean(axis=1, keepdims=True),
            "rsds": tr["rsds"].mean(axis=1, keepdims=True),
            "cumrsds": tr["cumrsds"].mean(axis=1, keepdims=True),
        }
        assert all(c.shape == shape[:1] + (1,) for c in cl.values()), \
            f"cell {s['cell']}: climatology shape mismatch"
        out.append({"cell": s["cell"], "clim": cl,
                    "train_mean": s["train_mean"]})
    return out


def _fit_cell(prog_spec, split, maxiter, ridge, crop):
    r = fit_score_cell(prog_spec, split, maxiter, ridge, crop)
    if r["status"] != "success":
        return None
    return r["p_opt"]


def _job(task, maxiter, ridge):
    """Picklable worker: (model, crop, cell) -> (model, crop, cell, params)."""
    m, crop, split = task
    return m, crop, split["cell"], _fit_cell(m, split, maxiter, ridge, crop)


def refit_params(model_specs, crops, splits_by_crop, n_workers, maxiter, ridge):
    """{crop: {model: {cell: params}}} for the requested sample."""
    out = {c: OrderedDict((m, {}) for m in model_specs) for c in crops}
    tasks = []
    for crop in crops:
        for m in model_specs:
            for s in splits_by_crop[crop]:
                tasks.append((m, crop, s))
    if n_workers and n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_job, tasks, [maxiter] * len(tasks),
                                  [ridge] * len(tasks)))
    else:
        results = [_job(t, maxiter, ridge) for t in tasks]
    for m, c, cell, p in results:
        if p is not None:
            out[c][m][cell] = p
    return out


def eval_relative_change(model, crop, cl, params) -> tuple[dict, dict]:
    """Model at climatological weather, co2 swept; returns pct change vs Cref
    (value in model units) and absolute raw value."""
    out = {}
    for c in CO2_SWEEP:
        out[c] = float(np.asarray(
            model.model(**cl, co2=np.array([c]), params=params,
                        **_kw(model.model, "crop", crop))
        ).ravel()[0])
    base = out[CREF]
    pct = {c: (out[c] / base - 1.0) * 100.0 if base != 0 else np.nan
           for c in CO2_SWEEP}
    return pct, out


def model_free_rows(prog, crop, cl, params):
    """Reference rows built on top of the incumbent model fits."""
    pct, _ = eval_relative_change(prog, crop, cl, params)
    return pct


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops", nargs="+", choices=["wheat", "maize"],
                    default=["wheat", "maize"])
    ap.add_argument("--n-cells", type=int, default=200)
    ap.add_argument("--models", nargs="+",
                    default=["24_co2_saturating_multiplier",
                             "25_co2_wue_drought_maize",
                             "26_co2_heat_amelioration",
                             "27_rue_co2"])
    ap.add_argument("--incumbents", nargs="+", default=["06_saturating_vpd",
                                                        "13_water_heat_bilinear"])
    ap.add_argument("--min-train-years", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--ridge", type=float, default=0.1)
    ap.add_argument("--n-workers", type=int, default=min(os.cpu_count() or 1, 8))
    args = ap.parse_args()

    t0 = time.time()
    model_specs = list(OrderedDict.fromkeys(args.incumbents + args.models))
    loaded = {}
    for m in model_specs:
        nm, mod = load_program(f"{m}.py")
        loaded[nm] = mod

    splits_by_crop, clima_by_crop = {}, {}
    for crop in args.crops:
        ds = load_crop(crop)
        cells = pick_cells(ds, args.n_cells, seed=args.seed,
                           train_years=DEFAULT_TRAIN_YEARS,
                           min_train_years=args.min_train_years)
        splits = per_cell_splits(ds, cells, DEFAULT_TRAIN_YEARS,
                                 DEFAULT_TEST_YEARS)
        splits_by_crop[crop] = splits
        clima_by_crop[crop] = climatology_inputs(splits)
        print(f"[{crop}] {len(splits)} cells for the response surface")

    print(f"refitting {len(model_specs)} models x {len(args.crops)} crops "
          f"(maxiter={args.maxiter}, ridge={args.ridge}) ...")
    params_by = refit_params(model_specs, args.crops, splits_by_crop,
                             args.n_workers, args.maxiter, args.ridge)

    rows = []
    for crop in args.crops:
        climas = clima_by_crop[crop]
        for m in model_specs:
            vals = {c: [] for c in CO2_SWEEP}
            for cl in climas:
                p = params_by[crop][m].get(cl["cell"])
                if p is None:
                    continue
                pct, _ = eval_relative_change(loaded[m], crop, cl["clim"], p)
                for c in CO2_SWEEP:
                    if np.isfinite(pct[c]):
                        vals[c].append(pct[c])
            if not vals[CREF]:
                continue
            rows.append({"crop": crop, "model": m,
                         "n_cells": len(vals[CREF]),
                         **{f"c{c:g}": np.median(vals[c]) for c in CO2_SWEEP}})

    df = pd.DataFrame(rows).sort_values(["crop", "model"]).reset_index(drop=True)
    md = df.to_markdown(index=False)
    print("\nMedian-implied % yield change vs 400 ppm (at climatological weather)\n")
    print(md)

    band_rows = []
    for _, r in df.iterrows():
        for c in (550.0, 1108.0):
            lo, hi = TARGETS[r["crop"]][c]
            v = r[f"c{c:g}"]
            ok = "OK" if (lo <= v <= hi) else ("low" if v < lo else "high")
            band_rows.append({"crop": r["crop"], "model": r["model"],
                              "co2": f"{c:g}", "implied%": round(v, 1),
                              "target%": f"{lo}..{hi}", "vs_target": ok})
    bands = pd.DataFrame(band_rows)
    print("\nTarget-band check (literature anchors)\n")
    print(bands.to_markdown(index=False))

    # EPIC-linear hazard reference (no fit; analytic multiplier)
    ref_rows = []
    for crop in args.crops:
        beta = EPIC_LINEAR_BETA[crop]
        line = {c: (1 + beta * (c - 350.0) / 350.0) / (1 + beta * (CREF - 350.0) / 350.0)
                - 1.0 for c in CO2_SWEEP}
        ref_rows.append({"crop": crop, "model": "EPIC-linear(ref)",
                         "note": "unbounded extrapolation hazard",
                         **{f"c{c:g}": 100 * line[c] for c in CO2_SWEEP}})
    ref = pd.DataFrame(ref_rows)
    print("\nEPIC-linear reference multiplier (no fitted model)\n")
    print(ref.to_markdown(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "co2_response.md"
    with open(out, "w") as f:
        f.write("# CO2 response surface (vs 400 ppm, climatological weather)\n\n")
        f.write("Model-implied % yield change at fixed climatological weather, "
                "median over fitted cells.\n\n")
        f.write("## Implied change\n\n" + md + "\n\n")
        f.write("## Target bands\n\n" + bands.to_markdown(index=False) + "\n\n")
        f.write("## EPIC-linear hazard reference\n\n" + ref.to_markdown(index=False) + "\n")
    print(f"\nwrote {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()