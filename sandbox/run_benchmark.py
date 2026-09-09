"""
run_benchmark.py
----------------
Runs every candidate program over a large, FIXED, seeded sample of locations
(per crop) and writes per-cell results to CSV for visualization and
statistics across all locations.

Default: 1000 cells per crop >= 20 train years, all programs + the two
model-free baselines (per_cell_train_mean, crop_mean).

Usage
-----
    python3 run_benchmark.py                          # wheat + maize, 1000 cells
    python3 run_benchmark.py --crops wheat --n-cells 500 --n-workers 12
    python3 viz.py                                    # then render figures
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import DEFAULT_TEST_YEARS, DEFAULT_TRAIN_YEARS, load_crop, per_cell_splits, pick_cells
from validate import baseline_metrics, cell_r2, run_program

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PROGRAMS_DIR = Path(__file__).resolve().parent / "programs"

BASELINE_MODELS = ["baseline_per_cell_mean", "baseline_crop_mean"]


def baseline_rows(splits, crop: str, crop_const: float) -> list[dict]:
    rows = []
    for s in splits:
        yte = s["test"]["y"]
        base = {
            "cell": s["cell"], "crop": crop,
            "n_train": s["n_train"], "n_test": s["n_test"],
            "train_mean": s["train_mean"], "status": "success", "reason": "",
        }
        for name, pred in [("baseline_per_cell_mean", np.full_like(yte, s["train_mean"])),
                           ("baseline_crop_mean", np.full_like(yte, crop_const))]:
            rows.append({
                **base, "model": name, "r2": cell_r2(yte, pred),
                "mse": float(np.mean((yte - pred) ** 2)), "n_params": 0,
                "converged": True,
            })
    return rows


def summarize(dfs: list[pd.DataFrame], out: Path) -> list[dict]:
    """Per-model aggregates from benchmark rows (per model+crop)."""
    summary_rows = []
    for df in dfs:
        crop = df["crop"].iloc[0]
        for nm in df["model"].unique():
            sub = df[df["model"] == nm]
            ok = sub[sub["r2"].notna()]
            if ok.empty:
                continue
            n = len(ok)
            pct = np.nan
            if nm not in BASELINE_MODELS:
                base = df[(df["model"] == "baseline_per_cell_mean") &
                          (df["cell"].isin(sub["cell"]))]
                m = ok.merge(base[["cell", "r2"]], on="cell", suffixes=("", "_base"))
                pct = float((m["r2"] > m["r2_base"]).mean())
            summary_rows.append({
                "crop": crop, "model": nm, "n_cells": n,
                "r2_cell_median": float(np.median(ok["r2"])),
                "r2_cell_mean": float(np.mean(ok["r2"])),
                "mse_pooled": float(np.average(ok["mse"], weights=ok["n_test"])),
                "pct_gt_baseline": pct,
            })
    if summary_rows:
        smry = pd.DataFrame(summary_rows)
        smry["r2_cell_median"] = smry["r2_cell_median"].map("{:+.3f}".format)
        smry["r2_cell_mean"] = smry["r2_cell_mean"].map("{:+.3f}".format)
        smry["mse_pooled"] = smry["mse_pooled"].map("{:.4f}".format)
        smry["pct_gt_baseline"] = smry["pct_gt_baseline"].map(
            lambda x: f"{x*100:.1f}%" if x == x else "-")
        smry.to_markdown(out, index=False)
        print(f"wrote {out}")
        print(smry.to_string(index=False))
    return summary_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops", nargs="+", choices=["wheat", "maize"], default=["wheat", "maize"])
    ap.add_argument("--n-cells", type=int, default=1000)
    ap.add_argument("--min-train-years", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--ridge", type=float, default=0.1)
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=None,
                    help="program names/paths to run (default: all programs)")
    ap.add_argument("--append", action="store_true",
                    help="merge into existing results CSVs instead of overwriting")
    ap.add_argument("--summary-only", action="store_true",
                    help="regenerate summary.md from existing benchmark_*.csv (no model runs)")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        dfs = []
        for crop in args.crops:
            p = args.out_dir / f"benchmark_{crop}.csv"
            if not p.exists():
                raise FileNotFoundError(f"no benchmark at {p}")
            dfs.append(pd.read_csv(p))
        summarize(dfs, args.out_dir / "summary.md")
        return

    if args.models is None:
        model_specs = sorted(p.stem for p in PROGRAMS_DIR.glob("*.py"))
    else:
        model_specs = args.models
    train_years, test_years = DEFAULT_TRAIN_YEARS, DEFAULT_TEST_YEARS

    all_dfs = []
    for crop in args.crops:
        ds = load_crop(crop)
        cells = pick_cells(ds, args.n_cells, seed=args.seed, train_years=train_years,
                           min_train_years=args.min_train_years)
        splits = per_cell_splits(ds, cells, train_years, test_years)
        print(f"[{crop}] picked {len(cells)} cells, {len(splits)} fittable "
              f"({len(set(cells)) - len(splits)} skipped)")

        crop_const = float(np.mean(np.concatenate([s["train"]["y"] for s in splits])))
        rows = baseline_rows(splits, crop, crop_const)

        for spec in model_specs:
            res = run_program(spec, splits, args.maxiter, args.ridge,
                              args.n_workers, return_per_cell=True, crop=crop)
            for r in res["per_cell"]:
                r["model"] = spec
                r["crop"] = crop
            rows += res["per_cell"]
            print(f"  {spec:22s} n={res['n_cells']:4d} "
                  f"r2_med={res['r2_cell_median']:+.3f} "
                  f"({res['seconds']:.1f}s)")

        df = pd.DataFrame(rows).sort_values(["model", "cell"]).reset_index(drop=True)
        path = args.out_dir / f"benchmark_{crop}.csv"
        if args.append and path.exists():
            old = pd.read_csv(path)
            rerun = set(model_specs) | set(BASELINE_MODELS)
            old = old[~old["model"].isin(rerun)]
            df = pd.concat([old, df], ignore_index=True)
            df = df.sort_values(["model", "cell"]).reset_index(drop=True)
        df.to_csv(path, index=False)
        print(f"[{crop}] wrote {path} ({len(df)} rows, {df['model'].nunique()} models)")
        all_dfs.append(df)

    if all_dfs:
        summarize(all_dfs, args.out_dir / "summary.md")


if __name__ == "__main__":
    main()