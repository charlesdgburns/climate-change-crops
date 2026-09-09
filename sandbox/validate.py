"""
validate.py
-----------
Fast, human-in-the-loop validation for scalar yield production functions.

A candidate is a script in programs/ defining `model()` and `estimate_params()`.
Fit is done per LOCATION (single cell): each sampled cell is fitted on its own
train years (only that cell's rows), then scored on that cell's test years.

Protocol
--------
1. Load one crop; sample `--n-cells` locations (default 10) that each have
   >= `--min-train-years` train-year rows (default 20).
2. Per location, split by year: train = `--train-years` (default 381-411),
   test = `--test-years` (default 412-419).
3. Per location: estimate_params(...) -> p0; scipy L-BFGS-B minimises
       MSE(model(...), y) + ridge * sum(params[1:]**2)     (intercept exempt)
4. Score per location on its OWN test years (per-cell R2, per-cell MSE).
5. Aggregate across locations: median / mean per-cell R2, pooled MSE, pooled R2.
6. Print a compact table with two model-free references:
       crop_mean           : global constant = mean of all sampled train years
       per_cell_train_mean : test years of a cell predicted by its train mean

Parallelism: fits run across CPU cores via ProcessPoolExecutor (--n-workers,
default = #cpus, capped at n cells).

Usage
-----
    python3 validate.py --crop wheat --model 04_saturating --n-cells 10
    python3 validate.py --crop maize --model all --n-cells 20   # all programs
"""

import argparse
import importlib.util
import inspect
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

PROGRAMS_DIR = Path(__file__).resolve().parent / "programs"

FEATURE_NAMES = ["tasmax", "tasmin", "pr", "rsds", "cumrsds", "co2"]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def cell_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Per-cell R2 over that location's test years (NaN if no variance)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return np.nan
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot)


def aggregate(per_cell: list[dict]) -> dict:
    """Collapse per-location results into pooled + aggregate metrics."""
    y_true = np.concatenate([r["y_true"] for r in per_cell])
    y_pred = np.concatenate([r["y_pred"] for r in per_cell])
    r2s = np.array([r["r2"] for r in per_cell])
    r2s = r2s[np.isfinite(r2s)]
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2_global = float("nan") if ss_tot < 1e-12 else \
        1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot
    return {
        "r2_cell_median": float(np.median(r2s)) if r2s.size else float("nan"),
        "r2_cell_mean": float(np.mean(r2s)) if r2s.size else float("nan"),
        "r2_cell_min": float(np.min(r2s)) if r2s.size else float("nan"),
        "mse_pooled": float(np.mean((y_true - y_pred) ** 2)),
        "r2_global": r2_global,
        "n_cells": len(per_cell),
    }


# ---------------------------------------------------------------------------
# Program loading
# ---------------------------------------------------------------------------

def load_program(spec: str) -> tuple[str, object]:
    """Load a program file by name ('04_saturating') or path."""
    raw = Path(spec)
    path = raw if raw.is_file() else PROGRAMS_DIR / (spec if spec.endswith(".py") else f"{spec}.py")
    if not path.exists():
        raise FileNotFoundError(f"no program at {path}")
    module_name = path.stem
    spec_loader = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec_loader)
    sys.modules[module_name] = module
    spec_loader.loader.exec_module(module)
    for fn in ("model", "estimate_params"):
        if not callable(getattr(module, fn, None)):
            raise AttributeError(f"{path} must define {fn}()")
    return module_name, module


def _accepts(fn, name: str) -> bool:
    """Whether a candidate function declares an optional keyword arg."""
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _kw(fn, name: str, value):
    """{name: value} for candidates that declare it, else {} (backward compat)."""
    return {name: value} if _accepts(fn, name) else {}


# ---------------------------------------------------------------------------
# Per-location fit + score (runs in worker processes)
# ---------------------------------------------------------------------------

def fit_score_cell(prog_spec: str, split: dict, maxiter: int,
                   ridge: float, crop: str | None = None) -> dict:
    """Fit one location on its train years, score its test years."""
    _, module = load_program(prog_spec)
    tr = {k: split["train"][k] for k in FEATURE_NAMES}
    ytr = split["train"]["y"]
    te = {k: split["test"][k] for k in FEATURE_NAMES}
    yte = split["test"]["y"]
    cw = _kw(module.estimate_params, "crop", crop)
    ck = _kw(module.model, "crop", crop)

    try:
        p0 = np.atleast_1d(np.asarray(module.estimate_params(**tr, y=ytr, **cw), dtype=float))
    except Exception as e:
        return {"status": "failed", "cell": split["cell"], "reason": f"estimate_params: {e}"}

    def loss(params):
        pred = np.asarray(module.model(**tr, params=params, **ck), dtype=np.float64).ravel()
        m = float(np.mean((pred - ytr) ** 2))
        if ridge > 0 and params.size > 1:
            m += ridge * float(np.sum(params[1:] ** 2))
        return m

    try:
        res = minimize(loss, p0, method="L-BFGS-B", options={"maxiter": maxiter})
    except Exception as e:
        return {"status": "failed", "cell": split["cell"], "reason": f"optimizer: {e}"}
    p_opt = res.x

    try:
        y_pred = np.asarray(module.model(**te, params=p_opt, **ck), dtype=np.float64).ravel()
        if y_pred.shape != yte.shape or not np.all(np.isfinite(y_pred)):
            return {"status": "failed", "cell": split["cell"],
                    "reason": "non-finite or wrong-shape predictions"}
    except Exception as e:
        return {"status": "failed", "cell": split["cell"], "reason": f"model eval: {e}"}

    return {
        "status": "success",
        "cell": split["cell"],
        "r2": cell_r2(yte, y_pred),
        "mse": float(np.mean((yte - y_pred) ** 2)),
        "y_true": yte,
        "y_pred": y_pred,
        "n_train": split["n_train"],
        "n_test": split["n_test"],
        "n_params": int(p_opt.size),
        "converged": bool(res.success),
        "p_opt": p_opt,
    }


def run_program(prog_spec: str, splits: list[dict], maxiter: int, ridge: float,
                n_workers: int, return_per_cell: bool = False,
                crop: str | None = None) -> dict:
    t0 = time.time()
    success = []
    failures = []
    if n_workers is None or n_workers < 1:
        n_workers = min(os.cpu_count() or 1, len(splits))
    n_workers = max(1, min(n_workers, len(splits)))

    if n_workers == 1:
        results = [fit_score_cell(prog_spec, s, maxiter, ridge, crop) for s in splits]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(fit_score_cell, [prog_spec] * len(splits),
                                  splits, [maxiter] * len(splits),
                                  [ridge] * len(splits), [crop] * len(splits)))
    for r in results:
        (success if r["status"] == "success" else failures).append(r)

    out = {"status": "failure" if not success else "success",
           "failures": failures}
    if success:
        out.update(aggregate(success))
        out["n_params"] = int(np.median([r["n_params"] for r in success]))
        out["n_fit_total"] = int(sum(r["n_train"] for r in success))
        out["n_test_total"] = int(sum(r["n_test"] for r in success))
        if return_per_cell:
            out["per_cell"] = [{
                "cell": r["cell"],
                "r2": r["r2"],
                "mse": r["mse"],
                "n_train": r["n_train"],
                "n_test": r["n_test"],
                "n_params": r["n_params"],
                "converged": r["converged"],
                "status": r["status"],
                "reason": "",
            } for r in success]
            out["per_cell"] += [{
                "cell": r["cell"], "r2": np.nan, "mse": np.nan,
                "n_train": r.get("n_train", 0), "n_test": 0,
                "n_params": 0, "converged": False,
                "status": "failed", "reason": r["reason"],
            } for r in failures]
    out["seconds"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------
# Reference baselines (no candidate program required)
# ---------------------------------------------------------------------------

def baseline_metrics(splits: list[dict]) -> dict:
    """Model-free references, computed per location and aggregated."""
    per_cell = []
    all_train = np.concatenate([s["train"]["y"] for s in splits])
    crop_const = float(np.mean(all_train))

    for s in splits:
        yte = s["test"]["y"]
        per_cell.append({
            "r2": cell_r2(yte, np.full_like(yte, s["train_mean"])),
            "y_true": yte, "y_pred": np.full_like(yte, s["train_mean"]),
            "_": s["cell"],
        })
    ref_cell_mean = aggregate(per_cell)

    per_cell = []
    for s in splits:
        yte = s["test"]["y"]
        per_cell.append({
            "r2": cell_r2(yte, np.full_like(yte, crop_const)),
            "y_true": yte, "y_pred": np.full_like(yte, crop_const),
            "_": s["cell"],
        })
    ref_crop = aggregate(per_cell)

    return {"per_cell_train_mean": ref_cell_mean, "crop_mean": ref_crop}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_row(name, m, extra=""):
    return (f"  {name:22s}  R2_cell_med={m['r2_cell_median']:+.3f}"
            f"   R2_cell_mean={m['r2_cell_mean']:+.3f}"
            f"   MSE={m['mse_pooled']:9.4f}   R2_glob={m['r2_global']:+.3f}"
            f"   {extra}")


def report(prog_name, crop, splits, res, baselines, ridge, verbose=False):
    print(f"\n=== {prog_name}   crop={crop}   fits={res.get('n_cells', 0)}/"
          f"{len(splits)} cells   params~{res.get('n_params', '?')}   "
          f"ridge={ridge:g}   {res['seconds']:.1f}s")
    print(f"  training rows (all cells): {res.get('n_fit_total', '?')}   "
          f"test rows: {res.get('n_test_total', '?')}")
    if res["status"] == "failure":
        print("  FAILED: all locations failed")
        for f in res["failures"][:5]:
            print(f"    cell {f['cell']}: {f['reason']}")
        return

    print(fmt_row(prog_name, res))
    print("  --- references ---")
    print(fmt_row("crop_mean", baselines["crop_mean"]))
    print(fmt_row("per_cell_train_mean", baselines["per_cell_train_mean"]))

    ref = baselines["per_cell_train_mean"]["r2_cell_median"]
    if res["r2_cell_median"] > ref + 1e-9:
        print("  >> beats the per-location-train-mean baseline")
    else:
        print("  >> no improvement over per-location-train-mean baseline")

    if verbose and res["status"] == "success":
        for f in res.get("failures", []):
            print(f"    fail cell {f['cell']}: {f['reason']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop", choices=["wheat", "maize"], default="wheat")
    ap.add_argument("--model", default="all",
                    help="program file (name, with/without .py, or path) or 'all'")
    ap.add_argument("--n-cells", type=int, default=10)
    ap.add_argument("--min-train-years", type=int, default=20,
                    help="exclude cells with fewer train-year rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--ridge", type=float, default=0.1,
                    help="L2 penalty on response params (intercept exempt); 0 disables")
    ap.add_argument("--n-workers", type=int, default=None,
                    help="parallel fits across cells (default: min(#cpu, n cells))")
    ap.add_argument("--train-years", type=int, nargs=2, default=list(map(int, "381 411".split())))
    ap.add_argument("--test-years", type=int, nargs=2, default=list(map(int, "412 419".split())))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from data import load_crop, pick_cells, per_cell_splits

    train_years, test_years = tuple(args.train_years), tuple(args.test_years)
    ds = load_crop(args.crop)
    cells = pick_cells(ds, args.n_cells, seed=args.seed, train_years=train_years,
                       min_train_years=args.min_train_years)
    splits = per_cell_splits(ds, cells, train_years, test_years)
    if not splits:
        print("no fittable cells for the requested windows - adjust "
              "--min-train-years / --n-cells")
        return
    baselines = baseline_metrics(splits)

    names = sorted(p.stem for p in PROGRAMS_DIR.glob("*.py")) if args.model == "all" \
        else [args.model]
    results = {}
    for nm in names:
        res = run_program(nm, splits, args.maxiter, args.ridge, args.n_workers,
                          crop=args.crop)
        report(nm, args.crop, splits, res, baselines, args.ridge, args.verbose)
        results[nm] = res

    ok = {k: v for k, v in results.items() if v["status"] == "success"}
    if len(ok) > 1:
        rank = sorted(ok.items(), key=lambda kv: -kv[1]["r2_cell_median"])
        print("\nRanking by median per-cell R2 (higher = better):")
        for i, (nm, v) in enumerate(rank, 1):
            print(f"  {i}. {nm:22s} {v['r2_cell_median']:+.3f}"
                  f"   (beats baseline: {v['r2_cell_median'] > baselines['per_cell_train_mean']['r2_cell_median']})")


if __name__ == "__main__":
    main()