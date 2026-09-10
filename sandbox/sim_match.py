"""
sim_match.py
------------
B-track "literal simulator" comparison (see ANALYSIS.md / GENESIS.md).

Runs the five candidate process-model families (CERES, STICS, APSIM, EPIC,
LPJmL - all parameters FIXED from the literature, see sim/families.py) on
per-location climate, and judges which family's weather response best matches
the observed yields, under the user's strict dual criterion (>= 0.10 margin on
BOTH legs before a winner is named):

  Leg 1 - held-out R2:  per cell, fit a per-location AFFINE map
      y_obs ~= a + b * y_sim                       (closed-form ridge, slope
      penalised, intercept exempt)
      on train years 381..411, score R2 on test years 412..419.
      Only a level+scale are coded; all climate-response structure is fixed
      from the literature.

  Leg 2 - fingerprint-of-simulator:  run the same F4/F2/F3/F6 weather
      diagnostics the B-track used on the OBSERVED yields, but applied to each
      family's SIMULATED yield series (same cells, same rows). A family whose
      simulated weather response reproduces the observed fingerprint signature
      (supply vs demand water channel, heat threshold, soil memory) scores high.

Report: sandbox/results/sim_match_{crop}.md (+ .json with the verdict).

Usage:
    python3 sim_match.py --crops wheat maize --n-cells 500 --n-workers 12
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from data import load_crop, pick_cells
from fingerprint import (build_year_features, fit_predict_cell, _winrate,
                         N_BASES, HEAT_THRESHOLDS)
from sim.core import simulate
from sim.families import FAMILIES

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MIN_TRAIN_YEARS = 15
MIN_TEST_YEARS = 3
GDD_BASE = {"wheat": 0, "maize": 10}


# ---------------------------------------------------------------------------
# Per-cell bundle: observed yields + weather features + sim series for all fams
# ---------------------------------------------------------------------------

def build_cell(ds, cell, fams, crop, ridge, train_years, test_years):
    rows = ds.cell_rows([int(cell)])
    clim, meta, y = ds.climate[rows], ds.meta[rows], ds.y[rows]
    years = meta[:, 3].astype(int)
    order = np.argsort(years)
    clim, y, years = clim[order], y[order], years[order]
    co2 = meta[order, 4].astype(float)
    n = int(meta[order, 5].mean())          # nitrogen is constant per location

    tr = (years >= train_years[0]) & (years <= train_years[1])
    te = (years >= test_years[0]) & (years <= test_years[1])
    if int(tr.sum()) < MIN_TRAIN_YEARS or int(te.sum()) < MIN_TEST_YEARS:
        return None

    # weather features on the same (year, day) layout fingerprint.py expects
    clim_yx = np.transpose(clim, (0, 2, 1))          # (years, 240, 5)
    feats = build_year_features(clim_yx, co2)
    idx = {int(yr): i for i, yr in enumerate(years)}
    for base_key, lag_key in (("wb", "wb_lag1"), ("prsum", "prsum_lag1")):
        lag = np.full(len(years), np.nan)
        for i, yr in enumerate(years):
            prev = idx.get(int(yr) - 1)
            if prev is not None:
                lag[i] = feats[base_key][prev]
        feats[lag_key] = lag

    feats["_y"] = y
    feats["_years"] = years

    # per-family simulated series over the SAME years (train + test stacked)
    train_clim = {k: clim[order][tr][:, :, j].T
                  for k, j in (("tasmax", 0), ("tasmin", 1), ("pr", 2), ("rsds", 3))}
    test_clim = {k: clim[order][te][:, :, j].T
                 for k, j in (("tasmax", 0), ("tasmin", 1), ("pr", 2), ("rsds", 3))}
    sim_by_fam = {}
    for fam in fams:
        s_tr = simulate(train_clim, co2[tr], n, fam, crop)
        s_te = simulate(test_clim, co2[te], n, fam, crop)
        sim = np.concatenate([s_tr, s_te])
        sim_by_fam[fam["id"]] = sim

    return {"cell": int(cell), "years": years, "feats": feats,
            "tr_mask": tr, "te_mask": te, "n": n, "sim": sim_by_fam}


# ---------------------------------------------------------------------------
# Leg 1: per-cell affine calibration
# ---------------------------------------------------------------------------

def _affine_r2(y_obs_train, y_obs_test, s_tr, s_te, ridge):
    """Closed-form ridge affine y ~= a + b*z; z standardised per cell so the
    slope penalty is unit-invariant across families; intercept exempt."""
    s_tr = np.asarray(s_tr, float)
    mu, sd = float(s_tr.mean()), float(s_tr.std())
    if sd < 1e-9:
        return np.nan
    z_tr = (s_tr - mu) / sd
    z_te = (s_te - mu) / sd
    Xtr = np.column_stack([np.ones_like(z_tr), z_tr])
    Xte = np.column_stack([np.ones_like(z_te), z_te])
    pen = np.array([[0.0, 0.0], [0.0, ridge]])
    try:
        theta = np.linalg.solve(Xtr.T @ Xtr + pen, Xtr.T @ y_obs_train)
    except np.linalg.LinAlgError:
        return np.nan
    pred = Xte @ theta
    ss_tot = float(np.sum((y_obs_test - y_obs_test.mean()) ** 2))
    if ss_tot < 1e-12:
        return np.nan
    return float(1.0 - np.sum((y_obs_test - pred) ** 2) / ss_tot)


# ---------------------------------------------------------------------------
# Leg 2: fingerprint-of-simulator
# ---------------------------------------------------------------------------

LEG2_MODELS = (["F4_none", "F4_pr", "F4_vpd", "F4_wb"]
               + ["F2_gdd"]
               + [f"F2_hdd{h}" for h in HEAT_THRESHOLDS]
               + [f"F3_gdd{b}" for b in N_BASES]
               + ["F6_weather", "F6_wb_lag"])


def _feature_keys(name, crop):
    return {
        "F4_none": [], "F4_pr": ["prsum"], "F4_vpd": ["vpd_sum"],
        "F4_wb": ["wb"], "F2_gdd": [f"gdd_{GDD_BASE[crop]}"],
        **{f"F2_hdd{h}": [f"hdd_{h}"] for h in HEAT_THRESHOLDS},
        **{f"F3_gdd{b}": [f"gdd_{b}"] for b in N_BASES},
        "F6_weather": ["prsum", "vpd_sum"],
        "F6_wb_lag": ["prsum", "vpd_sum", "wb_lag1"],
    }[name]


def _worker(args):
    ds, cell, fams, crop, ridge, train_years, test_years = args
    return build_cell(ds, cell, fams, crop, ridge, train_years, test_years)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops", nargs="+", choices=["wheat", "maize"],
                    default=["wheat", "maize"])
    ap.add_argument("--n-cells", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=0.1)
    ap.add_argument("--n-workers", type=int,
                    default=min(os.cpu_count() or 1, 8))
    ap.add_argument("--train-years", type=int, nargs=2, default=[381, 411])
    ap.add_argument("--test-years", type=int, nargs=2, default=[412, 419])
    args = ap.parse_args()
    train_years, test_years = tuple(args.train_years), tuple(args.test_years)

    fams = FAMILIES
    for crop in args.crops:
        t0 = time.time()
        ds = load_crop(crop)
        cells = pick_cells(ds, args.n_cells, seed=args.seed)
        tasks = [(ds, c, fams, crop, args.ridge, train_years, test_years)
                 for c in cells]
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            bundles = [b for b in ex.map(_worker, tasks) if b is not None]
        write_report(crop, bundles, args.ridge, fams, train_years, test_years)
        print(f"[{crop}] {len(cells)} sampled -> {len(bundles)} fittable, "
              f"{time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(crop, bundles, ridge, fams, train_years, test_years):
    fam_ids = [f["id"] for f in fams]
    n = len(bundles)

    # Leg 1: per-cell affine R2 + direct (un-fitted) corr(obs, sim) on train
    r2 = {fam: np.full(n, np.nan) for fam in fam_ids}
    corr = {fam: np.full(n, np.nan) for fam in fam_ids}
    for i, b in enumerate(bundles):
        tr, te = b["tr_mask"], b["te_mask"]
        ytr, yte = b["feats"]["_y"][tr], b["feats"]["_y"][te]
        for fam in fam_ids:
            s_tr = b["sim"][fam][tr]
            s_te = b["sim"][fam][te]
            r2[fam][i] = _affine_r2(ytr, yte, s_tr, s_te, ridge)
            if s_tr.std() > 1e-9:
                corr[fam][i] = float(np.corrcoef(ytr, s_tr)[0, 1])
    med = {fam: float(np.nanmedian(r2[fam])) for fam in fam_ids}

    # Leg 2: fingerprint-of-simulator (observed + each family's simulated series)
    obs_r2 = {m: [] for m in LEG2_MODELS}
    sim_r2 = {fam: {m: [] for m in LEG2_MODELS} for fam in fam_ids}
    n_feat = {m: 0 for m in LEG2_MODELS}
    for b in bundles:
        feats, years = b["feats"], b["years"]
        y0 = feats["_y"]
        lag_ok = np.isfinite(feats["wb_lag1"]) & np.isfinite(feats["prsum_lag1"])
        for m in LEG2_MODELS:
            keys = _feature_keys(m, crop)
            sub = {k: v[lag_ok] for k, v in feats.items() if not k.startswith("_")} \
                if m.startswith("F6") else feats
            yy, yr = y0, years
            if m.startswith("F6"):
                yy, yr = y0[lag_ok], years[lag_ok]
            r, _ = fit_predict_cell(sub, yy, yr, keys, ridge)
            obs_r2[m].append(r)
            for fam in fam_ids:
                s = b["sim"][fam]
                ss = s if not m.startswith("F6") else s[lag_ok]
                r, _ = fit_predict_cell(sub, ss, yr, keys, ridge)
                sim_r2[fam][m].append(r)

    def med_any(arr):
        arr = np.asarray(arr, float)
        return float(np.nanmedian(arr)) if np.isfinite(arr).any() else np.nan

    L = []
    A = L.append
    A(f"# sim_match: literal-simulator comparison ({crop})\n")
    A(f"\nCells: **{n}**. Train {train_years[0]}..{train_years[1]}, test "
      f"{test_years[0]}..{test_years[1]}. All family parameters FIXED from "
      f"literature (sim/families.py); per-cell fit is a 2-param affine "
      f"(closed-form ridge, ridge={ridge:g}, intercept exempt).\n")

    A("\n## Leg 1  held-out per-cell R2 (affine)")
    A("\n| family | median R2 | win vs CERES | win vs APSIM |\n|---|---|---|---|")
    order = sorted(fam_ids, key=lambda f: -med[f])
    for fam in order:
        wr_ceres, p_ce = _winrate(r2[fam], r2["ceres"])
        wr_apsim, p_ap = _winrate(r2[fam], r2["apsim"])
        A(f"| {fam} | {med[fam]:+.3f} | {wr_ceres:.3f} (p={p_ce:.2g}) | "
          f"{wr_apsim:.3f} (p={p_ap:.2g}) |")
    best = order[0]
    second = order[1] if len(order) > 1 else best
    margin = med[best] - med[second]
    A(f"\nBest: **{best}** (median {med[best]:+.3f}), next best {second} "
      f"({med[second]:+.3f}) -> margin {margin:+.3f}."
      f" {'MEETS >=0.10 margin' if margin >= 0.10 else 'below 0.10 margin'}.")

    A("\n## Leg 1b  direct correlation of observed vs simulated yields (train years)")
    A("\n| family | mean corr | median corr | % cells corr>0 |")
    A("|---|---|---|---|")
    for fam in fam_ids:
        c = corr[fam]
        c = c[np.isfinite(c)]
        A(f"| {fam} | {np.mean(c):+.3f} | {np.median(c):+.3f} | "
          f"{np.mean(c > 0):.2f} |")
    A("\n(no fitting here - just how well the literature-fixed simulator tracks "
      "the observed year-to-year yields within each cell).")

    A("\n## Leg 2  fingerprint-of-simulator")
    A("\nSame weather diagnostics as the yellow B-track, run on the OBSERVED "
      "yields and on each family's SIMULATED series (same cells, same rows).\n")
    A("\n| model | observed | " + " | ".join(fam_ids) + " |")
    A("|---|---|" + "---|" * len(fam_ids))
    for m in LEG2_MODELS:
        row = [m, f"{med_any(obs_r2[m]):+.3f}"]
        for fam in fam_ids:
            row.append(f"{med_any(sim_r2[fam][m]):+.3f}")
        A("| " + " | ".join(str(x) for x in row) + " |")

    A("\n### F4 water-channel signature (did the family pick the same stresses as observed?)")
    A("\nfingerprint F4 win rates (pr/vpd/wb vs none) + joint pr+vpd betas:\n")
    A("\n| metric | observed | " + " | ".join(fam_ids) + " |")
    A("|---|---|" + "---|" * len(fam_ids))
    def f4_stats(r2s):
        wr = {k: _winrate(r2s[k], r2s["F4_none"])[0] for k in ("F4_pr", "F4_vpd", "F4_wb")}
        return wr, None
    # joint betas not stored here; report win rates only
    for k in ("F4_pr", "F4_vpd", "F4_wb"):
        row = [k, f"{f4_stats(obs_r2)[0][k]:+.3f}"]
        for fam in fam_ids:
            row.append(f"{f4_stats(sim_r2[fam])[0][k]:+.3f}")
        A("| " + " | ".join(str(x) for x in row) + " |")

    # signature distance: mean |medianR2_sim - medianR2_obs| over F4+F2+F3 models
    A("\n### signature distance (mean |sim - observed| median R2 over F4+F2+F3)")
    dist = {}
    sig_models = [m for m in LEG2_MODELS if not m.startswith("F6")]
    for fam in fam_ids:
        d = np.mean([abs(med_any(sim_r2[fam][m]) - med_any(obs_r2[m]))
                     for m in sig_models])
        dist[fam] = float(d)
    best_sig = min(dist, key=dist.get)
    A("\n| family | signature distance |\n|---|---|")
    for fam in fam_ids:
        A(f"| {fam} | {dist[fam]:.3f} |")
    A(f"\nClosest signature: **{best_sig}** (dist {dist[best_sig]:.3f}).")

    # verdict
    A("\n## Verdict")
    second_sig = min(fam_ids, key=lambda f: dist[f] if f != best_sig else 1e9)
    sig_margin = dist[second_sig] - dist[best_sig] if len(fam_ids) > 1 else 0.0
    r2_win = margin >= 0.10
    sig_win = sig_margin >= 0.10
    verdict = "WINNER" if r2_win and sig_win else "NO WINNER"
    A(f"\nLeg 1 margin {margin:+.3f} (need >= +0.10) -> "
      f"{'PASS' if r2_win else 'FAIL'}")
    A(f"Leg 2 signature margin {sig_margin:+.3f} (need >= +0.10) -> "
      f"{'PASS' if sig_win else 'FAIL'}")
    A(f"\n**{verdict}** - keep 24/06 submission cores (no change) unless both "
      "legs pass for a single family.")

    md = "".join(L)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"sim_match_{crop}.md"
    out.write_text(md)
    print(md)
    summary = {
        "crop": crop, "n_cells": n,
        "leg1_median_r2": med,
        "leg1_best": best, "leg1_margin": float(margin),
        "leg1_pass": r2_win,
        "leg1b_corr_mean": {fam: float(np.mean(corr[fam][np.isfinite(corr[fam])]))
                            for fam in fam_ids},
        "signature_distance": dist,
        "leg2_best": best_sig, "leg2_margin": float(sig_margin),
        "leg2_pass": sig_win,
        "verdict": verdict,
    }
    jout = RESULTS_DIR / f"sim_match_{crop}.json"
    jout.write_text(json.dumps(summary, indent=2))
    return out


if __name__ == "__main__":
    main()