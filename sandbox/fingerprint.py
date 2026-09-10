"""
fingerprint.py
--------------
B-track genesis diagnostic (see ANALYSIS.md / GENESIS.md): characterise the
*generating* crop model's behavior from observed per-location yields.

Protocol (per-location, same philosophy as validate.py but diagnostic):
  - Per cell, build scalar year-series from the 240-day window.
  - Train = years 382..411 (year 381 only used as source for lag-1), test =
    412..419. A cell's own present years define the train/test masks; every
    model in a comparison group sees the SAME rows (lag models drop rows whose
    previous year is missing, and their lag-free comparator drops them too).
  - Every model is linear-in-params (intercept + standardized features),
    fitted per cell by closed-form ridge:
        theta = (X'X + ridge*diag(0,1,..))^-1 X'y   (intercept exempt)
    R2 is per-cell held-out on test years (1 - SSE/SST around cell test mean).
  - Reported per observable: median per-cell R2, paired win rates and
    Wilcoxon signed-rank p-values; standardized beta vectors where a profile
    is the point of interest.

Observables (F1-F8):
  F1 heat-sensitivity-by-season: 8 x 30-day segment means of daily mean T ->
     standardized partial betas -> which 30-day windows drive yield response.
  F2 heat non-linearity: tmean-sum vs GDD(base) vs heat-DD above threshold
     (22/26/30/34 C) vs hot-day count (>34) -> sharpness of heat penalty.
  F3 GDD base: sweep base {0..16}; median R2 profile + each cell's argmax
     base -> the development base temperature.
  F4 water channel: pr-sum vs VPD-sum vs (pr - 2.6*VPD) vs pr + vpd joint;
     standardized partial betas -> water-limited rainfed signature?
  F5 CO2 response shape: weather-controlled (tsum+prsum+vpdsum) plus one of
     {linear, log(CO2/380), saturating 1-exp} -> sign check only (narrow
     in-sample CO2 range ~341-415 ppm + level-drift identity).
  F6 soil-memory carryover: add lag-1 (pr-sum, water-balance) of the previous
     year -> inter-annual soil water memory?
  F7 nitrogen ladder (pooled cross-sectional): regress per-cell train-mean
     yield on nitrogen (linear/log/sqrt) with climate + texture controls.
  F8 window composition: pre-sowing buffer (days 1-30) vs season (31-240).

Usage:
    python3 fingerprint.py --crops wheat maize --n-cells 800 --n-workers 12
    python3 fingerprint.py --crop wheat --n-cells 200   # quick smoke test
Writes sandbox/results/fingerprint_{crop}.md and .json per crop.
"""

import argparse
import json
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from data import load_crop, pick_cells

RESULTS_DIR = Path(__file__).resolve().parent / "results"

HEAT_THRESHOLDS = [22, 26, 30, 34]
WB_VPD_K = 2.6                          # mm/day per kPa: crude PET tied to VPD
CO2_REF = 380.0
CO2_SAT_SCALE = 140.0                   # ppm, for 1 - exp(-(c - c0)/scale)
N_BASES = [0, 2, 4, 6, 8, 10, 12, 14, 16]

MIN_TRAIN_YEARS = 15
MIN_TEST_YEARS = 3


def esat(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


# ---------------------------------------------------------------------------
# Per-cell feature building
# ---------------------------------------------------------------------------

def build_year_features(clim, co2):
    """clim: (n_years, 240, 5). Returns dict of (n_years,) arrays, year-aligned."""
    tasmax = clim[:, :, 0].astype(np.float64)
    tasmin = clim[:, :, 1].astype(np.float64)
    pr = clim[:, :, 2].astype(np.float64)
    tmean = 0.5 * (tasmax + tasmin)
    vpd_day = np.clip(esat(tasmax) - esat(tasmin), 0.0, None)

    f = {}
    for k in range(8):
        seg = slice(30 * k, 30 * (k + 1))
        f[f"seg{k}"] = tmean[:, seg].mean(axis=1)
    f["tsum"] = tmean.sum(axis=1)
    f["tsum0"] = tmean[:, 0:30].sum(axis=1)
    f["tsum30"] = tmean[:, 30:240].sum(axis=1)
    f["prsum"] = pr.sum(axis=1)
    f["prsum0"] = pr[:, 0:30].sum(axis=1)
    f["prsum30"] = pr[:, 30:240].sum(axis=1)
    f["vpd_sum"] = vpd_day.sum(axis=1)
    f["wb"] = (pr - WB_VPD_K * vpd_day).sum(axis=1)
    for b in N_BASES:
        f[f"gdd_{b}"] = np.clip(tmean - b, 0.0, None).sum(axis=1)
    for thr in HEAT_THRESHOLDS:
        f[f"hdd_{thr}"] = np.clip(tmean - thr, 0.0, None).sum(axis=1)
        f[f"dcount_{thr}"] = (tmean >= thr).sum(axis=1)
    c = np.asarray(co2, dtype=np.float64)
    f["co2_lin"] = (c - 350.0) / 100.0
    f["co2_log"] = np.log(np.clip(c, 1.0, None) / CO2_REF)
    f["co2_sat"] = 1.0 - np.exp(-(c - 339.0) / CO2_SAT_SCALE)
    f["co2_sat"] = f["co2_sat"] - f["co2_sat"][0]
    return f


def build_cell_bundle(ds, cell):
    """Year-aligned feature bundle for one cell (None if too few train/test rows)."""
    rows = ds.cell_rows([int(cell)])
    clim, meta, y = ds.climate[rows], ds.meta[rows], ds.y[rows]
    years = meta[:, 3].astype(int)
    order = np.argsort(years)
    clim, y, years = clim[order], y[order], years[order]
    feats = build_year_features(clim, co2=meta[order, 4])

    tr = (years >= 382) & (years <= 411)
    te = (years >= 412) & (years <= 419)
    if int(tr.sum()) < MIN_TRAIN_YEARS or int(te.sum()) < MIN_TEST_YEARS:
        return None

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
    feats["_nitrogen"] = meta[order, 5]
    feats["_texture"] = meta[order, 6:19]
    return feats


# ---------------------------------------------------------------------------
# Closed-form ridge fit (per cell)
# ---------------------------------------------------------------------------

def design_matrix(feats, keys, mask):
    cols = [np.ones(int(mask.sum()))]
    for k in keys:
        a = np.asarray(feats[k], dtype=np.float64)
        tr = a[mask]
        mu, sd = float(tr.mean()), float(tr.std())
        if sd < 1e-9:
            sd = 1.0
        cols.append((a[mask] - mu) / sd)
    return np.column_stack(cols)


def fit_predict_cell(feats, y, years, keys, ridge):
    """Ridge fit on train (382..411), held-out R2 on test (412..419)."""
    tr = (years >= 382) & (years <= 411)
    te = (years >= 412) & (years <= 419)
    if int(tr.sum()) < 5 or int(te.sum()) < MIN_TEST_YEARS:
        return np.nan, []
    Xtr = design_matrix(feats, keys, tr)
    Xte = design_matrix(feats, keys, te)
    ytr, yte = y[tr], y[te]
    p = Xtr.shape[1]
    pen = np.eye(p) * ridge
    pen[0, 0] = 0.0
    theta = np.linalg.solve(Xtr.T @ Xtr + pen, Xtr.T @ ytr)
    pred = Xte @ theta
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    r2 = np.nan if ss_tot < 1e-12 else \
        float(1.0 - np.sum((yte - pred) ** 2) / ss_tot)
    return r2, theta[1:].tolist()


# ---------------------------------------------------------------------------
# Model families
# ---------------------------------------------------------------------------

def _models_for(crop):
    gb = 0 if crop == "wheat" else 10
    m = OrderedDict([
        ("F1_seg", ([f"seg{k}" for k in range(8)], True)),          # profile
        ("F2_tlin", (["tsum"], False)),
        ("F2_gdd", ([f"gdd_{gb}"], False)),
        *[(f"F2_hdd{thr}", ([f"hdd_{thr}"], False)) for thr in HEAT_THRESHOLDS],
        ("F2_dcount34", (["dcount_34"], False)),
        *[(f"F3_gdd{b}", ([f"gdd_{b}"], False)) for b in N_BASES],
        ("F4_none", ([], False)),
        ("F4_pr", (["prsum"], False)),
        ("F4_vpd", (["vpd_sum"], False)),
        ("F4_wb", (["wb"], False)),
        ("F4_pr_vpd", (["prsum", "vpd_sum"], True)),                 # profile
        ("F6_weather", (["prsum", "vpd_sum"], False)),               # F6 no-lag baseline
        ("F6_wb_lag", (["prsum", "vpd_sum", "wb_lag1"], False)),
        ("F6_pr_lag", (["prsum", "vpd_sum", "prsum_lag1"], False)),
        ("F5_none", (["tsum", "prsum", "vpd_sum"], False)),
        ("F5_lin", (["tsum", "prsum", "vpd_sum", "co2_lin"], False)),
        ("F5_log", (["tsum", "prsum", "vpd_sum", "co2_log"], False)),
        ("F5_sat", (["tsum", "prsum", "vpd_sum", "co2_sat"], False)),
        ("F8_full", (["tsum", "prsum"], False)),
        ("F8_pre", (["tsum0", "prsum0"], False)),
        ("F8_pos", (["tsum30", "prsum30"], False)),
        ("F8_both", (["tsum0", "prsum0", "tsum30", "prsum30"], False)),
    ])
    return m


def _worker(bundle, crop, ridge):
    """Fit all models for one cell. Returns (r2_by_name, coef_by_name)."""
    models = _models_for(crop)
    y, years = bundle["_y"], bundle["_years"]

    lag_ok = np.isfinite(bundle["wb_lag1"]) & np.isfinite(bundle["prsum_lag1"])

    r2, coefs = {}, {}
    for name, (keys, report) in models.items():
        if name.startswith("F6"):
            # F6 comparison group: all models on the SAME lag-available subset
            sub = {k: v[lag_ok] for k, v in bundle.items() if not k.startswith("_")}
            yy, yr = y[lag_ok], years[lag_ok]
        else:
            sub, yy, yr = bundle, y, years
        r, b = fit_predict_cell(sub, yy, yr, keys, ridge)
        r2[name] = r
        if report:
            coefs[name] = b
    return r2, coefs


def _worker_wrap(args):
    bundle, crop, ridge = args
    return _worker(bundle, crop, ridge)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _med(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan


def _winrate(a, b):
    """Paired win rate of A over B plus Wilcoxon signed-rank p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan, np.nan
    d = a[ok] - b[ok]
    wr = float(np.mean(d > 0))
    try:
        p = float(wilcoxon(d, zero_method="zsplit").pvalue)
    except ValueError:
        p = np.nan
    return wr, p


def _fw(x):
    return f"{x:+.3f}" if np.isfinite(x) else "  -  "


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(crop, ridge, bundles, results):
    n = len(results)
    r2 = {nm: np.array([r[nm] for r, _ in results]) for nm in _models_for(crop)}

    def prof(name, size):
        return np.vstack([c.get(name) for _, c in results
                          if c.get(name) and len(c[name]) == size]) \
            if n else np.empty((0, size))

    f1 = prof("F1_seg", 8)
    f4 = prof("F4_pr_vpd", 2)

    L = []
    A = L.append
    A(f"# {crop.capitalize()} fingerprints (B-track genesis analysis)\n")
    A(f"\nCells fitted: **{n}**. Train 382..411 (year 381 only as lag source), "
      f"test 412..419. Per-cell closed-form ridge (ridge={ridge:g}, intercept "
      "exempt), standardized features; R2 = held-out per-cell R2 on test years. "
      "Win rate A>B and p = Wilcoxon signed-rank p.\n")

    A("\n## F1  heat sensitivity by 30-day segment")
    if f1.size:
        med, q25, q75 = (np.nanmedian(f1, 0), np.nanpercentile(f1, 25, 0),
                         np.nanpercentile(f1, 75, 0))
        A("\n| days | median std. beta | [Q25, Q75] |")
        A("|---|---|---|")
        for k in range(8):
            A(f"| {30*k+1}-{30*(k+1)} | {med[k]:+.3f} | "
              f"[{q25[k]:+.3f}, {q75[k]:+.3f}] |")
        peak = int(np.nanargmin(med))
        A(f"\nMost heat-negative window: days {30*peak+1}-{30*(peak+1)} "
          "(heat there most hurts yield).")

    A("\n## F2  heat non-linearity / threshold")
    names = ["F2_tlin", "F2_gdd", "F2_hdd22", "F2_hdd26", "F2_hdd30",
             "F2_hdd34", "F2_dcount34"]
    order = sorted(names, key=lambda nm: -_med(r2[nm]))
    A("\nmedian per-cell R2 (desc):  " + "  ".join(
        f"{nm}={_fw(_med(r2[nm]))}" for nm in order))
    best = order[0]
    wr, p = _winrate(r2[best], r2["F2_tlin"])
    A(f"\nbest ({best}) vs tmean-linear: win {wr:.3f}, p={p:.4g}")
    for thr in HEAT_THRESHOLDS:
        wr, p = _winrate(r2[f"F2_hdd{thr}"], r2["F2_gdd"])
        A(f"hdd>={thr}C vs gdd: win {wr:.3f}, p={p:.4g}")

    A("\n## F3  GDD base sweep")
    A("\n| base C | median R2 | frac cells picking base |")
    A("|---|---|---|")
    argmax = np.full(n, -1)
    col = [r2[f"F3_gdd{b}"] for b in N_BASES]
    for i in range(n):
        v = [c[i] for c in col]
        if np.isfinite(v).all():
            argmax[i] = int(np.argmax(v))
    frac = {b: float(np.mean(argmax == j)) for j, b in enumerate(N_BASES)}
    frac = {b: v for b, v in frac.items() if v > 0}
    for b in N_BASES:
        A(f"| {b} | {_fw(_med(r2[f'F3_gdd{b}']))} | {frac.get(b, 0.0):.2f} |")
    top = max(N_BASES, key=lambda b: _med(r2[f"F3_gdd{b}"]))
    modal = max(N_BASES, key=lambda b: frac.get(b, 0.0))
    A(f"\nmedian-R2 best base: {top} C; modal per-cell pick: {modal} C.")

    A("\n## F4  water-stress channel")
    for nm in ["F4_none", "F4_pr", "F4_vpd", "F4_wb", "F4_pr_vpd"]:
        A(f"- {nm}: median R2 = {_fw(_med(r2[nm]))}")
    for a, b in [("F4_pr", "F4_none"), ("F4_vpd", "F4_none"),
                 ("F4_wb", "F4_none"), ("F4_pr_vpd", "F4_none"),
                 ("F4_pr_vpd", "F4_pr"), ("F4_wb", "F4_vpd")]:
        wr, p = _winrate(r2[a], r2[b])
        A(f"- {a} vs {b}: win {wr:.3f}, p={p:.4g}")
    if f4.size:
        mb = np.nanmedian(f4, 0)
        A(f"\njoint pr+vpd standardized betas: pr {mb[0]:+.3f}, "
          f"vpd {mb[1]:+.3f}; frac(pr>0)={np.mean(f4[:, 0] > 0):.2f}, "
          f"frac(vpd<0)={np.mean(f4[:, 1] < 0):.2f} "
          "(water-limited signature: wet helps, dry hurts)")

    A("\n## F5  CO2 response shape (in-sample sign check)")
    A("\ntrain CO2 range is narrow and level-drift confounds a separate CO2 "
      "signal - sign check only.")
    for nm in ["F5_none", "F5_lin", "F5_log", "F5_sat"]:
        A(f"- {nm}: median R2 = {_fw(_med(r2[nm]))}")
    for a in ["F5_lin", "F5_log", "F5_sat"]:
        wr, p = _winrate(r2[a], r2["F5_none"])
        A(f"- {a} vs none: win {wr:.3f}, p={p:.4g}")

    A("\n## F6  soil-memory carryover (lag-1 weather, same rows for all 3)")
    for nm in ["F6_weather", "F6_wb_lag", "F6_pr_lag"]:
        A(f"- {nm}: median R2 = {_fw(_med(r2[nm]))}")
    for a in ["F6_wb_lag", "F6_pr_lag"]:
        wr, p = _winrate(r2[a], r2["F6_weather"])
        A(f"- {a} vs no-lag: win {wr:.3f}, p={p:.4g}")

    A("\n## F7  nitrogen ladder (pooled cross-sectional)")
    f7 = _f7(A, bundles)

    A("\n## F8  window composition (buffer days 1-30 vs season 31-240)")
    for nm in ["F8_full", "F8_pre", "F8_pos", "F8_both"]:
        A(f"- {nm}: median R2 = {_fw(_med(r2[nm]))}")
    for a, b in [("F8_both", "F8_full"), ("F8_both", "F8_pre"),
                 ("F8_pos", "F8_full"), ("F8_pre", "F8_pos")]:
        wr, p = _winrate(r2[a], r2[b])
        A(f"- {a} vs {b}: win {wr:.3f}, p={p:.4g}")

    A("\n\n_Interpreting: per-cell held-out weather skill is small (level-drift "
      "identity), so win rates near 0.5 and |dR2| < ~0.03 are not informative; "
      "the era-stability gate (era_stability.py) stays the arbiter of candidate "
      "promotion._\n")

    md = "".join(L)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"fingerprint_{crop}.md"
    out.write_text(md)
    print(f"\n===== {crop} =====")
    print(md)

    summary = {
        "crop": crop, "n_cells": n,
        "F1_profile": med.tolist() if f1.size else [],
        "F1_peak_slice": int(np.nanargmin(med)) if f1.size else None,
        "F2_best": best,
        "F2_hdd_vs_gdd_winrate": {str(thr): _winrate(
            r2[f"F2_hdd{thr}"], r2["F2_gdd"])[0] for thr in HEAT_THRESHOLDS},
        "F3_median_best_base": top,
        "F3_modal_base": modal,
        "F3_med_r2": {str(b): _med(r2[f"F3_gdd{b}"]) for b in N_BASES},
        "F4_pr_win": _winrate(r2["F4_pr"], r2["F4_none"])[0],
        "F4_vpd_win": _winrate(r2["F4_vpd"], r2["F4_none"])[0],
        "F4_wb_win": _winrate(r2["F4_wb"], r2["F4_none"])[0],
        "F4_pr_beta": float(np.nanmedian(f4[:, 0])) if f4.size else None,
        "F4_vpd_beta": float(np.nanmedian(f4[:, 1])) if f4.size else None,
        "F6_wb_lag_win": _winrate(r2["F6_wb_lag"], r2["F6_weather"])[0],
        "F6_pr_lag_win": _winrate(r2["F6_pr_lag"], r2["F6_weather"])[0],
        "F8_pre_win": _winrate(r2["F8_pre"], r2["F8_pos"])[0],
        "F8_both_win": _winrate(r2["F8_both"], r2["F8_pos"])[0],
        "F7": f7,
    }
    jout = RESULTS_DIR / f"fingerprint_{crop}.json"
    jout.write_text(json.dumps(summary, indent=2))
    return out, jout


def _f7(A, bundles):
    if not bundles:
        A("\n(no cells)")
        return {}
    ymean = np.array([np.mean(b["_y"]) for b in bundles])
    nvec = np.array([np.mean(b["_nitrogen"]) for b in bundles])
    X = np.column_stack([np.ones(len(bundles)),
                         [np.mean(b["tsum"]) for b in bundles],
                         [np.mean(b["prsum"]) for b in bundles],
                         [np.mean(b["vpd_sum"]) for b in bundles],
                         *[[np.mean(b["_texture"][:, j]) for b in bundles]
                           for j in range(bundles[0]["_texture"].shape[1])]])
    yc = ymean - ymean.mean()

    def ols_r2(extra):
        Z = np.column_stack([X, extra]) if extra.size else X
        beta = np.linalg.lstsq(Z, ymean, rcond=None)[0]
        pred = Z @ beta
        return float(1 - np.sum((ymean - pred) ** 2) / np.sum(yc ** 2))

    r2_no = ols_r2(np.empty((len(ymean), 0)))
    r2_lin = ols_r2(nvec[:, None])
    r2_log = ols_r2(np.log(nvec[:, None] + 1))
    r2_sq = ols_r2(np.sqrt(nvec[:, None]))
    A(f"\nPooled regression R2 (climate+texture + N-shape): "
      f"no-N {r2_no:+.3f}, linear-N {r2_lin:+.3f}, "
      f"log-N {r2_log:+.3f}, sqrt-N {r2_sq:+.3f}")
    A(f"\nraw corr(train-mean yield, N) = {np.corrcoef(ymean, nvec)[0, 1]:+.3f} "
      "(N constant per cell -> cross-sectional fingerprint only).")
    return {"no": r2_no, "linear": r2_lin, "log": r2_log, "sqrt": r2_sq}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops", nargs="+", choices=["wheat", "maize"],
                    default=["wheat", "maize"])
    ap.add_argument("--n-cells", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=0.1)
    ap.add_argument("--n-workers", type=int,
                    default=min(os.cpu_count() or 1, 12))
    args = ap.parse_args()

    t0 = time.time()
    for crop in args.crops:
        ds = load_crop(crop)
        cells = pick_cells(ds, args.n_cells, seed=args.seed)
        bundles = [b for c in cells if (b := build_cell_bundle(ds, c)) is not None]
        print(f"[{crop}] {len(cells)} sampled -> {len(bundles)} fittable")
        tasks = [(b, crop, args.ridge) for b in bundles]
        if args.n_workers > 1:
            with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
                results = list(ex.map(_worker_wrap, tasks))
        else:
            results = [_worker_wrap(t) for t in tasks]
        outcrop = write_report(crop, args.ridge, bundles, results)
        print(f"[{crop}] wrote {outcrop[0]}")
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()