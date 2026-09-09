"""
era_stability.py
----------------
Anti-winner's-curse gate for the wheat CO2 branch.

The main benchmark selects on the single held-out window 412-419 (8 years).
This script re-fits wheat candidates 13 (weather-only) and 24 (+literature
saturating CO2) on an EARLIER era (train 381-400) and scores them on 401-419,
so the promotion decision does not rest on the one 8-year window.

Decision rule (pre-registered): promote 24 over 13 only if BOTH hold
  1. no pairwise-majority loss   -> pct(24 better) >= 50% AND median dR2 >= 0
  2. the win is not a tail artifact (cf. model 27 on maize: 46.6% better cells
     but mean dR2 = -0.002)      -> mean dR2 >= 0
Otherwise keep 13 for the weather-only branch. Local data can only VETO a CO2
candidate, never confirm it (the CO2 signal lives at 420-1108 ppm, beyond the
validation window).

Usage:
    python3 era_stability.py --n-cells 1000 --n-workers 8
Writes sandbox/results/era_stability.md.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from data import load_crop, pick_cells, per_cell_splits
from validate import run_program

RESULTS_DIR = Path(__file__).resolve().parent / "results"
WHEAT_A = "13_water_heat_bilinear"   # weather-only core
WHEAT_CO2 = "24_co2_saturating_multiplier"


def pair_table(a, b):
    """Pairwise comparison on cells valid for both (finite R2)."""
    ca = {r["cell"]: r["r2"] for r in a if np.isfinite(r.get("r2", np.nan))}
    cb = {r["cell"]: r["r2"] for r in b if np.isfinite(r.get("r2", np.nan))}
    cells = sorted(set(ca) & set(cb))
    da = np.array([ca[c] for c in cells])
    db = np.array([cb[c] for c in cells])
    d = db - da
    return {
        "n": len(cells),
        "med_a": float(np.median(da)), "med_b": float(np.median(db)),
        "mean_a": float(np.mean(da)), "mean_b": float(np.mean(db)),
        "med_d": float(np.median(d)), "mean_d": float(np.mean(d)),
        "pct_b_better": float(100.0 * np.mean(d > 0)),
        "wins_b": int(np.sum(d > 0.05)), "wins_a": int(np.sum(d < -0.05)),
        "ties": int(np.sum(np.abs(d) <= 0.05)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop", choices=["wheat"], default="wheat")
    ap.add_argument("--n-cells", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-train-years", type=int, default=15)
    ap.add_argument("--n-workers", type=int, default=None)
    args = ap.parse_args()

    train_years, test_years = (381, 400), (401, 419)
    ds = load_crop(args.crop)
    cells = pick_cells(ds, args.n_cells, seed=args.seed, train_years=train_years,
                       min_train_years=args.min_train_years)
    splits = per_cell_splits(ds, cells, train_years, test_years)
    print(f"{args.crop}: {len(splits)} fittable cells "
          f"(train {train_years[0]}-{train_years[1]}, test {test_years[0]}-{test_years[1]})")

    res_a = run_program(WHEAT_A, splits, maxiter=60, ridge=0.1,
                        n_workers=args.n_workers, return_per_cell=True,
                        crop=args.crop)
    res_b = run_program(WHEAT_CO2, splits, maxiter=60, ridge=0.1,
                        n_workers=args.n_workers, return_per_cell=True,
                        crop=args.crop)
    p = pair_table(res_a["per_cell"], res_b["per_cell"])

    passed = p["pct_b_better"] >= 50.0 and p["med_d"] >= 0.0 and p["mean_d"] >= 0.0
    print(f"\n{args.crop} era-stability  train {train_years[0]}-{train_years[1]} / "
          f"test {test_years[0]}-{test_years[1]}")
    print(f"  matched cells: {p['n']}")
    print(f"  {WHEAT_A:32s}  med {p['med_a']:+.3f}   mean {p['mean_a']:+.3f}")
    print(f"  {WHEAT_CO2:32s}  med {p['med_b']:+.3f}   mean {p['mean_b']:+.3f}")
    print(f"  delta (CO2 - base): med {p['med_d']:+.3f}   mean {p['mean_d']:+.3f}"
          f"   pct CO2 better {p['pct_b_better']:.1f}%")
    print(f"  wins |dR2|>0.05: {WHEAT_CO2} {p['wins_b']}  vs  {WHEAT_A} {p['wins_a']}"
          f"   ties {p['ties']}")
    verdict = ("PASS - CO2 candidate survives era shift" if passed
               else "FAIL - keep weather-only core for this branch")
    print(f"  GATE: {verdict}")

    md = f"""# Era-stability gate (wheat CO2 branch)

Split captured here: **train {train_years[0]}–{train_years[1]} /
test {test_years[0]}–{test_years[1]}**  ({p['n']} matched cells), compared with the
main benchmark's train 381–411 / test 412–419. This checks that promoting the
CO2 candidate does not rest on the single 8-year validation window.

| | median R2 | mean R2 |
|---|---|---|
| {WHEAT_A} | {p['med_a']:+.3f} | {p['mean_a']:+.3f} |
| {WHEAT_CO2} | {p['med_b']:+.3f} | {p['mean_b']:+.3f} |
| delta (CO2 − base) | {p['med_d']:+.3f} | {p['mean_d']:+.3f} |

- pairwise: {WHEAT_CO2} better on {p['pct_b_better']:.1f}% of cells
- wins at |dR2| > 0.05: {WHEAT_CO2} {p['wins_b']} / {WHEAT_A} {p['wins_a']} /
  ties {p['ties']}

**Gate (pre-registered): PASS iff no pairwise-majority loss AND median dR2 >= 0
AND mean dR2 >= 0.  → {('PASS' if passed else 'FAIL')}**

{'The CO2 candidate survives the era shift; the wheat comparison is not an '
 '8-year-window artifact.' if passed else 'The CO2 candidate loses majority '
 'support on an earlier era; keep the weather-only core for the submission '
 'branch (local data can veto a CO2 candidate but never confirm it).'}
"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "era_stability.md"
    with open(out, "w") as f:
        f.write(md)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()