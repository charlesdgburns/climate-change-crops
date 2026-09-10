"""
era_gate_maize_flowering.py
---------------------------
Era-stability gate (pre-registered, same rule as era_stability.py) for the
maize flowering-heat diagnostic candidate 32_maize_flower_plus vs the maize
incumbent 06_saturating_vpd, on an EARLIER era (train 381-400 / test 401-419)
so the promotion decision does not rest on the single 8-year window 412-419.

Decision rule: PASS iff
  1. no pairwise-majority loss   -> pct(32 better) >= 50% AND median dR2 >= 0
  2. not a tail artifact         -> mean dR2 >= 0

Usage:
    python3 era_gate_maize_flowering.py --n-cells 1000 --n-workers 16
Writes sandbox/results/era_gate_maize_flowering.md.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from data import load_crop, pick_cells, per_cell_splits
from validate import run_program
from era_stability import pair_table

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASE = "06_saturating_vpd"
CAND = "32_maize_flower_plus"
CROP = "maize"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-cells", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-train-years", type=int, default=15)
    ap.add_argument("--n-workers", type=int, default=None)
    args = ap.parse_args()

    train_years, test_years = (381, 400), (401, 419)
    ds = load_crop(CROP)
    cells = pick_cells(ds, args.n_cells, seed=args.seed, train_years=train_years,
                       min_train_years=args.min_train_years)
    splits = per_cell_splits(ds, cells, train_years, test_years)
    print(f"{CROP}: {len(splits)} fittable cells "
          f"(train {train_years[0]}-{train_years[1]}, test {test_years[0]}-{test_years[1]})")

    t0 = time.time()
    res_a = run_program(BASE, splits, maxiter=60, ridge=0.1,
                        n_workers=args.n_workers, return_per_cell=True, crop=CROP)
    res_b = run_program(CAND, splits, maxiter=60, ridge=0.1,
                        n_workers=args.n_workers, return_per_cell=True, crop=CROP)
    p = pair_table(res_a["per_cell"], res_b["per_cell"])
    print(f"  ({time.time()-t0:.0f}s)")

    passed = p["pct_b_better"] >= 50.0 and p["med_d"] >= 0.0 and p["mean_d"] >= 0.0
    print(f"\n{CROP} era-stability  train {train_years[0]}-{train_years[1]} / "
          f"test {test_years[0]}-{test_years[1]}")
    print(f"  matched cells: {p['n']}")
    print(f"  {BASE:26s}  med {p['med_a']:+.3f}   mean {p['mean_a']:+.3f}")
    print(f"  {CAND:26s}  med {p['med_b']:+.3f}   mean {p['mean_b']:+.3f}")
    print(f"  delta ({CAND} - {BASE}): med {p['med_d']:+.3f}   mean {p['mean_d']:+.3f}"
          f"   pct {CAND} better {p['pct_b_better']:.1f}%")
    print(f"  wins |dR2|>0.05: {CAND} {p['wins_b']}  vs  {BASE} {p['wins_a']}"
          f"   ties {p['ties']}")
    verdict = ("PASS - flowering heat survives era shift" if passed
               else "FAIL - keep 06 for maize")
    print(f"  GATE: {verdict}")

    md = f"""# Era-stability gate (maize flowering-heat candidate)

Split captured here: **train {train_years[0]}–{train_years[1]} /
test {test_years[0]}–{test_years[1]}**  ({p['n']} matched cells), compared with the
main benchmark's train 381–411 / test 412–419.

| | median R2 | mean R2 |
|---|---|---|
| {BASE} | {p['med_a']:+.3f} | {p['mean_a']:+.3f} |
| {CAND} | {p['med_b']:+.3f} | {p['mean_b']:+.3f} |
| delta ({CAND} − {BASE}) | {p['med_d']:+.3f} | {p['mean_d']:+.3f} |

- pairwise: {CAND} better on {p['pct_b_better']:.1f}% of cells
- wins at |dR2| > 0.05: {CAND} {p['wins_b']} / {BASE} {p['wins_a']} /
  ties {p['ties']}

**Gate (pre-registered): PASS iff no pairwise-majority loss AND median dR2 >= 0
AND mean dR2 >= 0.  → {('PASS' if passed else 'FAIL')}**

{'The additive flowering-heat term survives the era shift; the benchmark win '
 'is not an 8-year-window artifact.' if passed else 'The flowering-heat '
 'candidate loses majority support on an earlier era; keep 06 for maize.'}
Note: the additive term (32) passed the main-window pairwise gate but the
bare additive delta is tiny (~+0.002). This gate is the arbiter.
"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "era_gate_maize_flowering.md"
    with open(out, "w") as f:
        f.write(md)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()