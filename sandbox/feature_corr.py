"""
feature_corr.py
---------------
Pooled within-cell cross-correlation of the candidate feature set.

Why within-cell: every per-cell-constant feature (nitrogen, texture) is
inert under the per-location protocol, so only year-to-year variation
matters. Features are computed per location from the raw (240, T) series,
then the Pearson / Spearman correlation is computed ACROSS THAT CELL'S YEARS,
pooled over all sampled cells (each cell contributes one correlation at a
time; we pool the mean-centered products). This measures the redundancy the
optimizer sees.

Canonical feature set (the union used across candidates 06-27):
    gdd        sum(max(tmean-8, 0))                        [13, 06, ...]
    prec       sum(pr)                                     [06, 19]
    prc_mid    sum(pr[80:160])                             [13, 17, 19]
    heat30     count(tasmax > 30)                          [06, 25, 26, 27]
    ht         sum(max(tmean-ogT, 0)), crop-aware (27/32C) [16]
    spell3     >=3-day runs of tasmax>30                   [11]
    vd         sum((vpd-2)+), es(tasmax)-es(tasmin)        [12, 13, 17, 19]
    vpd_mean   mean(es(tasmax)-es(tasmin))                 [06, 25, 26, 27]
    water_bal  prec - PET (Hargreaves)                     [19]
    lgt        mean rsds on GDD>0 days                     [10, 27]
    cum_end    final cumulative rsds                       [not used directly]
    co2        annual atmospheric CO2                      [all CO2 candidates]

Interpretation rule: |r| >= 0.7 (Pearson, within-cell pooled) marks a pair as
redundant *candidates for removal*; removal is only justified if a slimmed
model passes the same held-out gate (pairwise-majority + era-stability). High
between-feature correlation is expected between different flavours of the same
physical driver (e.g. vd vs vpd_mean); the interesting question is which
variant generalises, which the candidates already answer.

Usage:
    python3 feature_corr.py --crops wheat maize --n-cells 500 --n-workers 8
Writes sandbox/results/feature_corr.md (and prints the matrix + report).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_crop, pick_cells, per_cell_splits

RESULTS_DIR = Path(__file__).resolve().parent / "results"

FEATURES = ["gdd", "prec", "prc_mid", "heat30", "ht", "spell3",
            "vd", "vpd_mean", "water_bal", "lgt", "cum_end", "co2"]

T_HTTP = {"wheat": 27.0, "maize": 32.0}


def _count_runs_geq(cond, k):
    cond2d = cond.T
    n = cond2d.shape[0]
    b = np.hstack([np.zeros((n, 1), bool), cond2d, np.zeros((n, 1), bool)])
    s = b[:, 1:] & ~b[:, :-1]
    runid = np.cumsum(s, axis=1)
    rlen = np.bincount(runid.ravel())[runid]
    return np.sum((rlen >= k) & s, axis=1).astype(float)


def compute_features(split, crop):
    """Year-feature matrix (T, n_features) for one cell's train years."""
    tr = split["train"]
    tasmax, tasmin, pr, rsds, cumrsds, co2 = (
        tr["tasmax"], tr["tasmin"], tr["pr"], tr["rsds"], tr["cumrsds"], tr["co2"])
    tmean = 0.5 * (tasmax + tasmin)
    f = {}
    f["gdd"] = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    f["prec"] = np.sum(pr, axis=0)
    f["prc_mid"] = np.sum(pr[80:160], axis=0)
    f["heat30"] = np.sum(tasmax > 30.0, axis=0)
    f["ht"] = np.sum(np.maximum(tmean - T_HTTP[crop], 0.0), axis=0)
    f["spell3"] = _count_runs_geq(tasmax > 30.0, 3)
    es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
    vpd = es(tasmax) - es(tasmin)
    f["vd"] = np.sum(np.maximum(vpd - 2.0, 0.0), axis=0)
    f["vpd_mean"] = np.mean(vpd, axis=0)
    f["water_bal"] = f["prec"] - np.sum(0.0023 * (tmean + 17.8) * rsds * 0.408, axis=0)
    grow = tmean - 8.0 > 0.0
    f["lgt"] = np.sum(np.where(grow, rsds, 0.0), axis=0) / np.maximum(np.sum(grow, axis=0), 1)
    f["cum_end"] = cumrsds[-1, :]
    f["co2"] = co2
    return np.column_stack([f[k] for k in FEATURES])


def pooled_corr(matrices, method="pearson"):
    """Pooled correlation across cells: mean-center each cell's columns, then
    compute one correlation from the concatenated centered matrix."""
    parts = [m - m.mean(axis=0, keepdims=True) for m in matrices]
    X = np.concatenate(parts, axis=0)
    if method == "spearman":
        X = pd.DataFrame(X).rank().to_numpy()
    C = np.corrcoef(X.T)
    return C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops", nargs="+", choices=["wheat", "maize"],
                    default=["wheat", "maize"])
    ap.add_argument("--n-cells", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-train-years", type=int, default=20)
    ap.add_argument("--n-workers", type=int, default=min(os.cpu_count() or 1, 8),
                    help="accepted for interface parity; corr computed serially")
    args = ap.parse_args()

    sections = []
    for crop in args.crops:
        ds = load_crop(crop)
        cells = pick_cells(ds, args.n_cells, seed=args.seed,
                           train_years=(381, 411),
                           min_train_years=args.min_train_years)
        splits = per_cell_splits(ds, cells, (381, 411), (412, 419))
        mats = [compute_features(s, crop) for s in splits]
        P = pooled_corr(mats, "pearson")
        S = pooled_corr(mats, "spearman")
        dfp = pd.DataFrame(P, index=FEATURES, columns=FEATURES)
        dfs = pd.DataFrame(S, index=FEATURES, columns=FEATURES)

        n = len(mats)
        lines = [f"## {crop}  ({n} cells, pooled within-cell)",
                 f"Pearson (lower triangle) / Spearman (upper triangle) of the "
                 f"year-features (mean-centered per cell, then pooled):",
                 "",
                 "| feature | " + " | ".join(FEATURES) + " |",
                 "|---|" + "---|" * len(FEATURES)]
        for i, f in enumerate(FEATURES):
            row = []
            for j in range(len(FEATURES)):
                if j < i:
                    row.append(f"{P[i][j]:+.2f}")
                elif j > i:
                    row.append(f"{S[i][j]:+.2f}")
                else:
                    row.append("·")
            lines.append(f"| **{f}** | " + " | ".join(row) + " |")
        redundant = []
        for i in range(len(FEATURES)):
            for j in range(i + 1, len(FEATURES)):
                rp, rs = P[i][j], S[i][j]
                if max(abs(rp), abs(rs)) >= 0.7:
                    redundant.append((FEATURES[i], FEATURES[j],
                                      f"{rp:+.2f}", f"{rs:+.2f}"))
        lines.append("")
        lines.append("### Redundancy report (|r| >= 0.7)")
        if redundant:
            lines.append("| pair | pearson | spearman |")
            lines.append("|---|---|---|")
            for a, b, rp, rs in redundant:
                lines.append(f"| {a} ~ {b} | {rp} | {rs} |")
        else:
            lines.append("none.")
        lines.append("")
        lines.append("Notes: vd vs vpd_mean are the same physical driver "
                     "(exceedance vs mean of daily VPD); ht vs gdd/heat30 are "
                     "both thermal accumulation; water_bal vs prec overlap by "
                     "construction. A correlated pair only justifies removal if "
                     "a slimmed model passes the held-out gate.")
        sections.append("\n".join(lines))

    md = ("# Feature cross-correlations (pooled within-cell)\n\n"
          "Year-feature redundancy as seen by the per-location fits. "
          "|r| >= 0.7 (either metric) flags a candidate-removal pair.\n\n"
          + "\n\n".join(sections) + "\n")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "feature_corr.md"
    with open(out, "w") as f:
        f.write(md)
    print(md[:3000])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()