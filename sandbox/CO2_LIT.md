# CO₂ response: literature digest and candidate rationale

Origin: "broad research into models that explore the 1100 ppm CO₂ scenario, and
implement some of the literature models." This file records what the research
established, which literature numbers we encode, the four candidates built in
`programs/24_…`–`27_…`, and how they were judged (response surface + held-out
gate). Raw outputs: `results/co2_response.md`, `results/summary.md`,
`results/benchmark_{crop}.csv`.

## 1. The generating model (what we are trying to imitate)

The FutureCrop dataset (Sweet et al. 2024, Kaggle) is **simulated by a single
biophysical crop model** — "a crop model that has been thoroughly studied and
validated" (vs. an ensemble): rainfed/water-limited everywhere, N fertilised at
a per-country constant, 30 days pre-sowing + 210 days of daily weather per cell;
train 1980–2020 (real years ∼ 1982–...), test 2021–2100 under a **high-emissions
scenario (test CO₂ reaches 1108 ppm)**, so the test window is far outside the
training CO₂ range (~341–415 ppm).

The exact model is not disclosed, so we encode CO₂ response structures that are
**shared across the plausible candidate families** (DSSAT/CERES, EPIC/GEPIC,
APSIM, STICS, LPJmL/GEPIC family — all present in GGCMI Phase 2, Franke et al.
2020). These models agree on the mechanism: CO₂ enters as a **saturating
(hyperbolic/asymptotic) multiplier on radiation-use efficiency or
photosynthesis**, with the *direct* C3 benefit plus an *indirect* water-use
efficiency benefit; C4 (maize) has essentially no direct response.

Key model-side references:
- CERES-Wheat applies an **asymptotic (saturating) RUE multiplier** defined in
  the species file (`WHCER045.spe`).
- EPIC/GEPIC use a linear **or** hyperbolic multiplier
  `f = 1 + B1·(CO2−350)/(B2 + CO2−350)`; linear extrapolation past ~700 ppm is
  the known hazard (see below).
- RUE at 20 °C increases ~21% from 350 to 700 ppm; transpiration efficiency
  ~+37% (model review, Bullock et al. in "Novel multimodel ensemble approach…").
- GGCMI Phase 2 emulators (Franke 2020; Liu 2023) give per-cell polynomial /
  ML response surfaces to CO₂, T, water, N for cross-checking model CO₂
  sensitivity.

## 2. Empirical anchors (FACE and optimality)

- FACE (free-air CO2 enrichment), ~353→550 ppm: **C3 yields +19% on average**
  (Kimball 2016); wheat +10% well-watered, up to +20% under water limitation
  (Kimball 2010). **C4 maize: no direct photosynthetic/yield response** in the
  absence of drought; benefit only via stomatal conductance → WUE under water
  stress (Leakey et al. 2006; Ainsworth & Long 2021).
- Optimality/PC model (Ainsworth & Long 2020): predicts wheat **+11.7%** for
  380→550 ppm vs. observed FACE median **+11.16%** — our mid target.
- NBER/OCO-2 econometric (Taylor, Wolfram, Schlenker): corn +0.5%, wheat +0.8%
  per ppm — clearly at the **high end** vs FACE (~0.09%/ppm); treat as an upper
  bound, not a default.
- Rubisco-limited theory: beyond ~700 ppm the photosynthetic gain saturates.
  A saturated multiplier **must** flatten toward an asymptote; a linear form
  (EPIC-linear, β=0.77 for C3 → **+140% at 1108 ppm**) is physically
  unjustified and is exactly the extrapolation failure we avoid.

## 3. The identification problem that forced fixed constants

Per cell, CO₂ moves only ~75 ppm over 39 train years (341→415) and is collinear
with the trend; the data-driven CO₂ slope is unidentifiable (benchmarks:
`06`'s fitted `f·log(co2/380)` ≈ 0, `19`'s fitted CO₂ ≈ 0, universal across the
candidate set; §7 of ANALYSIS.md). Fitting the CO₂ response per cell from this
record would be fitting noise and would not transfer to 1108 ppm.

**Decision (user): freeze CO₂ parameters as literature constants** — zero CO₂
regression DOF. The weather block keeps its per-location fitted params under
the no-pooling protocol, and the CO₂ factor is a deterministic, physically
anchored transfer function applied at test time.

## 4. Encoding: shared saturating shape

All four candidates use the same hyperbolic factor

```
h(C) = (C − 400) / (C − 400 + 350)          Cref = 400 ppm, K = 350 ppm
h(380) = −0.054   h(440) = +0.103   h(550) = +0.300   h(700) = +0.375   h(1108) = +0.669
```

| candidate | CO₂ channel | fixed constant | implied gain @550 / @1108 |
|---|---|---|---|
| `24_co2_saturating_multiplier` | whole-production multiplier (RUE/photosynthetic scaling, C3) | γ wheat 0.40, maize 0.03 | +12% / +27% (wheat, in-target); ~+2% maize |
| `25_co2_wue_drought_maize` | drought-penalty attenuation (WUE, C4 channel) | β maize 0.30, wheat 0.10 | small; drought-contingent by construction |
| `26_co2_heat_amelioration` | heat-penalty attenuation (stomatal cooling) | β wheat 0.30, maize 0.15 | small (wheat heat term weak in-record) |
| `27_rue_co2` | solar-radiation term × (1+δ·h) (RUE coupling) | δ wheat 0.50, maize 0.10 | small (radiation term ≈ 0 when fitted) |
| EPIC-linear (reference, not a candidate) | `f = 1 + β(C−350)/350` | β 0.77 / 0.11 | **+30% @550 / +140% @1108** — the hazard |

Scales chosen so wheat `24` lands on the Ainsworth & Long 2020 midpoint
(+12% @550) and saturates to ≈ +27% @1108 (within the +20..+35% band); maize
direct effect ≈ 0 (FACE), any maize benefit routed through the WUE/heat
channels where the literature puts it.

## 5. Response-surface check (results/co2_response.md, 200 cells/crop)

Implied median % yield change vs 400 ppm at each cell's climatological weather:

| crop | model | @550 | @1108 | vs bands |
|---|---|---|---|---|
| wheat | 06 (incumbent) | +0.3 | +0.9 | low (as expected — unidentifiable in-sample) |
| wheat | 13 (incumbent) | 0 | 0 | low |
| wheat | **24 multiplier** | **+12** | **+26.8** | **in 10..19 / 20..35** |
| wheat | 25 / 26 / 27 | ≈0 | ≈0 | low |
| maize | 06 / 13 | ≈0 | ≈0 | OK (C4 target ≈ low) |
| maize | 24 | +0.9 | +2.0 | OK (direct ≈ 0) |
| maize | 25 / 26 / 27 | ≈0 | ≈0 | OK |
| EPIC-linear ref (wheat) | — | +30 | +140 | high — the avoided hazard |

The interactive channels (`25–27`) imply near-zero *mean* response because at
climatological weather their fitted weather coefficients are small; their value,
if any, is year-to-year within the 1108-ppm regime (e.g., hot/dry years where
the penalty term is large), not in the mean level change. The sole candidate
carrying the C3 mean CO₂ benefit is `24`.

## 6. Held-out gate (1000 cells/crop, unchanged headline metric)

Per the protocol the CO₂ candidates must at least not degrade held-out R²:

| crop | model | R² med | pairwise vs incumbent |
|---|---|---|---|
| wheat | 13 (incumbent) | **−0.112** | — |
| wheat | 24 | −0.121 | worse |
| wheat | 27 | −0.118 | worse (mean δ R² = −0.002, 46% cells >0) |
| wheat | 25 / 26 | −0.129 / −0.130 | worse |
| maize | 06 (incumbent) | **−0.130** | — |
| maize | 27 | −0.123 | **not robust**: 46.6% cells better, mean δ = −0.002 (tail artifact of the median) |
| maize | 25 / 24 / 26 | −0.129 / −0.130 / −0.137 | tie or worse |

**Verdict: none of the literature-CO₂ candidates clears the held-out gate.**
The val window (CO₂ ~420–440 ppm) barely exercises the multiplier (h ≈ 0.1),
so this is the expected outcome of a conservative gate — the CO₂ structure is
not detected by an 8-year, small-CO₂ holdout. The apparent `27`-on-maize median
win is a tail artifact (46.6% of cells improve; mean δ R² = −0.002) and is not
treated as a win. Wheat `13` and maize `06` remain the submission models.

## 7. What this does (and does not) say for the 2021–2100 test

- The val gate cannot see CO₂ extrapolation by construction; the response
  surface is the evidence that `24`'s wheat multiplier is right-sized
  (+12% @550, +26.8% @1108 — dead in the literature bands) while the incumbent
  models embed ~0 CO₂ and the EPIC-linear form would overshoot ~5× at 1108.
- If a submission is built from `24`, the per-location CO₂ behaviour is fully
  determined by the frozen literature constants — no in-sample overfit, no
  pooled fit, transferable by design.
- Open questions: (a) whether the generating model's CO₂ response follows the
  C3 mean (+12% @550) or the upper end of model spreads; (b) year-to-year
  interplay of the CO₂ factor with heat/drought penalties at 1108 ppm (the
  channels `25`/`26`/`27` encode, currently ≪ mean-level); (c) whether a CO₂
  × water-stress *level* interaction should be added once the regime is reached.