# GENESIS.md — B-track: identifying the generating crop model

The FutureCrop generator is a biophysical crop model (rainfed, water-limited,
per-country constant N). The competition deliberately withholds its identity.
This B-track recorded our attempt to identify it by *behavioral
fingerprinting*, decided the verdict, and pinned the guarded design priors
that the fingerprint evidence supports.

**Verdict (B3, `fingerprint_match.md`): NO distinct winner.**
CERES-DSSAT and STICS tie at 0.755; APSIM is inside the indistinguishable
bucket (0.645, gap ≤ 0.10). Because the adoption guardrail requires a distinct
winner, **no single family is adopted as "the model"** and no priors-based
submission candidate was built (B4 conditional = not triggered). What carries
forward is the *robust common set* below.

---

## 1. Method (B1)

`fingerprint.py` fits per-location, linear-in-params models (intercept +
standardized features, closed-form ridge) with train 382-411 / test 412-419
and reports median per-cell held-out R2 + paired win rates (Wilcoxon p). A
cell enters only with all 39 years present. 1109 wheat / 1200 maize cells.
Year-381 only serves as the lag source.

Observables F1-F8 (full tables in `results/fingerprint_{crop}.md`):

| obs | wheat (1109 cells) | maize (1200 cells) |
|---|---|---|
| **F1 heat, 30d slices** | weak; most-neg seg 211-240 (-0.031) | **peak days 121-150 (-0.091, IQR upper −0.013): mid-season** |
| **F2 heat shape** | hdd≥30/34 best (win vs gdd 0.52 n.s.); linear weak | **hdd≥22/26 clearly best (win vs tlin 0.60, p<1e-19)** |
| **F3 GDD base** | flat; modal pick 0 & 16 C | flat; modal pick 0 & 16 C |
| **F4 water** | **pr win 0.601 (p<1e-9); vpd/wb inert; joint β pr +0.026/vpd −0.017** | **vpd win 0.614 & wb 0.620 (p<1e-12); joint β vpd −0.074 (76% of cells neg)** |
| F5 CO2 in-sample | any term hurts (win ~0.40, p<1e-17) | any term hurts (win ~0.41, p<1e-15) |
| F6 soil memory (lag1) | lag wb/pr hurt (win 0.40, p<1e-19) | lag hurt (win 0.41, p<1e-15) |
| **F7 N ladder (pooled)** | saturating (log/sqrt best; corr 0.49) | saturating (sqrt 0.514; corr 0.63) |
| **F8 window** | buffer ~inert; 240d full best | 240d full ≥ others |

Interpretation caveats:
- Per-cell held-out weather skill is small (level-drift identity, ANALYSIS.md
  §1-2), so win rates near 0.5 are noise; the *significant* contrasts above
  (F1 maize, F2 maize, F4 both) are the reliable drivers of the match.
- F5's "CO2 hurts" is the level-drift identity acting again — *not* evidence
  against a CO2 effect; in-sample CO2 is unidentifiable (see CO2_LIT.md).
- F6 "no memory" is about *weather* carryover between consecutive years; a
  model with strong soil-water pools should have shown lag value, so families
  whose priors demand inter-annual soil memory scored down.

## 2. Family priors and scoring (B2/B3)

`models_library.py` encodes published mechanistic signatures for 7 candidate
families (CERES, EPIC, APSIM, STICS, LPJmL, AquaCrop, WOFOST) in the same
categorical space as the observables; `fingerprint_match.py` scores weighted
agreement + leave-one-observable-out sensitivity + indistinguishable set.

| family | score | note |
|---|---|---|
| CERES / STICS | 0.755 | agree on wheat-pr supply, heat, N; disagree maize-VPD demand |
| APSIM | 0.645 | in overlap bucket; agrees maize-VPD, disagrees wheat-pr |
| EPIC | 0.593 | agrees maize-VPD + N, disagrees soil-memory "yes" |
| LPJmL | 0.522 | VPD-based but smooth heat, strong soil memory |
| WOFOST | 0.475 | wheat-supply ok, but **no N** |
| AquaCrop | 0.436 | **no heat mechanism**, fertility constant |

The decisive disagreement: observed **wheat is precipitation-supply driven and
maize is VPD-demand driven**. No single library family claims that split, so
no family wins outright; an emulator should simply include **both** channels.

## 3. Robust common set (guarded priors for candidate design)

1. **Water**: wet-helps via precipitation (wheat) AND dryness-hurts via VPD
   (maize), both significant. A candidate with only one channel leaves signal
   on the table for the other crop.
2. **Heat**: flowering-window heat for **maize at low threshold** (~22-26 C
   mean; hdd≥22/26) — current `06` uses tmax>30, likely too coarse a
   threshold. Wheat: high-threshold (≥30-34) grain-fill heat, weak signal, so
   heat terms should **differ by crop** (or be dropped for wheat).
3. **Nitrogen**: saturating cross-sectionally. Mitscherlich-type N form OK;
   under the per-location protocol N is constant per cell, so only usable
   pooled (FunSearch `meta` col 5) — no per-location fit can use it.
4. **Soil memory**: no inter-annual weather carryover patsible → do not add
   lag terms in per-location candidates.
5. **CO2**: unidentifiable in-sample; rely on literature-saturating C3 /
   near-C4 values already in candidates 24-27 (CO2_LIT.md).
6. **Window**: the full 240-day window (30d pre-sowing + 210d season) is
   adequate; no evidence the pre-sowing buffer carries independent signal.

## 4. Alignment with the current portfolio

The incumbent submission cores already embody the strongest findings:
- wheat `13` (pr + VD + mid-window water) matches wheat's pr-supply channel;
- maize `06` (VPD + pr + heat30) matches maize's VPD-demand channel.
The single most actionable (unexploited) insight for a future candidate: a
**maize low-threshold flowering heat** term (hdd≥22/26, days ~120-150) is the
strongest unused signature. It was pursued (candidates 28–32) and **does not
beat 06** on the benchmark + era gate — see below.

## 5. What would falsify this (and what the maize-heat test found)

- A maize candidate with hdd≥22/26 + mid-window weighting beating 06 on
  benchmark + era gate would confirm the F1/F2 maize heat finding.
  **Tested (programs 28–32, 1000 cells/crop): NOT confirmed.**
  - Replacement variants (28 window-only hdd22, 29 full-window hdd22,
    30 window-only count≥26, 31 full-window hdd26) all **lose** to 06 on
    median per-cell R² and pairwise majority (pct-better 37–43%, median
    δR² −0.03…−0.06). The F2 hdd22 win was measured as a *lone feature*
    (intercept-only baseline); inside 06's GDD+VPD+pr package the
    low-threshold hdd is redundant with GDD and the windowed tanh shape
    underperforms 06's linear heat30/8.
  - Only the **additive** variant (32 = 06 + thin flower-heat term) passes:
    pairwise 51.8% better on bench (mean δR² +0.002), era gate PASS (55.0%
    better, mean δR² +0.001). The delta is within noise vs the incumbent;
    a submission change is not justified (keeps an extra parameter for ~0).
  - **Falsification verdict:** the strong version of the maize flowering-heat
    claim (replace heat30 with low-threshold flowering hdd) is falsified by
    the benchmark. The additive version survives only at noise level and is
    NOT adopted. `06` stays the maize submission core.
- A family would become a distinct winner only if the CERES/STICS tie
  breaks — e.g., if an external reveal names the generator, or if a finer
  water-channel splitter (supply vs demand within one crop) separates the
  top two with >0.10 margin.

## 6. Literal-simulator comparison: do the process models themselves match?

A separate, deliberately-parametric test (user decision, after the 024
submission improved the leaderboard and we wanted a mechanism-anchored CO₂
form for the 1100 ppm test regime): run simplified but faithful versions of
the top-5 candidate families *literally*, with **all** parameters fixed from
the literature (`sim/families.py`), and check which family's weather response
best reproduces the observed yields. No climate-response parameter is fitted:
the per-location fit is only a 2-param affine `y_obs ≈ a + b·y_sim`.

- **Simulator engine** (`sim/core.py`): GDD phenology → trapezoid LAI →
  Priestley-Taylor (supply-type: CERES/STICS) or VPD-driven (demand-type:
  APSIM/EPIC/LPJmL) PET → single-bucket soil water (residual carried across
  seasons for demand families) → RUE × intercepted light × water-stress ×
  family heat-window damage × saturating N × saturating CO₂ → harvest index.
- **Judged on a strict dual criterion (≥0.10 margin needed on BOTH legs**
  before a winner is named):
  - Leg 1 — held-out per-cell R² of the affine-fit onto observed yields.
  - Leg 2 — "fingerprint-of-simulator": re-run the B-track F4 (water channel),
    F2 (heat), F3 (GDD base), F6 (memory) weather diagnostics on each
    family's *simulated* series and compare against the same-cell observed
    fingerprint (signature distance).
  - Leg 1b (supporting) — direct unfitted within-cell corr(obs, sim).

**Result (500 cells/crop, `results/sim_match_{crop}.md`): NO WINNER.** No
family clears both 0.10 margins for either crop. But there is a consistent
directional signal:

| crop | Leg 1 best | Leg 1b mean corr (supply vs demand) | Leg 2 closest |
|---|---|---|---|
| wheat | stics −0.232, margin +0.013 | +0.06 (63%) vs −0.08 (37%) | apsim (dist 0.024, margin +0.005) |
| maize | ceres −0.284, margin +0.003 | +0.16 (75%) vs +0.04 (63%) | ceres (dist 0.090, margin +0.030) |

- For **maize**, the supply-driven, radiation-PET family group (CERES/STICS)
  is closer than the demand/VPD group on all three legs — ceres beats apsim
  pairwise 0.656 (p≈2e-15) on held-out R² and 75% of cells show positive
  unfitted correlation. SAPSIM-type demand (VPD+strong soil memory) is the
  poorest maize match.
- For **wheat**, all simulators are near-noise: supply corr ~+0.06, demand ~
  −0.08; nothing reproduces the observed few-percent weather signal. This is
  the same null the fingerprints reported for wheat (no distinct weather
  channel).
- The method does **not** break the CERES/STICS tie: for maize both sit at the
  top of every leg, separated by ~0.003–0.030 — far below 0.10. This is
  *consistent* with §2/§5: the generator's discriminating signature already
  sat in the CERES/STICS water/supply/radiation space, and literal simulation
  cannot split the two.

**Decision under the guardrails:** keep `submission_baseline_13_06.ipynb` and
`submission_co2_24_06.ipynb` unchanged (maize `06`, wheat `13`/`24`). The
literature-CO₂ arm (`24`) stays the scientifically-justified CO₂ form for the
1100 ppm test years; the simulator test found no family capable of replacing
its weather package. If a winner had emerged we would have built a candidate
from its CO₂ constants and run benchmark + pairwise + era gate first.

## 7. Reproduce

```
cd sandbox
python3 fingerprint.py --crops wheat maize --n-cells 1200 --n-workers 16
python3 fingerprint_match.py          # needs models_library.py
python3 run_benchmark.py --crops maize --models 06_saturating_vpd 32_maize_flower_plus
python3 era_gate_maize_flowering.py   # era gate for 32 vs 06 on maize
python3 sim_match.py --crops wheat maize --n-cells 500 --n-workers 8   # literal simulators
```
Outputs (gitignored `results/`): `fingerprint_{crop}.md/.json`,
`fingerprint_match.md`, `era_gate_maize_flowering.md`,
`sim_match_{crop}.md/.json`. The maize-heat candidates live in
`programs/28_maize_flowering_heat.py` … `32_maize_flower_plus.py`; the
literal simulators live in `sim/core.py`, `sim/families.py`.