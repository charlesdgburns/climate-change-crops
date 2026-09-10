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
strongest unused signature. If pursued, it must clear the standard gates
(benchmark, pairwise-majority, era-stability) before touching the submission
portfolio.

## 5. What would falsify this

- A maize candidate with hdd≥22/26 + mid-window weighting beating 06 on
  benchmark + era gate would confirm the F1/F2 maize heat finding.
- A family would become a distinct winner only if the CNES/STICS tie
  breaks — e.g., if an external reveal names the generator, or if a finer
  water-channel splitter (supply vs demand within one crop) separates the
  top two with >0.10 margin.

## 6. Reproduce

```
cd sandbox
python3 fingerprint.py --crops wheat maize --n-cells 1200 --n-workers 16
python3 fingerprint_match.py          # needs models_library.py
```
Outputs (gitignored `results/`): `fingerprint_{crop}.md/.json`,
`fingerprint_match.md`.