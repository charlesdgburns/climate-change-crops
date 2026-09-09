# Analysis: per-cell held-out R² and the level-drift problem

Session findings that shaped the sandbox's modelling direction. Headline metric
is unchanged: **per-cell R² on the held-out 8 test years (412–419)**, median
across cells, fitted per location on train years 381–411.

## TL;DR

- The negative per-cell R² is **two additive terms**: an unforecastable
  `level-anchor` penalty from the drift between the train-window mean and the
  held-out-window mean, plus a `weather-term` from year-varying weather
  response. Predictions are *not* constant — they just carry a level anchor.
- The 8-year held-out level **cannot be estimated from any per-location input
  we have** (verified: trend, CO₂, recent-window means all fail; cross-cell
  drift is ~5–16% predictable from window-mean climate/CO₂ changes). Beating
  this drift is, by design, the FutureCrop challenge — we keep the metric and
  work on the weather term.
- Weather skill is **real on maize** (≈ +0.11 R²-points of the held-out, and
  similar or better under LOO / block-CV evaluations) and **~absent on wheat**.
- Static per-cell attributes are screened out: soil texture is constant per cell
  and explains <5% of response variation (no USDA/AWC moderation in this data);
  nitrogen tags cross-cell levels (Spearman 0.55–0.57) but is per-cell constant
  and subsumed by the per-location baseline. (Section 3.)
- Utility of extra weather structure is crop-specific. Current best on the
  metric: wheat `11_heat_intensity` / `13_water_heat_bilinear` / `17_co2_water`
  (−0.112 / −0.112 / −0.113), maize `06_saturating_vpd` (−0.130).
  Physics-grounded candidates (continuous VPD, water balance, VPD×water)
  all tie or regress — the VD/50 exceedance index and GDD×water bilinear
  already capture the dominant signals.
- Structural alternatives tested (multiplicative yield-ratio, Lag-1 weather
  carry-over, both combined, log-space fit, fixed intercept) do **not** beat
  the additive+intercept form on the headline metric. The intercept dominance
  is irreducible: with intercept=0, R² is −22 (wheat) / −82 (maize); the
  weather terms cannot predict absolute yield. Yield lag-1 autocorrelation is
  ~0; weather lag-1 persistence (GDD +0.17, rsds +0.24) carries no extra yield
  signal; and the heat→yield response is itself non-stationary (stronger over
  the record). Only `21_yield_ratio`'s multiplicative form posts the best
  *pooled* MSE on maize (0.838 vs 06's 0.936) without winning per-cell median.

## 1. Metric and the baseline identity

Per cell, held-out R² is defined against the held-out window's own mean:

```
R² = 1 − Σ(y−ŷ)² / Σ(y−ȳ_te)²          ȳ_te = held-out-window mean
```

For a **constant prediction** c (including the per-location train mean):

```
Σ(y−c)² = Σ(y−ȳ_te)² + n(ȳ_te − c)²   ⇒   R² = − n(ȳ_te − c)² / SST ≤ 0
```

So a level-anchored predictor has R² ≤ 0 **in every location, by construction**:
the penalty is quadratic in the level gap, so *signs of drift cancel across
cells, the penalty never does*. 0% of cells can exceed R² = 0 with a constant
predictor. Measured drift: wheat 61% of cells drift up vs down 39%, maize 29% up
vs 71% down; median |drift| ≈ 0.18 yield units both crops.

For a real model, decompose additively:

```
R²_model  =  level-anchor term  +  weather term
```

- level-anchor: same model with its year-varying weather frozen at the test
  window mean (const-baseline equivalent): wheat −0.14, maize −0.16.
- weather term: the net contribution of year-varying weather predictions.
  Wheat **−0.007** (no signal), maize **+0.11** (real signal).
- sanity: the models are not constant — prediction std over the 8 test years is
  42% (wheat) / 61% (maize) of held-out yield std.

## 2. Level-drift forensics (all levers tested, median per-cell R²)

| lever | wheat | maize |
|---|---|---|
| held-out baseline (train-window mean) | −0.147 | −0.294 |
| + per-location linear trend extrapolation | −0.329 | −0.286 |
| + per-location CO₂ regression | −0.386 | −0.331 |
| + trend AND CO₂ | −1.033 | −0.840 |
| level = nearest-2/3/5/8-yr train mean | ≥ full-mean in all cases | ≥ full-mean |
| cross-cell drift ~ Δ(window-mean gdd,prec,heat,logCO₂) | R²=0.045 | R²=0.161 |

The window-mean climate/CO₂ betas are unstable across crops (CO₂ slope +0.66
wheat, −9.7 maize), i.e., collinear + not a real driver of the drift here.
Conclusion: the held-out window's **level is effectively unpredictable** from
the 39-year per-location record; the drift is decadal weather realisation.

## 3. Static per-cell attributes: soil texture and nitrogen (screened)

**Data facts.** `texture_class` is constant within every cell — 0 of 8,663
(wheat) and 0 of 9,303 (maize) cells change texture across years; 11–12 classes,
class 9 dominant (~57% / 40% of cells). Train and test define the same cells, so
soil is knowable at prediction time (moot under the per-location protocol).
`nitrogen` shares the same structure (constant per cell, range ~0.4–650).

**Why static covariates can't move the metric.** Every sandbox model is fitted
per location, so per-cell constants are absorbed by the intercept and the
per-location response coefficients. A covariate that never varies within a cell
is structurally inert for per-cell held-out R² — unless it is used in a *pooled*
fit, which we deliberately don't do.

**Screen 1 — weak response moderation, strong level association** (per-cell OLS
gdd+prec+heat30, 2,520 wheat / 2,929 maize cells):

| Spearman | wheat | maize |
|---|---|---|
| precip slope ~ texture | +0.048 | +0.077 |
| yield level ~ texture | +0.157 | −0.064 |
| yield level ~ **nitrogen** | **+0.552** | **+0.571** |
| precip slope ~ nitrogen | −0.058 | −0.005 |

Median precip slopes are flat (≈ −0.000 … +0.001) across all texture classes.

**Screen 2 — strong-driver check** (per-cell OLS gdd + prc_mid + heat + vd,
2,358 wheat / 2,941 maize cells, held-out R²):

| driver | Spearman(slope ~ texture) | η² (texture) |
|---|---|---|
| water (prc_mid) | +0.018 / +0.014 | 0.5% / 4.5% |
| VPD exceedance (vd) | +0.009 / +0.055 | 0.6% / 0.2% |
| heat | −0.027 / −0.000 | 0.7% / 1.7% |
| held-out R² | ≈ +0.016 | 0.7% |

Texture explains ≤ 4.5% of between-cell response variation (mostly < 1%). The
sole 4.5% comes from small high-leverage classes (e.g., maize class 2, n=56,
water slope 0.745 vs 0.02–0.15 elsewhere) — small-sample outliers, not a
gradient; held-out R² shows no soil effect.

**Conclusion.** There is no soil moderation of the water / VPD / heat response to
exploit in this data — a USDA available-water-capacity-scaled water term is not
supported, and we will not pool. Nitrogen is by far the strongest static *level*
tag, but it is constant per cell and already subsumed by the per-location
baseline (`meta[:,20]`) in any pooled interface. If a future *pooled* submission
model is ever built (FunSearch already exposes soil/nitrogen via `x["meta"]`),
**nitrogen** is the covariate worth trying there — not soil.

## 4. Weather skill is real on maize, absent on wheat

| evaluation | metric | wheat | maize |
|---|---|---|---|
| fixed 8-yr holdout | median per-cell R², OLS gdd+prec+heat | −0.213 | −0.166 |
| same, but weather frozen at window mean (level term) | | −0.139 | −0.164 |
| leave-one-out (≈30 scored yrs/cell) | median per-cell R² | −0.128 (31%>0) | **+0.066 (59%>0)** |
| 4×8-yr in-record blocks | median per-cell R² | −0.325 (15%>0) | −0.151 (34%>0) |

Maize generalises in every scheme; richer raw-linear driver sets (7 features)
*overfit* on both crops (−0.34/−0.31 vs −0.21/−0.17), and bounded composite
candidates outperform the raw-linear form.

## 5. Current benchmark (1000 cells/crop, fixed held-out headline metric)

| model | wheat | maize | notes |
|---|---|---|---|
| baseline per_cell_train_mean | −0.157 | −0.311 | |
| 04_saturating / 05_weather_only | −0.155 / −0.154 | −0.139 / −0.140 | |
| 06_saturating_vpd | −0.130 (66%) | **−0.130 (72%)** | best maize; VPD stress |
| 07_heat_spell | −0.115 (69%) | −0.190 | best wheat among 4-part |
| 08_water_window | −0.122 (66%) | −0.213 | |
| 09_full_mechanistic (7 params) | −0.122 (65%) | −0.199 | never beats its parts |
| 10_light_water (solar rsds) | −0.133 (70%) | −0.263 (76%) | solar term harmful both crops |
| 11_heat_intensity (degree-heat) | **−0.112 (69%)** | −0.188 (78%) | new wheat co-best |
| 12_vpd_exceedance (>2 hPa) | −0.115 (69%) | −0.190 (78%) | ~tied w/ 07; maize worse than 06 |
| 13_water_heat_bilinear | **−0.112 (66%)** | −0.188 (78%) | 06's simple mean-VPD still unbeaten on maize |
| 14_heat_threshold (crop-aware) | −0.123 (66%) | −0.200 (76%) | wheat 27C/maize 32C per literature; no gain |
| 15_anthesis_window | −0.133 (65%) | −0.201 (73%) | GDD-fraction midpoint heuristic; disappointing |
| 16_multiplicative (FAO/Liebig) | −0.126 (65%) | −0.196 (73%) | product form; no gain over additive |
| 17_co2_water (CO2×drought) | **−0.113 (66%)** | −0.197 (76%) | near-tied wheat; weak maize |
| 18_vpd_continuous (daily VPD) | −0.124 (66%) | −0.215 (75%) | continuous VPD worse than exceedance |
| 19_water_balance (prec−PET) | −0.115 (68%) | −0.188 (78%) | ties 13 on both crops |
| 20_vpd_water_stress (VPD×P) | −0.123 (66%) | −0.215 (74%) | interaction doesn't help |
| 21_yield_ratio (multiplicative) | −0.130 (63%) | −0.163 (78%) | non-additive structure; best maize pooled MSE (0.838) but worse median R² than 06 |
| 22_lag1_weather (GDD,VD prev yr) | −0.119 (65%) | −0.204 (75%) | Lag-1 carry-over adds little; maize worse than 06 |
| 23_ratio_lag1 (multiplicative + Lag-1) | −0.133 (61%) | −0.178 (74%) | both innovations together still lose to 06 maize |

(% = cells beating the per-location baseline.)

Wheat winners: heat *intensity* (11) and GDD×water bilinear (13) remain
co-best at −0.112; `17_co2_water` is a near-tie at −0.113; `19_water_balance`
ties at −0.115 with the highest pct_gt_baseline (68.4%). `18` (continuous VPD)
and `20` (VPD×water) regress vs the mechanistic candidates. The multiplicative
`21_yield_ratio` (−0.130) and Lag-1 `22` (−0.119) are respectable but neither
beats `13`. Wheat stays additive-friendly: all weather terms near zero, so the
intercept-dominated fit cannot be improved by non-additive structure.

Maize: `06_saturating_vpd` at −0.130 is **still unbeaten**. Of the structural
candidates, `21_yield_ratio` (−0.163) is the best multi-param newcomer and
posts the *lowest pooled MSE of any candidate* (0.838 vs 06's 0.9355) with the
highest pct_gt_baseline (77.5%) — by pooled/aggregate metrics it edges 06, but
its per-cell median is worse, meaning ~half the cells do fine but more cells do
worse. `22` (Lag-1 weather, −0.204) and `23` (−0.178) don't help. The
temperature-derived Lag-1 autocorrelation (~+0.17) that motivated carry-over
does not carry yield signal: precipitation is iid year-to-year, and the yield
response to persistent GDD/VPD is non-stationary (heat sensitivity strengthens
over the record — see below), so the past is a weak guide to the present.

### Structural-alternative diagnostics (why non-additive models don't lift the headline)

- **Fixed-intercept test**: intercept=0 → R² = −22.6 (wheat) / −81.9 (maize);
  intercept=global-mean → −4.1 / −24.5. Weather terms alone cannot predict the
  absolute yield level; the per-cell mean is what carries held-out R². Any
  structure that downweights the intercept trades a huge level-anchor
  guaranteed term for a small weather term.
- **Log-space fit**: fitting in log(y) is *worse* than level-space on both
  crops (wheat −0.17 vs −0.09; maize −0.27 vs −0.18, model 06 structure).
- **Yield Lag-1 autocorrelation**: raw yield lag-1 ~ −0.03 (wheat) / +0.03
  (maize); residual lag-1 after model 06 ~ −0.03 / −0.02. No exploitable
  inter-annual yield memory.
- **Weather Lag-1 autocorrelation**: GDD +0.17, rsds +0.24, VPD +0.11 — real
  persistence; precipitation ~0 (iid). But this persistence is already *in*
  the provided weather inputs for future years, so carry-over terms can only
  add a small soil-moisture/pest proxy — measured here and negligible.
- **Non-stationary response**: weather→yield correlation shifts across the
  record's halves, GDD going *more negative* (wheat −0.06→−0.15;
  maize −0.27→−0.36). The heat-stress signal is strengthening over time —
  consistent with climate-change amplification. A fixed intercept
  (long-run mean) increasingly drifts from the current yield level, which is
  exactly the level-anchor problem from §2.

## 6. Data quirks

- `tasmax…cumrsds` day axis first `(240, T)`; `co2 (T,)` per location.
- Units are not obviously physical: seasonal precip ~3–20 (data units). The
  prepared-cache `co2` column is atmospheric (341–415 ppm), matching the
  problem description; `log(co2/380)` is valid in-record.  The `soil_co2`
  parquet file's co2 range differs — see FreeFunSearch prep docs.
- rsds on growing days ~125–250 W/m²; VPD (hPa) daily ~1–3.

## 7. Implications for FutureCrop

- The competition scores median per-cell R² over future decades (~80 yrs), vs
  our 8-yr held-out. An 80-yr window mean ≈ a smooth trajectory: the
  level-error is far smaller and its remaining part IS extrapolatable
  (CO₂→560 ppm, technology) — so level-following mechanisms matter more there
  than this proxy credits.
- On maize, median per-cell R² on the true (long-window) metric should be
  materially above the −0.13 we measure here; wheat is a genuinely harder
  level+weather problem.
- Physics-grounded candidates (14–20) all fail to beat the existing best.
  Key findings: (a) continuous daily VPD is the single strongest in-sample
  predictor (r=−0.45 with maize yield) but the exceedance index (VD/50)
  generalizes better — the *tail* of the VPD distribution matters more than
  the baseline; (b) water balance (prec−PET) ties the GDD×water bilinear
  (13) on both crops but doesn't beat 06; (c) the VPD×water interaction
  doesn't help beyond the individual terms.
- Fitted parameter analysis reveals that GDD is near-zero or negative on
  both crops (30–35% of cells), CO₂ is universally dead (90%+ of cells
  ignore it), and the anthesis-window heuristic doesn't locate flowering
  accurately enough. The intercept dominates on wheat (weather terms used
  by <50% of cells); on maize, PRCMID and VD/50 are the only consistently
  active terms (45–62% of cells).
- Direction: the current candidates already capture the dominant physics
  for maize (water + VPD + interaction). Further improvement likely
  requires either (a) external data (flowering dates, soil moisture
  measurements), (b) a process-based crop model structure, or (c) pooling
  across cells (rejected). For wheat, the weather response is too weak to
  detect with our current data and methods.