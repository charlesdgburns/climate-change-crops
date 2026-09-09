"""
seed_programs.py
----------------
Starting programs for the FutureCrop FunSearch run.

Interface (MUST match what the LLM will be told, and evaluate_programs.py):
    x = {"climate": float32 (T, 240, 5), "meta": float32 (T, 21)}
    y = (T,) simulated crop yield
    model(x, params)  -> (N,) yield predictions
    estimate_params(x, y) -> starting parameter vector (no iterative fitting)

climate channels: 0 tasmax (degC), 1 tasmin (degC), 2 pr (mm/day),
                  3 rsds (W/m2), 4 cumulative rsds
meta columns:     0 cell_id, 1 lon, 2 lat, 3 year, 4 co2 (ppm),
                  5 nitrogen, 6..18 texture one-hot, 19 crop(0 wheat/1 maize),
                  20 yield_train_cell_mean (per-location baseline)

Seed 1 = the mandated baseline: predict the per-location train mean yield.
Seeds 2-4 add mechanistic structure (CO2 response, growing-degree days,
water, heat stress) that the LLM should generalise and improve on.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Seed 1: per-location mean baseline
# ---------------------------------------------------------------------------

def model(x, params):
    """Predict each location's TRAIN mean yield (the competition baseline).
    Captures the location fixed effect but no weather variability, so its
    per-cell R2 should be ~0 (score ~100)."""
    return x["meta"][:, 20].astype(np.float64)


def estimate_params(x, y):
    """No free parameters needed for the mean baseline."""
    return np.array([0.0])


# ---------------------------------------------------------------------------
# Seed 2: baseline x CO2 fertilisation response
# ---------------------------------------------------------------------------

def model_v2(x, params):
    """Per-location mean scaled by a CO2 fertilisation response.
    yield = mean_loc * (1 + a * log(co2 / co2_ref)).
    As atmospheric CO2 rises (train 340->415 ppm, test higher), yield grows."""
    a, co2_ref = params
    meta = x["meta"]
    loc_mean = meta[:, 20]
    co2 = meta[:, 4]
    return loc_mean * (1.0 + a * np.log(co2 / co2_ref))


def estimate_params_v2(x, y):
    """Modest positive CO2 response; reference = mean train CO2."""
    co2_mean = float(np.mean(x["meta"][:, 4]))
    return np.array([0.3, co2_mean])


# ---------------------------------------------------------------------------
# Seed 3: baseline + linear responses to GDD, water, heat, CO2
# ---------------------------------------------------------------------------

def model_v3(x, params):
    """Mechanistic equation on top of the per-location baseline:
        yield = mean_loc
              + a * GDD            (growing degree days, base 8 degC)
              + b * PREC           (total precipitation, mm)
              + c * HEAT           (days with tasmax > 30 degC)
              + d * CO2            (annual co2, ppm)
    Aggregates are computed inside the model from the 240-day climate series."""
    a, b, c, d = params
    clim = x["climate"]
    meta = x["meta"]

    tmean = 0.5 * (clim[:, :, 0] + clim[:, :, 1])
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=1)          # (N,)
    prec = np.sum(clim[:, :, 2], axis=1)                         # (N,)
    heat = np.sum(clim[:, :, 0] > 30.0, axis=1).astype(np.float64)  # (N,)
    co2 = meta[:, 4]                                             # (N,)

    return meta[:, 20] + a * gdd + b * prec + c * heat + d * co2


def estimate_params_v3(x, y):
    """Small starting coefficients (GDD ~ 2000, precip ~ several hundred mm,
    heat ~ tens of days, CO2 ~ 380 ppm)."""
    return np.array([1e-4, 1e-4, 1e-3, 1e-2])


# ---------------------------------------------------------------------------
# Seed 4: baseline + saturating GDD and water terms
# ---------------------------------------------------------------------------

def model_v4(x, params):
    """Saturating response to heat/water, combined with CO2:
    yield = mean_loc
          + a * tanh(GDD / 1000 - 1)
          + b * log1p(PREC)
          - c * HEAT
          + d * log(CO2 / 380)"""
    a, b, c, d = params
    clim = x["climate"]
    meta = x["meta"]

    tmean = 0.5 * (clim[:, :, 0] + clim[:, :, 1])
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=1)
    prec = np.sum(clim[:, :, 2], axis=1)
    heat = np.sum(clim[:, :, 0] > 30.0, axis=1).astype(np.float64)

    return (meta[:, 20]
            + a * np.tanh(gdd / 1000.0 - 1.0)
            + b * np.log1p(prec)
            - c * heat
            + d * np.log(meta[:, 4] / 380.0))


def estimate_params_v4(x, y):
    return np.array([1.0, 0.4, 0.1, 2.0])


# ---------------------------------------------------------------------------
# Standalone check — parse and report what program_parser finds
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "FunSearch"))
    from FunSearch.program_parser import script_to_strings

    programs = script_to_strings(__file__)
    print(f"{len(programs)} seed program(s) found:")
    for i, p in enumerate(programs):
        print(f"\n--- Seed {i + 1} ---")
        print("model_src     :", p.model_src[:60], "...")
        print("estimator_src :", p.estimator_src[:60], "...")