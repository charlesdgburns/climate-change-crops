"""
16_multiplicative.py
--------------------
Single location, multiplicative (Liebig / Schutz / FAO-33) production function.
Yield is the product of bounded dimensionless factors:
    yield = a * G * W * H
where:
    G = 1 + b * tanh(GDD/1000 - 1)               # thermal growth (1 +/- b)
    W = 1 + c * tanh(PREC/5)                     # water adequacy (1 +/- c)
    H = 1 - d * tanh(HT_CROP/DEN)                # heat penalty (0 to 1-d)
HT_CROP is crop-aware: wheat sum max(tmean-27,0) / 100; maize sum max(tmean-32,0) / 100.
Tests whether non-additivity matters for these data.
"""

import numpy as np

THRESHOLDS = {"wheat": 27.0, "maize": 32.0}


def _drivers(tasmax, tasmin, pr, rsds, cumrsds, crop):
    tmean = 0.5 * (tasmax + tasmin)
    T_heat = THRESHOLDS.get(crop, 30.0)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prec = np.sum(pr, axis=0)
    ht = np.sum(np.maximum(tmean - T_heat, 0.0), axis=0)
    return gdd, prec, ht


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params, crop="wheat"):
    a, b, c, d = params
    gdd, prec, ht = _drivers(tasmax, tasmin, pr, rsds, cumrsds, crop)
    G = 1.0 + b * np.tanh(gdd / 1000.0 - 1.0)
    W = 1.0 + c * np.tanh(prec / 5.0)
    H = 1.0 - d * np.tanh(ht / 100.0)
    return a * G * W * H


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y, crop="wheat"):
    return np.array([np.mean(y), 1.0, 0.5, 0.3])
