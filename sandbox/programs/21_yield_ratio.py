"""
21_yield_ratio.py
-----------------
Single location, yield-ratio (multiplicative) structure.
Each weather factor is a dimensionless multiplier ~1; the intercept is the
potential yield (seeded to cell mean). Weather effects are percentage changes,
not additive corrections -- more physically grounded than Y = a + weather.
    Y = a * G * W * H
    G = 1 + b * tanh(GDD/1000 - 1)      # thermal growth (1 +/- b)
    W = 1 + c * tanh(PRCMID/3)          # water supply (1 +/- c)
    H = 1 - d * tanh(VD/50)             # VPD stress (0 to 1-d)
VD = sum of (VPD - 2 hPa)+ over season; med wheat ~3, maize ~32.
PRCMID = precipitation in calendar mid-season (days 80-160).
Derived features are computed inline from the raw (240, T) series.
"""

import numpy as np


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prc_mid = np.sum(pr[80:160], axis=0)
    es = lambda T_: 0.6108 * np.exp(17.27 * T_ / (T_ + 237.3))
    vpd = es(tasmax) - es(tasmin)
    vd = np.sum(np.maximum(vpd - 2.0, 0.0), axis=0)
    return gdd, prc_mid, vd


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d = params
    gdd, prc_mid, vd = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    G = 1.0 + b * np.tanh(gdd / 1000.0 - 1.0)
    W = 1.0 + c * np.tanh(prc_mid / 3.0)
    H = 1.0 - d * np.tanh(vd / 50.0)
    return a * G * W * H


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.3])
