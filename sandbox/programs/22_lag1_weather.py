"""
22_lag1_weather.py
------------------
Single location, additive model with Lag-1 weather carry-over.
Previous year's GDD and VPD are added as predictors to capture soil
moisture carry-over, pest pressure, and other inter-annual memory.
GDD and VPD have significant Lag-1 autocorrelation (r ~ +0.17 / +0.11);
precipitation does not (r ~ 0), so only temperature-derived terms lagged.
    Y = a
      + b * tanh(GDD/1000 - 1)          # current thermal growth
      + c * log1p(PRCMID)               # current flowering-window water
      - d * tanh(VD/50)                 # current VPD exceedance
      + e * tanh(GDD_PREV/1000 - 1)     # previous year thermal
      - f * tanh(VD_PREV/50)            # previous year VPD
Lag-1 computed by shifting arrays by 1 year (assumes consecutive years;
correct for ~66% maize, ~27% wheat cells; noisy for others).
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
    # Lag-1: shift by 1 year; fill first year with current value
    gdd_prev = np.concatenate([[gdd[0]], gdd[:-1]])
    vd_prev = np.concatenate([[vd[0]], vd[:-1]])
    return gdd, prc_mid, vd, gdd_prev, vd_prev


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d, e, f = params
    gdd, prc_mid, vd, gdd_prev, vd_prev = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prc_mid)
            - d * np.tanh(vd / 50.0)
            + e * np.tanh(gdd_prev / 1000.0 - 1.0)
            - f * np.tanh(vd_prev / 50.0))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.3, 0.3, 0.3])
