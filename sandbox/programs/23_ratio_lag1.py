"""
23_ratio_lag1.py
----------------
Single location, yield-ratio (multiplicative) structure with Lag-1 weather
carry-over. Combines the multiplicative yield-fraction model (21) with
inter-annual memory from previous year's GDD and VPD (22).
    Y = a * G * W * H * L
    G = 1 + b * tanh(GDD/1000 - 1)      # thermal growth
    W = 1 + c * tanh(PRCMID/3)          # water supply
    H = 1 - d * tanh(VD/50)             # VPD stress
    L = 1 + e * tanh(GDD_PREV/1000 - 1) - f * tanh(VD_PREV/50)  # carry-over
GDD/VPD Lag-1 autocorrelation ~ +0.17/+0.11; precipitation ~0.
8 params. Derived features computed inline from the raw (240, T) series.
"""

import numpy as np


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prc_mid = np.sum(pr[80:160], axis=0)
    es = lambda T_: 0.6108 * np.exp(17.27 * T_ / (T_ + 237.3))
    vpd = es(tasmax) - es(tasmin)
    vd = np.sum(np.maximum(vpd - 2.0, 0.0), axis=0)
    gdd_prev = np.concatenate([[gdd[0]], gdd[:-1]])
    vd_prev = np.concatenate([[vd[0]], vd[:-1]])
    return gdd, prc_mid, vd, gdd_prev, vd_prev


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d, e, f = params
    gdd, prc_mid, vd, gdd_prev, vd_prev = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    G = 1.0 + b * np.tanh(gdd / 1000.0 - 1.0)
    W = 1.0 + c * np.tanh(prc_mid / 3.0)
    H = 1.0 - d * np.tanh(vd / 50.0)
    L = 1.0 + e * np.tanh(gdd_prev / 1000.0 - 1.0) - f * np.tanh(vd_prev / 50.0)
    return a * G * W * H * L


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.3, 0.3, 0.3])
