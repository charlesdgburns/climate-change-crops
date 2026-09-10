"""
31_maize_hdd26.py
-----------------
Single location, saturating responses + VPD stress, heat penalty as a
FULL-WINDOW low-threshold heat degree-day term (hdd>=26 over all 240 days).
B-track F2 for maize: hdd>=26 is second-best heat shape vs tmean-linear;
hdd>=22 was best but is redundant with GDD, so try the 26 threshold.
Mode: 06 core with heat30 -> tanh(hdd26 full window).
"""

import numpy as np


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prec = np.sum(pr, axis=0)
    hdd26 = np.sum(np.maximum(tmean - 26.0, 0.0), axis=0)
    es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
    vpd = np.mean(es(tasmax) - es(tasmin), axis=0)
    return gdd, prec, hdd26, vpd


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d, e, f = params
    gdd, prec, hdd26, vpd = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prec)
            - d * np.tanh(hdd26 / 900.0)
            - e * np.tanh(vpd / 2.5)
            + f * np.log(co2 / 380.0))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.1, 0.3, 2.0])