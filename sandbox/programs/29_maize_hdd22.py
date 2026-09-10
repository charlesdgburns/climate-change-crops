"""
29_maize_hdd22.py
-----------------
Single location, saturating responses + VPD stress, with the heat penalty as a
FULL-WINDOW low-threshold heat degree-day term (hdd>=22 over all 240 days).
B-track F2 for maize: full-window hdd>=22 is the best heat shape (win vs
tmean-linear 0.60, p<1e-19), better than hdd>=26/30/34. Model 06 core with
heat30 -> tanh(hdd22 full window):
    yield = a
          + b * tanh(GDD/1000 - 1)      # growing season warmth, saturating
          + c * log1p(PREC)             # water, sublinear
          - d * tanh(HDD22/1000)        # low-threshold heat, saturating
          - e * tanh(VPD/2.5)           # vapour pressure deficit stress
          + f * log(co2 / 380)          # co2 fertilisation
"""

import numpy as np


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prec = np.sum(pr, axis=0)
    hdd22 = np.sum(np.maximum(tmean - 22.0, 0.0), axis=0)
    es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
    vpd = np.mean(es(tasmax) - es(tasmin), axis=0)
    return gdd, prec, hdd22, vpd


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d, e, f = params
    gdd, prec, hdd22, vpd = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prec)
            - d * np.tanh(hdd22 / 1000.0)
            - e * np.tanh(vpd / 2.5)
            + f * np.log(co2 / 380.0))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.1, 0.3, 2.0])