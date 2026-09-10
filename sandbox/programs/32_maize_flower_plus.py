"""
32_maize_flower_plus.py
-----------------------
Single location, 06_saturating_vpd UNCHANGED plus an ADDITIVE flowering-window
low-threshold heat stress (B-track F1/F2 for maize: days 121-150 most
heat-negative; hdd>=22/26 best shape). Tests incremental signal beyond 06's
heat30+VPD+GDD versus pure replacement (models 28-31, which all lost):
    yield = a
          + b * tanh(GDD/1000 - 1)      # growing season warmth, saturating
          + c * log1p(PREC)             # water, sublinear
          - d * (HEAT30/8)              # 06's high-threshold count, linear
          - e * tanh(VPD/2.5)           # vapour pressure deficit stress
          + f * log(co2 / 380)          # co2 fertilisation
          - g * tanh(CFL26/6)           # additive flowering heat (days 121-150, tmean>=26C)
"""

import numpy as np


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prec = np.sum(pr, axis=0)
    heat30 = np.sum(tasmax > 30.0, axis=0)
    cfl26 = np.sum(tmean[120:150] >= 26.0, axis=0)
    es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
    vpd = np.mean(es(tasmax) - es(tasmin), axis=0)
    return gdd, prec, heat30, vpd, cfl26


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d, e, f, g = params
    gdd, prec, heat30, vpd, cfl26 = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prec)
            - d * (heat30 / 8.0)
            - e * np.tanh(vpd / 2.5)
            + f * np.log(co2 / 380.0)
            - g * np.tanh(cfl26 / 6.0))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.1, 0.3, 2.0, 0.1])