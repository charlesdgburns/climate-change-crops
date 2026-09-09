"""
06_saturating_vpd.py
--------------------
Single location, saturating responses + atmospheric-demand (VPD) stress:
    yield = a
          + b * tanh(GDD/1000 - 1)      # growing season warmth, saturating
          + c * log1p(PREC)             # water, sublinear
          - d * HEAT30/8                 # heat stress, linear penalty
          - e * tanh(VPD/2.5)           # vapour pressure deficit stress
          + f * log(co2 / 380)          # co2 fertilisation
VPD proxy: daytime es(tasmax) - es(tasmin) with es(T)=0.6108*exp(17.27T/(T+237.3)).
Everything bounded (tanh/log1p) for extrapolation past the training window.
Derived features are computed inline from the raw (240, T) series.
"""

import numpy as np


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prec = np.sum(pr, axis=0)
    heat30 = np.sum(tasmax > 30.0, axis=0)
    es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
    vpd = np.mean(es(tasmax) - es(tasmin), axis=0)
    return gdd, prec, heat30, vpd


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d, e, f = params
    gdd, prec, heat30, vpd = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prec)
            - d * (heat30 / 8.0)
            - e * np.tanh(vpd / 2.5)
            + f * np.log(co2 / 380.0))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.1, 0.3, 2.0])