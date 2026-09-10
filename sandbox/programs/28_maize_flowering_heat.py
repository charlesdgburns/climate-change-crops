"""
28_maize_flowering_heat.py
--------------------------
Single location, saturating responses + VPD stress, with the heat penalty
replaced by a FLOWERING-WINDOW low-threshold heat degree-day term (B-track
finding for maize: F1 peak heat sensitivity days 121-150; F2 hdd>=22/26 is the
best heat shape, win vs tmean-linear 0.60, p<1e-19). Model 06 core:
    yield = a
          + b * tanh(GDD/1000 - 1)      # growing season warmth, saturating
          + c * log1p(PREC)             # water, sublinear
          - d * tanh(HFL/150)           # flowering heat, hdd>=22 over days 121-150
          - e * tanh(VPD/2.5)           # vapour pressure deficit stress
          + f * log(co2 / 380)          # co2 fertilisation
VPD proxy: daytime es(tasmax) - es(tasmin) with es(T)=0.6108*exp(17.27T/(T+237.3)).
Everything bounded (tanh/log1p) for extrapolation past the training window.
"""

import numpy as np


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prec = np.sum(pr, axis=0)
    hfl = np.sum(np.maximum(tmean[120:150] - 22.0, 0.0), axis=0)
    es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
    vpd = np.mean(es(tasmax) - es(tasmin), axis=0)
    return gdd, prec, hfl, vpd


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d, e, f = params
    gdd, prec, hfl, vpd = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prec)
            - d * np.tanh(hfl / 150.0)
            - e * np.tanh(vpd / 2.5)
            + f * np.log(co2 / 380.0))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.1, 0.3, 2.0])