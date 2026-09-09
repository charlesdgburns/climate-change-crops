"""
12_vpd_exceedance.py
--------------------
Single location, saturating responses + vapour-pressure-deficit stress as a
seasonal *exceedance* score (capped sum of (VPD - 2 hPa)+ over the season):
    yield = a
          + b * tanh(GDD/1000 - 1)      # growing season warmth, saturating
          + c * log1p(PREC_MID)         # water in the mid-season window (days 80-160)
          - d * tanh(VD/50)             # sum of (vpd - 2 hPa)+ over the season
          + e * log(co2 / 380)          # co2 fertilisation
VD uses daytime es(tasmax) - es(tasmin) with es(T)=0.6108*exp(17.27T/(T+237.3));
med ~32, p75 ~80 per year, so /50 -- it grows with both hotter days and drier air.
Derived features are computed inline from the raw (240, T) series.
"""

import numpy as np


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prc_mid = np.sum(pr[80:160], axis=0)
    es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
    vpd = es(tasmax) - es(tasmin)
    vd = np.sum(np.maximum(vpd - 2.0, 0.0), axis=0)
    return gdd, prc_mid, vd


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d, e = params
    gdd, prc_mid, vd = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prc_mid)
            - d * np.tanh(vd / 50.0)
            + e * np.log(co2 / 380.0))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.3, 2.0])