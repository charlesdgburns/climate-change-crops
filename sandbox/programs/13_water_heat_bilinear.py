"""
13_water_heat_bilinear.py
-------------------------
Single location, saturating responses + an explicit thermal x water
co-limitation term on the GDD anchor (growing-season light-and-moisture):
    yield = a
          + b * tanh(GDD/1000 - 1)                        # thermal response
          + c * tanh(GDD/1000 - 1) * tanh(PREC/10)        # GDD x water interaction
          + d * log1p(PREC_MID)                           # flowering-window water
          - e * tanh(VD/50)                               # VPD-exceedance stress
The interaction lets warmth only help when water is available (drought-limited
warm years do not translate into yield). VD as in 12: sum of (vpd-2 hPa)+,
med ~32 / year. Derived features computed inline from the raw (240, T) series.
"""

import numpy as np


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prec = np.sum(pr, axis=0)
    prc_mid = np.sum(pr[80:160], axis=0)
    es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
    vpd = es(tasmax) - es(tasmin)
    vd = np.sum(np.maximum(vpd - 2.0, 0.0), axis=0)
    return gdd, prec, prc_mid, vd


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d, e = params
    gdd, prec, prc_mid, vd = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    th = np.tanh(gdd / 1000.0 - 1.0)
    wt = np.tanh(prec / 10.0)
    return (a
            + b * th
            + c * th * wt
            + d * np.log1p(prc_mid)
            - e * np.tanh(vd / 50.0))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.3, 0.3])