"""
10_light_water.py
-----------------
Single location, saturating responses + solar-radiation (light) and
light x water co-limitation (uses rsds for the first time):
    yield = a
          + b * tanh(GDD/1000 - 1)                  # growing season warmth
          + c * tanh((L - 120)/60)                  # mean rsds on gdd-active days
          + d * tanh(PREC/10)                        # water, saturating
          + e * tanh((L - 120)/60) * tanh(PREC/10)   # light x water co-limitation
L = seasonal-mean solar radiation (W/m2) over days with GDD>0; the (L-120)/60
form spreads the 125-250 W/m2 range. Everything bounded for 2051-2100
extrapolation. Derived features are computed inline from the raw (240, T) series.
"""

import numpy as np


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.maximum(tmean - 8.0, 0.0)
    grow = gdd > 0.0
    lgt = np.sum(np.where(grow, rsds, 0.0), axis=0) / np.maximum(np.sum(grow, axis=0), 1)
    grd = np.sum(gdd, axis=0)
    prec = np.sum(pr, axis=0)
    return grd, lgt, prec


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d, e = params
    grd, lgt, prec = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    li = np.tanh((lgt - 120.0) / 60.0)
    wt = np.tanh(prec / 10.0)
    return (a
            + b * np.tanh(grd / 1000.0 - 1.0)
            + c * li
            + d * wt
            + e * li * wt)


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.3, 0.3])