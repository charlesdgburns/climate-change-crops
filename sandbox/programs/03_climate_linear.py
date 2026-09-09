"""
03_climate_linear.py
---------------------
Single location, linear response to derived climate drivers:
    yield = a + b*GDD + c*PREC + d*HEAT + e*RSDS + f*log(co2/380)
GDD = growing degree days (base 8C), PREC = season total precipitation,
HEAT = count of days above 30C, RSDS = season mean radiation.
"""

import numpy as np


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d, e, f = params
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prec = np.sum(pr, axis=0)
    heat = np.sum(tasmax > 30.0, axis=0)
    rad = np.mean(rsds, axis=0)
    return a + b * gdd + c * prec + d * heat + e * rad + f * np.log(co2 / 380.0)


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 5e-4, 1e-3, -0.05, 5e-4, 1.0])