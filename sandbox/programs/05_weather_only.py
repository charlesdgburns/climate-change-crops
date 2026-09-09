"""
05_weather_only.py
------------------
Single location, saturating weather responses without CO2:
    yield = a + b*tanh(GDD/1000 - 1) + c*log1p(PREC) - d*HEAT
`a` is the per-location level. Tests how much of within-location variability
is explained by weather alone (no secular CO2 trend).
"""

import numpy as np


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d = params
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prec = np.sum(pr, axis=0)
    heat = np.sum(tasmax > 30.0, axis=0)
    return a + b * np.tanh(gdd / 1000.0 - 1.0) + c * np.log1p(prec) - d * heat


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.05])