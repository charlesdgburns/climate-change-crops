"""
01_constant.py
--------------
Single-location intercept only:
    yield = a
Fitted per location, `a` converges to that location's train mean, so this is
the per-location-mean baseline - a harness sanity check.
"""

import numpy as np


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    return np.full(co2.shape, params[0])


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y)])