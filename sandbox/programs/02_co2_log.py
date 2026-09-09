"""
02_co2_log.py
-------------
Single location, CO2 fertilisation only:
    yield = a + b * log(co2)
`a` is the per-location level (a reference concentration is folded into `a`,
so no separate c0 param). Within one location co2 is the only non-climate
input that drifts year to year.
"""

import numpy as np


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b = params
    return a + b * np.log(np.maximum(co2, 1e-3))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 3.0])