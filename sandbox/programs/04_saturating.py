"""
04_saturating.py
----------------
Single location, saturating mechanistic responses:
    yield = a
          + b * tanh(GDD/1000 - 1)      # growing season warmth, saturating
          + c * log1p(PREC)             # water, sublinear
          - d * HEAT                    # heat stress, linear penalty
          + e * log(co2 / 380)          # co2 fertilisation
`a` is the per-location level. Saturating forms keep predictions bounded as
climate runs beyond the training window (projections are warmer / higher-CO2).
"""

import numpy as np


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d, e = params
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prec = np.sum(pr, axis=0)
    heat = np.sum(tasmax > 30.0, axis=0)
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prec)
            - d * heat
            + e * np.log(co2 / 380.0))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.05, 2.0])