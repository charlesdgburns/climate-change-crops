"""
25_co2_wue_drought_maize.py
---------------------------
Single location, CO2 x drought (WUE / stomatal-closure) channel only.
C4 crops get no direct photosynthetic benefit from elevated CO2, but reduced
stomatal conductance raises water-use efficiency and *attenuates drought
stress* (FACE: maize unchanged unless water-limited - Leakey 2006; Ainsworth
& Long 2021). The CO2 response is therefore forced to act ON the drought
penalty, never on the level:
    yield = a
          + b * tanh(GDD/1000 - 1)          # thermal, saturating
          + c * log1p(PREC)                 # water, sublinear
          - d * (HEAT30/8)                  # heat stress
          - e * tanh(VPD/2.5) * (1 - beta * h(C))   # drought penalty shrinks
h(C) = (C-400)/(C-400+350), saturating ~0.67 at 1108 ppm.
beta fixed from the WUE literature: maize 0.30 (strongest channel, only
visible in dry years), wheat 0.10 (C3 also has a small indirect component -
Kimball 2010 reports up to ~20% gain under water limitation). Well-watered
years (VPD penalty ~0) are unaffected at any CO2 level.
"""

import numpy as np

CREF = 400.0
KHALF = 350.0
BETA = {"wheat": 0.10, "maize": 0.30}


def _h(co2):
    d = co2 - CREF
    return d / (d + KHALF)


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prec = np.sum(pr, axis=0)
    heat30 = np.sum(tasmax > 30.0, axis=0)
    es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
    vpd = np.mean(es(tasmax) - es(tasmin), axis=0)
    return gdd, prec, heat30, vpd


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params, crop="maize"):
    a, b, c, d, e = params
    gdd, prec, heat30, vpd = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    wue = (1.0 - BETA[crop] * _h(co2))
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prec)
            - d * (heat30 / 8.0)
            - e * np.tanh(vpd / 2.5) * wue)


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y, crop="maize"):
    return np.array([np.mean(y), 1.0, 0.3, 0.1, 0.3])