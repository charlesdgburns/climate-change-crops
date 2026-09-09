"""
19_water_balance.py
-------------------
Single location, water balance (prec - PET) as the primary water metric.
Raw precipitation ignores evaporative demand: a wet year with high PET
(hot, sunny) may have less available water than a dry year with low PET
(cool, cloudy). Water balance captures the net.
    yield = a
          + b * tanh(GDD/1000 - 1)          # thermal growth, saturating
          + c * tanh(WATER_BAL / 1000)       # prec - PET (Hargreaves proxy)
          + d * log1p(PREC_MID)             # flowering-window water (calendar)
          - e * tanh(VD / 50)               # VPD exceedance stress
          + f * log(co2 / 380)              # co2 fertilisation
PET = sum(0.0023*(Tmean+17.8)*Rsds*0.408) per year (Hargreaves, rough).
WATER_BAL median: maize ~-1752, wheat ~-699; /1000 centers tanh at -0.7 to -1.8.
Derived features are computed inline from the raw (240, T) series.
"""

import numpy as np


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prc_mid = np.sum(pr[80:160], axis=0)
    pet = np.sum(0.0023 * (tmean + 17.8) * rsds * 0.408, axis=0)
    prec = np.sum(pr, axis=0)
    water_bal = prec - pet
    es = lambda T_: 0.6108 * np.exp(17.27 * T_ / (T_ + 237.3))
    vpd = es(tasmax) - es(tasmin)
    vd = np.sum(np.maximum(vpd - 2.0, 0.0), axis=0)
    return gdd, prc_mid, water_bal, vd


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d, e, f = params
    gdd, prc_mid, water_bal, vd = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.tanh(water_bal / 1000.0)
            + d * np.log1p(prc_mid)
            - e * np.tanh(vd / 50.0)
            + f * np.log(co2 / 380.0))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.3, 0.3, 2.0])
