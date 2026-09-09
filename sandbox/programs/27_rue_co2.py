"""
27_rue_co2.py
-------------
Single location, radiation-use-efficiency x CO2 channel.
Process models couple the CO2-fertilisation factor to carbon assimilation by
scaling radiation use efficiency (EPIC/STICS/CERES: RUE increases ~21% from
350 to 700 ppm at 20 degC for C3 crops; C4 RUE barely affected). Here the
solar-radiation term carries a fixed saturating CO2 multiplier:
    yield = a
          + b * tanh(GDD/1000 - 1)
          + c * log1p(PREC)
          - d * (HEAT30/8)
          - e * tanh(VPD/2.5)
          + f * light * (1 + delta * h(C))
light = tanh((L-120)/60), L = mean rsds (W/m2) on GDD-active days (as 10).
h(C) = (C-400)/(C-400+350). delta fixed: wheat 0.50 (~+19% RUE at 700 ppm,
matching the ~21% literature), maize 0.10 (C4 RUE nearly CO2-insensitive).
Only the radiation term is boosted - the heat/water terms keep their own
saturating weather responses, so CO2 acts where models say it acts (on the
light-limited photosynthesis, RUE).
"""

import numpy as np

CREF = 400.0
KHALF = 350.0
DELTA = {"wheat": 0.50, "maize": 0.10}


def _h(co2):
    d = co2 - CREF
    return d / (d + KHALF)


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    gdd_d = np.maximum(tmean - 8.0, 0.0)
    grow = gdd_d > 0.0
    lgt = np.sum(np.where(grow, rsds, 0.0), axis=0) / np.maximum(np.sum(grow, axis=0), 1)
    gdd = np.sum(gdd_d, axis=0)
    prec = np.sum(pr, axis=0)
    heat30 = np.sum(tasmax > 30.0, axis=0)
    es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
    vpd = np.mean(es(tasmax) - es(tasmin), axis=0)
    return gdd, prec, heat30, vpd, lgt


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params, crop="wheat"):
    a, b, c, d, e, f = params
    gdd, prec, heat30, vpd, lgt = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    light = np.tanh((lgt - 120.0) / 60.0)
    rue = light * (1.0 + DELTA[crop] * _h(co2))
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prec)
            - d * (heat30 / 8.0)
            - e * np.tanh(vpd / 2.5)
            + f * rue)


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y, crop="wheat"):
    return np.array([np.mean(y), 1.0, 0.3, 0.1, 0.3, 0.3])