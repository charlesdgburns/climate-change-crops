"""
26_co2_heat_amelioration.py
---------------------------
Single location, CO2 x heat (stomatal-cooling) channel.
Elevated CO2 reduces stomatal conductance (FACE: g_s -22%; Ainsworth & Rogers
2007) which lowers canopy temperature and can soften heat-stress penalties;
heat-wave impact has been shown to shrink under eCO2 (Long et al. 2006).
The heat penalty itself is attenuated by the saturating factor as CO2 rises:
    yield = a
          + b * tanh(GDD/1000 - 1)
          + c * log1p(PREC)
          - d * (HEAT30/8) * (1 - beta * h(C))   # heat penalty shrinks
          - e * tanh(VPD/2.5)
h(C) = (C-400)/(C-400+350), saturating ~0.67 at 1108 ppm.
beta fixed from literature: wheat 0.30 (large stomatal-cooling effect;
amelioration documented for wheat heat waves), maize 0.15 (C4 g_s response
smaller). At val CO2 (~420-440 ppm) h~0.1 so the val effect is tiny (~3%):
the gain is concentrated in the 700-1108 ppm tail it is designed for.
"""

import numpy as np

CREF = 400.0
KHALF = 350.0
BETA = {"wheat": 0.30, "maize": 0.15}


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


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params, crop="wheat"):
    a, b, c, d, e = params
    gdd, prec, heat30, vpd = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    heat = (heat30 / 8.0) * (1.0 - BETA[crop] * _h(co2))
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prec)
            - d * heat
            - e * np.tanh(vpd / 2.5))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y, crop="wheat"):
    return np.array([np.mean(y), 1.0, 0.3, 0.1, 0.3])