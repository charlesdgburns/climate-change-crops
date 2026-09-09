"""
24_co2_saturating_multiplier.py
-------------------------------
Single location, mechanical multiplicative CO2-fertilisation multiplier.
Crop models scale RUE / photosynthetic carbon gain by a saturating factor of
ambient CO2 (CERES uses an asymptotic lookup multiplier; EPIC+GEPIC a
hyperbolic B1/B2 form). Here the multiplier is FIXED from literature - no CO2
parameters are fitted - while the additive weather core is the per-crop best
(wheat: 13_water_heat_bilinear core; maize: 06_saturating_vpd core):
    yield = weather_base * (1 + gamma * h(C)),   h(C) = (C-400)/(C-400+350)
h saturates at ~0.67 by 1108 ppm. gamma fixed per crop: wheat 0.40 (C3 direct
photosynthetic response: ~+12% at 550 ppm, ~+27% at 1108 ppm - FACE/Ainsworth
& Long 2020 optimality target +11.7%); maize 0.03 (C4 photosynthesis is CO2
saturated - FACE Leakey 2006 shows no direct effect; kept as a near-control).
At val CO2 (~420-440 ppm) the multiplier moves only +/-3-4%, so this is a
real, small test against the held-out window.
"""

import numpy as np

CREF = 400.0
KHALF = 350.0
GAMMA = {"wheat": 0.40, "maize": 0.03}


def _h(co2):
    d = co2 - CREF
    return d / (d + KHALF)


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prec = np.sum(pr, axis=0)
    es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
    vpd = es(tasmax) - es(tasmin)
    return {"gdd": gdd, "prec": prec, "vpd": vpd}


def _wheat_core(tasmax, tasmin, pr, rsds, cumrsds, params):
    a, b, c, d, e = params
    drv = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    gdd, prec, vpd = drv["gdd"], drv["prec"], drv["vpd"]
    prc_mid = np.sum(pr[80:160], axis=0)
    vd = np.sum(np.maximum(vpd - 2.0, 0.0), axis=0)
    th = np.tanh(gdd / 1000.0 - 1.0)
    return (a
            + b * th
            + c * th * np.tanh(prec / 10.0)
            + d * np.log1p(prc_mid)
            - e * np.tanh(vd / 50.0))


def _maize_core(tasmax, tasmin, pr, rsds, cumrsds, params):
    a, b, c, d, e = params
    drv = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    gdd, prec, vpd = drv["gdd"], drv["prec"], drv["vpd"]
    heat30 = np.sum(tasmax > 30.0, axis=0)
    vpd_m = np.mean(vpd, axis=0)
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prec)
            - d * (heat30 / 8.0)
            - e * np.tanh(vpd_m / 2.5))


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params, crop="wheat"):
    base = _wheat_core(tasmax, tasmin, pr, rsds, cumrsds, params) if crop == "wheat" \
        else _maize_core(tasmax, tasmin, pr, rsds, cumrsds, params)
    return base * (1.0 + GAMMA[crop] * _h(co2))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y, crop="wheat"):
    return np.array([np.mean(y), 1.0, 0.3, 0.3, 0.3])