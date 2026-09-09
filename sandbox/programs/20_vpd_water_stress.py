"""
20_vpd_water_stress.py
----------------------
Single location, VPD x water interaction: atmospheric demand x supply.
High VPD (dry, hot air) drives transpiration; when soil moisture is low,
the plant cannot replace lost water -> multiplicative stress.
More physically grounded than GDD x water (13) because VPD is the direct
driver of transpiration, not GDD (which is a phenological proxy).
    yield = a
          + b * tanh(GDD/1000 - 1)                          # thermal growth
          + c * log1p(PREC_MID)                              # water supply
          - d * tanh(SUM_VPD_TANH / 120)                    # atmospheric demand
          + e * tanh(PREC_MID / 3) * tanh(SUM_VPD_TANH / 150) # supply x demand
          + f * log(co2 / 380)                               # co2 fertilisation
SUM_VPD_TANH = sum(tanh(vpd_daily / 2.0)); med maize ~159, wheat ~76.
The interaction is positive when supply AND demand are both ample
(hot-wet years -> growth), and small when either is limiting.
"""

import numpy as np


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prc_mid = np.sum(pr[80:160], axis=0)
    es = lambda T_: 0.6108 * np.exp(17.27 * T_ / (T_ + 237.3))
    vpd = es(tasmax) - es(tasmin)
    sum_vpd_tanh = np.sum(np.tanh(vpd / 2.0), axis=0)
    return gdd, prc_mid, sum_vpd_tanh


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d, e, f = params
    gdd, prc_mid, sum_vpd_tanh = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    supply = np.tanh(prc_mid / 3.0)
    demand = np.tanh(sum_vpd_tanh / 150.0)
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prc_mid)
            - d * np.tanh(sum_vpd_tanh / 120.0)
            + e * supply * demand
            + f * np.log(co2 / 380.0))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.3, 0.3, 2.0])
