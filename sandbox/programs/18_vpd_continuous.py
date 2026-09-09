"""
18_vpd_continuous.py
--------------------
Single location, continuous daily VPD penalty (replaces VD/50 exceedance).
The literature (PMC8192978) shows VPD has a monotonic negative effect on
corn yield across the full range -- not just above a threshold. Our VD/50
exceedance index only captures VPD > 2 hPa; this captures every day.
    yield = a
          + b * tanh(GDD/1000 - 1)          # thermal growth, saturating
          + c * log1p(PREC_MID)             # mid-season water
          - d * tanh(SUM_VPD_TANH / 120)    # continuous VPD: sum(tanh(vpd/2))
          + e * log(co2 / 380)              # co2 fertilisation
SUM_VPD_TANH = sum over season of tanh(vpd_daily / 2.0).
Maize median ~159, wheat ~76; /120 centers tanh at ~0.5-0.8 for both crops.
Derived features are computed inline from the raw (240, T) series.
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
    a, b, c, d, e = params
    gdd, prc_mid, sum_vpd_tanh = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prc_mid)
            - d * np.tanh(sum_vpd_tanh / 120.0)
            + e * np.log(co2 / 380.0))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.3, 2.0])
