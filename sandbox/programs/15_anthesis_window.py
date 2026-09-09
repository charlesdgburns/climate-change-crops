"""
15_anthesis_window.py
---------------------
Single location, phenology-scaled stress windows centered on anthesis (flowering).
Grain number is set in a short window around anthesis; irreversible heat/drought
damage in this window is the dominant yield loss mechanism per the literature.

Anthesis per year = first day cumulative GDD reaches 50% of the season total.
Stress window = [anthesis - 30, anthesis + 30] clipped to [0, 239].
    yield = a
          + b * tanh(GDD/1000 - 1)          # full-season thermal, saturating
          + c * log1p(PREC_MID)              # mid-season (calendar) water
          + d * log1p(PR_ANTE)               # anthesis-window precip, sublinear
          - e * tanh(HT_ANTE/50)             # anthesis-window heat exceedance (>30C)
          - f * tanh(VPD_ANTE/2.5)           # anthesis-window mean VPD
Derived features are computed inline from the raw (240, T) series.
"""

import numpy as np


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    D, T = tmean.shape
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prc_mid = np.sum(pr[80:160], axis=0)

    gdd_daily = np.maximum(tmean - 8.0, 0.0)
    cumgdd = np.cumsum(gdd_daily, axis=0)           # (D, T)
    total = cumgdd[-1, :]                            # (T,)
    ant = np.argmax(cumgdd >= 0.5 * total[None, :], axis=0)  # (T,)
    lo = np.maximum(ant - 30, 0)
    hi = np.minimum(ant + 30, D)

    es = lambda T_: 0.6108 * np.exp(17.27 * T_ / (T_ + 237.3))
    vpd = es(tasmax) - es(tasmin)
    ht = np.maximum(tmean - 30.0, 0.0)

    pr_ant = np.zeros(T)
    ht_ant = np.zeros(T)
    vpd_ant = np.zeros(T)
    for t in range(T):
        s, e = int(lo[t]), int(hi[t])
        pr_ant[t] = np.sum(pr[s:e, t])
        ht_ant[t] = np.sum(ht[s:e, t])
        vpd_ant[t] = np.mean(vpd[s:e, t]) if e > s else 0.0
    return gdd, prc_mid, pr_ant, ht_ant, vpd_ant


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params, crop="wheat"):
    a, b, c, d, e, f = params
    gdd, prc_mid, pr_ant, ht_ant, vpd_ant = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prc_mid)
            + d * np.log1p(pr_ant)
            - e * np.tanh(ht_ant / 50.0)
            - f * np.tanh(vpd_ant / 2.5))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y, crop="wheat"):
    return np.array([np.mean(y), 1.0, 0.3, 0.3, 0.3, 0.3])
