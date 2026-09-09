"""
14_heat_threshold.py
--------------------
Single location, crop-specific heat thresholds per Schlenker-Roberts / PMC10721465.
Wheat critical T = 27C (flowering/seed-set); maize critical T = 32C.
    yield = a
          + b * tanh(GDD_CAPPED/1000 - 1)   # capped GDD (above T_heat excluded)
          + c * log1p(PREC_MID)              # flowering-window water (calendar)
          - d * tanh(HT/DEN)                 # sum of max(tmean - T_heat, 0)
          - e * tanh(SPELL3/DEN_SPELL)       # >=3-day runs of tmean > T_heat
          + f * log(co2 / 380)               # co2 fertilisation
GDD_CAPPED = sum max(min(tmean-8, T_heat-8), 0) -- degree days stop above threshold.
DEN: wheat ht27 median ~18.6 -> /100; maize ht32 median ~30.2 -> /100.
Derived features are computed inline from the raw (240, T) series.
"""

import numpy as np

THRESHOLDS = {"wheat": 27.0, "maize": 32.0}


def _count_runs_geq(cond, k):
    cond2d = cond.T
    n = cond2d.shape[0]
    b = np.hstack([np.zeros((n, 1), bool), cond2d, np.zeros((n, 1), bool)])
    s = b[:, 1:] & ~b[:, :-1]
    runid = np.cumsum(s, axis=1)
    rlen = np.bincount(runid.ravel())[runid]
    return np.sum((rlen >= k) & s, axis=1).astype(float)


def _drivers(tasmax, tasmin, pr, rsds, cumrsds, crop):
    tmean = 0.5 * (tasmax + tasmin)
    T_heat = THRESHOLDS.get(crop, 30.0)
    gdd_capped = np.sum(np.maximum(np.minimum(tmean - 8.0, T_heat - 8.0), 0.0), axis=0)
    prc_mid = np.sum(pr[80:160], axis=0)
    ht = np.sum(np.maximum(tmean - T_heat, 0.0), axis=0)
    spell3 = _count_runs_geq(tmean > T_heat, 3)
    return gdd_capped, prc_mid, ht, spell3


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params, crop="wheat"):
    a, b, c, d, e, f = params
    gdd_capped, prc_mid, ht, spell3 = _drivers(tasmax, tasmin, pr, rsds, cumrsds, crop)
    return (a
            + b * np.tanh(gdd_capped / 1000.0 - 1.0)
            + c * np.log1p(prc_mid)
            - d * np.tanh(ht / 100.0)
            - e * np.tanh(spell3 / 3.0)
            + f * np.log(co2 / 380.0))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y, crop="wheat"):
    return np.array([np.mean(y), 1.0, 0.3, 0.3, 0.3, 2.0])
