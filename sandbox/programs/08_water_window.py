"""
08_water_window.py
------------------
Single location, saturating responses + flowering-window drought stress:
    yield = a
          + b * tanh(GDD/1000 - 1)      # growing season warmth, saturating
          + c * log1p(PREC_MID)         # water in the mid-season window (days 80-160)
          - d * tanh(DSPELL5/6)         # >=5-day dry spells (pr < 1 mm)
          - e * tanh(VPD/2.5)           # vapour pressure deficit stress
          + f * log(co2 / 380)          # co2 fertilisation
PREC_MID targets the flowering / grain-fill period; dry spells count runs of
5+ days with < 1 mm rain. Bounded terms for 2051-2100 extrapolation. Derived
features are computed inline from the raw (240, T) series.
"""

import numpy as np


def _count_runs_geq(cond, k):
    """Number of >=k-day consecutive runs of True, along rows (time axis first)."""
    cond2d = cond.T  # (N, D)
    n = cond2d.shape[0]
    b = np.hstack([np.zeros((n, 1), bool), cond2d, np.zeros((n, 1), bool)])
    s = b[:, 1:] & ~b[:, :-1]
    runid = np.cumsum(s, axis=1)
    rlen = np.bincount(runid.ravel())[runid]
    return np.sum((rlen >= k) & s, axis=1).astype(float)


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prc_mid = np.sum(pr[80:160], axis=0)
    dspell5 = _count_runs_geq(pr < 1.0, 5)
    es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
    vpd = np.mean(es(tasmax) - es(tasmin), axis=0)
    return gdd, prc_mid, dspell5, vpd


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d, e, f = params
    gdd, prc_mid, dspell5, vpd = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prc_mid)
            - d * np.tanh(dspell5 / 6.0)
            - e * np.tanh(vpd / 2.5)
            + f * np.log(co2 / 380.0))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.3, 0.3, 2.0])