"""
11_heat_intensity.py
--------------------
Single location, saturating responses + heat stress as an *intensity* score
(degree-heat days above 30C) alongside >=3-day heat-event counting:
    yield = a
          + b * tanh(GDD/1000 - 1)      # growing season warmth, saturating
          + c * log1p(PREC)             # water, sublinear
          - d * tanh(HT/200)            # sum of (tasmax-30)+ over the season
          - e * tanh(SPELL3/4)          # >=3-day heat events (tasmax>30)
          + f * log(co2 / 380)          # co2 fertilisation
HT accumulates exceedance above 30C (median ~92, p75 ~331 per year, so /200).
That rewards distribution: many modestly-hot days vs a few extremely hot ones.
Derived features are computed inline from the raw (240, T) series.
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
    prec = np.sum(pr, axis=0)
    ht = np.sum(np.maximum(tasmax - 30.0, 0.0), axis=0)
    spell3 = _count_runs_geq(tasmax > 30.0, 3)
    return gdd, prec, ht, spell3


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d, e, f = params
    gdd, prec, ht, spell3 = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prec)
            - d * np.tanh(ht / 200.0)
            - e * np.tanh(spell3 / 4.0)
            + f * np.log(co2 / 380.0))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.3, 0.3, 2.0])