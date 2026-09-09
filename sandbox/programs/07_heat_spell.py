"""
07_heat_spell.py
----------------
Single location, saturating responses + heat-event / extreme-day stress:
    yield = a
          + b * tanh(GDD/1000 - 1)      # growing season warmth, saturating
          + c * log1p(PREC)             # water, sublinear
          - d * tanh(SPELL3/4)          # number of >=3-day heat events (tasmax>30)
          - e * tanh(HEAT34/8)          # days above 34C
          + f * log(co2 / 380)          # co2 fertilisation
SPELL3 counts consecutive-day runs of tasmax>30 lasting >= 3 days. Bounded
forms keep predictions sane for 2051-2100 climate. Derived features are
computed inline from the raw (240, T) series.
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
    spell3 = _count_runs_geq(tasmax > 30.0, 3)
    heat34 = np.sum(tasmax > 34.0, axis=0)
    return gdd, prec, spell3, heat34


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params):
    a, b, c, d, e, f = params
    gdd, prec, spell3, heat34 = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prec)
            - d * np.tanh(spell3 / 4.0)
            - e * np.tanh(heat34 / 8.0)
            + f * np.log(co2 / 380.0))


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y):
    return np.array([np.mean(y), 1.0, 0.3, 0.3, 0.3, 2.0])