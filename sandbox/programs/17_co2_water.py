"""
17_co2_water.py
---------------
Single location, CO2 x water (stomatal / WUE) interaction.
Elevated CO2 reduces stomatal conductance -> higher water-use efficiency ->
drought stress attenuated.  In-record CO2 spans 341-415 ppm.
    yield = a
          + b * tanh(GDD/1000 - 1)                        # thermal, saturating
          + c * log1p(PREC_MID)                           # mid-season water
          + d * tanh(log1p(PREC_MID) * log1p(GDD) / 40)  # water x warmth synergy
          - e * tanh(VD/50) * (1 - f * tanh((co2 - 340) / 50))
VD = sum of (VPD - 2 hPa)+ over the season; med wheat ~3, maize ~32.
At co2=341, atten~0.02 -> full penalty; at co2=415, atten~0.90 -> 10% penalty.
CO2 center at 340 ppm (minimum in record). Crop-aware via VD scale (fixed /50).
"""

import numpy as np


def _drivers(tasmax, tasmin, pr, rsds, cumrsds):
    tmean = 0.5 * (tasmax + tasmin)
    gdd = np.sum(np.maximum(tmean - 8.0, 0.0), axis=0)
    prc_mid = np.sum(pr[80:160], axis=0)
    es = lambda T_: 0.6108 * np.exp(17.27 * T_ / (T_ + 237.3))
    vpd = es(tasmax) - es(tasmin)
    vd = np.sum(np.maximum(vpd - 2.0, 0.0), axis=0)
    return gdd, prc_mid, vd


def model(tasmax, tasmin, pr, rsds, cumrsds, co2, params, crop="wheat"):
    a, b, c, d, e, f = params
    gdd, prc_mid, vd = _drivers(tasmax, tasmin, pr, rsds, cumrsds)
    co2_atten = np.tanh((co2 - 340.0) / 50.0)
    drought = e * np.tanh(vd / 50.0) * (1.0 - f * co2_atten)
    return (a
            + b * np.tanh(gdd / 1000.0 - 1.0)
            + c * np.log1p(prc_mid)
            + d * np.tanh(np.log1p(prc_mid) * np.log1p(gdd) / 40.0)
            - drought)


def estimate_params(tasmax, tasmin, pr, rsds, cumrsds, co2, y, crop="wheat"):
    return np.array([np.mean(y), 1.0, 0.3, 1.0, 0.3, 0.5])
