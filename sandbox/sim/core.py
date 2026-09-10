"""
sim/core.py
-----------
Shared daily crop-physiology engine for the B-track "literal simulator" test
(sandbox/sim_match.py). One location's record at a time:

    clim:  dict with tasmax, tasmin, pr, rsds each (240, T)  [day, year]
    co2:   (T,) ppm
    nitrogen: float                  (constant per location)
    fam:   dict from families.py     (all parameters fixed from literature)
    crop:  "wheat" | "maize"

Returns the simulated grain-yield series (T,) in arbitrary units (the
per-location affine calibration in sim_match.py absorbs level+scale).

Engine (reduced, but faithful to each family's published signature):
  - thermal-time phenology (GDD) -> anthesis, maturity
  - Priestley-Taylor PET (supply-type) or VPD/demand-driven PET (demand-type)
  - single soil-water bucket; residual SWC carried across seasons for memory
    families (sequential-year path), reset each season otherwise (vectorised)
  - RUE x intercepted light x water-stress x heat-damage x N x CO2 -> biomass
  - harvest index -> grain yield
"""

import numpy as np

PET_PT_K = 0.018   # approx Priestley-Taylor: PET mm/d ~ 0.018 * rsds[W/m2]


def _es(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def _vpd(tasmax, tasmin):
    return np.maximum(_es(tasmax) - _es(tasmin), 0.0)


def daily_pet(tasmax, tasmin, rsds, fam):
    """Reference daily PET (mm/d): 'pt' (Priestley-Taylor) or 'vpd' (demand)."""
    if fam["pet"] == "pt":
        return PET_PT_K * rsds
    return fam["pet_vpd_k"] * _vpd(tasmax, tasmin)


def _stress_et(w_avail, kc_full, fam):
    """(water-stress scalar, actual ET) given plant-available water (mm)."""
    whc = fam["whc"]
    if fam["water_channel"] == "supply":
        # plant-available water fraction, smooth exponential stress (CERES/STICS)
        fa = np.clip(w_avail / whc, 0.0, 1.0)
        ws = (1.0 - np.exp(-fam["stress_k"] * fa)).clip(0.0, 1.0)
        return ws, kc_full * ws
    # demand: full ET above a critical water level, linear collapse below
    # (APSIM/EPIC style; kc_full already carries the VPD demand signal)
    w_crit = fam.get("w_crit", 0.4) * whc
    ws = np.clip((w_avail - w_crit) / max(whc - w_crit, 1e-6), 0.0, 1.0)
    return ws, kc_full * ws


# ---------------------------------------------------------------------------

def simulate(clim, co2, nitrogen, fam, crop, sow_day=30):
    c = fam[crop]                          # crop-specific parameter block
    tasmax, tasmin = clim["tasmax"], clim["tasmin"]   # (240, T)
    pr, rsds = clim["pr"], clim["rsds"]
    T = tasmax.shape[1]

    tmean = 0.5 * (tasmax + tasmin)
    gdd_t = np.maximum(tmean - c["gdd_base"], 0.0)                    # (240, T)
    cum_gdd = np.cumsum(gdd_t, axis=0)                                # (240, T)
    cum_since_sow = np.maximum(cum_gdd - cum_gdd[sow_day - 1][None, :], 0.0)

    tt_ant, tt_mat = c["tt_ant"], c["tt_mat"]
    tt_lo, tt_hi = c["tt_heat_lo"], c["tt_heat_hi"]
    dev = np.clip(cum_since_sow / tt_mat, 0.0, 1.0)
    dev_ant = tt_ant / max(tt_mat, 1e-6)

    lai = np.minimum(dev / max(dev_ant, 1e-6),
                     np.clip((1.0 - dev) / max(1.0 - dev_ant, 1e-6), 0.0, 1.0))
    lai = np.minimum(lai, 1.0) * c["lai_max"]

    C0, K = fam["co2_c0"], fam["co2_k"]
    h_c = np.clip((co2 - C0) / (co2 - C0 + K), 0.0, None)
    gamma = c["co2_gamma"]
    co2_mult = (1.0 + gamma * h_c).clip(None, 1.0 + gamma * 0.9)  # (T,)

    n_mult = 1.0 - np.exp(-nitrogen / fam["n_half"])

    heat = fam["heat"][crop]
    if fam["soil_memory"] == "no":
        ass = _run_array(tasmax, tasmin, pr, rsds, lai, tmean, cum_since_sow,
                         tt_lo, tt_hi, co2_mult, n_mult, fam, heat, c, sow_day)
        return c["hi"] * ass
    grain = np.zeros(T)
    w_state = fam["whc"] * fam["w_init"]           # soil memory across seasons
    for y in range(T):
        ass = 0.0
        for d in range(sow_day, tasmax.shape[0]):
            in_win = tt_lo <= cum_since_sow[d, y] <= tt_hi
            w_state, tmp = _run_day(
                tasmax[d, y], tasmin[d, y], pr[d, y], rsds[d, y],
                lai[d, y], tmean[d, y], in_win,
                co2_mult[y], n_mult, w_state, fam, heat, c)
            ass += tmp
        grain[y] = c["hi"] * ass
    return grain


# ---------------------------------------------------------------------------

def _run_array(tasmax, tasmin, pr, rsds, lai, tmean, cum_since_sow,
               tt_lo, tt_hi, co2_mult, n_mult, fam, heat, c, sow_day):
    """Fast no-memory path: all years simulated in parallel ((T,) arrays)."""
    whc = fam["whc"]
    kext = fam["kext"]
    rue = c["rue"]
    T = tasmax.shape[1]
    w_state = np.full(T, whc * fam["w_init"])
    ass = np.zeros(T)
    for d in range(sow_day, tasmax.shape[0]):
        kc_full = daily_pet(tasmax[d], tasmin[d], rsds[d], fam) * fam["kc"] \
            * np.clip(lai[d] / fam["lai_ref"], 0.05, 1.0)
        wstress, et = _stress_et(w_state, kc_full, fam)
        in_win = (cum_since_sow[d] >= tt_lo) & (cum_since_sow[d] <= tt_hi)
        hstress = heat_scalar(tmean[d], in_win, heat)
        fpar = 1.0 - np.exp(-kext * lai[d])
        par = 0.5 * rsds[d] * (86400.0 / 4.18e6)
        ass += rue * fpar * par * wstress * hstress * co2_mult * n_mult
        w_state = np.clip(w_state + pr[d] - et, 0.0, whc)
    return ass


def _run_day(tmax_d, tmin_d, pr_d, rsd_d, lai_d, tmean_d, in_win,
             co2_mult_y, n_mult, w_state, fam, heat, c):
    """One (day, year) step for the memory path. Returns (w_state, d_assim)."""
    kc_full = daily_pet(tmax_d, tmin_d, rsd_d, fam) * fam["kc"] \
        * np.clip(lai_d / fam["lai_ref"], 0.05, 1.0)
    wstress, et = _stress_et(w_state, kc_full, fam)
    hstress = heat_scalar(tmean_d, in_win, heat)
    fpar = 1.0 - np.exp(-fam["kext"] * lai_d)
    par = 0.5 * rsd_d * (86400.0 / 4.18e6)
    w_state = np.clip(w_state + pr_d - et, 0.0, fam["whc"])
    return w_state, c["rue"] * fpar * par * wstress * hstress \
        * co2_mult_y * n_mult


def heat_scalar(tmean_d, in_win, heat):
    """Family heat-damage factor (1 = no stress, <1 = damage)."""
    mode, thr, k = heat["mode"], heat["thr"], heat["k"]
    if mode == "none":
        return np.ones_like(tmean_d) if hasattr(tmean_d, "shape") else 1.0
    dmg = np.maximum(tmean_d - thr, 0.0) if mode == "hdd" else \
        (tmean_d >= thr).astype(float)
    dmg = np.where(in_win, dmg, 0.0)
    return 1.0 / (1.0 + k * dmg)