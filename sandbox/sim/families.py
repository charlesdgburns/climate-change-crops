"""
sim/families.py
---------------
Family configs for the B-track literal-simulator comparison (sim_match.py).
All values are FIXED from the published model descriptions (see models_library.py
citations); nothing here is fitted to the competition data. Smaller values
reflect real physiological spread; the comparison's power comes from the
structural switches (water channel, soil memory, heat window/threshold, PET
physics), not from fine numeric tuning.

Key literature anchors (from models_library.py / CO2_LIT.md citations):
  CERES (Jones 2003; Ritchie & Otter 1985): supply-type water, weak memory,
      wheat grain-fill heat at high threshold, maize flowering heat at low
      threshold, asymptotic RUE CO2 multiplier.
  STICS (Brisson 2003): same water/heating signature as CERES in these dims.
  APSIM (Keating 2003; Asseng 2015): demand-type (VPD-driven), strong
      soil memory, maize flowering heat low threshold, wheat flowering heat
      high threshold.
  EPIC/GEPIC (Williams 1984/1989): demand-type, strong memory, wheat grain-fill
      high threshold, maize flowering HIGH threshold (flo_high), hyperbolic
      CO2 (B1/B2).
  LPJmL (Bondeau 2007): demand-type, strong memory, SMOOTH thermal response
      (no sharp threshold), saturating CO2 in RUE.

Units: tt_* are GDD (degC-day) from sowing; whc mm plant-available water;
rue g biomass / MJ PAR; hi fraction; co2_gamma fractional RUE gain at CO2->inf
(C3 ~0.4, C4 ~0.03); n_half kg N/ha for Mitscherlich.
"""

W = {  # wheat
    "gdd_base": 0.0,
    "tt_ant": 2100.0,
    "tt_mat": 2800.0,
    "tt_heat_lo": 2100.0,   # grain fill onward
    "tt_heat_hi": 9999.0,
    "lai_max": 5.5,
    "rue": 2.9,             # C3
    "hi": 0.45,
    "co2_gamma": 0.40,      # C3 saturating gain (CO2_LIT.md)
}
M = {  # maize
    "gdd_base": 8.0,
    "tt_ant": 1050.0,
    "tt_mat": 1550.0,
    "tt_heat_lo": 820.0,    # flowering window
    "tt_heat_hi": 1300.0,
    "lai_max": 5.5,
    "rue": 3.6,             # C4
    "hi": 0.50,
    "co2_gamma": 0.03,      # C4 near-null direct response (FACE Leakey 2006)
}


# --- families --------------------------------------------------------------

CERES = dict(
    id="ceres", name="CERES-DSSAT (CERES-Wheat / CERES-Maize)",
    cite="Jones et al. 2003; Ritchie & Otter 1985; Jones & Kiniry 1986",
    water_channel="supply",
    pet="pt",
    soil_memory="no",            # bucket is effectively empty between seasons
    w_init=0.9,
    stress_k=4.0,
    kc=1.0,
    lai_ref=4.0,
    whc=150.0,
    kext=0.55,
    n_half=70.0,
    co2_c0=350.0,
    co2_k=300.0,
    vpd_ref=2.0,
    pet_vpd_k=2.5,
    heat={"wheat": dict(mode="hdd", thr=31.0, k=0.08),
          "maize": dict(mode="hdd", thr=24.0, k=0.10)},
    wheat=dict(W), maize=dict(M),
)

STICS = dict(
    id="stics", name="STICS",
    cite="Brisson et al. 2003; Casadebaig et al. 2016",
    water_channel="supply",
    pet="pt",
    soil_memory="no",
    w_init=0.9,
    stress_k=4.5,
    kc=1.0,
    lai_ref=4.0,
    whc=140.0,
    kext=0.55,
    n_half=65.0,
    co2_c0=340.0,
    co2_k=320.0,
    vpd_ref=2.0,
    pet_vpd_k=2.5,
    heat={"wheat": dict(mode="hdd", thr=32.0, k=0.08),
          "maize": dict(mode="hdd", thr=23.0, k=0.11)},
    wheat=dict(W), maize=dict(M),
)

APSIM = dict(
    id="apsim", name="APSIM (APSIM-Wheat / APSIM-Maize)",
    cite="Keating et al. 2003; Carberry et al. 1999; Asseng et al. 2015",
    water_channel="demand",
    pet="vpd",
    soil_memory="yes",           # soil-water pool persists across seasons
    w_init=0.9,
    stress_k=4.0,
    kc=1.0,
    lai_ref=4.0,
    whc=160.0,
    kext=0.55,
    n_half=70.0,
    co2_c0=350.0,
    co2_k=280.0,
    vpd_ref=2.0,
    pet_vpd_k=2.5,
    heat={"wheat": dict(mode="hdd", thr=30.0, k=0.07),
          "maize": dict(mode="hdd", thr=24.0, k=0.10)},
    wheat=dict(W), maize=dict(M),
)

EPIC = dict(
    id="epic", name="EPIC / GEPIC",
    cite="Williams et al. 1984/1989; Liu et al. 2007",
    water_channel="demand",
    pet="vpd",
    soil_memory="yes",
    w_init=0.9,
    stress_k=4.0,
    kc=1.0,
    lai_ref=4.0,
    whc=170.0,
    kext=0.55,
    n_half=75.0,
    co2_c0=350.0,
    co2_k=300.0,
    vpd_ref=2.0,
    pet_vpd_k=2.5,
    heat={"wheat": dict(mode="hdd", thr=31.0, k=0.08),
          "maize": dict(mode="hdd", thr=28.0, k=0.09)},  # flo_high
    wheat=dict(W), maize=dict(M),
)

LPJML = dict(
    id="lpjml", name="LPJmL",
    cite="Bondeau et al. 2007; Schaphoff et al. 2018",
    water_channel="demand",
    pet="vpd",
    soil_memory="yes",
    w_init=0.9,
    stress_k=4.0,
    kc=1.0,
    lai_ref=4.0,
    whc=180.0,
    kext=0.55,
    n_half=60.0,
    co2_c0=300.0,
    co2_k=350.0,
    vpd_ref=2.0,
    pet_vpd_k=2.4,
    heat={"wheat": dict(mode="none", thr=0.0, k=0.0),    # smooth thermal opt,
          "maize": dict(mode="none", thr=0.0, k=0.0)},   # handled by LAI/phot
    wheat=dict(W), maize=dict(M),
)

FAMILIES = [CERES, STICS, APSIM, EPIC, LPJML]
FAMILY_BY_ID = {f["id"]: f for f in FAMILIES}


def family(id_):
    return FAMILY_BY_ID[id_]