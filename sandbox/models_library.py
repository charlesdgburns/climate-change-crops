"""
models_library.py
-----------------
Reference library of biophysical crop-model "fingerprint priors" used by
fingerprint_match.py (B3) to rank which generating-model family best explains
the OBSERVED fingerprints computed by fingerprint.py (B1).

A family's *prior profile* is a set of categorical expectations, one per
observable the fingerprint harness can resolve:

  water_wheat / water_maize : which weather variable drives water-limited
    yield in one season.
        "supply"  -> precipitation-driven soil-water deficit (stress =
                      f(available water)); VPD channel is inert.
        "demand"  -> evaporative demand (VPD / potential ET) drives stress;
                      precipitation secondary (through supply on top).
  heat_wheat  / heat_maize   : timing + sharpness of the heat-damage term.
        "late_high"  -> damage concentrated late season (grain fill) at high
                        thresholds (~30-34 C mean).
        "flo_low"    -> damage concentrated at flowering with a low threshold
                        (~22-26 C mean), i.e. moderately hot days already hurt.
        "flo_high"   -> flowering heat but only at very hot extremes.
        "smooth"     -> smooth thermal response (optimum curve), no sharp
                        threshold-window.
        "none"       -> no temperature-damage mechanism.
  n_response           : nitrogen response in the yield surface.
        "saturating" -> Mitscherlich-type saturating (log/sqrt best in pooled
                        regressions).
        "constant"  -> fertility is an input constant (no N dynamics), so N
                       shows at most a constant level offset.
        "none"      -> no nitrogen term at all.
  soil_memory         : inter-annual soil-water carryover.
        "yes" / "weak" / "no"
  buffer              : whether the 30-day pre-sowing "buffer" carries signal.
        "inert" (buffer adds nothing) vs "responsive".

The numeric anchors (bases/thresholds) are formatted ranges from the standard
model documentation cited below; they are priors, NOT calibrated parameters.

Families (7), with canonical references:
  CERES : Jones et al. 2003 (DSSAT v4); Ritchie & Otter 1985; Jones & Kiniry 1986
  EPIC  : Williams et al. 1984/1989; GEPIC: Liu et al. 2007
  APSIM : Keating et al. 2003; Carberry et al. 1999 (maize); Asseng et al. 2015
  STICS : Brisson et al. 2003; Casadebaig et al. 2016 (heat)
  LPJmL : Bondeau et al. 2007; Schaphoff et al. 2018
  AQUACROP : Steduto et al. 2009; Raes et al. 2009
  WOFOST : de Wit et al. 2019 (WOFOST-GT); van Diepen et al. 1989

Attribution note for the CO2 axis (see CO2_LIT.md): C3 (wheat) shows strong
saturating CO2 response in essentially all of these families (FACE-based
+10-19% at 550 ppm); C4 (maize) shows a small response only through water-use
efficiency under drought. Family-specific CO2 shapes live in CO2_LIT.md and
the candidates 24-27; this library skips CO2 because the in-sample fingerprint
cannot resolve it (level-drift identity).
"""

from dataclasses import dataclass, field, asdict


@dataclass
class Family:
    id: str
    name: str
    cite: str
    profile: dict = field(default_factory=dict)

    def as_dict(self):
        return {"id": self.id, "name": self.name, "cite": self.cite,
                **self.profile}


FAMILIES = [
    Family(
        id="ceres", name="CERES-DSSAT (CERES-Wheat / CERES-Maize)",
        cite="Jones et al. 2003; Ritchie & Otter 1985; Jones & Kiniry 1986",
        profile=dict(
            water_wheat="supply", water_maize="supply",
            heat_wheat="late_high", heat_maize="flo_low",
            n_response="saturating", soil_memory="weak", buffer="inert")),
    Family(
        id="epic", name="EPIC / GEPIC",
        cite="Williams et al. 1984/1989; Liu et al. 2007",
        profile=dict(
            water_wheat="demand", water_maize="demand",
            heat_wheat="late_high", heat_maize="flo_high",
            n_response="saturating", soil_memory="yes", buffer="inert")),
    Family(
        id="apsim", name="APSIM (APSIM-Wheat / APSIM-Maize)",
        cite="Keating et al. 2003; Asseng et al. 2015; Carberry et al. 1999",
        profile=dict(
            water_wheat="demand", water_maize="demand",
            heat_wheat="flo_high", heat_maize="flo_low",
            n_response="saturating", soil_memory="yes", buffer="inert")),
    Family(
        id="stics", name="STICS",
        cite="Brisson et al. 2003; Casadebaig et al. 2016",
        profile=dict(
            water_wheat="supply", water_maize="supply",
            heat_wheat="late_high", heat_maize="flo_low",
            n_response="saturating", soil_memory="weak", buffer="inert")),
    Family(
        id="lpjml", name="LPJmL",
        cite="Bondeau et al. 2007; Schaphoff et al. 2018",
        profile=dict(
            water_wheat="demand", water_maize="demand",
            heat_wheat="smooth", heat_maize="smooth",
            n_response="saturating", soil_memory="yes", buffer="inert")),
    Family(
        id="aquacrop", name="AquaCrop",
        cite="Steduto et al. 2009; Raes et al. 2009",
        profile=dict(
            water_wheat="demand", water_maize="demand",
            heat_wheat="none", heat_maize="none",
            n_response="constant", soil_memory="no", buffer="inert")),
    Family(
        id="wofost", name="WOFOST",
        cite="de Wit et al. 2019; van Diepen et al. 1989",
        profile=dict(
            water_wheat="supply", water_maize="supply",
            heat_wheat="late_high", heat_maize="flo_high",
            n_response="none", soil_memory="weak", buffer="inert")),
]

# -- ground truth obtained from fingerprint.py (B1), with confidence weights ---

# Weights 0..1 reflect how reliably each observable is pinned by the held-out
# statistics (win-rate significance in fingerprint_{crop}.md).
OBSERVED = {
    "water_wheat": {"obs": "supply", "w": 1.0},    # F4_pr win 0.601 p~1e-9, vpd inert
    "water_maize": {"obs": "demand", "w": 1.0},    # F4_vpd/wb win ~0.62 p~1e-13
    "heat_wheat": {"obs": "late_high", "w": 0.5},  # hdd30/34 best but weak abs. R2
    "heat_maize": {"obs": "flo_low", "w": 0.9},    # F1 peak 121-150; hdd22/26 win big
    "n_response": {"obs": "saturating", "w": 1.0}, # log/sqrt best both crops, pooled
    "soil_memory": {"obs": "no", "w": 0.7},         # lag-1 terms uniformly hurt
    "buffer": {"obs": "inert", "w": 0.4},           # buffer adds ~nothing (weak)
}

# categorical scoring: how well family value F explains observed value O
AGREE_MATRIX = {
    # (family_value, observed_value) -> score 0..1
    ("supply", "supply"): 1.0, ("demand", "demand"): 1.0,
    ("supply", "demand"): 0.0, ("demand", "supply"): 0.0,
    ("late_high", "late_high"): 1.0, ("late_high", "flo_high"): 0.5,
    ("flo_low", "flo_low"): 1.0, ("flo_low", "flo_high"): 0.5,
    ("flo_high", "flo_high"): 1.0, ("flo_high", "flo_low"): 0.4,
    ("flo_high", "late_high"): 0.5, ("late_high", "flo_low"): 0.2,
    ("flo_low", "late_high"): 0.2,
    ("smooth", "smooth"): 1.0, ("smooth", "flo_low"): 0.3,
    ("smooth", "flo_high"): 0.4, ("smooth", "late_high"): 0.4,
    ("none", "none"): 1.0, ("none", "flo_low"): 0.0, ("none", "flo_high"): 0.0,
    ("none", "late_high"): 0.0, ("none", "smooth"): 0.0,
    ("saturating", "saturating"): 1.0, ("constant", "saturating"): 0.3,
    ("none", "saturating"): 0.0,
    ("yes", "yes"): 1.0, ("weak", "yes"): 0.5, ("no", "yes"): 0.0,
    ("yes", "no"): 0.0, ("weak", "no"): 0.5, ("no", "no"): 1.0,
    ("yes", "weak"): 0.5, ("weak", "weak"): 1.0, ("no", "weak"): 0.0,
    ("inert", "inert"): 1.0, ("responsive", "inert"): 0.0,
    ("responsive", "responsive"): 1.0, ("inert", "responsive"): 0.0,
}


def agree(fvalue: str, ovalue: str) -> float:
    return AGREE_MATRIX.get((fvalue, ovalue), AGREE_MATRIX.get((ovalue, fvalue), 0.5))


def match_score(family: Family):
    """Weighted average agreement between family priors and observed fingerprints."""
    fam = family.profile
    num = den = 0.0
    breakdown = {}
    for key, o in OBSERVED.items():
        fval = fam.get(key)
        if fval is None:
            continue
        a = agree(fval, o["obs"])
        num += a * o["w"]
        den += o["w"]
        breakdown[key] = {"family": fval, "observed": o["obs"],
                          "agree": round(a, 2), "w": o["w"]}
    return (num / den if den else 0.0), breakdown


def ranked_families():
    rows = []
    for f in FAMILIES:
        score, bd = match_score(f)
        rows.append((score, f, bd))
    return sorted(rows, key=lambda t: -t[0])


if __name__ == "__main__":
    print(f"{'family':10s} {'score':>6s}  top agree / disagree\n" + "=" * 75)
    for score, fam, bd in ranked_families():
        flags = ["+" + k for k, v in bd.items() if v["agree"] >= 0.8]
        flags += ["-" + k for k, v in bd.items() if v["agree"] <= 0.2]
        print(f"{fam.id:10s} {score:6.3f}  {flags}")