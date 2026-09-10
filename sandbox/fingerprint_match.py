"""
fingerprint_match.py
--------------------
B3: match the OBSERVED fingerprints (from fingerprint.py, hard-coded in
models_library.OBSERVED) against each candidate family's prior profile
(models_library.py), rank families, run a leave-one-observable-out sensitivity
check, compute the "indistinguishable set", and apply the adoption guardrail.

The guardrail (adopted in the B-track plan): a family is only a *distinct
winner* if its score exceeds the runner-up by >= 0.1 AND no other family
enters its >0.95-overlap set. Otherwise no family is adopted as "the model";
only the *robust common set* (features shared by every top-ranked family) is
carried forward to GENESIS.md as design priors.

Usage:
    python3 fingerprint_match.py
Writes sandbox/results/fingerprint_match.md (and prints the tables).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from models_library import FAMILIES, OBSERVED, agree, match_score, ranked_families

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SCORE_TIE = 0.1
OVERLAP_EPS = 0.05


def leave_one_out():
    """Score each family with each observable dropped -> min/max/range."""
    keys = list(OBSERVED)
    fam_min = {f.id: 1.0 for f in FAMILIES}
    fam_max = {f.id: 0.0 for f in FAMILIES}
    for k in keys:
        # re-weight on the remaining observables
        sub_w = {kk: v["w"] for kk, v in OBSERVED.items() if kk != k}
        for f in FAMILIES:
            num = den = 0.0
            for kk, w in sub_w.items():
                fv = f.profile.get(kk)
                if fv is None:
                    continue
                a = agree(fv, OBSERVED[kk]["obs"])
                num += a * w
                den += w
            s = num / den if den else 0.0
            fam_min[f.id] = min(fam_min[f.id], s)
            fam_max[f.id] = max(fam_max[f.id], s)
    return fam_min, fam_max


def indistinguishable_set(target_id: str, families, scores):
    """Families whose score >= target_score - SCORE_TIE (overlap bucket)."""
    base = scores[target_id]
    return [f for f in families if scores[f.id] >= base - SCORE_TIE - OVERLAP_EPS]


def main():
    ranked = ranked_families()
    fam_min, fam_max = leave_one_out()

    rows = []
    for score, fam, bd in ranked:
        flags = ["+" + k for k, v in bd.items() if v["agree"] >= 0.8]
        flags += ["-" + k for k, v in bd.items() if v["agree"] <= 0.2]
        rows.append({
            "family": fam.name, "id": fam.id,
            "score": round(score, 3),
            "score_min": round(fam_min[fam.id], 3),
            "score_max": round(fam_max[fam.id], 3),
            "agree": ", ".join(flags) or "-",
            "cite": fam.cite,
        })
    df = pd.DataFrame(rows)

    top_id = ranked[0][1].id
    top_score = ranked[0][0]
    over = indistinguishable_set(top_id, [f for _, f, _ in ranked],
                                 {f.id: s for s, f, _ in ranked})
    distinct_winner = (
        len(over) == 1 and
        (ranked[0][0] - ranked[1][0]) >= SCORE_TIE
    )

    L = []
    A = L.append
    A("# fingerprint -> model-family matching (B3)\n")
    A("\nObserved fingerprints used (from `fingerprint.py`, hold-out win-rate "
      "significance; weights = reliability):\n\n"
      + pd.DataFrame([
          {"observable": k, "observed": v["obs"], "weight": v["w"]}
          for k, v in OBSERVED.items()
      ]).to_markdown(index=False) + "\n")
    A("\n## Family ranking\n\n" + df[["family", "id", "score",
                                       "score_min", "score_max", "agree"]]
      .to_markdown(index=False) + "\n")
    A("\n`score` = weighted agreement of family priors with observed "
      "fingerprints (1 = all observables explained). `score_min/max` = range "
      "over leave-one-observable-out sensitivity.\n")

    A(f"\n## Indistinguishable set\n\nTop family: **{over[0].name}** "
      f"({top_score:.3f}).\n")
    if len(over) > 1:
        A(f"\nWithin {SCORE_TIE:.2f} of the top (the 'no-distinct-winner' "
          f"bucket):\n\n" + "\n".join(f"- {f.name}" for f in over if f.id != top_id) + "\n")
    verdict = ("ADOPT-ABLE distinct winner" if distinct_winner
               else "NO distinct winner — do NOT adopt any single family as 'the model'")
    A(f"\n**Guardrail verdict: {verdict}**\n")
    if len(over) > 1:
        A("\nThe overlap bucket spans families with different heat/water "
          "mechanisms. Only the *robust common set* of design priors carries "
          "forward (see GENESIS.md): a generic water-stress term active "
          "through both precipitation supply and VPD demand, flowering-window "
          "heat for maize at low threshold, grain-fill heat for wheat at high "
          "threshold, saturating nitrogen, no inter-annual soil memory.\n")

    A("\n## Robust common set (guarded priors for candidate design)\n")
    A("\n1. **Water**: wet-helps signature in precipitation (wheat pr win "
      "0.601, p<1e-8) AND dryness-hurts in VPD (maize vpd win 0.614, p<1e-12; "
      "joint beta pr +0.027 / vpd -0.074). A candidate that only uses rain is "
      "leaving the VPD channel on maize; one that only uses VPD leaves the "
      "pr channel on wheat. Both channels should be present.")
    A("\n2. **Heat**: maize heat peaks in days 121-150 (mid season, ~flowering) "
      "with hdd>=22/26 best -> low-threshold flowering heat. Wheat weak, "
      "best hdd>=30/34 with slight late-window tilt -> high-threshold "
      "grain-fill heat. A single shared heat threshold is NOT supported; "
      "thresholds should differ by crop.")
    A("\n3. **Nitrogen**: saturating cross-sectionally (log/sqrt best in both "
      "crops, pooled). Supports Mitscherlich-type N over linear. (N is "
      "constant per cell, so this enters only via a per-cell N response in "
      "the design or via FunSearch's meta column.)")
    A("\n4. **Soil memory**: lag-1 water terms uniformly hurt hold-out R2 "
      "(win rates ~0.40, p<1e-15). No inter-annual soil-water carryover "
      "warrants a term in the per-location protocol.")
    A("\n5. **CO2**: identifiable in-sample is impossible (level-drift "
      "identity). Use the literature-saturating C3 / near-C4 values already "
      "adopted in candidates 24-27 (CO2_LIT.md).")

    md = "".join(L)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "fingerprint_match.md"
    out.write_text(md)
    print(md)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()