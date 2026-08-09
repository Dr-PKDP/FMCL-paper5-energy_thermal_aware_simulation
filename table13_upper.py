"""
Reconstruction of Table 13's upper block: a Monte Carlo sweep over the full
uncertain-coefficient space feeding fleet_charging.py's simulation (11
charging.py module constants + the 3 coefficients.py Coefficient objects
fleet_charging.py actually reads: STATIC_POWER_FRACTION, LEAKAGE_TEMP_CONSTANT,
DVFS_EXPONENT), testing whether CATS's dominance claims from Table 4 survive
coefficient uncertainty.

This script did not exist in the repository as pushed; it was written to
close that gap, then re-run at N=70 (up from an initial N=20) to test
whether sample size explained an observed discrepancy on two specific
claims. It did not.

RESULTS, N=70, against the paper's Section 9 text:

  Delay/fatigue/compute dominance over all 7 baselines: 95.7-100%,
    matching "the first five [claims] hold everywhere."           CONFIRMED
  CATS energy <= Oort:                              95.7% vs documented 89%. CLOSE
  Charger-signal delay reduction vs charger-blind:  82.9% vs documented 87%. CLOSE
  CATS energy <= static_score (linear score):       88.6% vs documented 67%. NOT CLOSED
  CATS energy <= WILF-Q-analog:                     88.6% vs documented 72%. NOT CLOSED

The last two do not close. Going from N=20 to N=70 moved them by under two
percentage points (90.0% -> 88.6%), which rules out sample size as the
explanation -- a genuine ~20-point gap does not shrink to nothing from
3.5x more draws if it were sampling noise.

Best-supported (not confirmed) explanation: this repository's coefficient
library is independently known to have evolved after Table 13's original
figures were produced -- R_th was retightened per finding F2, changing the
library's own R_th.value from 3.6/4.5 K/W to 5.0/6.0 K/W (see
COEFFICIENTS_FINDINGS.md), and that correction is explicitly labelled as
such in coefficients.py's source comment. The 11 charging.py coefficients
swept here carry no equivalent "corrected per finding X" comment, so this
cannot be confirmed the same way -- but given one coefficient in this
exact model demonstrably moved between when Table 13 was written and now,
it is the most likely explanation for why CATS's measured energy advantage
over static_score/WILF-Q-analog is now stronger (88.6%) than documented
(67%/72%), rather than a bug in this reconstruction.

What this means practically: if the paper's 67%/72% figures predate the
current, more corrected coefficient library, then 88.6% is arguably the
more accurate current figure, not merely a different guess.
"""
import dataclasses
import pickle
import time
import numpy as np

import charging as CH
import coefficients as C
import fleet_charging as FC

N_DRAWS = 70
K = 30
HOURS = 6.0
W_BUDGET = 0.05
T_ROUND = 65.0
T_GAP = 15.0
SEED = 0  # fixed: isolates coefficient-draw effect from seed effect

BASELINES = ["random", "static_score", "energy_only", "oort", "eafl", "wilfq", "fedcs"]

# --- Combined uncertain coefficient space (14 total) ---------------------
# 11 from charging.py (same set/ranges as uq_charging.py's CHG_COEFFS)
CHG_BOUNDS = {
    "ETA_CHG": (0.88, 0.96),
    "R_S": (7.0, 14.0),
    "R_B": (7.0, 14.0),
    "R_SB": (3.0, 9.0),
    "C_S": (10.0, 25.0),
    "C_B": (30.0, 100.0),
    "T_CHG_DERATE": (37.0, 42.0),
    "T_CHG_STOP": (44.0, 48.0),
    "T_COMPUTE_CEILING": (52.0, 62.0),
    "T_COMPUTE_KILL": (58.0, 70.0),
    "E_BATT_WH": (14.0, 22.0),
}
# 3 from coefficients.py (Coefficient objects actually read by fleet_charging.py)
THERMAL_BOUNDS = {
    "STATIC_POWER_FRACTION": (C.STATIC_POWER_FRACTION.low, C.STATIC_POWER_FRACTION.high),
    "LEAKAGE_TEMP_CONSTANT": (C.LEAKAGE_TEMP_CONSTANT.low, C.LEAKAGE_TEMP_CONSTANT.high),
    "DVFS_EXPONENT": (C.DVFS_EXPONENT.low, C.DVFS_EXPONENT.high),
}

RNG = np.random.default_rng(20260809)


def sample_draw():
    p = {}
    for k, (lo, hi) in CHG_BOUNDS.items():
        p[k] = RNG.uniform(lo, hi)
    if p["T_COMPUTE_KILL"] <= p["T_COMPUTE_CEILING"] + 2.0:
        p["T_COMPUTE_KILL"] = p["T_COMPUTE_CEILING"] + 2.0
    if p["T_CHG_STOP"] <= p["T_CHG_DERATE"] + 2.0:
        p["T_CHG_STOP"] = p["T_CHG_DERATE"] + 2.0
    for k, (lo, hi) in THERMAL_BOUNDS.items():
        p[k] = RNG.uniform(lo, hi)
    return p


def apply_draw(p):
    for k in CHG_BOUNDS:
        setattr(CH, k, p[k])
    # Coefficient is a frozen dataclass; replace the module attribute with a
    # new instance carrying the drawn value rather than mutating in place.
    for k in THERMAL_BOUNDS:
        setattr(C, k, dataclasses.replace(getattr(C, k), value=p[k]))


def restore(orig_chg, orig_thermal):
    for k, v in orig_chg.items():
        setattr(CH, k, v)
    for k, coef in orig_thermal.items():
        setattr(C, k, coef)


def run_one(policy, V, nu, seed=SEED):
    return FC.run(policy, n=100, hours=HOURS, K=K, seed=seed, V=V, nu=nu,
                  W_budget=W_BUDGET, t_round=T_ROUND, t_gap=T_GAP)


def main(resume=True):
    orig_chg = {k: getattr(CH, k) for k in CHG_BOUNDS}
    orig_thermal = {k: getattr(C, k) for k in THERMAL_BOUNDS}

    rows = []
    start_i = 0
    if resume:
        try:
            rows = pickle.load(open("table13_upper_partial.pkl", "rb"))
            start_i = len(rows)
            print(f"Resuming from draw {start_i}/{N_DRAWS} (loaded {len(rows)} prior draws)")
        except FileNotFoundError:
            pass

    t0 = time.time()
    for i in range(start_i, N_DRAWS):
        p = sample_draw()
        apply_draw(p)
        try:
            cats = run_one("charger_aware", 5.0, 0.5)
            blind = run_one("charger_aware", 5.0, 0.0)
            base = {b: run_one(b, 1.0, 1.0) for b in BASELINES}
            rows.append({"cats": cats, "blind": blind, "base": base, "params": p})
        finally:
            restore(orig_chg, orig_thermal)
        print(f"  draw {i+1}/{N_DRAWS} done ({time.time()-t0:.0f}s elapsed)", flush=True)
        pickle.dump(rows, open("table13_upper_partial.pkl", "wb"))

    pickle.dump(rows, open("table13_upper_full.pkl", "wb"))
    print(f"\nAll {N_DRAWS} draws complete. Saved table13_upper_full.pkl")


if __name__ == "__main__":
    main()
