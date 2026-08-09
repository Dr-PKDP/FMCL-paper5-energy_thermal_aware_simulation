"""
Reconstruction of Table 13's upper block: a Monte Carlo sweep over the full
uncertain-coefficient space feeding fleet_charging.py's simulation (11
charging.py module constants + the 3 coefficients.py Coefficient objects
fleet_charging.py actually reads: STATIC_POWER_FRACTION, LEAKAGE_TEMP_CONSTANT,
DVFS_EXPONENT), testing whether CATS's dominance claims from Table 4 survive
coefficient uncertainty.

This script did not exist in the repository as pushed; it is being written
now to close that gap. Settings (K=30, n=100, hours=6.0) match reproduce_table4.py
exactly. N=20 draws, one seed per draw, matching the smaller of the two block
sizes the paper describes (justified here by including WILF-Q-analog, whose
per-draw index reconstruction is the same cost driver the paper cites).

Checkpointed after every draw so a partial run is resumable.
"""
import dataclasses
import pickle
import time
import numpy as np

import charging as CH
import coefficients as C
import fleet_charging as FC

N_DRAWS = 20
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


def main():
    orig_chg = {k: getattr(CH, k) for k in CHG_BOUNDS}
    orig_thermal = {k: getattr(C, k) for k in THERMAL_BOUNDS}

    rows = []
    t0 = time.time()
    for i in range(N_DRAWS):
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
    print(f"\nAll {N_DRAWS} draws complete in {time.time()-t0:.0f}s. Saved table13_upper_full.pkl")


if __name__ == "__main__":
    main()
