"""
Reproduce Table 4 (main policy comparison) and Table 7 (classical scheduling
metrics), and write cats_tuned_full.pkl, the reference file several other
scripts in this repository read (baseline_tuning.py's sanity check,
run_sweeps.py's CATS reference row).

Settings match Section 7.1 and the tuned operating point of Section 8.3/8.5:
K=30, W_budget=0.05, 65 s round / 15 s aggregation, six-hour campaign,
six seeds. CATS is evaluated at its tuned weights, V=5.0 (energy-vs-wear)
and nu=0.5 (charger-headroom); "cats_orig" is the same policy at the
untuned V=1.0 reported once in Section 8.2 before tuning, kept here only
because Table 12's ablation and baseline_tuning.py's sanity check compare
against it.

Runtime: well under a minute on a single CPU core (Section 7.6).

Usage:
    python reproduce_table4.py
"""
import pickle

import numpy as np

import fleet_charging as FC

SEEDS = range(6)
K = 30
W_BUDGET = 0.05
T_ROUND = 65.0
T_GAP = 15.0
HOURS = 6.0

# (policy key in fleet_charging.POLICIES, V, nu) -- mu is left at run()'s
# default (0.3) throughout, as in the paper.
RUNS = {
    "random":       ("random", 1.0, 1.0),
    "static_score": ("static_score", 1.0, 1.0),
    "energy_only":  ("energy_only", 1.0, 1.0),
    "oort":         ("oort", 1.0, 1.0),
    "eafl":         ("eafl", 1.0, 1.0),
    "wilfq":        ("wilfq", 1.0, 1.0),
    "fedcs":        ("fedcs", 1.0, 1.0),
    "cats_tuned":   ("charger_aware", 5.0, 0.5),
    # "cats_orig": CATS at the untuned energy weight (V=1) reported once in
    # Section 8.2/8.3 before the V-sweep of Table 5 found V=5 optimal; the
    # charger weight nu=0.5 was already fixed at this stage. Kept here
    # because baseline_tuning.py's sanity check and run_sweeps.py's
    # reference row both read it from this file.
    "cats_orig":    ("charger_aware", 1.0, 0.5),
}


def run_all():
    results = {}
    for label, (policy, V, nu) in RUNS.items():
        results[label] = [
            FC.run(policy, n=100, hours=HOURS, K=K, seed=s, V=V, nu=nu,
                   W_budget=W_BUDGET, t_round=T_ROUND, t_gap=T_GAP)
            for s in SEEDS
        ]
    return results


def summarize(runs):
    return dict(
        energy_MJ=np.mean([r["energy_J"] for r in runs]) / 1e6,
        chg_pen_pct=np.mean([r["chg_penalty_mean"] for r in runs]) * 100,
        fatigue=np.mean([r["cycling_max"] for r in runs]),
        work_ks=np.mean([r["work_s"] for r in runs]) / 1000,
        jain=np.mean([r["jain"] for r in runs]),
        utilisation=np.mean([r["utilisation"] for r in runs]),
        touched=np.mean([r["n_touched"] for r in runs]),
    )


if __name__ == "__main__":
    results = run_all()

    print(f"{'policy':<14}{'energy(MJ)':>11}{'chg_pen':>9}{'fatigue':>9}"
          f"{'work(ks)':>10}{'jain':>7}{'util':>7}{'touched':>9}")
    print("-" * 82)
    for label in RUNS:
        s = summarize(results[label])
        print(f"{label:<14}{s['energy_MJ']:>11.2f}{s['chg_pen_pct']:>8.1f}%"
              f"{s['fatigue']:>9.0f}{s['work_ks']:>10.1f}{s['jain']:>7.3f}"
              f"{s['utilisation']:>7.3f}{s['touched']:>9.0f}")

    pickle.dump(results, open("cats_tuned_full.pkl", "wb"))
    print("\nSaved: cats_tuned_full.pkl")
    print("(baseline_tuning.py and run_sweeps.py read this file directly)")
