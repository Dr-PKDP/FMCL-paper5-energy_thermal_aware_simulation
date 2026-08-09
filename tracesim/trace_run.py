"""
Run the main policy comparison with real trace-derived availability,
substituted for the i.i.d. Bernoulli(0.85) draw, everything else identical
(K=30, W_budget=0.05, V=5/nu=0.5 for CATS, 270 rounds, n=407 registered
devices calibrated to ~85 mean concurrent availability).

Safety: ChargingFleet.available is overridden on the INSTANCE only, via a
bound closure reading a precomputed matrix; the class definition and every
other physics method are untouched.
"""
import os
import sys

# Make the parent directory (repo root) importable, regardless of where
# this repository is cloned, so this script can be run directly from
# within tracesim/ via `python trace_run.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import fleet_charging as FC
import trace_availability as TA

N_REG = 407
CAMPAIGN_LEN = 270 * 80.0

_behave = None
def get_behave():
    global _behave
    if _behave is None:
        _behave = TA.load_trace()
    return _behave


def run_trace(policy, seed=0, K=30, W_budget=0.05, V=5.0, mu=0.3, nu=0.5,
              t_round=65.0, t_gap=15.0):
    behave = get_behave()
    eligible_ids = [k for k, v in behave.items()
                    if v.get("finish_time", 0) >= CAMPAIGN_LEN]
    rng_sel = np.random.default_rng(seed)
    chosen_ids = rng_sel.choice(eligible_ids, size=N_REG, replace=False)
    mat, _ = TA.build_availability_matrix(behave, chosen_ids, t_round=t_round,
                                           t_gap=t_gap, seed=seed)
    n_rounds = mat.shape[1]

    rng = np.random.default_rng(seed + 100000)   # separate stream for physics
    fleet = FC.ChargingFleet(N_REG, rng, t_round=t_round, t_gap=t_gap)
    fleet.total_rounds = n_rounds

    # Instance-level override: read the precomputed row for the current round
    round_ctr = {"i": 0}
    def available_from_trace(p=0.85):
        row = mat[:, round_ctr["i"]]
        round_ctr["i"] += 1
        return row
    fleet.available = available_from_trace

    base_full = FC.counterfactual_full_time(fleet)
    round_ctr["i"] = 0   # counterfactual consumed the counter; reset for the real run
    fleet.available = available_from_trace

    st = {"Q": np.zeros(N_REG), "V": V, "mu": mu, "nu": nu, "_round": 0}
    if policy == "wilfq":
        st["_wilfq_idx"] = FC._build_wilfq_index(t_round, t_gap)
    fn = FC.POLICIES[policy]
    for _ in range(n_rounds):
        pool = np.where(fleet.available())[0]
        if len(pool) == 0:
            fleet.step_round(np.array([], dtype=int))
            continue
        chosen = np.asarray(fn(fleet, pool, K, st, rng), dtype=int)
        w = fleet.marginal_wear()
        fleet.step_round(chosen)
        st["Q"] = np.maximum(st["Q"] - W_budget, 0.0)
        st["Q"][chosen] += w[chosen]
    fleet.close()

    horizon = n_rounds * (t_round + t_gap)
    ft = np.where(np.isnan(fleet.full_time), horizon, fleet.full_time)
    bf = np.where(np.isnan(base_full), horizon, base_full)
    pen = ft / np.maximum(bf, 1.0) - 1.0
    touched = fleet.selections > 0

    return {
        "energy_J": float(fleet.energy.sum()),
        "work_s": float(fleet.work.sum()),
        "cycling_max": float(fleet.cycling.max()),
        "chg_penalty_mean_frac": float(pen[touched].mean()) if touched.any() else 0.0,
        "chg_penalty_max_frac": float(pen.max()),
        "n_touched": int(touched.sum()),
        "mean_avail_per_round": float(mat.sum(axis=0).mean()),
    }
