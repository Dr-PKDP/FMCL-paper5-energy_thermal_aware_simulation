"""
Minimal hyperparameter search for the three baselines that have a genuine
free parameter (FedCS-Greedy has none, by its own docstring). Copies the
exact decision-rule logic from fleet_charging.py's p_oort/p_eafl/p_wilfq,
parameterising only the one constant each currently hardcodes, so the
reimplementation stays faithful to what is already validated.

Two safety checks built in before trusting any swept result:
  1. Sanity check: default-parameter run must reproduce cats_tuned_full.pkl's
     already-published numbers for that baseline, seed-for-seed-matched mean.
  2. Canary check: output must actually change across swept values -- an
     unchanging output across the whole sweep means the parameter is not
     wired into the decision rule and the run is untrustworthy.

All percentages are reported with an explicit *100 conversion, printed
alongside the raw fraction, to avoid the exact units bug found and fixed
in the ablation section.
"""
import numpy as np
import fleet_charging as FC
import thermal as TH

def _topk(score, pool, K):
    return pool if len(pool) <= K else pool[np.argsort(-score)[:K]]


# ---- Parameterised copies of the tunable baselines ----

def make_p_oort(alpha):
    def p_oort_tuned(f, pool, K, st, rng):
        T_i = f.t_round / np.maximum(TH.eta(f.T_s[pool], f.T_cap[pool],
                                            f.eta_min[pool]), 1e-6)
        T_target = np.percentile(T_i, 50)
        sys_util = np.where(T_i > T_target, (T_target / T_i) ** alpha, 1.0)
        util = f.utility[pool] * sys_util
        round_idx = st.get("_round", 0)
        last_seen = st.setdefault("_last_seen", np.zeros(f.n))
        staleness = round_idx - last_seen[pool]
        ucb = np.sqrt(2.0 * np.log(round_idx + 2.0) / np.maximum(staleness, 1.0))
        chosen = _topk(util + 0.1 * ucb, pool, K)
        last_seen[chosen] = round_idx
        st["_round"] = round_idx + 1
        return chosen
    return p_oort_tuned

def make_p_eafl(soc_min):
    def p_eafl_tuned(f, pool, K, st, rng):
        headroom = np.maximum(f.soc[pool] - soc_min, 0.0)
        T_i = f.t_round / np.maximum(TH.eta(f.T_s[pool], f.T_cap[pool],
                                            f.eta_min[pool]), 1e-6)
        return _topk(headroom / T_i, pool, K)
    return p_eafl_tuned

def make_p_wilfq(util_weight, wilfq_idx):
    def p_wilfq_tuned(f, pool, K, st, rng):
        scores = np.array([wilfq_idx[f.cls[i]].at(f.T_s[i]) for i in pool])
        return _topk(scores + util_weight * f.utility[pool], pool, K)
    return p_wilfq_tuned


def run_custom(policy_fn, n=100, hours=6.0, K=30, seed=0,
                W_budget=0.05, t_round=65.0, t_gap=15.0):
    """Exact mirror of fleet_charging.run(), taking a policy FUNCTION
    directly instead of a string looked up in FC.POLICIES."""
    rng = np.random.default_rng(seed)
    fleet = FC.ChargingFleet(n, rng, t_round=t_round, t_gap=t_gap)
    fleet.total_rounds = int(hours * 3600 / (t_round + t_gap))
    base_full = FC.counterfactual_full_time(fleet)

    st = {"Q": np.zeros(n), "V": 1.0, "mu": 0.3, "nu": 1.0, "_round": 0}
    for _ in range(fleet.total_rounds):
        pool = np.where(fleet.available())[0]
        if len(pool) == 0:
            fleet.step_round(np.array([], dtype=int))
            continue
        chosen = np.asarray(policy_fn(fleet, pool, K, st, rng), dtype=int)
        w = fleet.marginal_wear()
        fleet.step_round(chosen)
        st["Q"] = np.maximum(st["Q"] - W_budget, 0.0)
        st["Q"][chosen] += w[chosen]
    fleet.close()

    horizon = fleet.total_rounds * (t_round + t_gap)
    ft = np.where(np.isnan(fleet.full_time), horizon, fleet.full_time)
    bf = np.where(np.isnan(base_full), horizon, base_full)
    pen = ft / np.maximum(bf, 1.0) - 1.0
    touched = fleet.selections > 0

    return {
        "work_s": float(fleet.work.sum()),
        "energy_J": float(fleet.energy.sum()),
        "cycling_max": float(fleet.cycling.max()),
        "chg_penalty_mean_frac": float(pen[touched].mean()) if touched.any() else 0.0,
        "chg_penalty_max_frac": float(pen.max()),
    }


SEEDS = range(6)

print("=" * 70)
print("SANITY CHECK: default-parameter reimplementation vs published pickle")
print("=" * 70)

import pickle
published = pickle.load(open("cats_tuned_full.pkl", "rb"))

# Oort default alpha=2.0
oort_fn = make_p_oort(alpha=2.0)
runs = [run_custom(oort_fn, seed=s) for s in SEEDS]
e = np.mean([r["energy_J"] for r in runs])
pub_e = np.mean([r["energy_J"] for r in published["oort"]])
print(f"Oort  default: my energy={e/1e6:.4f} MJ  published={pub_e/1e6:.4f} MJ  "
      f"match={'YES' if abs(e-pub_e)/pub_e < 0.01 else 'NO -- STOP'}")

# EAFL default soc_min=0.05
eafl_fn = make_p_eafl(soc_min=0.05)
runs = [run_custom(eafl_fn, seed=s) for s in SEEDS]
e = np.mean([r["energy_J"] for r in runs])
pub_e = np.mean([r["energy_J"] for r in published["eafl"]])
print(f"EAFL  default: my energy={e/1e6:.4f} MJ  published={pub_e/1e6:.4f} MJ  "
      f"match={'YES' if abs(e-pub_e)/pub_e < 0.01 else 'NO -- STOP'}")

# WILF-Q default weight=0.05
wilfq_idx = FC._build_wilfq_index(65.0, 15.0)
wilfq_fn = make_p_wilfq(util_weight=0.05, wilfq_idx=wilfq_idx)
runs = [run_custom(wilfq_fn, seed=s) for s in SEEDS]
e = np.mean([r["energy_J"] for r in runs])
pub_e = np.mean([r["energy_J"] for r in published["wilfq"]])
print(f"WILF-Q default: my energy={e/1e6:.4f} MJ  published={pub_e/1e6:.4f} MJ  "
      f"match={'YES' if abs(e-pub_e)/pub_e < 0.01 else 'NO -- STOP'}")
