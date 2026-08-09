"""
Device-churn / flash-crowd scenario: instead of i.i.d. availability at
p=0.85 throughout, half the fleet drops out simultaneously and abruptly
partway through the campaign (e.g. a phone call, an app launch across a
correlated population -- think 9pm TV-ad break), then recovers. Tests
whether CATS's wear queues and delivered outcomes are robust to a
correlated shock rather than only to independent per-round draws.
"""
import numpy as np
import fleet_charging as FC

def run_with_churn(policy, n=100, hours=6.0, K=10, seed=0, V=5.0, mu=0.3, nu=1.0,
                    W_budget=0.05, t_round=65.0, t_gap=15.0,
                    churn_start_frac=0.30, churn_duration_frac=0.10,
                    churn_availability=0.35):
    """Identical to fleet_charging.run(), except availability drops to
    churn_availability for a churn_duration_frac window starting at
    churn_start_frac through the campaign."""
    rng = np.random.default_rng(seed)
    fleet = FC.ChargingFleet(n, rng, t_round=t_round, t_gap=t_gap)
    fleet.total_rounds = int(hours * 3600 / (t_round + t_gap))
    base_full = FC.counterfactual_full_time(fleet)

    r0 = int(fleet.total_rounds * churn_start_frac)
    r1 = int(fleet.total_rounds * (churn_start_frac + churn_duration_frac))

    st = {"Q": np.zeros(n), "V": V, "mu": mu, "nu": nu, "_round": 0}
    if policy == "wilfq":
        st["_wilfq_idx"] = FC._build_wilfq_index(t_round, t_gap)
    fn = FC.POLICIES[policy]
    for rnd in range(fleet.total_rounds):
        p_avail = churn_availability if r0 <= rnd < r1 else 0.85
        pool = np.where(fleet.available(p=p_avail))[0]
        if len(pool) == 0:
            fleet.step_round(np.array([], dtype=int))
            continue
        chosen = np.asarray(fn(fleet, pool, K, st, rng), dtype=int)
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
        "policy": policy,
        "energy_J": float(fleet.energy.sum()),
        "work_s": float(fleet.selections.sum() * t_round),
        "cycling_max": float(fleet.cycling.max()),
        "chg_penalty_mean": float(np.mean(pen[touched]) * 100) if touched.any() else 0.0,
        "chg_penalty_max": float(np.max(pen[touched]) * 100) if touched.any() else 0.0,
        "n_unfinished": int(np.sum(np.isnan(fleet.full_time))),
        "Q_final_mean": float(st["Q"].mean()),
        "Q_final_max": float(st["Q"].max()),
    }

POLICIES = ["random", "static_score", "energy_only", "oort", "eafl",
            "wilfq", "fedcs", "cats"]
SEEDS = range(6)

results = {}
for p in POLICIES:
    # V and nu only affect the "cats" policy's drift-plus-penalty score;
    # passed explicitly here at the paper's tuned operating point (Section
    # 8.3/8.5) rather than relying on run_with_churn's defaults, which are
    # the pre-tuning values and do not reproduce Section 7.8's numbers.
    runs = [run_with_churn(p, seed=s, V=5.0, nu=0.5) for s in SEEDS]
    results[p] = {
        "energy_J": np.mean([r["energy_J"] for r in runs]),
        "work_ks": np.mean([r["work_s"] for r in runs]) / 1000,
        "fatigue": np.mean([r["cycling_max"] for r in runs]),
        "delay_pct": np.mean([r["chg_penalty_mean"] for r in runs]),
        "delay_max_pct": np.mean([r["chg_penalty_max"] for r in runs]),
        "unfinished": np.mean([r["n_unfinished"] for r in runs]),
        "Q_final_max": np.mean([r["Q_final_max"] for r in runs]),
    }

print(f"\n{'policy':<16}{'energy(MJ)':>11}{'work(ks)':>10}{'fatigue':>10}"
      f"{'delay%':>9}{'delaymax%':>11}{'unfinished':>12}{'Qmax':>10}")
print("-" * 90)
for p in POLICIES:
    a = results[p]
    print(f"{p:<16}{a['energy_J']/1e6:>11.3f}{a['work_ks']:>10.1f}{a['fatigue']:>10.0f}"
          f"{a['delay_pct']:>9.2f}{a['delay_max_pct']:>11.2f}{a['unfinished']:>12.1f}"
          f"{a['Q_final_max']:>10.3f}")

import pickle
pickle.dump(results, open("churn_experiment.pkl", "wb"))
print("\nSaved: churn_experiment.pkl")
