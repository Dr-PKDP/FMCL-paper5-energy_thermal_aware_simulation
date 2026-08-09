"""
Extreme non-IID sweep: rerun the fixed-budget quality comparison at
Dirichlet concentration alpha = 0.3 (baseline, already in the paper),
0.1, and 0.05, to check CATS's thermal-aware selection does not
compromise convergence when local distributions are far more skewed.
"""
import numpy as np
from simulation import make_federated_data
from converge_charging import run, rate_and_floor

N, K, HOURS = 100, 30, 6.0
SEEDS = range(3)
POLICIES = ["random", "static_score", "energy_only", "lyapunov", "charger_aware"]
ALPHAS = [0.3, 0.1, 0.05]

results = {}
for alpha in ALPHAS:
    clients, NC, DIM = make_federated_data(N=N, K=4, dim=10, n_per=50,
                                            alpha=alpha, seed=0)
    results[alpha] = {}
    for p in POLICIES:
        # V=5.0, nu=0.5: the paper's tuned CATS operating point (Section
        # 8.3/8.5). Passed explicitly rather than relying on
        # converge_charging.run()'s defaults, which are the pre-tuning
        # values and do not reproduce Section 8.8's reported ratios.
        runs = [run(p, clients, NC, DIM, n=N, K=K, hours=HOURS, seed=s,
                     V=5.0, nu=0.5)
                for s in SEEDS]
        g = np.mean([r["grads"] for r in runs], axis=0)
        rate_, floor_ = rate_and_floor(g)
        results[alpha][p] = {
            "final_quality": float(np.mean([r["final_quality"] for r in runs])),
            "rate": rate_, "floor": floor_,
            "energy_J": float(np.mean([r["energy_J"] for r in runs])),
            "cycling_max": float(np.mean([r["cycling_max"] for r in runs])),
            "dropout_rate": float(np.mean([r["dropout_rate"] for r in runs])),
            "n_contributors": float(np.mean([r["n_contributors"] for r in runs])),
        }

print(f"\n{'alpha':<8}{'policy':<16}{'final ||g||^2':>15}{'rate':>10}{'floor':>13}"
      f"{'fatigue':>10}{'dropout':>9}{'contrib':>9}")
print("-" * 92)
for alpha in ALPHAS:
    best = min(results[alpha][p]["final_quality"] for p in POLICIES)
    for p in POLICIES:
        a = results[alpha][p]
        rel = a["final_quality"] / best
        print(f"{alpha:<8}{p:<16}{a['final_quality']:>15.3e}{a['rate']:>10.4f}"
              f"{a['floor']:>13.3e}{a['cycling_max']:>10.0f}{a['dropout_rate']:>9.1%}"
              f"{a['n_contributors']:>9.0f}   ({rel:.2f}x best)")
    print()

import pickle
pickle.dump(results, open("noniid_sweep.pkl", "wb"))
print("Saved: noniid_sweep.pkl")
