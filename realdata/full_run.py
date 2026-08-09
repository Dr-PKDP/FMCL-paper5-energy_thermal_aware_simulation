"""
Real-dataset validation of the Table 9 quality comparison. Same
methodology, same policy set, same alpha=0.3, same N/K/hours as the
synthetic-data result already in the paper -- only the data source
changes.
"""
import os
import sys
import time

# Make the parent directory (repo root) importable, regardless of where
# this repository is cloned, so this script can be run directly from
# within realdata/ via `python full_run.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import real_data as R
from converge_charging import run, rate_and_floor

N, K, HOURS = 100, 30, 6.0
SEEDS = range(3)
POLICIES = ["random", "static_score", "energy_only", "lyapunov", "charger_aware"]

results = {}
for ds, n_per in [("digits", 17), ("mnist", 50)]:
    t0 = time.time()
    clients, NC, DIM = R.make_real_federated_data(ds, N=N, n_per=n_per,
                                                    alpha=0.3, seed=0)
    results[ds] = {}
    for p in POLICIES:
        # V=5.0, nu=0.5: the paper's tuned CATS operating point (Section
        # 8.3/8.5), matching Table 9's synthetic-data setting exactly so
        # this real-dataset run is a like-for-like comparison.
        runs = [run(p, clients, NC, DIM, n=N, K=K, hours=HOURS, seed=s,
                     V=5.0, nu=0.5)
                for s in SEEDS]
        g = np.mean([r["grads"] for r in runs], axis=0)
        rate_, floor_ = rate_and_floor(g)
        results[ds][p] = {
            "final_quality": float(np.mean([r["final_quality"] for r in runs])),
            "rate": rate_, "floor": floor_,
            "energy_J": float(np.mean([r["energy_J"] for r in runs])),
            "work_s": float(np.mean([r["work_s"] for r in runs])),
            "cycling_max": float(np.mean([r["cycling_max"] for r in runs])),
            "dropout_rate": float(np.mean([r["dropout_rate"] for r in runs])),
            "n_contributors": float(np.mean([r["n_contributors"] for r in runs])),
            "contrib_gini": float(np.mean([r["contrib_gini"] for r in runs])),
        }
    print(f"{ds}: {time.time()-t0:.1f}s")

print(f"\n{'dataset':<9}{'policy':<16}{'final_q':>12}{'vs best':>9}{'rate':>9}"
      f"{'floor':>12}{'fatigue':>9}{'dropout':>9}{'contrib':>9}{'gini':>7}")
print("-" * 101)
for ds in results:
    best = min(results[ds][p]["final_quality"] for p in POLICIES)
    for p in POLICIES:
        a = results[ds][p]
        rel = a["final_quality"] / best
        print(f"{ds:<9}{p:<16}{a['final_quality']:>12.3e}{rel:>8.2f}x{a['rate']:>9.4f}"
              f"{a['floor']:>12.3e}{a['cycling_max']:>9.0f}{a['dropout_rate']:>9.1%}"
              f"{a['n_contributors']:>9.0f}{a['contrib_gini']:>7.3f}")
    print()

import pickle
pickle.dump(results, open("realdata_results.pkl", "wb"))
print("Saved: realdata_results.pkl")
