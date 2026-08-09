import numpy as np, pickle, importlib
import baseline_tuning as BT
importlib.reload(BT)
import fleet_charging as FC

SEEDS = range(6)
published = pickle.load(open("cats_tuned_full.pkl", "rb"))

def summarize(runs):
    return {
        "energy_MJ": np.mean([r["energy_J"] for r in runs]) / 1e6,
        "delay_pct": np.mean([r["chg_penalty_mean_frac"] for r in runs]) * 100,
        "delay_max_pct": np.mean([r["chg_penalty_max_frac"] for r in runs]) * 100,
        "fatigue": np.mean([r["cycling_max"] for r in runs]),
        "work_ks": np.mean([r["work_s"] for r in runs]) / 1000,
    }

results = {}

# ---- Oort: sweep alpha (speed-discount exponent) ----
print("=== Oort: sweeping alpha ===")
results["oort"] = {}
for alpha in [0.5, 1.0, 2.0, 3.0, 4.0]:
    fn = BT.make_p_oort(alpha=alpha)
    runs = [BT.run_custom(fn, seed=s) for s in SEEDS]
    s = summarize(runs)
    results["oort"][alpha] = s
    tag = " (default)" if alpha == 2.0 else ""
    print(f"  alpha={alpha:<4} energy={s['energy_MJ']:.3f}MJ delay={s['delay_pct']:.2f}% "
          f"fatigue={s['fatigue']:.0f} work={s['work_ks']:.1f}ks{tag}")

# ---- EAFL: sweep soc_min (safety floor) ----
print("\n=== EAFL: sweeping soc_min ===")
results["eafl"] = {}
for soc_min in [0.02, 0.05, 0.10, 0.15, 0.20]:
    fn = BT.make_p_eafl(soc_min=soc_min)
    runs = [BT.run_custom(fn, seed=s) for s in SEEDS]
    s = summarize(runs)
    results["eafl"][soc_min] = s
    tag = " (default)" if soc_min == 0.05 else ""
    print(f"  soc_min={soc_min:<5} energy={s['energy_MJ']:.3f}MJ delay={s['delay_pct']:.2f}% "
          f"fatigue={s['fatigue']:.0f} work={s['work_ks']:.1f}ks{tag}")

# ---- WILF-Q: sweep utility-blend weight ----
print("\n=== WILF-Q-analog: sweeping utility weight ===")
results["wilfq"] = {}
wilfq_idx = FC._build_wilfq_index(65.0, 15.0)
for w in [0.0, 0.05, 0.10, 0.20, 0.50]:
    fn = BT.make_p_wilfq(util_weight=w, wilfq_idx=wilfq_idx)
    runs = [BT.run_custom(fn, seed=s) for s in SEEDS]
    s = summarize(runs)
    results["wilfq"][w] = s
    tag = " (default)" if w == 0.05 else ""
    print(f"  weight={w:<5} energy={s['energy_MJ']:.3f}MJ delay={s['delay_pct']:.2f}% "
          f"fatigue={s['fatigue']:.0f} work={s['work_ks']:.1f}ks{tag}")

# ---- Canary check: did the parameter actually change anything? ----
print("\n=== CANARY CHECK: outputs must vary across the sweep ===")
for pol in results:
    energies = [v["energy_MJ"] for v in results[pol].values()]
    spread = max(energies) - min(energies)
    print(f"  {pol}: energy range = {spread*1000:.1f} kJ across sweep  "
          f"{'OK -- parameter is wired in' if spread > 0.001 else 'FLAT -- SUSPICIOUS'}")

# CATS tuned reference
cats_e = np.mean([r["energy_J"] for r in published["cats_tuned"]]) / 1e6
cats_delay = np.mean([r["chg_penalty_mean"] for r in published["cats_tuned"]]) * 100
cats_fatigue = np.mean([r["cycling_max"] for r in published["cats_tuned"]])
print(f"\nCATS tuned reference: energy={cats_e:.3f}MJ delay={cats_delay:.2f}% fatigue={cats_fatigue:.0f}")

pickle.dump(results, open("baseline_sweep_results.pkl", "wb"))
print("\nSaved: baseline_sweep_results.pkl")
