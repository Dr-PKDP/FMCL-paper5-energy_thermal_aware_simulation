"""
FMCL Paper 5 -- Uncertainty quantification over the CHARGING coefficients.

The charging model was built with point estimates, violating the library's
standing rule that no T3 or T4 coefficient may appear as a point estimate in a
reported result. This file closes that gap: it assigns tiers and ranges, then
tests whether the ordinal claims F26-F29 survive.

Sensitivity is reported as standardised regression coefficients rather than
Sobol indices. SRC is computed from the same Monte Carlo sample at no extra
cost, and is valid where the response is monotone in each input; the model
here is, except near the admission thresholds. Where the SRC model's R^2 is
low, the attribution should not be trusted and a Sobol design is needed.
"""

import numpy as np
import charging as CH
import fleet_charging as FC

RNG = np.random.default_rng(20260803)

# name -> (low, high, tier, note)
CHG_COEFFS = {
    "ETA_CHG": (0.88, 0.96, "T3",
                "Derived by matching the measured 8-10 C fast-vs-standard "
                "battery rise (arXiv:1607.06402). Not independently measured."),
    "R_S": (7.0, 14.0, "T3",
            "SoC-to-ambient resistance. Constrained jointly with R_SB and R_B "
            "by the compute-only steady state."),
    "R_B": (7.0, 14.0, "T3",
            "Battery-to-ambient resistance, constrained by the charging rise."),
    "R_SB": (3.0, 9.0, "T3",
             "SoC-to-battery coupling. Sets the derived k_batt; the prior on "
             "k_batt was [0.30, 0.85] and this range keeps it inside."),
    "C_S": (10.0, 25.0, "T3",
            "SoC heat capacity, tied loosely to tau_heat via tau = R*C."),
    "C_B": (30.0, 100.0, "T4",
            "Battery heat capacity. NOT MEASURED. Battery is the larger "
            "thermal mass but the ratio to C_S is assumed."),
    "T_CHG_DERATE": (37.0, 42.0, "T2",
                     "Governor begins cutting charge current. Literature "
                     "places this in the high 30s to low 40s C."),
    "T_CHG_STOP": (44.0, 48.0, "T2",
                   "Charging suspended. Battery damage thresholds cited "
                   "around 45 C."),
    "T_COMPUTE_CEILING": (52.0, 62.0, "T4",
                          "OS begins suspending background work. INVENTED -- "
                          "no source. Drives F24 and the suspension term in "
                          "F27. Directly measurable on a real device."),
    "T_COMPUTE_KILL": (58.0, 70.0, "T4",
                       "Background work fully suspended. INVENTED. Constrained "
                       "to exceed T_COMPUTE_CEILING."),
    "E_BATT_WH": (14.0, 22.0, "T1",
                  "Battery capacity, 3700-5700 mAh at 3.85 V. Product spec."),
}

NAMES = list(CHG_COEFFS)


def sample_params():
    p = {}
    for k, (lo, hi, _, _) in CHG_COEFFS.items():
        p[k] = RNG.uniform(lo, hi)
    # keep the kill threshold above the ceiling
    if p["T_COMPUTE_KILL"] <= p["T_COMPUTE_CEILING"] + 2.0:
        p["T_COMPUTE_KILL"] = p["T_COMPUTE_CEILING"] + 2.0
    if p["T_CHG_STOP"] <= p["T_CHG_DERATE"] + 2.0:
        p["T_CHG_STOP"] = p["T_CHG_DERATE"] + 2.0
    return p


def apply(p):
    for k, v in p.items():
        setattr(CH, k, v)


def derived(p):
    r_s_eff = 1.0 / (1.0 / p["R_S"] + 1.0 / (p["R_SB"] + p["R_B"]))
    k_batt = p["R_B"] / (p["R_SB"] + p["R_B"])
    return r_s_eff, k_batt


def evaluate(n=50, hours=3.0, K=15, seed=0, nu=0.5):
    """Run the policy set on one parameter draw."""
    out = {}
    for pol, kw in [("random", {}), ("static_score", {}),
                    ("energy_only", {}), ("lyapunov", {"nu": 0.0}),
                    ("charger_blind", {"nu": 0.0}), ("charger_aware", {"nu": nu})]:
        name = "charger_aware" if pol == "charger_blind" else pol
        r = FC.run(name, n=n, hours=hours, K=K, seed=seed, **kw)
        out[pol] = r
    return out


def main():
    print("=" * 78)
    print("FMCL PAPER 5 -- CHARGING COEFFICIENT PROVENANCE AND ROBUSTNESS")
    print("=" * 78)

    tiers = {}
    for k, (lo, hi, t, _) in CHG_COEFFS.items():
        tiers[t] = tiers.get(t, 0) + 1
    print(f"\nCharging coefficient census: {dict(sorted(tiers.items()))}")
    print("\nT4 (unmeasured) charging coefficients:")
    for k, (lo, hi, t, note) in CHG_COEFFS.items():
        if t == "T4":
            print(f"  {k:<20} [{lo}, {hi}]")
            print(f"     {note}")

    N_MC = 70
    print(f"\n{'-'*78}\nMONTE CARLO: {N_MC} draws over 11 charging coefficients\n{'-'*78}")

    keep = {k: getattr(CH, k) for k in NAMES}
    X, rows = [], []
    for i in range(N_MC):
        p = sample_params()
        apply(p)
        try:
            res = evaluate()
        except Exception:
            continue
        r_s_eff, k_batt = derived(p)
        X.append([p[k] for k in NAMES])
        rows.append({
            "work_static": res["static_score"]["work_s"],
            "work_random": res["random"]["work_s"],
            "pen_static": res["static_score"]["chg_penalty_mean"],
            "pen_random": res["random"]["chg_penalty_mean"],
            "pen_blind": res["charger_blind"]["chg_penalty_mean"],
            "pen_aware": res["charger_aware"]["chg_penalty_mean"],
            "work_blind": res["charger_blind"]["work_s"],
            "work_aware": res["charger_aware"]["work_s"],
            "fat_aware": res["charger_aware"]["cycling_max"],
            "fat_energy": res["energy_only"]["cycling_max"],
            "fat_random": res["random"]["cycling_max"],
            "fast_aware": res["charger_aware"]["fast_share"],
            "fast_blind": res["charger_blind"]["fast_share"],
            "susp_static": res["static_score"]["suspended_s"],
            "susp_random": res["random"]["suspended_s"],
            "r_s_eff": r_s_eff, "k_batt": k_batt,
        })
    for k, v in keep.items():
        setattr(CH, k, v)

    X = np.array(X)
    n = len(rows)
    col = lambda k: np.array([r[k] for r in rows])
    print(f"  {n} successful draws\n")

    print(f"  Derived quantities across the sample:")
    print(f"    R_s_eff  median {np.median(col('r_s_eff')):.2f} K/W  "
          f"[{np.percentile(col('r_s_eff'),5):.2f}, {np.percentile(col('r_s_eff'),95):.2f}]")
    print(f"    k_batt   median {np.median(col('k_batt')):.3f}      "
          f"[{np.percentile(col('k_batt'),5):.3f}, {np.percentile(col('k_batt'),95):.3f}]"
          f"   prior was [0.30, 0.85]")

    checks = [
        ("F27a static_score delivers less work than random",
         col("work_static") < col("work_random")),
        ("F27b static_score charging penalty exceeds random",
         col("pen_static") > col("pen_random")),
        ("F27c static_score suspension exceeds random",
         col("susp_static") > col("susp_random")),
        ("F28a charger signal cuts fast-charger share",
         col("fast_aware") < col("fast_blind")),
        ("F28b charger signal cuts charging penalty",
         col("pen_aware") < col("pen_blind")),
        ("F28c charger signal costs less than 5% of the work",
         col("work_aware") > 0.95 * col("work_blind")),
        ("F29a charger_aware fatigue below energy_only",
         col("fat_aware") < col("fat_energy")),
        ("F29b charger_aware fatigue below random",
         col("fat_aware") < col("fat_random")),
    ]
    print(f"\n  ORDINAL ROBUSTNESS across the charging parameter space:")
    print("  " + "-" * 68)
    for label, arr in checks:
        f = float(np.mean(arr))
        mark = "ok  " if f >= 0.95 else ("weak" if f >= 0.70 else "FAIL")
        print(f"  [{mark}] {label:<52} {100*f:5.1f}%")
    print("  " + "-" * 68)

    print(f"\n  Effect magnitudes (median, 5th-95th):")
    print("  " + "-" * 68)
    for label, num, den in [
            ("static_score work vs random", "work_static", "work_random"),
            ("static_score penalty vs random", "pen_static", "pen_random"),
            ("charger_aware penalty vs blind", "pen_aware", "pen_blind"),
            ("charger_aware fast share vs blind", "fast_aware", "fast_blind")]:
        r = col(num) / np.maximum(col(den), 1e-12)
        print(f"  {label:<36} {np.median(r):>7.2f}x  "
              f"[{np.percentile(r,5):>6.2f}, {np.percentile(r,95):>6.2f}]")
    print("  " + "-" * 68)

    # --- standardised regression coefficients -------------------------
    print("\n" + "-" * 78)
    print("SENSITIVITY (standardised regression coefficients)")
    print("-" * 78)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    for target in ["pen_aware", "fat_aware", "susp_static", "work_aware"]:
        y = col(target)
        ys = (y - y.mean()) / (y.std() + 1e-12)
        beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
        r2 = 1.0 - np.sum((ys - Xs @ beta) ** 2) / np.sum(ys ** 2)
        order = np.argsort(-np.abs(beta))[:5]
        flag = "" if r2 > 0.6 else "   <-- LOW R^2, attribution unreliable"
        print(f"\n  {target}   R^2 = {r2:.3f}{flag}")
        for i in order:
            print(f"     {NAMES[i]:<22} {beta[i]:>+7.3f}")
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
