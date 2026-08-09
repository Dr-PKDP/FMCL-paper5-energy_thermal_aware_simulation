"""
FMCL Paper 5 -- Uncertainty quantification.

Two jobs:

  1. A fast campaign model, validated against the full step-wise simulation in
     thermal.py, cheap enough for tens of thousands of evaluations.
  2. Monte Carlo + Sobol over the coefficient library, testing whether the six
     findings in COEFFICIENTS_v0.1.md survive coefficient uncertainty.

The standing rule from Paper 3 applies: claims are reported as ORDINAL
invariances across the parameter space, not as point estimates.

The fast model exploits a structural fact about the campaign patterns being
compared: inter-batch gaps run to hours while tau_cool is order 100 s, so
every batch begins from ambient and all batches within a campaign are
identical. One batch therefore determines the whole campaign.
"""

import numpy as np
import coefficients as C
import thermal as TH

RNG = np.random.default_rng(20260802)


# ---------------------------------------------------------------------
# Fast campaign model
# ---------------------------------------------------------------------

def fast_campaign(n_rounds, m, t_round, P_train, R_th, tau_heat, tau_cool,
                  T_amb, T_cap, eta_min, f_static, theta_leak, p_dvfs,
                  Ea, k_batt, n_cm, dt=1.0, slope=0.04):
    """
    Campaign outcome under batching pattern m, assuming gaps long enough that
    every batch starts from ambient. Simulates ONE batch and scales.
    """
    n_batches = int(np.ceil(n_rounds / m))
    rounds_in_batch = n_rounds / n_batches   # fractional, keeps work exact

    # --- one batch, active phase ---
    T = T_amb
    work_remaining = rounds_in_batch * t_round
    energy = 0.0
    T_peak = T_amb
    arr_active = 0.0
    a_ref = 1.0
    T_prev = -1e9
    while work_remaining > 1e-9:
        e = float(TH.eta(T, T_cap, eta_min, slope))
        # Once the device has settled, temperature and eta are constant, so the
        # remainder of the batch can be completed in closed form.
        if abs(T - T_prev) < 1e-5 and work_remaining > dt * e:
            step = work_remaining / e
            P = float(TH.power_at(T, e, P_train, f_static, theta_leak,
                                  p_dvfs, T_amb))
            energy += P * step
            T_b = T_amb + k_batt * (T - T_amb)
            arr_active += (float(TH.arrhenius_ratio(T_b, T_amb, Ea)) - a_ref) * step
            T = TH.heat(T, P_train, step, R_th, tau_heat, T_amb)
            T_peak = max(T_peak, T)
            break
        T_prev = T
        work_step = min(dt * e, work_remaining)
        step = work_step / e
        P = float(TH.power_at(T, e, P_train, f_static, theta_leak, p_dvfs, T_amb))
        energy += P * step
        T_b = T_amb + k_batt * (T - T_amb)
        arr_active += (float(TH.arrhenius_ratio(T_b, T_amb, Ea)) - a_ref) * step
        T = TH.heat(T, P_train, step, R_th, tau_heat, T_amb)
        T_peak = max(T_peak, T)
        work_remaining -= work_step

    # --- decay tail, integrated to 10 tau_cool (residual < 5e-5 of peak) ---
    n_tail = 400
    dt_tail = 10.0 * tau_cool / n_tail
    t_tail = (np.arange(n_tail) + 0.5) * dt_tail
    T_tail = TH.cool(T_peak, t_tail, tau_cool, T_amb)
    T_tail_b = T_amb + k_batt * (T_tail - T_amb)
    arr_tail = float(np.sum(TH.arrhenius_ratio(T_tail_b, T_amb, Ea) - 1.0) * dt_tail)

    dT = T_peak - T_amb
    return {
        "T_peak": T_peak,
        "energy_J": energy * n_batches,
        "cycling": n_batches * 2.0 * (dT ** n_cm if dT >= 1.0 else 0.0),
        "arrhenius_excess": (arr_active + arr_tail) * n_batches,
        "n_batches": n_batches,
    }


# ---------------------------------------------------------------------
# Validation of the fast model against the full simulation
# ---------------------------------------------------------------------

N_ROUNDS, T_ROUND = 200, 25.0
CAMPAIGN_S = 30 * 24 * 3600.0
PATTERNS = [1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 200]

NOMINAL = dict(
    t_round=T_ROUND, P_train=C.P_TRAIN.value, R_th=C.R_TH.value,
    tau_heat=C.TAU_HEAT.value, tau_cool=C.TAU_COOL.value, T_amb=C.T_AMB.value,
    T_cap=C.T_CAP_CONSERVATIVE.value, eta_min=C.ETA_MIN_TOLERANT.value,
    f_static=C.STATIC_POWER_FRACTION.value,
    theta_leak=C.LEAKAGE_TEMP_CONSTANT.value, p_dvfs=C.DVFS_EXPONENT.value,
    Ea=C.EA_CAPACITY_FADE.value, k_batt=C.BATT_SOC_THERMAL_COUPLING.value,
    n_cm=C.CM_EXPONENT.value,
)


def validate_fast_model(verbose=True):
    """Compare fast model against the full step-wise simulation at nominal."""
    rows, worst = [], 0.0
    for m in PATTERNS:
        full = TH.simulate_campaign(N_ROUNDS, m, campaign_duration=CAMPAIGN_S,
                                    **NOMINAL)
        fast = fast_campaign(N_ROUNDS, m, **NOMINAL)
        errs = {}
        for key in ("energy_J", "cycling", "arrhenius_excess", "T_peak"):
            a, b = full[key], fast[key]
            errs[key] = abs(a - b) / max(abs(a), 1e-12)
            worst = max(worst, errs[key])
        rows.append((m, full, fast, errs))
    if verbose:
        print("  " + "-" * 72)
        print(f"  {'m':>4} | {'energy':>9} {'cycling':>9} {'arrhen.':>9} "
              f"{'T_peak':>9}   (relative error)")
        print("  " + "-" * 72)
        for m, _, _, e in rows:
            print(f"  {m:>4} | {e['energy_J']:>9.2e} {e['cycling']:>9.2e} "
                  f"{e['arrhenius_excess']:>9.2e} {e['T_peak']:>9.2e}")
        print("  " + "-" * 72)
    return worst, rows


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------

UNCERTAIN = [
    ("P_train", C.P_TRAIN), ("R_th", C.R_TH), ("tau_heat", C.TAU_HEAT),
    ("tau_cool", C.TAU_COOL), ("T_amb", C.T_AMB),
    ("T_cap", C.T_CAP_CONSERVATIVE), ("eta_min", C.ETA_MIN_TOLERANT),
    ("f_static", C.STATIC_POWER_FRACTION),
    ("theta_leak", C.LEAKAGE_TEMP_CONSTANT), ("p_dvfs", C.DVFS_EXPONENT),
    ("Ea", C.EA_CAPACITY_FADE), ("k_batt", C.BATT_SOC_THERMAL_COUPLING),
    ("n_cm", C.CM_EXPONENT), ("t_round", C.T_ROUND),
]
NAMES = [n for n, _ in UNCERTAIN]
BOUNDS = [[c.low, c.high] for _, c in UNCERTAIN]


def evaluate(params):
    """
    Run the full pattern sweep for one parameter draw and return the four
    quantities the findings depend on.
    """
    kw = dict(zip(NAMES, params))
    kw["n_cm"] = float(kw["n_cm"])
    out = [fast_campaign(N_ROUNDS, m, **kw) for m in PATTERNS]
    eng = np.array([o["energy_J"] for o in out])
    cyc = np.array([o["cycling"] for o in out])
    arr = np.array([o["arrhenius_excess"] for o in out])

    eng_n = eng / eng[0] if eng[0] > 0 else eng
    cyc_n = cyc / cyc[0] if cyc[0] > 0 else np.zeros_like(cyc)
    i_worst = int(np.argmax(cyc_n))
    interior_max = 1 if 0 < i_worst < len(PATTERNS) - 1 else 0
    corr = float(np.corrcoef(eng_n, cyc_n)[0, 1]) if np.std(cyc_n) > 0 else 0.0
    arr_spread = (arr.max() - arr.min()) / max(arr.max(), 1e-30)

    return {
        "energy_premium": float(eng[-1] / eng[0]),
        "worst_m": float(PATTERNS[i_worst]),
        "worst_cost": float(cyc_n[i_worst]),
        "interior_max": float(interior_max),
        "corr_energy_cycling": corr,
        "arrhenius_spread": float(arr_spread),
        "energy_monotone": float(np.all(np.diff(eng) >= -1e-9)),
    }


def monte_carlo(n=2000):
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    X = RNG.uniform(lo, hi, size=(n, len(BOUNDS)))
    return X, [evaluate(x) for x in X]


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def pct(x):
    return f"{100.0 * x:5.1f}%"


def main():
    print("=" * 74)
    print("FMCL PAPER 5 -- UNCERTAINTY QUANTIFICATION")
    print(f"coefficient library v{C.LIBRARY_VERSION} (R_th corrected per F2)")
    print("=" * 74)

    print("\n" + "-" * 74)
    print("0. FAST MODEL VALIDATION against full step-wise simulation")
    print("-" * 74)
    worst, _ = validate_fast_model()
    ok = worst < 0.05
    print(f"  worst relative error across all outputs and patterns: {worst:.3%}")
    print(f"  [{'ok' if ok else 'FAIL'}] fast model {'accepted' if ok else 'REJECTED'} "
          f"for Monte Carlo use (threshold 5%)")
    if not ok:
        print("  Aborting: fast model not accurate enough to substitute.")
        return

    print("\n" + "-" * 74)
    print("1. MONTE CARLO over 14 uncertain coefficients")
    print("-" * 74)
    N_MC = 2000
    X, res = monte_carlo(N_MC)
    print(f"  {N_MC} draws, uniform over each coefficient's plausible range.\n")

    def col(k):
        return np.array([r[k] for r in res])

    prem = col("energy_premium")
    corr = col("corr_energy_cycling")
    imax = col("interior_max")
    wm = col("worst_m")
    wc = col("worst_cost")
    aspread = col("arrhenius_spread")
    mono = col("energy_monotone")

    print("  " + "-" * 68)
    print(f"  {'quantity':<34} {'median':>10} {'5th':>10} {'95th':>10}")
    print("  " + "-" * 68)
    for label, v in [("energy premium (batch/spread)", prem),
                     ("corr(energy, cycling)", corr),
                     ("worst-case batch size m*", wm),
                     ("cost at m* (x spread case)", wc),
                     ("Arrhenius spread across m", aspread)]:
        print(f"  {label:<34} {np.median(v):>10.3f} "
              f"{np.percentile(v, 5):>10.3f} {np.percentile(v, 95):>10.3f}")
    print("  " + "-" * 68)

    print("\n  ORDINAL ROBUSTNESS -- fraction of draws in which each finding holds:")
    print("  " + "-" * 68)
    checks = [
        ("F6a energy monotone increasing in batch size", float(np.mean(mono))),
        ("F6b energy premium > 1 (throttling costs energy)", float(np.mean(prem > 1.0))),
        ("F6c energy and cycling anti-correlated", float(np.mean(corr < 0.0))),
        ("F5a cycling has an interior maximum", float(np.mean(imax))),
        ("F5b worst-case m* in [2, 20]", float(np.mean((wm >= 2) & (wm <= 20)))),
        ("F5c Arrhenius near-flat (spread < 25%)", float(np.mean(aspread < 0.25))),
    ]
    for label, frac in checks:
        mark = "ok  " if frac >= 0.95 else ("weak" if frac >= 0.70 else "FAIL")
        print(f"  [{mark}] {label:<48} {pct(frac)}")
    print("  " + "-" * 68)

    print("\n  Distribution of the fatigue-worst batch size m*:")
    vals, counts = np.unique(wm, return_counts=True)
    for v, c in zip(vals, counts):
        bar = "#" * int(round(50 * c / len(wm)))
        print(f"    m* = {int(v):>3}  {pct(c/len(wm))}  {bar}")

    # -----------------------------------------------------------------
    print("\n" + "-" * 74)
    print("2. SOBOL SENSITIVITY")
    print("-" * 74)
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze

    problem = {"num_vars": len(NAMES), "names": NAMES, "bounds": BOUNDS}
    Xs = sobol_sample.sample(problem, 256, calc_second_order=False)
    print(f"  Saltelli design: {Xs.shape[0]} evaluations")
    Ys = [evaluate(x) for x in Xs]

    for target, label in [("energy_premium", "ENERGY PREMIUM"),
                          ("worst_cost", "COST AT THE FATIGUE-WORST BATCH SIZE"),
                          ("corr_energy_cycling", "ENERGY-CYCLING CORRELATION")]:
        y = np.array([r[target] for r in Ys])
        if np.std(y) < 1e-12:
            print(f"\n  {label}: constant across the parameter space "
                  f"(value {y[0]:.4f}) -- no variance to decompose.")
            continue
        Si = sobol_analyze.analyze(problem, y, calc_second_order=False,
                                   print_to_console=False)
        order = np.argsort(-np.array(Si["ST"]))
        print(f"\n  {label}   (mean {y.mean():.3f}, sd {y.std():.3f})")
        print("  " + "-" * 52)
        print(f"  {'parameter':<14} {'S1':>10} {'ST':>10}")
        print("  " + "-" * 52)
        for i in order[:7]:
            print(f"  {NAMES[i]:<14} {Si['S1'][i]:>10.3f} {Si['ST'][i]:>10.3f}")
        print("  " + "-" * 52)
        neg = [NAMES[i] for i in range(len(NAMES)) if Si["ST"][i] < 0.01]
        if neg:
            print(f"  Negligible (ST < 0.01): {', '.join(neg)}")

    print("\n" + "=" * 74)
    print("Interpretation belongs in COEFFICIENTS_v0.2.md, not here.")
    print("=" * 74)


if __name__ == "__main__":
    main()
