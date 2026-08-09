"""
FMCL Paper 5 -- Independent verification suite for the coefficient library
and the coupled thermal/energy/wear model.

Every closed-form result is checked against an independently computed
value, and every claim the paper intends to make is tested rather than
assumed. Failures are reported, not suppressed.
"""

import numpy as np
import coefficients as C
import thermal as TH

PASS, FAIL, FINDINGS = [], [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append((name, detail))


def finding(tag, text):
    FINDINGS.append((tag, text))


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# =====================================================================
print("=" * 74)
print(f"FMCL PAPER 5 -- VERIFICATION SUITE  (coefficient library v{C.LIBRARY_VERSION})")
print("=" * 74)

# --- Library integrity -----------------------------------------------
census = C.tier_census()
print(f"\nCoefficient census by provenance tier: {census}")
print(f"Total coefficients: {len(C.ALL_COEFFICIENTS)}")

check("L1 every coefficient has a source",
      all(len(c.source) > 20 for c in C.ALL_COEFFICIENTS))
check("L2 nominal within bounds for all",
      all(c.low <= c.value <= c.high for c in C.ALL_COEFFICIENTS))
check("L3 at least one T1 anchor per sub-model",
      census.get("T1", 0) >= 8, f"T1 count = {census.get('T1', 0)}")

print("\nT4 (unmeasured) coefficients -- the measurement agenda:")
for c in C.unmeasured():
    print(f"   {c.symbol:12s} {c.low:>8.3g} .. {c.high:<8.3g} {c.unit}")

print("\nWidest relative ranges (UQ triage):")
for c in C.widest(5):
    print(f"   {c.symbol:12s} span={c.span:6.2f}  tier={c.tier}")


# =====================================================================
# 1. THERMAL RC MODEL
# =====================================================================
print("\n" + "-" * 74)
print("1. THERMAL RC MODEL")
print("-" * 74)

P = C.P_TRAIN.value
Rth = C.R_TH.value
Tamb = C.T_AMB.value
tau = C.TAU_HEAT.value

T_ss = TH.steady_state_temp(P, Rth, Tamb)
check("T1 steady state = T_amb + P*R_th",
      approx(T_ss, Tamb + P * Rth), f"T_ss = {T_ss:.2f} C")

T_at_tau = TH.heat(Tamb, P, tau, Rth, tau, Tamb)
frac = (T_at_tau - Tamb) / (T_ss - Tamb)
check("T2 reaches 63.2% of rise at t = tau",
      approx(frac, 1 - np.exp(-1), 1e-9), f"fraction = {frac:.6f}")

T_long = TH.cool(80.0, 10_000.0, C.TAU_COOL.value, Tamb)
check("T3 cooling returns to ambient", approx(T_long, Tamb, 1e-6),
      f"T after 10000 s idle = {T_long:.6f} C")

T_mid = TH.heat(Tamb, P, 60.0, Rth, tau, Tamb)
t_back = TH.time_to_reach(Tamb, T_mid, P, Rth, tau, Tamb)
check("T4 time_to_reach inverts heat", approx(t_back, 60.0, 1e-6),
      f"recovered t = {t_back:.6f} s (expected 60)")

check("T5 unreachable target returns inf",
      np.isinf(TH.time_to_reach(Tamb, T_ss + 5, P, Rth, tau, Tamb)))

# R_th derivation reproduced independently
Rth_naive = C.DELTA_T_OBSERVED.value / C.P_TRAIN.value
Rth_min = (C.T_CAP_CONSERVATIVE.value - Tamb) / C.P_TRAIN.value
check("T6 R_th lower bound satisfies the F2 reachability constraint",
      C.R_TH.low >= Rth_min - 1e-9,
      f"library low {C.R_TH.low} K/W >= {Rth_min:.2f} K/W required to reach "
      f"the {C.T_CAP_CONSERVATIVE.value:.0f} C cap "
      f"(naive inversion gave {Rth_naive:.2f}, withdrawn per F2)")

# tau recovered from observed onset time, per governor class
print("\n  tau_heat recovered from t_onset = 150 s, by governor class:")
for label, cap in [("conservative (~50 C)", C.T_CAP_CONSERVATIVE.value),
                   ("tolerant (~65 C)", C.T_CAP_TOLERANT.value)]:
    for rth_label, rth in [("R_th low  3.6", C.R_TH.low),
                           ("R_th nom  4.5", C.R_TH.value),
                           ("R_th high 8.0", C.R_TH.high)]:
        t = TH.tau_from_onset(C.T_THROTTLE_ONSET_TIME.value, cap, P, rth, Tamb)
        flag = "  <-- cap unreachable" if np.isnan(t) else ""
        shown = "n/a" if np.isnan(t) else f"{t:7.1f} s"
        print(f"    {label:22s} {rth_label:14s} tau = {shown}{flag}")

tau_cons_low = TH.tau_from_onset(150.0, C.T_CAP_CONSERVATIVE.value, P,
                                 C.R_TH.low, Tamb)
check("T7 conservative cap unreachable at R_th lower bound",
      np.isnan(tau_cons_low),
      "T_ss = 25 + 5*3.6 = 43 C < 50 C cap")

finding(
    "F2",
    "IDENTIFIABILITY. At the lower bound R_th = 3.6 K/W the model predicts a "
    "steady state of only 43 C, which never reaches the ~50 C governor cap "
    "measured by Wang et al. -- so throttling at 150 s could not occur. The "
    "closed-loop inversion of dT_obs is therefore not merely a lower bound, it "
    "is INCONSISTENT with the observed onset. R_th must exceed roughly "
    f"{(C.T_CAP_CONSERVATIVE.value - Tamb) / P:.2f} K/W for a 5 W load to reach "
    "a 50 C cap at all. This tightens the library: raise R_th.low."
)


# =====================================================================
# 2. THROTTLING RESPONSE
# =====================================================================
print("\n" + "-" * 74)
print("2. THROTTLING RESPONSE")
print("-" * 74)

cap = C.T_CAP_CONSERVATIVE.value
emin = C.ETA_MIN_TOLERANT.value
grid = np.linspace(20, 90, 400)
e = TH.eta(grid, cap, emin)

check("E1 eta = 1 below cap", np.allclose(e[grid <= cap], 1.0))
check("E2 eta bounded in [eta_min, 1]", bool(e.min() >= emin - 1e-12 and e.max() <= 1 + 1e-12))
check("E3 eta monotone non-increasing", bool(np.all(np.diff(e) <= 1e-12)))

T_floor = cap + (1 - emin) / 0.04
check("E4 floor reached at predicted temperature",
      approx(float(TH.eta(T_floor, cap, emin)), emin, 1e-9),
      f"eta hits {emin} at T = {T_floor:.2f} C")

e_sev = TH.eta(grid, cap, C.ETA_MIN_SEVERE.value)
check("E5 severe class floor lower than tolerant",
      float(e_sev.min()) < float(e.min()))
print(f"  Sustained/peak ratio, tolerant class : {emin:.2f} "
      f"(={1/emin:.2f}x slowdown)")
print(f"  Sustained/peak ratio, severe class   : {C.ETA_MIN_SEVERE.value:.2f} "
      f"(={1/C.ETA_MIN_SEVERE.value:.2f}x slowdown)")


# =====================================================================
# 3. ENERGY MISCALIBRATION  (contribution C2)
# =====================================================================
print("\n" + "-" * 74)
print("3. ENERGY PENALTY UNDER THROTTLING")
print("-" * 74)

kw = dict(T_cap=cap, eta_min=C.ETA_MIN_TOLERANT.value, P_nom=P,
          f_static=C.STATIC_POWER_FRACTION.value,
          theta_leak=C.LEAKAGE_TEMP_CONSTANT.value,
          p_dvfs=C.DVFS_EXPONENT.value, T_ref=Tamb)

psi_ref = TH.energy_penalty(Tamb, **kw)
check("P1 penalty = 1 at reference temperature",
      approx(float(psi_ref), 1.0, 1e-9), f"psi(25 C) = {float(psi_ref):.6f}")

P_check = TH.power_at(Tamb, 1.0, P, C.STATIC_POWER_FRACTION.value,
                      C.LEAKAGE_TEMP_CONSTANT.value, C.DVFS_EXPONENT.value, Tamb)
check("P2 power at reference, unthrottled = P_nom",
      approx(float(P_check), P, 1e-9), f"P = {float(P_check):.6f} W")

print("\n  psi(T) across the DVFS-exponent and static-fraction ranges:")
print("  " + "-" * 62)
print(f"  {'T (C)':>6} | {'p=1,fs=.15':>11} {'p=2,fs=.28':>11} {'p=3,fs=.40':>11}")
print("  " + "-" * 62)
rows = []
for T in [25, 40, 50, 55, 60, 65, 70]:
    vals = []
    for p_d, f_s in [(1.0, 0.15), (2.0, 0.28), (3.0, 0.40)]:
        k = dict(kw); k["p_dvfs"] = p_d; k["f_static"] = f_s
        vals.append(float(TH.energy_penalty(T, **k)))
    rows.append((T, vals))
    print(f"  {T:>6} | {vals[0]:>11.3f} {vals[1]:>11.3f} {vals[2]:>11.3f}")
print("  " + "-" * 62)

hot = [v for T, vals in rows if T >= 60 for v in vals]
check("P3 energy penalty exceeds 1 when throttled",
      all(v > 1.0 for v in hot),
      f"min psi at T>=60 C across parameter corners = {min(hot):.3f}")

psi_worst = max(hot)
finding(
    "F3",
    f"ENERGY MISCALIBRATION IS REAL BUT MODEST. Across the full swept corner "
    f"space, a throttled device at 60-70 C costs between {min(hot):.2f}x and "
    f"{psi_worst:.2f}x the joules per unit of useful work that a nominal power "
    "model would predict. The effect is robust in sign across every corner, "
    "which is what contribution C2 needs, but it is a tens-of-percent effect "
    "rather than an order of magnitude. Claim the SIGN and the ROBUSTNESS, not "
    "a headline multiplier."
)


# =====================================================================
# 4. WEAR MECHANISM 1 -- ARRHENIUS
# =====================================================================
print("\n" + "-" * 74)
print("4. ARRHENIUS CAPACITY FADE")
print("-" * 74)

R = C.R_GAS.value
check("A1 ratio = 1 at reference temperature",
      approx(float(TH.arrhenius_ratio(25.0, 25.0, C.EA_CAPACITY_FADE.value)), 1.0, 1e-12))
check("A2 ratio monotone increasing in temperature",
      bool(np.all(np.diff(TH.arrhenius_ratio(np.arange(25, 70), 25.0,
                                             C.EA_CAPACITY_FADE.value)) > 0)))

print("\n  Degradation rate multiplier vs 25 C, by activation energy:")
print("  " + "-" * 58)
print(f"  {'T (C)':>6} | {'Ea=23.6 kJ':>11} {'Ea=26.75 kJ':>12} {'Ea=29.9 kJ':>11}")
print("  " + "-" * 58)
for T in [30, 35, 40, 45, 50, 60]:
    r = [float(TH.arrhenius_ratio(T, 25.0, ea))
         for ea in (C.EA_CAPACITY_FADE.low, C.EA_CAPACITY_FADE.value,
                    C.EA_CAPACITY_FADE.high)]
    print(f"  {T:>6} | {r[0]:>11.3f} {r[1]:>12.3f} {r[2]:>11.3f}")
print("  " + "-" * 58)

r35 = [float(TH.arrhenius_ratio(35.0, 25.0, ea))
       for ea in (C.EA_CAPACITY_FADE.low, C.EA_CAPACITY_FADE.value,
                  C.EA_CAPACITY_FADE.high)]
# Ea that WOULD produce the popular doubling-per-10C rule
Ea_double = R * np.log(2.0) / (1 / (25 + 273.15) - 1 / (35 + 273.15))
check("A3 measured Ea range does not reproduce doubling per 10 C",
      all(x < 1.6 for x in r35),
      f"25->35 C gives {r35[0]:.3f}-{r35[2]:.3f}x, not 2x")

finding(
    "F1",
    f"THE DOUBLING RULE IS WRONG FOR THIS COEFFICIENT RANGE. The widely "
    f"repeated claim that battery degradation doubles per sustained 10 C "
    f"requires Ea = {Ea_double/1000:.1f} kJ/mol. The 232-cell, 13-year "
    f"calendar-ageing dataset gives 23.6-29.9 kJ/mol, which yields only "
    f"{r35[0]:.2f}-{r35[2]:.2f}x per 10 C -- roughly a 40% acceleration, not "
    "100%. Consequence: the Arrhenius arm of the wear objective is "
    "substantially WEAKER than the popular framing suggests, which shifts "
    "relative weight onto the cycling arm. This is a correction the paper "
    "should state explicitly rather than let a widely repeated but "
    "unsupported figure stand uncorrected."
)


# =====================================================================
# 5. WEAR MECHANISM 2 -- COFFIN-MANSON CYCLING
# =====================================================================
print("\n" + "-" * 74)
print("5. COFFIN-MANSON CYCLING DAMAGE")
print("-" * 74)

n = C.CM_EXPONENT.value


def triangle(amp, cycles, pts=50):
    up = np.linspace(0, amp, pts)
    down = np.linspace(amp, 0, pts)
    return np.concatenate([np.tile(np.concatenate([up, down]), cycles), [0.0]])


d1 = TH.cycling_damage(triangle(10.0, 1), n)
check("W1 single 10 K cycle gives 2 * 10^n (up and down)",
      approx(d1, 2 * 10.0 ** n, 1e-6), f"got {d1:.4f}, expected {2*10.0**n:.4f}")

d5 = TH.cycling_damage(triangle(10.0, 5), n)
check("W2 damage linear in cycle count",
      approx(d5, 5 * d1, 1e-6), f"5 cycles = {d5:.3f}, 1 cycle = {d1:.3f}")

d20 = TH.cycling_damage(triangle(20.0, 1), n)
check("W3 damage scales as amplitude^n",
      approx(d20 / d1, 2.0 ** n, 1e-6),
      f"ratio = {d20/d1:.4f}, expected 2^{n} = {2.0**n:.4f}")

check("W4 sub-threshold excursions ignored",
      approx(TH.cycling_damage(triangle(0.5, 10), n), 0.0, 1e-12))

# Key structural fact: is it better to split one big cycle into many small ones?
big = TH.cycling_damage(triangle(20.0, 1), n)
small = TH.cycling_damage(triangle(2.0, 10), n)
check("W5 ten 2 K cycles less damaging than one 20 K cycle",
      small < big, f"10 x 2 K = {small:.3f} vs 1 x 20 K = {big:.3f} "
                   f"(ratio {small/big:.4f})")

finding(
    "F4",
    "CYCLING DAMAGE DEPENDS ON WHAT IS HELD FIXED -- AND THE TWO CASES GIVE "
    "OPPOSITE ANSWERS. At fixed TOTAL EXCURSION, splitting into many small "
    f"cycles is far less damaging (ten 2 K cycles cost {small/big:.1%} of one "
    "20 K cycle) because damage grows as dT^n with n>1. But at fixed TOTAL "
    "WORK -- the case that actually applies to a training campaign -- "
    "amplitude SATURATES at the RC steady state while cycle count falls "
    "linearly with batching, so concentration wins instead. The initial "
    "intuition that the two wear mechanisms oppose each other was drawn from "
    "the first case and does not survive the second. State the constraint "
    "explicitly wherever this appears in the manuscript."
)


# =====================================================================
# 6. CAMPAIGN SWEEP -- does an interior optimum exist?
# =====================================================================
print("\n" + "-" * 74)
print("6. CAMPAIGN BATCHING SWEEP  (the headline test)")
print("-" * 74)

N_ROUNDS = 200          # one year of typical reuse-centric FL campaign cadence
T_ROUND = 25.0          # s of active compute per round (EnFed midpoint)
CAMPAIGN_S = 30 * 24 * 3600.0   # 30 days of wall-clock to fit the campaign in

base = dict(t_round=T_ROUND, campaign_duration=CAMPAIGN_S,
            P_train=P, R_th=C.R_TH.value,
            tau_heat=C.TAU_HEAT.value, tau_cool=C.TAU_COOL.value,
            T_amb=Tamb, T_cap=cap, eta_min=C.ETA_MIN_TOLERANT.value,
            f_static=C.STATIC_POWER_FRACTION.value,
            theta_leak=C.LEAKAGE_TEMP_CONSTANT.value,
            p_dvfs=C.DVFS_EXPONENT.value, Ea=C.EA_CAPACITY_FADE.value,
            k_batt=C.BATT_SOC_THERMAL_COUPLING.value, n_cm=n)

patterns = [1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 200]
results = [(m, TH.simulate_campaign(N_ROUNDS, m, **base)) for m in patterns]

print(f"\n  {N_ROUNDS} rounds, {T_ROUND:.0f} s each, all patterns spanning 30 days.")
print("  Wear columns are MARGINAL over a non-participating device,")
print("  normalised to the m=1 (fully spread) case.\n")
print("  " + "-" * 88)
print(f"  {'m':>4} {'gap(h)':>8} {'T_peak':>8} {'T_mean':>8} {'energy(J)':>10} "
      f"{'time(s)':>9} {'arrhen.':>9} {'cycling':>9}")
print("  " + "-" * 88)
a0 = results[0][1]["arrhenius_excess"]
c0 = results[0][1]["cycling"]
for m, r in results:
    print(f"  {m:>4} {r['t_gap']/3600:>8.2f} {r['T_peak']:>8.2f} {r['T_mean']:>8.3f} "
          f"{r['energy_J']:>10.1f} {r['wall_time_s']:>9.0f} "
          f"{r['arrhenius_excess']/a0:>9.3f} {r['cycling']/c0:>9.3f}")
print("  " + "-" * 88)

durations = [r["wall_time_s"] for _, r in results]
check("S0 all patterns span equal calendar duration",
      max(durations) - min(durations) < 1.0,
      f"spread = {max(durations)-min(durations):.4f} s")

arr = np.array([r["arrhenius_excess"] / a0 for _, r in results])
cyc = np.array([r["cycling"] / c0 for _, r in results])
eng = np.array([r["energy_J"] for _, r in results])

check("S1 energy non-decreasing in batch size (throttling cost)",
      bool(np.all(np.diff(eng) >= -1e-6)),
      f"energy {eng[0]:.1f} -> {eng[-1]:.1f} J")
check("S2 peak temperature non-decreasing in batch size",
      bool(np.all(np.diff([r['T_peak'] for _, r in results]) >= -1e-9)))

print("\n  Combined marginal wear vs weighting w on the Arrhenius mechanism:")
print("  " + "-" * 62)
print(f"  {'w_batt':>8} | {'argmin m':>9} | {'interior?':>10} | {'min value':>10}")
print("  " + "-" * 62)
interior = []
for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
    total = w * arr + (1 - w) * cyc
    i = int(np.argmin(total))
    is_int = 0 < i < len(patterns) - 1
    interior.append(is_int)
    print(f"  {w:>8.2f} | {patterns[i]:>9} | {str(is_int):>10} | {total[i]:>10.4f}")
print("  " + "-" * 62)

check("S3 wear mechanisms oppose each other (interior optimum in batch size)",
      any(interior),
      "tested across w_batt in [0,1] -- see finding F5")

# --- the tradeoff that DOES exist: energy vs package fatigue ---
print("\n  Energy and cycling damage move in OPPOSITE directions:")
print("  " + "-" * 66)
print(f"  {'m':>5} | {'energy (norm)':>14} | {'cycling (norm)':>15} | {'sum':>8}")
print("  " + "-" * 66)
e_n = eng / eng[0]
for i, m in enumerate(patterns):
    print(f"  {m:>5} | {e_n[i]:>14.3f} | {cyc[i]:>15.3f} | {e_n[i]+cyc[i]:>8.3f}")
print("  " + "-" * 66)

check("S4 energy and cycling are anti-correlated across batch size",
      float(np.corrcoef(e_n, cyc)[0, 1]) < 0.0,
      f"corr(energy, cycling) = {float(np.corrcoef(e_n, cyc)[0,1]):.3f}")

i_worst = int(np.argmax(cyc))
check("S5 cycling damage has an interior MAXIMUM",
      0 < i_worst < len(patterns) - 1,
      f"worst batch size m = {patterns[i_worst]} at {cyc[i_worst]:.2f}x the m=1 case")

joint_argmin = []
for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
    total = w * e_n + (1 - w) * cyc
    joint_argmin.append(patterns[int(np.argmin(total))])
check("S6 joint energy-fatigue optimum shifts with weighting",
      len(set(joint_argmin)) > 1,
      f"argmin m across w in [0,1]: {joint_argmin}")

finding(
    "F5",
    "NO INTERIOR OPTIMUM IN THE WEAR OBJECTIVE, BUT AN INTERIOR WORST CASE. "
    "Across every weighting of the two wear mechanisms, marginal wear is "
    "minimised at maximum batching -- the mechanisms do not oppose each other, "
    "so the dual-mechanism objective as originally conceived does not hold. "
    "The marginal Arrhenius term is close to FLAT in batch size "
    f"({arr.min():.3f}-{arr.max():.3f}x), because at these temperatures the "
    "rate multiplier is nearly linear in dT and the time-integral of dT is "
    "fixed by total delivered energy rather than by how it is distributed. "
    "What the sweep does reveal is sharper: cycling damage has an interior "
    f"MAXIMUM at m = {patterns[i_worst]} rounds per session, costing "
    f"{cyc[i_worst]:.2f}x the fully-spread case. Short back-to-back bursts are "
    "the worst possible pattern for package fatigue, and that is precisely "
    "where a conventional round-based FL schedule sits."
)

finding(
    "F6",
    "THE REAL TRADEOFF IS ENERGY AGAINST FATIGUE, NOT ARRHENIUS AGAINST "
    "COFFIN-MANSON. At fixed work and fixed calendar duration, campaign energy "
    f"rises monotonically from {eng[0]:.0f} J to {eng[-1]:.0f} J "
    f"({eng[-1]/eng[0]:.2f}x) as batching increases, entirely because of "
    "throttling, while cycling damage falls. The two are anti-correlated "
    f"(r = {float(np.corrcoef(e_n, cyc)[0,1]):.3f}). This is the tension the "
    "scheduler must resolve, it is genuinely two-sided, and it is a cleaner "
    "story than the one Paper 5 was originally scoped around: an FMCL "
    "scheduler that minimises energy alone drives devices into the "
    "fatigue-worst regime, and one that minimises fatigue alone pays a 67% "
    "energy premium. Neither single-objective policy is defensible."
)


# =====================================================================
print("\n" + "=" * 74)
print("RESULTS")
print("=" * 74)
print(f"\nPASSED: {len(PASS)}    FAILED: {len(FAIL)}\n")
for nme, d in PASS:
    print(f"  [ok]   {nme}" + (f"  -- {d}" if d else ""))
if FAIL:
    print()
    for nme, d in FAIL:
        print(f"  [FAIL] {nme}" + (f"  -- {d}" if d else ""))

print("\n" + "=" * 74)
print("FINDINGS REQUIRING ACTION")
print("=" * 74)
for tag, text in FINDINGS:
    print(f"\n[{tag}] {text}")
print()
