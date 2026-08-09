"""
FMCL Paper 5 -- Coupled thermal / throughput / energy / wear device model.

Four coupled sub-models:

  1. Thermal   : lumped first-order RC, heating under load, passive cooling.
  2. Throughput: governor derating eta(T), piecewise-linear to a floor.
  3. Energy    : leakage rises with T while clock falls, so joules per useful
                 gradient rise under throttling. This is the miscalibration
                 that energy-aware schedulers using nominal power models miss.
  4. Wear      : two mechanisms with different arguments --
                 Arrhenius in absolute battery temperature (dwell),
                 Coffin-Manson in thermal cycling amplitude (excursions).

All state is per-device. Nothing here is FL-specific; the scheduler sits on
top of this.
"""

import numpy as np

KELVIN = 273.15


# ---------------------------------------------------------------------
# 1. Thermal
# ---------------------------------------------------------------------

def steady_state_temp(P, R_th, T_amb):
    """Open-loop steady-state temperature under constant power P."""
    return T_amb + P * R_th


def heat(T0, P, dt, R_th, tau, T_amb):
    """Temperature after dt seconds of load at power P, starting from T0."""
    T_ss = steady_state_temp(P, R_th, T_amb)
    return T_ss + (T0 - T_ss) * np.exp(-dt / tau)


def cool(T0, dt, tau_cool, T_amb):
    """Temperature after dt seconds idle, starting from T0."""
    return T_amb + (T0 - T_amb) * np.exp(-dt / tau_cool)


def time_to_reach(T0, T_target, P, R_th, tau, T_amb):
    """Seconds of load at power P to go from T0 to T_target. inf if unreachable."""
    T_ss = steady_state_temp(P, R_th, T_amb)
    if T_target >= T_ss or T0 >= T_target:
        return np.inf if T_target >= T_ss else 0.0
    return -tau * np.log((T_ss - T_target) / (T_ss - T0))


def tau_from_onset(t_onset, T_cap, P, R_th, T_amb, T0=None):
    """
    Invert the RC solution to recover tau from an observed throttle-onset time.

    t_onset = -tau * ln( (T_ss - T_cap) / (T_ss - T0) )
    """
    T0 = T_amb if T0 is None else T0
    T_ss = steady_state_temp(P, R_th, T_amb)
    if T_cap >= T_ss:
        return np.nan  # device never reaches the cap; onset unexplained
    return t_onset / np.log((T_ss - T0) / (T_ss - T_cap))


# ---------------------------------------------------------------------
# 2. Throughput derating
# ---------------------------------------------------------------------

def eta(T, T_cap, eta_min, slope=0.04):
    """
    Throughput multiplier relative to unthrottled peak.

    Full speed below the governor cap; linear derate above it down to a floor.
    slope has units 1/K. Default 0.04 gives a fall to 0.75 about 6 K above cap,
    consistent with the 20-30% sustained discount Wang et al. report.
    """
    T = np.asarray(T, dtype=float)
    over = np.maximum(T - T_cap, 0.0)
    return np.clip(1.0 - slope * over, eta_min, 1.0)


# ---------------------------------------------------------------------
# 3. Energy under throttling
# ---------------------------------------------------------------------

def power_at(T, eta_val, P_nom, f_static, theta_leak, p_dvfs, T_ref):
    """
    Instantaneous device power, decomposed into leakage and dynamic terms.

    Leakage rises exponentially with temperature and does NOT scale with clock.
    Dynamic power scales as eta^p under DVFS.
    """
    P_stat0 = f_static * P_nom
    P_dyn0 = (1.0 - f_static) * P_nom
    # Clamp the leakage exponent. Beyond ~60 K above reference the model is
    # extrapolating well past any measurement it was calibrated on, and the
    # unclamped term is a positive feedback that diverges numerically.
    P_stat = P_stat0 * np.exp(np.clip((T - T_ref) / theta_leak, -20.0, 4.0))
    P_dyn = P_dyn0 * np.power(eta_val, p_dvfs)
    return P_stat + P_dyn


def energy_penalty(T, T_cap, eta_min, P_nom, f_static, theta_leak,
                   p_dvfs, T_ref, slope=0.04):
    """
    psi(T): joules per unit of useful work at temperature T, relative to the
    same work done at the reference temperature unthrottled.

    psi > 1 means an energy-aware scheduler using a NOMINAL power model
    underestimates the true energy cost of selecting this device.
    """
    e = eta(T, T_cap, eta_min, slope)
    P_hot = power_at(T, e, P_nom, f_static, theta_leak, p_dvfs, T_ref)
    P_ref = power_at(T_ref, 1.0, P_nom, f_static, theta_leak, p_dvfs, T_ref)
    return (P_hot / e) / P_ref


# ---------------------------------------------------------------------
# 4. Wear
# ---------------------------------------------------------------------

def arrhenius_ratio(T_hot_C, T_ref_C, Ea, R=8.314462618):
    """
    Degradation rate at T_hot relative to T_ref. Both temperatures in Celsius.
    """
    Th = np.asarray(T_hot_C, dtype=float) + KELVIN
    Tr = T_ref_C + KELVIN
    return np.exp((Ea / R) * (1.0 / Tr - 1.0 / Th))


def arrhenius_damage(T_trace_C, dt, T_ref_C, Ea, k_batt, R=8.314462618):
    """
    Accumulated calendar-ageing damage over a temperature trace, expressed in
    equivalent-seconds-at-reference-temperature.

    k_batt maps SoC temperature rise to battery cell temperature rise.
    """
    T_soc = np.asarray(T_trace_C, dtype=float)
    T_batt = T_ref_C + k_batt * (T_soc - T_ref_C)
    return float(np.sum(arrhenius_ratio(T_batt, T_ref_C, Ea, R)) * dt)


def cycling_damage(T_trace_C, n_cm, min_amplitude=1.0):
    """
    Coffin-Manson damage from thermal cycling, in arbitrary units proportional
    to accumulated fatigue.

    Cycles are extracted by simple peak-valley alternation on the trace, and
    each half-excursion of amplitude dT contributes dT^n. Excursions below
    min_amplitude are ignored as measurement noise.

    Note: this is a rainflow-lite counter. Full rainflow counting is the
    correct method for irregular loading and should replace this before any
    load-bearing claim; the ordering it produces on the regular batch/idle
    patterns used here is the same.
    """
    T = np.asarray(T_trace_C, dtype=float)
    if T.size < 3:
        return 0.0
    # collapse plateaus, otherwise zero-slope segments break sign detection
    keep = np.concatenate([[True], np.abs(np.diff(T)) > 1e-12])
    T = T[keep]
    if T.size < 3:
        return 0.0
    sign = np.sign(np.diff(T))
    turns = [0]
    for i in range(1, len(sign)):
        if sign[i] != sign[i - 1]:
            turns.append(i)
    turns.append(len(T) - 1)
    ext = T[turns]
    amps = np.abs(np.diff(ext))
    amps = amps[amps >= min_amplitude]
    return float(np.sum(np.power(amps, n_cm)))


# ---------------------------------------------------------------------
# Campaign simulation: a duty-cycle pattern over a fixed round budget
# ---------------------------------------------------------------------

def simulate_campaign(n_rounds, rounds_per_batch, campaign_duration,
                      t_round, P_train, R_th, tau_heat, tau_cool, T_amb,
                      T_cap, eta_min, f_static, theta_leak, p_dvfs,
                      Ea, k_batt, n_cm, dt=1.0, dt_idle=30.0, slope=0.04,
                      _active_time=None):
    """
    Simulate a full training campaign under a given batching pattern.

    Two invariants are enforced so that patterns are comparable:

      (i)  EQUAL WORK  -- n_rounds is fixed. Throttling therefore lengthens
           wall-clock time rather than reducing the work done.
      (ii) EQUAL CALENDAR DURATION -- every pattern spans campaign_duration
           seconds. Idle gaps are sized to fill it, and any residual is padded
           at the end. Without this, a heavily batched pattern would be
           compared over a shorter trace and would appear to accumulate less
           calendar ageing purely because less calendar time had elapsed.

    Wear is reported as MARGINAL over a non-participating device:

      arrhenius_excess = (damage with training) - (damage of the same device
                          idling at ambient for the same duration)

    which is the quantity FMCL's assumption A2 actually concerns. Total
    calendar ageing accrues whether or not the device trains, so charging it
    to the learning task would violate the marginal-allocation principle the
    whole series rests on.
    """
    n_batches = int(np.ceil(n_rounds / rounds_per_batch))
    # Two-pass gap sizing. Throttling lengthens the active phase, so gaps sized
    # from the UNTHROTTLED work estimate leave the campaign duration slightly
    # over-run (0.05% at nominal). Pass 1 measures the true active time, pass 2
    # sizes the gaps against it, so the equal-duration invariant holds exactly.
    if _active_time is None:
        probe = simulate_campaign(
            n_rounds, rounds_per_batch, campaign_duration, t_round, P_train,
            R_th, tau_heat, tau_cool, T_amb, T_cap, eta_min, f_static,
            theta_leak, p_dvfs, Ea, k_batt, n_cm, dt, dt_idle, slope,
            _active_time=n_rounds * t_round)
        _active_time = probe["active_time_s"]
    idle_total = max(campaign_duration - _active_time, 0.0)
    t_gap = idle_total / n_batches   # n_batches gaps: one after each batch

    T = T_amb
    trace = [T]
    dts = [dt]
    total_energy = 0.0
    elapsed = 0.0
    active_time = 0.0
    rounds_done = 0

    for _ in range(n_batches):
        r_this = min(rounds_per_batch, n_rounds - rounds_done)
        # --- active phase: work-conserving, throttling extends wall time ---
        work_remaining = r_this * t_round
        while work_remaining > 1e-9:
            e = float(eta(T, T_cap, eta_min, slope))
            work_step = min(dt * e, work_remaining)
            step = work_step / e
            P = power_at(T, e, P_train, f_static, theta_leak, p_dvfs, T_amb)
            total_energy += float(P) * step
            elapsed += step
            active_time += step
            T = heat(T, P_train, step, R_th, tau_heat, T_amb)
            trace.append(T)
            dts.append(step)
            work_remaining -= work_step
        rounds_done += r_this
        # --- idle phase: cooling is analytic, so evaluate it vectorised ---
        # The step must resolve the decay, not just the gap: most of the
        # marginal Arrhenius damage accrues in the first few tau_cool. A fixed
        # 30 s step under-resolved this by ~9% in library v0.1 (finding F7).
        step_cap = min(dt_idle, tau_cool / 20.0)
        n_steps = max(int(round(t_gap / step_cap)), 1)
        step = t_gap / n_steps
        t_grid = np.arange(1, n_steps + 1) * step
        T_idle = cool(T, t_grid, tau_cool, T_amb)
        trace.append(T_idle)
        dts.append(np.full(n_steps, step))
        elapsed += t_gap
        T = float(T_idle[-1])

    trace = np.concatenate([np.atleast_1d(x) for x in trace])
    dts = np.concatenate([np.atleast_1d(x) for x in dts])

    T_batt = T_amb + k_batt * (trace - T_amb)
    dmg = float(np.sum(arrhenius_ratio(T_batt, T_amb, Ea) * dts))
    baseline = float(np.sum(dts))          # ratio == 1 at ambient
    return {
        "T_peak": float(trace.max()),
        "T_mean": float(np.sum(trace * dts) / np.sum(dts)),
        "energy_J": total_energy,
        "wall_time_s": elapsed,
        "active_time_s": active_time,
        "arrhenius_excess": dmg - baseline,
        "cycling": cycling_damage(trace, n_cm),
        "n_batches": n_batches,
        "t_gap": t_gap,
        "trace": trace,
    }
