"""
FMCL Paper 5 -- Thermal-aware scheduling.

PROBLEM
-------
N reused consumer devices, T federated rounds, K participants per round.
Selecting device i for a round costs energy that depends on its CURRENT
temperature (throttled devices burn more joules per unit of useful work) and
raises that temperature; not selecting it lets it cool. Device state is
therefore path-dependent in the scheduler's own past decisions -- an arm that
degrades when pulled and recovers when rested.

    minimise   long-run average campaign energy
    subject to long-run average marginal wear per device <= W_budget
               K participants per round
               participation only from available devices

This is a restless multi-armed bandit with recovering arms. Two policies are
derived for it:

  WHITTLE   Solve the single-arm subsidy problem on a discretised temperature
            grid by value iteration, binary-searching the subsidy that makes
            passivity indifferent. Indexability is checked numerically rather
            than assumed. Requires knowing the arm dynamics.

  LYAPUNOV  Drift-plus-penalty with a per-device virtual wear-debt queue.
            Requires no transition model; the queue backlog is a stochastic
            estimate of the same Lagrange multiplier the Whittle subsidy
            represents, which is why the two should agree.

Baselines: random, Paper 1's static linear score (thermally blind, uses a
NOMINAL power model), and greedy energy-only (true instantaneous cost, but no
forward cost of heating).
"""

import numpy as np
import thermal as TH
import coefficients as C


# =====================================================================
# Device classes
# =====================================================================

DEVICE_CLASSES = {
    # label:        R_th, tau_heat, tau_cool, T_cap, eta_min, P_train
    "flagship":   dict(R_th=5.2, tau_heat=110.0, tau_cool=150.0,
                       T_cap=52.0, eta_min=0.80, P_train=6.0),
    "midrange":   dict(R_th=6.0, tau_heat=90.0, tau_cool=135.0,
                       T_cap=50.0, eta_min=0.75, P_train=5.0),
    "lowend":     dict(R_th=7.5, tau_heat=70.0, tau_cool=120.0,
                       T_cap=48.0, eta_min=0.35, P_train=4.0),
}

SHARED = dict(T_amb=C.T_AMB.value,
              f_static=C.STATIC_POWER_FRACTION.value,
              theta_leak=C.LEAKAGE_TEMP_CONSTANT.value,
              p_dvfs=C.DVFS_EXPONENT.value,
              Ea=C.EA_CAPACITY_FADE.value,
              k_batt=C.BATT_SOC_THERMAL_COUPLING.value,
              n_cm=C.CM_EXPONENT.value)


class Fleet:
    """A heterogeneous population of reused consumer devices."""

    def __init__(self, n_devices, t_round, t_gap, rng,
                 mix=(("flagship", 0.25), ("midrange", 0.50), ("lowend", 0.25)),
                 p_avail=0.6):
        self.n = n_devices
        self.t_round = t_round
        self.t_gap = t_gap
        self.rng = rng
        self.p_avail = p_avail
        labels, probs = zip(*mix)
        self.cls = rng.choice(labels, size=n_devices, p=probs)
        self.par = [dict(DEVICE_CLASSES[c], **SHARED) for c in self.cls]
        self.T = np.full(n_devices, SHARED["T_amb"], dtype=float)
        # data utility, fixed per device (stand-in for divergence of the local
        # distribution from the global one; replaced by real utilities when the
        # FedProx engine is wired in)
        self.utility = rng.uniform(0.4, 1.0, size=n_devices)
        self.reset_stats()

    def reset_stats(self):
        T_amb = SHARED["T_amb"]
        self.T[:] = T_amb
        self.energy = np.zeros(self.n)
        self.arrhenius = np.zeros(self.n)
        self.selections = np.zeros(self.n, dtype=int)
        self.session_peak = np.full(self.n, T_amb)
        self.in_session = np.zeros(self.n, dtype=bool)
        self.T_max_ever = np.full(self.n, T_amb)
        self.cycling = np.zeros(self.n)
        self.violation_s = np.zeros(self.n)
        self.wall_time = 0.0

    # ---- per-device physics -----------------------------------------
    def round_cost(self, i, T=None):
        """Energy (J) and duration (s) of one round on device i at temp T."""
        p = self.par[i]
        T = self.T[i] if T is None else T
        e = float(TH.eta(T, p["T_cap"], p["eta_min"]))
        dur = self.t_round / e
        P = float(TH.power_at(T, e, p["P_train"], p["f_static"],
                              p["theta_leak"], p["p_dvfs"], p["T_amb"]))
        return P * dur, dur

    def next_temp(self, i, T=None):
        p = self.par[i]
        T = self.T[i] if T is None else T
        _, dur = self.round_cost(i, T)
        return TH.heat(T, p["P_train"], dur, p["R_th"], p["tau_heat"], p["T_amb"])

    def marginal_wear(self, i, T=None):
        """
        Marginal wear of running one more round on device i now.

        Cycling: damage of a session is 2*(peak rise)^n, so one more round adds
        the EXACT increment 2*[(new peak)^n - (old peak)^n], not the derivative
        n*dT^(n-1)*delta. The distinction matters at the boundary: the
        derivative vanishes at ambient, so cycle INITIATION appears free and a
        greedy policy rotates devices into many small excursions -- landing the
        fleet in the fatigue-worst duty cycle identified as F9. The exact
        increment charges a fresh excursion its full (delta)^n on the first
        round of a session. Recorded as finding F12.

        Arrhenius: dwell contribution over the round.
        """
        p = self.par[i]
        T = self.T[i] if T is None else T
        T_next = self.next_temp(i, T)
        _, dur = self.round_cost(i, T)
        peak_now = max(self.session_peak[i], T) if self.in_session[i] else T
        dT_old = max(peak_now - p["T_amb"], 0.0)
        dT_new = max(max(T_next, peak_now) - p["T_amb"], 0.0)
        cyc = 2.0 * (dT_new ** p["n_cm"] - dT_old ** p["n_cm"])
        T_b = p["T_amb"] + p["k_batt"] * (T - p["T_amb"])
        arr = (float(TH.arrhenius_ratio(T_b, p["T_amb"], p["Ea"])) - 1.0) * dur
        return cyc, arr

    # ---- stepping ----------------------------------------------------
    def available(self):
        return self.rng.random(self.n) < self.p_avail

    def step(self, chosen):
        """
        Advance one synchronous round: chosen devices train, everyone cools.

        The round ends when the slowest selected device finishes, so a throttled
        straggler extends the idle time of every other device. Chosen device i
        is active for dur_i, then idle for (round_time - dur_i) + t_gap;
        unselected devices are idle for the whole round_time + t_gap.
        """
        chosen = np.asarray(chosen, dtype=int)
        dur_of = {}
        for i in chosen:
            i = int(i)
            p = self.par[i]
            E, dur = self.round_cost(i)
            cyc, arr = self.marginal_wear(i)
            self.energy[i] += E
            self.arrhenius[i] += arr
            self.selections[i] += 1
            if self.T[i] > C.T_SKIN_CAP.value:
                self.violation_s[i] += dur
            self.T[i] = self.next_temp(i)
            self.session_peak[i] = max(self.session_peak[i], self.T[i])
            self.T_max_ever[i] = max(self.T_max_ever[i], self.T[i])
            self.in_session[i] = True
            dur_of[i] = dur

        round_time = max(dur_of.values()) if dur_of else 0.0

        for i in range(self.n):
            p = self.par[i]
            idle = (round_time - dur_of[i] if i in dur_of else round_time) + self.t_gap
            self.T[i] = TH.cool(self.T[i], max(idle, 0.0),
                                p["tau_cool"], p["T_amb"])
            # close a session once the device has essentially returned to base
            if self.in_session[i] and self.T[i] < p["T_amb"] + 1.0:
                amp = self.session_peak[i] - p["T_amb"]
                if amp >= 1.0:
                    self.cycling[i] += 2.0 * amp ** p["n_cm"]
                self.in_session[i] = False
                self.session_peak[i] = p["T_amb"]

        self.wall_time += round_time + self.t_gap

    def close_sessions(self):
        for i in range(self.n):
            if self.in_session[i]:
                p = self.par[i]
                amp = self.session_peak[i] - p["T_amb"]
                if amp >= 1.0:
                    self.cycling[i] += 2.0 * amp ** p["n_cm"]
                self.in_session[i] = False


# =====================================================================
# Whittle index
# =====================================================================

class WhittleIndex:
    """
    Single-arm subsidy problem, solved per device class on a temperature grid.

    Active : cost  c(T) = E(T) + lam * w(T),   T -> heat(T)
    Passive: cost  -nu  (subsidy nu),          T -> cool(T)

    nu(T) is the subsidy at which the two are indifferent. Larger nu means
    activation stays attractive under a larger subsidy, so the K arms with the
    LARGEST index are activated.
    """

    def __init__(self, params, t_round, t_gap, lam, beta=0.97,
                 n_grid=120, n_subsidy=600):
        self.p = params
        self.t_round, self.t_gap, self.lam, self.beta = t_round, t_gap, lam, beta
        T_amb = params["T_amb"]
        # Only temperatures the device can actually reach are meaningful; the
        # open-loop steady state under sustained load is the ceiling.
        self.T_ss = TH.steady_state_temp(params["P_train"], params["R_th"], T_amb)
        self.grid = np.linspace(T_amb, self.T_ss + 0.5, n_grid)
        self._precompute()
        self._sweep(n_subsidy)

    def _sweep(self, n_subsidy):
        """
        One pass over subsidies gives every state's threshold at once, and the
        passive sets it produces are exactly what indexability requires.

        nu(T) = smallest subsidy at which passivity becomes optimal in state T.
        """
        span = float(np.max(self.c_active) - np.min(self.c_active))
        lo = -3.0 * span - 10.0
        hi = 3.0 * span + 10.0
        self.subsidies = np.linspace(lo, hi, n_subsidy)
        sets = [self._value_iterate(nu)[1] for nu in self.subsidies]
        self.passive_sets = np.array(sets)              # (n_subsidy, n_grid)
        idx = np.full(len(self.grid), np.nan)
        for k in range(len(self.grid)):
            hits = np.where(self.passive_sets[:, k])[0]
            idx[k] = self.subsidies[hits[0]] if len(hits) else self.subsidies[-1]
        self.index = idx

    def indexability(self):
        """
        Indexability: the passive set must be non-decreasing in the subsidy.
        Checked over the same sweep used to build the index, so the two cannot
        disagree.
        """
        P = self.passive_sets
        ok = bool(np.all(P[1:] | ~P[:-1]))
        return ok, [int(s.sum()) for s in P[::max(len(P) // 12, 1)]]

    def _precompute(self):
        p, g = self.p, self.grid
        e = TH.eta(g, p["T_cap"], p["eta_min"])
        dur = self.t_round / e
        P = TH.power_at(g, e, p["P_train"], p["f_static"], p["theta_leak"],
                        p["p_dvfs"], p["T_amb"])
        self.E = P * dur
        T_next = TH.heat(g, p["P_train"], dur, p["R_th"], p["tau_heat"], p["T_amb"])
        # Exact increment, matching Fleet.marginal_wear (finding F12). Within
        # the index's 1-D state the session peak is approximated by the current
        # temperature, which is exact while a device is heating.
        dT_old = np.maximum(g - p["T_amb"], 0.0)
        dT_new = np.maximum(T_next - p["T_amb"], 0.0)
        cyc = 2.0 * (dT_new ** p["n_cm"] - dT_old ** p["n_cm"])
        T_b = p["T_amb"] + p["k_batt"] * (g - p["T_amb"])
        arr = (TH.arrhenius_ratio(T_b, p["T_amb"], p["Ea"]) - 1.0) * dur
        self.W = cyc + arr
        self.nxt_a = np.clip(np.searchsorted(self.grid, T_next), 0, len(self.grid) - 1)
        T_cool = TH.cool(g, self.t_gap, p["tau_cool"], p["T_amb"])
        self.nxt_p = np.clip(np.searchsorted(self.grid, T_cool), 0, len(self.grid) - 1)
        self.c_active = self.E + self.lam * self.W

    def _value_iterate(self, nu, iters=600, tol=1e-10):
        V = np.zeros(len(self.grid))
        for _ in range(iters):
            Qa = self.c_active + self.beta * V[self.nxt_a]
            Qp = -nu + self.beta * V[self.nxt_p]
            Vn = np.minimum(Qa, Qp)
            if np.max(np.abs(Vn - V)) < tol:
                V = Vn
                break
            V = Vn
        Qa = self.c_active + self.beta * V[self.nxt_a]
        Qp = -nu + self.beta * V[self.nxt_p]
        return V, Qp <= Qa   # True where passive is optimal

    def indexability(self):
        """
        Indexability: the passive set must be non-decreasing in the subsidy.
        Checked over the same sweep used to build the index, so the two cannot
        disagree.
        """
        P = self.passive_sets
        ok = bool(np.all(P[1:] | ~P[:-1]))
        return ok, [int(s.sum()) for s in P[::max(len(P) // 12, 1)]]

    def _precompute(self):
        p, g = self.p, self.grid
        e = TH.eta(g, p["T_cap"], p["eta_min"])
        dur = self.t_round / e
        P = TH.power_at(g, e, p["P_train"], p["f_static"], p["theta_leak"],
                        p["p_dvfs"], p["T_amb"])
        self.E = P * dur
        T_next = TH.heat(g, p["P_train"], dur, p["R_th"], p["tau_heat"], p["T_amb"])
        # Exact increment, matching Fleet.marginal_wear (finding F12). Within
        # the index's 1-D state the session peak is approximated by the current
        # temperature, which is exact while a device is heating.
        dT_old = np.maximum(g - p["T_amb"], 0.0)
        dT_new = np.maximum(T_next - p["T_amb"], 0.0)
        cyc = 2.0 * (dT_new ** p["n_cm"] - dT_old ** p["n_cm"])
        T_b = p["T_amb"] + p["k_batt"] * (g - p["T_amb"])
        arr = (TH.arrhenius_ratio(T_b, p["T_amb"], p["Ea"]) - 1.0) * dur
        self.W = cyc + arr
        self.nxt_a = np.clip(np.searchsorted(self.grid, T_next), 0, len(self.grid) - 1)
        T_cool = TH.cool(g, self.t_gap, p["tau_cool"], p["T_amb"])
        self.nxt_p = np.clip(np.searchsorted(self.grid, T_cool), 0, len(self.grid) - 1)
        self.c_active = self.E + self.lam * self.W

    def _value_iterate(self, nu, iters=600, tol=1e-10):
        V = np.zeros(len(self.grid))
        for _ in range(iters):
            Qa = self.c_active + self.beta * V[self.nxt_a]
            Qp = -nu + self.beta * V[self.nxt_p]
            Vn = np.minimum(Qa, Qp)
            if np.max(np.abs(Vn - V)) < tol:
                V = Vn
                break
            V = Vn
        Qa = self.c_active + self.beta * V[self.nxt_a]
        Qp = -nu + self.beta * V[self.nxt_p]
        return V, Qp <= Qa   # True where passive is optimal

    def _solve(self, k, lo=-5000.0, hi=5000.0, iters=40):
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            _, passive = self._value_iterate(mid)
            if passive[k]:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def indexability(self, subsidies=None):
        """Passive set must grow monotonically with the subsidy."""
        if subsidies is None:
            subsidies = np.linspace(self.index.min() - 50,
                                    self.index.max() + 50, 40)
        sets = [self._value_iterate(nu)[1] for nu in subsidies]
        ok = all(bool(np.all(sets[j + 1] | ~sets[j])) for j in range(len(sets) - 1))
        sizes = [int(s.sum()) for s in sets]
        return ok, sizes

    def at(self, T):
        return float(np.interp(T, self.grid, self.index))


# =====================================================================
# Policies
# =====================================================================

def _topk(scores, pool, K):
    pool = np.asarray(pool)
    if len(pool) <= K:
        return pool
    return pool[np.argsort(-scores)[:K]]


def policy_random(fleet, pool, K, state, rng):
    return rng.choice(pool, size=min(K, len(pool)), replace=False)


def policy_static_score(fleet, pool, K, state, rng):
    """
    Paper 1's rule: s_i = a*U_data + b*U_sys - g*E_marginal, with E_marginal
    from a NOMINAL power model. Thermally blind by construction -- this is the
    baseline whose miscalibration the paper is about.
    """
    a, b, g = 0.4, 0.3, 0.3
    U_data = fleet.utility[pool]
    U_sys = np.array([1.0 / (fleet.t_round / fleet.par[i]["eta_min"]) for i in pool])
    U_sys = U_sys / U_sys.max()
    E_nom = np.array([fleet.par[i]["P_train"] * fleet.t_round for i in pool])
    E_nom = E_nom / E_nom.max()
    return _topk(a * U_data + b * U_sys - g * E_nom, pool, K)


def policy_energy_only(fleet, pool, K, state, rng):
    """Greedy on TRUE instantaneous energy. Sees heat, ignores its future cost."""
    cost = np.array([fleet.round_cost(i)[0] for i in pool])
    return _topk(-cost / cost.max() + 0.3 * fleet.utility[pool], pool, K)


def policy_lyapunov(fleet, pool, K, state, rng):
    """
    Drift-plus-penalty. Minimise V*energy + Q_i*wear - mu*utility, where Q_i is
    a virtual wear-debt queue draining at the per-device budget.
    """
    V, mu = state["V"], state["mu"]
    Q = state["Q"]
    sc = []
    for i in pool:
        E, _ = fleet.round_cost(i)
        cyc, arr = fleet.marginal_wear(i)
        sc.append(-(V * E + Q[i] * (cyc + arr)) + mu * fleet.utility[i])
    return _topk(np.array(sc), pool, K)


def policy_whittle(fleet, pool, K, state, rng):
    """Activate the arms with the largest Whittle index, offset by utility."""
    idx = state["whittle"]
    mu = state["mu_w"]
    sc = np.array([idx[fleet.cls[i]].at(fleet.T[i]) + mu * fleet.utility[i]
                   for i in pool])
    return _topk(sc, pool, K)


POLICIES = {
    "random": policy_random,
    "static_score": policy_static_score,
    "energy_only": policy_energy_only,
    "lyapunov": policy_lyapunov,
    "whittle": policy_whittle,
}


# =====================================================================
# Campaign runner
# =====================================================================

def run(policy_name, n_devices=100, n_rounds=200, K=10, t_round=25.0,
        t_gap=60.0, seed=0, W_budget=None, V=1.0, mu=None, lam=None):
    rng = np.random.default_rng(seed)
    fleet = Fleet(n_devices, t_round, t_gap, rng)

    state = {"Q": np.zeros(n_devices), "V": V,
             "mu": 0.0 if mu is None else mu, "mu_w": 0.0}
    if policy_name == "whittle":
        lam = 1.0 if lam is None else lam
        state["whittle"] = {c: WhittleIndex(dict(DEVICE_CLASSES[c], **SHARED),
                                            t_round, t_gap, lam)
                            for c in DEVICE_CLASSES}
        state["mu_w"] = 0.0 if mu is None else mu
    if W_budget is None:
        W_budget = 0.0

    fn = POLICIES[policy_name]
    for _ in range(n_rounds):
        avail = np.where(fleet.available())[0]
        if len(avail) == 0:
            continue
        chosen = fn(fleet, avail, K, state, rng)
        pre = {int(i): fleet.marginal_wear(int(i)) for i in chosen}
        fleet.step(chosen)
        # virtual queue update
        state["Q"] = np.maximum(state["Q"] - W_budget, 0.0)
        for i in chosen:
            c, a = pre[int(i)]
            state["Q"][int(i)] += c + a
    fleet.close_sessions()

    sel = fleet.selections
    gini = _gini(sel)
    return {
        "policy": policy_name,
        "energy_J": float(fleet.energy.sum()),
        "cycling": float(fleet.cycling.sum()),
        # For a REUSE architecture the fleet total is the wrong burden measure:
        # devices belong to users who can withdraw, so what matters is the wear
        # borne by the worst-affected participant, not the sum over the fleet.
        "cycling_max": float(fleet.cycling.max()),
        "cycling_p95": float(np.percentile(fleet.cycling, 95)),
        "energy_max": float(fleet.energy.max()),
        "arrhenius": float(fleet.arrhenius.sum()),
        "T_peak": float(fleet.T_max_ever.max()),
        "T_peak_mean": float(fleet.T_max_ever.mean()),
        "violation_s": float(fleet.violation_s.sum()),
        "wall_time_s": float(fleet.wall_time),
        "gini_participation": gini,
        "max_share": float(sel.max() / max(sel.sum(), 1)),
        "n_touched": int((sel > 0).sum()),
        "selections": sel,
    }


def _gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
