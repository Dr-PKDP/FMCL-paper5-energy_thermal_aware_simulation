"""
FMCL Paper 5 -- Charging-coupled fleet and scheduling.

Replaces the ambient-baseline Fleet in scheduler.py. Devices are now plugged in
and charging throughout an overnight participation window, which is the setting
Paper 1's availability definition actually specifies. Each device carries a
charger rate, a state of charge, and two coupled temperatures.

The two-node dynamics of charging.py are vectorised across the fleet, since a
per-second scalar loop over 100 devices across a six-hour window is intractable.

New outcome, reported alongside energy and wear: the CHARGING PENALTY, measured
against a per-device counterfactual in which that device does not participate.
"""

import numpy as np
import thermal as TH
import charging as CH
import coefficients as C
import scheduler as S

CHARGER_CLASSES = {"fast": 25.0, "mid": 15.0, "standard": 5.0}


class ChargingFleet:
    def __init__(self, n, rng, t_round=65.0, t_gap=15.0, dt=5.0,
                 mix=(("flagship", .25), ("midrange", .50), ("lowend", .25)),
                 chg_mix=(("fast", .40), ("mid", .40), ("standard", .20)),
                 T_amb=25.0, soc0=(0.20, 0.50)):
        self.n, self.rng, self.dt = n, rng, dt
        self.t_round, self.t_gap = t_round, t_gap
        self.T_amb = T_amb
        labels, probs = zip(*mix)
        self.cls = rng.choice(labels, n, p=probs)
        clab, cprob = zip(*chg_mix)
        self.chg_cls = rng.choice(clab, n, p=cprob)
        self.P_chg_max = np.array([CHARGER_CLASSES[c] for c in self.chg_cls])

        par = [dict(S.DEVICE_CLASSES[c], **S.SHARED) for c in self.cls]
        self.T_cap = np.array([p["T_cap"] for p in par])
        self.eta_min = np.array([p["eta_min"] for p in par])
        self.P_comp = np.array([p["P_train"] for p in par])
        self.n_cm = S.SHARED["n_cm"]

        self.soc0 = rng.uniform(*soc0, size=n)
        self.utility = rng.uniform(0.4, 1.0, size=n)
        self.reset()

    def reset(self):
        n, T = self.n, self.T_amb
        self.T_s = np.full(n, T)
        self.T_b = np.full(n, T)
        self.soc = self.soc0.copy()
        self.work = np.zeros(n)          # useful compute delivered, s
        self.energy = np.zeros(n)        # J
        self.suspended = np.zeros(n)     # s of OS-suspended training
        self.above40 = np.zeros(n)       # s battery cell above 40 C
        self.above_cap = np.zeros(n)     # s processor above its throttle cap
        self.selections = np.zeros(n, int)
        self.T_s_peak = np.full(n, T)
        self.T_b_peak = np.full(n, T)
        self.session_peak = np.full(n, T)
        self.in_session = np.zeros(n, bool)
        self.cycling = np.zeros(n)
        self.full_time = np.full(n, np.nan)   # s to reach full charge
        self.elapsed = 0.0

    # ---- vectorised physics -----------------------------------------
    def _advance(self, active, seconds):
        """Advance the fleet by `seconds`, with `active` devices training."""
        steps = max(int(round(seconds / self.dt)), 1)
        dt = seconds / steps
        for _ in range(steps):
            eta = TH.eta(self.T_s, self.T_cap, self.eta_min)
            admit = np.clip((CH.T_COMPUTE_KILL - self.T_s) /
                            (CH.T_COMPUTE_KILL - CH.T_COMPUTE_CEILING), 0.0, 1.0)
            P = TH.power_at(self.T_s, eta, self.P_comp,
                            C.STATIC_POWER_FRACTION.value,
                            C.LEAKAGE_TEMP_CONSTANT.value,
                            C.DVFS_EXPONENT.value, self.T_amb) * admit
            P = np.where(active, P, 0.0)
            self.work += np.where(active, eta * admit * dt, 0.0)
            self.energy += P * dt
            self.suspended += np.where(active, (1 - admit) * dt, 0.0)
            self.above_cap += np.where(active & (self.T_s > self.T_cap), dt, 0.0)

            taper = np.where(self.soc < 0.8, 1.0,
                             np.maximum((1 - self.soc) / 0.2, 0.05))
            derate = np.clip((CH.T_CHG_STOP - self.T_b) /
                             (CH.T_CHG_STOP - CH.T_CHG_DERATE), 0.0, 1.0)
            P_chg = self.P_chg_max * taper * derate * (self.soc < 0.999)
            P_loss = (1 - CH.ETA_CHG) * P_chg

            dTs = (P - (self.T_s - self.T_amb) / CH.R_S
                   - (self.T_s - self.T_b) / CH.R_SB) / CH.C_S
            dTb = (P_loss - (self.T_b - self.T_amb) / CH.R_B
                   + (self.T_s - self.T_b) / CH.R_SB) / CH.C_B
            was_full = self.soc >= 0.999
            self.soc = np.minimum(
                self.soc + P_chg * CH.ETA_CHG * dt / (CH.E_BATT_WH * 3600.0), 1.0)
            newly_full = (~was_full) & (self.soc >= 0.999)
            self.full_time = np.where(newly_full & np.isnan(self.full_time),
                                      self.elapsed, self.full_time)
            self.T_s = self.T_s + dTs * dt
            self.T_b = self.T_b + dTb * dt
            self.above40 += (self.T_b > 40.0) * dt
            self.T_s_peak = np.maximum(self.T_s_peak, self.T_s)
            self.T_b_peak = np.maximum(self.T_b_peak, self.T_b)
            self.session_peak = np.maximum(self.session_peak, self.T_s)
            self.elapsed += dt

        # close thermal cycles for devices that have returned to baseline
        done = self.in_session & (self.T_s < self.T_amb + 1.0)
        amp = np.maximum(self.session_peak - self.T_amb, 0.0)
        self.cycling += np.where(done & (amp >= 1.0), 2.0 * amp ** self.n_cm, 0.0)
        self.session_peak = np.where(done, self.T_amb, self.session_peak)
        self.in_session = self.in_session & ~done

    def step_round(self, chosen):
        act = np.zeros(self.n, bool)
        act[chosen] = True
        self.selections[chosen] += 1
        self.in_session |= act
        self._advance(act, self.t_round)
        self._advance(np.zeros(self.n, bool), self.t_gap)

    def close(self):
        amp = np.maximum(self.session_peak - self.T_amb, 0.0)
        self.cycling += np.where(self.in_session & (amp >= 1.0),
                                 2.0 * amp ** self.n_cm, 0.0)
        self.in_session[:] = False

    # ---- signals used by policies ------------------------------------
    def round_energy(self):
        eta = TH.eta(self.T_s, self.T_cap, self.eta_min)
        P = TH.power_at(self.T_s, eta, self.P_comp,
                        C.STATIC_POWER_FRACTION.value,
                        C.LEAKAGE_TEMP_CONSTANT.value,
                        C.DVFS_EXPONENT.value, self.T_amb)
        return P * self.t_round / eta

    def marginal_wear(self):
        eta = TH.eta(self.T_s, self.T_cap, self.eta_min)
        dur = self.t_round / eta
        T_next = TH.heat(self.T_s, self.P_comp, dur, np.array(
            [S.DEVICE_CLASSES[c]["R_th"] for c in self.cls]), np.array(
            [S.DEVICE_CLASSES[c]["tau_heat"] for c in self.cls]), self.T_amb)
        peak = np.where(self.in_session,
                        np.maximum(self.session_peak, self.T_s), self.T_s)
        d_old = np.maximum(peak - self.T_amb, 0.0)
        d_new = np.maximum(np.maximum(T_next, peak) - self.T_amb, 0.0)
        return 2.0 * (d_new ** self.n_cm - d_old ** self.n_cm)

    def charge_headroom(self):
        """
        Thermal headroom the charging load leaves for compute. A fast charger
        has already spent much of the budget; a slow one has not. This is the
        signal finding F25 identified and no FL scheduler uses.
        """
        return (CH.T_CHG_STOP - self.T_b) / (CH.T_CHG_STOP - self.T_amb) \
            * (1.0 - self.P_chg_max / max(CHARGER_CLASSES.values()))

    def available(self, p=0.85):
        """Plugged in and idle. Devices already full remain eligible."""
        return self.rng.random(self.n) < p


# =====================================================================
# Policies (vectorised)
# =====================================================================

def _topk(score, pool, K):
    return pool if len(pool) <= K else pool[np.argsort(-score)[:K]]


def p_random(f, pool, K, st, rng):
    return rng.choice(pool, min(K, len(pool)), replace=False)


def p_static(f, pool, K, st, rng):
    """Paper 1's rule with a nominal power model: thermally blind."""
    U = f.utility[pool]
    Us = 1.0 / (f.t_round / f.eta_min[pool]); Us /= Us.max()
    En = f.P_comp[pool] * f.t_round; En /= En.max()
    return _topk(0.4 * U + 0.3 * Us - 0.3 * En, pool, K)


def p_energy(f, pool, K, st, rng):
    e = f.round_energy()[pool]
    return _topk(-e / e.max() + 0.3 * f.utility[pool], pool, K)


def _norm(x):
    """Scale to [0,1] so the policy weights are commensurable. Without this the
    unbounded virtual queue swamps every other term and the weights are inert."""
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def p_lyapunov(f, pool, K, st, rng):
    w = f.marginal_wear()[pool]
    e = f.round_energy()[pool]
    drift = _norm(st["Q"][pool] * w)
    return _topk(-(st["V"] * _norm(e) + drift) + st["mu"] * f.utility[pool],
                 pool, K)


def p_fedcs(f, pool, K, st, rng):
    """
    FedCS (Nishio & Yonetani, IEEE ICC 2019). The canonical deadline- and
    resource-aware FL scheduler: the server estimates each candidate's round
    completion time from its profiled resource state and greedily selects the
    clients predicted to finish soonest, dropping likely stragglers so that
    the round completes within a fixed deadline. This harness does not model
    heterogeneous network bandwidth (Section 11.3), so the estimated
    completion time reduces to the compute term alone, T_i = t_round / eta_i.

    FedCS has NO energy, thermal, wear, or charging term of any kind. It
    exists purely to answer whether optimising the classical scheduling
    objective -- makespan, or equivalently minimising predicted completion
    time -- is itself sufficient to avoid the thermal failure mode of
    Section 9.2. It is the purest possible test of that question, since it is
    the one baseline here built with no resource-conservation goal at all.
    """
    T_i = f.t_round / np.maximum(TH.eta(f.T_s[pool], f.T_cap[pool],
                                        f.eta_min[pool]), 1e-6)
    return _topk(-T_i, pool, K)


def p_charger_aware(f, pool, K, st, rng):
    """
    CATS -- Charger- and Thermal-Aware Scheduler. The algorithm proposed in
    this paper: drift-plus-penalty selection (Eq. 17-18) over energy and
    marginal wear, extended with the charger-headroom term of Eq. 19 (finding
    F25). This is the only policy compared here that observes processor
    temperature, cell temperature, AND charger rating.
    """
    w = f.marginal_wear()[pool]
    e = f.round_energy()[pool]
    drift = _norm(st["Q"][pool] * w)
    h = _norm(f.charge_headroom()[pool])
    return _topk(-(st["V"] * _norm(e) + drift) + st["mu"] * f.utility[pool]
                 + st["nu"] * h, pool, K)


# =====================================================================
# Baselines from the published participant-selection literature.
#
# None of the source papers release code compatible with this harness, and
# none report per-device physical state at the granularity this simulation
# tracks, so each reimplementation is SIMPLIFIED FROM the published algorithm
# rather than a byte-for-byt reproduction. What is preserved in every case is
# the DECISION RULE -- which signals the policy is allowed to observe and how
# it combines them -- since that is what determines whether thermal or
# charging state can be represented at all. Where the original paper is
# specific enough to fix a formula, that formula is used; where it is not
# (e.g. Oort's system utility discount is annealed by hyperparameters not
# specified precisely for consumer-charging fleets), a standard form for that
# structural element is substituted and stated here.
# =====================================================================

def p_oort(f, pool, K, st, rng):
    """
    Oort (Lai, Zhu, Madhyastha & Chowdhury, OSDI 2021). Combines a
    STATISTICAL UTILITY (this simulation uses the same per-device data-utility
    proxy used throughout this paper, since no per-sample loss is tracked by
    the physical fleet model) with a SYSTEM UTILITY that penalises clients
    whose expected round completion time exceeds a target, following the
    speed-discount form Util_sys = (T_target / T_i)^alpha for T_i > T_target
    reported in the paper. A UCB-style exploration bonus over selection
    staleness reproduces Oort's stated exploration-exploitation split, so that
    long-idle clients are not permanently excluded.

    Oort has NO energy, NO thermal, and NO charging term. It is included as
    the reference point every energy-aware or thermal-aware method in
    Section 3.1 positions itself against.
    """
    T_i = f.t_round / np.maximum(TH.eta(f.T_s[pool], f.T_cap[pool],
                                        f.eta_min[pool]), 1e-6)
    T_target = np.percentile(T_i, 50)
    alpha = 2.0
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


def p_eafl(f, pool, K, st, rng):
    """
    EAFL (Arouj & Abdelmoneem, ACM FedEdge 2022). A power-aware selector that
    "cherry-picks clients with higher battery levels" and trades remaining
    power against expected completion time to reduce mid-training dropout, per
    the published design goal. Implemented as headroom above a safety floor
    (state of charge in excess of a minimum reserve) divided by expected
    completion time, which is the direct simulation-facing form of that
    trade-off. EAFL's battery variable maps exactly onto the state-of-charge
    state this fleet already tracks.

    EAFL observes BATTERY LEVEL but has no notion of processor or cell
    temperature, so it cannot see throttling or the charging competition of
    Eq. 4 even though it is, in the sense of Section 3.1, the closest prior
    method in spirit to the algorithm proposed here.
    """
    soc_min = 0.05
    headroom = np.maximum(f.soc[pool] - soc_min, 0.0)
    T_i = f.t_round / np.maximum(TH.eta(f.T_s[pool], f.T_cap[pool],
                                        f.eta_min[pool]), 1e-6)
    return _topk(headroom / T_i, pool, K)


def p_wilfq(f, pool, K, st, rng):
    """
    Restless-bandit selection in the spirit of WILF-Q (arXiv:2509.13933,
    2025), which formulates client selection as a restless multi-armed bandit
    over UNOBSERVABLE client computation and communication state, learning a
    Whittle index by Q-learning, and targets time-to-accuracy rather than
    energy, wear, or carbon.

    This simulation does not reproduce the Q-learning estimator, since the
    reward shaping needed to do so is not specified precisely enough in the
    published description to reimplement faithfully. What is reproduced is
    the STRUCTURAL claim the comparison needs to test: a restless-bandit
    formulation over LATENCY state alone, with no energy or thermal term,
    computed here by the same value-iteration technique used for the Whittle
    relaxation in Section 7.2, substituting expected round latency for the
    energy-plus-wear cost of Eq. 15. The state each arm evolves in is exactly
    the processor temperature of Eq. 1, but the POLICY never observes
    temperature directly -- it observes only the round latency that
    temperature happens to produce, which mirrors WILF-Q's premise that
    client state is inferred from timing rather than measured directly.
    """
    idx = st["_wilfq_idx"]
    scores = np.array([idx[f.cls[i]].at(f.T_s[i]) for i in pool])
    return _topk(scores + 0.05 * f.utility[pool], pool, K)


def _build_wilfq_index(t_round, t_gap, T_amb=25.0):
    """
    Whittle index over latency alone (no wear, no energy weighting), one per
    device class, used by p_wilfq. Built once per evaluation and cached by the
    caller, since value iteration is not free.
    """
    import scheduler as sch

    idx = {}
    for cname, base in sch.DEVICE_CLASSES.items():
        p = dict(base, **sch.SHARED)
        w = sch.WhittleIndex.__new__(sch.WhittleIndex)
        w.p, w.t_round, w.t_gap = p, t_round, t_gap
        w.lam, w.beta = 0.0, 0.97   # lam=0: no wear term, latency only
        w.T_ss = TH.steady_state_temp(p["P_train"], p["R_th"], T_amb)
        w.grid = np.linspace(T_amb, w.T_ss + 0.5, 120)
        w._precompute()
        # overwrite the active-arm cost with latency, not energy+wear
        eta_grid = TH.eta(w.grid, p["T_cap"], p["eta_min"])
        w.c_active = t_round / np.maximum(eta_grid, 1e-6)
        w._sweep(150)
        idx[cname] = w
    return idx


POLICIES = {"random": p_random, "static_score": p_static,
            "energy_only": p_energy, "oort": p_oort, "eafl": p_eafl,
            "wilfq": p_wilfq, "fedcs": p_fedcs, "lyapunov": p_lyapunov,
            "cats": p_charger_aware, "charger_aware": p_charger_aware}


# =====================================================================
# Campaign
# =====================================================================

def counterfactual_full_time(fleet):
    """Charge completion time for each device with NO participation."""
    f = ChargingFleet.__new__(ChargingFleet)
    f.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else v)
                       for k, v in fleet.__dict__.items()})
    f.reset()
    n_rounds = int(fleet.total_rounds)
    for _ in range(n_rounds):
        f.step_round(np.array([], dtype=int))
    return f.full_time.copy()


def run(policy, n=100, hours=6.0, K=10, seed=0, V=1.0, mu=0.3, nu=1.0,
        W_budget=0.05, t_round=65.0, t_gap=15.0):
    rng = np.random.default_rng(seed)
    fleet = ChargingFleet(n, rng, t_round=t_round, t_gap=t_gap)
    fleet.total_rounds = int(hours * 3600 / (t_round + t_gap))
    base_full = counterfactual_full_time(fleet)

    st = {"Q": np.zeros(n), "V": V, "mu": mu, "nu": nu, "_round": 0}
    if policy == "wilfq":
        st["_wilfq_idx"] = _build_wilfq_index(t_round, t_gap)
    fn = POLICIES[policy]
    for _ in range(fleet.total_rounds):
        pool = np.where(fleet.available())[0]
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

    # Classical scheduling metrics, computed from state already collected.
    # Jain's fairness index (Jain, Chiu & Hawe 1984), over per-device work
    # delivered: J = (sum x_i)^2 / (n * sum x_i^2), in (0, 1], 1 = perfectly
    # even. Reported alongside the Gini coefficient already used elsewhere in
    # this paper because Jain's index is the form standard in the networking
    # and systems literature and is what EAFL's own evaluation reports [6],
    # which makes it the more direct point of comparison for that baseline.
    w = fleet.work
    jain = float((w.sum() ** 2) / (n * np.sum(w ** 2) + 1e-12)) if w.sum() > 0 else 0.0
    # Resource utilisation: useful compute delivered as a fraction of the
    # round-slot-seconds actually allocated to selected devices. Distinct from
    # work_s, which reports the absolute quantity delivered and says nothing
    # about how much of what was allocated came back as useful throughput.
    allocated = float(fleet.selections.sum()) * t_round
    utilisation = float(w.sum() / allocated) if allocated > 0 else 0.0

    return {
        "policy": policy,
        "work_s": float(fleet.work.sum()),
        "energy_J": float(fleet.energy.sum()),
        "suspended_s": float(fleet.suspended.sum()),
        "above_cap_s": float(fleet.above_cap.sum()),
        "cycling_max": float(fleet.cycling.max()),
        "chg_penalty_mean": float(pen[touched].mean()) if touched.any() else 0.0,
        "chg_penalty_max": float(pen.max()),
        "n_harmed": int((pen > 0.10).sum()),
        "n_unfinished": int(np.isnan(fleet.full_time).sum()),
        "above40_s": float(fleet.above40.sum()),
        "T_s_peak": float(fleet.T_s_peak.max()),
        "T_b_peak": float(fleet.T_b_peak.max()),
        "gini": S._gini(fleet.selections),
        "jain": jain,
        "utilisation": utilisation,
        "n_touched": int(touched.sum()),
        "fast_share": float(fleet.selections[fleet.chg_cls == "fast"].sum()
                            / max(fleet.selections.sum(), 1)),
        "std_share": float(fleet.selections[fleet.chg_cls == "standard"].sum()
                           / max(fleet.selections.sum(), 1)),
    }
