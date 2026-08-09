"""
FMCL Paper 5 -- Convergence on the charging-coupled fleet.

Resolves F19 (the matched-quality metric is ill-posed when policies converge to
different floors) by changing the question rather than patching the metric.

FOUR CANDIDATE RESOLUTIONS, AND WHY THIS ONE
--------------------------------------------
(a) Fix the BUDGET, compare quality.  Always defined; no censoring.
(b) Decompose into convergence RATE and FLOOR, reported separately.
(c) Area under the quality-versus-cumulative-energy curve.
(d) Survival analysis treating "never reached" as right-censored.

(a) is primary here because the charging fleet supplies a natural budget: a
participant gives one overnight window. "Quality reached per overnight session"
is what a deployment actually faces, and the floor problem does not arise.
(b) is reported alongside as a diagnostic, since a policy can lose on rate,
on floor, or on both, and the three cases call for different fixes.
(c) is computed as a robustness check on (a).

THERMAL DROPOUT
---------------
When the OS suspends background work the device does not merely complete fewer
local steps -- it fails to return an update at all. That is a dropout, not a
degraded contribution, and it is correlated with device class and charger rate
rather than random. This is the mechanism linking thermal state to the
participation bias that sets the floor, and it connects to Paper 2's dropout
analysis rather than sitting beside it.
"""

import numpy as np

from simulation import make_federated_data, _fedprox_local, _global_grad  # noqa

import thermal as TH  # noqa
import charging as CH  # noqa
import coefficients as C  # noqa
import fleet_charging as FC  # noqa

LOCAL_STEPS = 5


def device_capability(fleet):
    """Throughput multiplier and OS admission fraction, per device."""
    eta = TH.eta(fleet.T_s, fleet.T_cap, fleet.eta_min)
    admit = np.clip((CH.T_COMPUTE_KILL - fleet.T_s) /
                    (CH.T_COMPUTE_KILL - CH.T_COMPUTE_CEILING), 0.0, 1.0)
    return eta, admit


def run(policy, clients, n_classes, dim, n=100, hours=6.0, K=30, seed=0,
        V=1.0, mu=0.3, nu=0.5, W_budget=0.05, t_round=65.0, t_gap=15.0,
        prox_mu=0.1, clip=5.0, dropout_threshold=0.9, debias=False,
        propensity_floor=0.02):
    """
    One overnight campaign. Returns the quality trajectory against both round
    index and cumulative energy, plus dropout and participation diagnostics.
    """
    rng = np.random.default_rng(seed)
    fleet = FC.ChargingFleet(n, rng, t_round=t_round, t_gap=t_gap)
    fleet.total_rounds = int(hours * 3600 / (t_round + t_gap))
    st = {"Q": np.zeros(n), "V": V, "mu": mu, "nu": nu, "_round": 0}
    if policy == "wilfq":
        st["_wilfq_idx"] = FC._build_wilfq_index(t_round, t_gap)
    fn = FC.POLICIES[policy]

    W = np.zeros((dim, n_classes))
    grads, energy_curve = [], []
    n_selected = n_dropped = 0
    dropped_by_class = {}
    contributions = np.zeros(n)
    # Running selection-propensity estimate, used to remove the participation
    # bias that any state-dependent selection rule introduces. This is the same
    # correction Paper 2 applies analytically for Bernoulli participation, done
    # empirically here because the scheduler's propensities are not in closed
    # form.
    sel_count = np.zeros(n)
    rounds_seen = 0

    for _ in range(fleet.total_rounds):
        pool = np.where(fleet.available())[0]
        if len(pool) == 0:
            fleet.step_round(np.array([], dtype=int))
            grads.append(float(np.linalg.norm(_global_grad(W, clients, n_classes)) ** 2))
            energy_curve.append(float(fleet.energy.sum()))
            continue

        chosen = np.asarray(fn(fleet, pool, K, st, rng), dtype=int)
        eta, admit = device_capability(fleet)

        agg = np.zeros((dim, n_classes))
        wsum = 0.0
        for i in chosen:
            n_selected += 1
            # THERMAL DROPOUT: below the admission threshold the OS suspends the
            # job and no update is returned at all.
            if admit[i] < dropout_threshold:
                n_dropped += 1
                cls = fleet.chg_cls[i]
                dropped_by_class[cls] = dropped_by_class.get(cls, 0) + 1
                continue
            ls = max(int(np.floor(LOCAL_STEPS * eta[i] * admit[i])), 1)
            X, y = clients[i]
            delta = _fedprox_local(W, X, y, n_classes, mu=prox_mu,
                                   local_steps=ls, lr=0.5)
            nr = np.linalg.norm(delta)
            if nr > clip:
                delta = delta * clip / nr
            wt = float(len(y))
            if debias:
                p_hat = max(sel_count[i] / max(rounds_seen, 1), propensity_floor)
                wt = wt / p_hat
            agg += wt * delta
            wsum += wt
            contributions[i] += 1
        if wsum > 0:
            W = W + agg / wsum

        rounds_seen += 1
        sel_count[chosen] += 1
        w = fleet.marginal_wear()
        fleet.step_round(chosen)
        st["Q"] = np.maximum(st["Q"] - W_budget, 0.0)
        st["Q"][chosen] += w[chosen]

        grads.append(float(np.linalg.norm(_global_grad(W, clients, n_classes)) ** 2))
        energy_curve.append(float(fleet.energy.sum()))

    fleet.close()
    g = np.array(grads)
    e = np.array(energy_curve)
    return {
        "policy": policy, "grads": g, "energy_curve": e,
        "final_quality": float(g[-1]),
        "energy_J": float(fleet.energy.sum()),
        "work_s": float(fleet.work.sum()),
        "cycling_max": float(fleet.cycling.max()),
        "dropout_rate": n_dropped / max(n_selected, 1),
        "dropped_by_class": dropped_by_class,
        "n_contributors": int((contributions > 0).sum()),
        "contrib_gini": FC.S._gini(contributions),
        "auc": float(np.trapezoid(g, e) / max(e[-1], 1e-9)),
    }


def rate_and_floor(g):
    """
    Decompose the trajectory into a convergence rate and an asymptotic floor by
    fitting ||grad||^2 = A*exp(-r*t) + F on a log grid of candidate floors.

    A policy can lose on rate, on floor, or on both, and the three cases call
    for different remedies -- which is why they are reported separately rather
    than folded into one scalar.
    """
    t = np.arange(len(g), dtype=float)
    best = (np.inf, np.nan, np.nan)
    lo = max(g.min() * 1e-3, 1e-16)
    for F in np.geomspace(lo, max(g.min() * 0.999, lo * 1.001), 60):
        y = g - F
        m = y > 0
        if m.sum() < 10:
            continue
        A = np.polyfit(t[m], np.log(y[m]), 1)
        resid = np.sum((np.log(y[m]) - np.polyval(A, t[m])) ** 2)
        if resid < best[0]:
            best = (resid, -A[0], F)
    return best[1], best[2]


def main():
    print("=" * 82)
    print("FMCL PAPER 5 -- CONVERGENCE ON THE CHARGING-COUPLED FLEET")
    print("Resolving F19 by fixing the budget instead of the target")
    print("=" * 82)

    N, K, HOURS = 100, 30, 6.0
    SEEDS = range(3)
    POLICIES = ["random", "static_score", "energy_only", "lyapunov",
                "charger_aware"]
    ALL = None
    clients, NC, DIM = make_federated_data(N=N, K=4, dim=10, n_per=50,
                                           alpha=0.3, seed=0)

    VARIANTS = [(p, False) for p in POLICIES] + \
               [("charger_aware", True), ("lyapunov", True)]
    res = {}
    for p, db in VARIANTS:
        key = p + (" +debias" if db else "")
        runs = [run(p, clients, NC, DIM, n=N, K=K, hours=HOURS, seed=s,
                    debias=db) for s in SEEDS]
        g = np.mean([r["grads"] for r in runs], axis=0)
        r_, f_ = rate_and_floor(g)
        res[key] = {"grads": g, "rate": r_, "floor": f_,
                  "plateaued": bool(abs(g[-1] - g[int(0.8*len(g))]) / g[-1] < 0.25),
                  **{k: float(np.mean([r[k] for r in runs]))
                     for k in ("final_quality", "energy_J", "work_s",
                               "cycling_max", "dropout_rate",
                               "n_contributors", "contrib_gini", "auc")}}

    print(f"\n(a) PRIMARY -- QUALITY AT A FIXED BUDGET (one {HOURS:.0f}-hour "
          f"overnight window)\n")
    print(f"  {'policy':<22}{'final ||g||^2':>15}{'vs best':>10}{'energy(MJ)':>12}"
          f"{'work(ks)':>10}{'max fatigue':>12}")
    print("  " + "-" * 81)
    ALL = list(res)
    best = min(res[p]["final_quality"] for p in ALL)
    for p in ALL:
        a = res[p]
        print(f"  {p:<22}{a['final_quality']:>15.3e}"
              f"{a['final_quality']/best:>10.2f}x{a['energy_J']/1e6:>12.2f}"
              f"{a['work_s']/1000:>10.1f}{a['cycling_max']:>12.0f}")
    print("  " + "-" * 74)
    print("  No censoring, no unreachable targets: every policy gets the same")
    print("  window and is scored on what it achieved inside it.")

    print(f"\n(b) DIAGNOSTIC -- RATE versus FLOOR\n")
    print(f"  {'policy':<22}{'rate':>10}{'fitted floor':>15}{'ident?':>8}"
          f"{'dropout':>9}{'contrib':>9}{'Gini':>7}")
    print("  " + "-" * 80)
    for p in ALL:
        a = res[p]
        print(f"  {p:<22}{a['rate']:>10.4f}{a['floor']:>15.3e}"
              f"{'yes' if a['plateaued'] else 'NO':>8}"
              f"{a['dropout_rate']:>9.1%}{a['n_contributors']:>9.0f}"
              f"{a['contrib_gini']:>7.3f}")
    print("  " + "-" * 78)
    print("  A policy can lose on rate, on floor, or on both. Thermal dropout")
    print("  raises the floor by narrowing who actually contributes.")

    print(f"\n(c) ROBUSTNESS -- AUC of quality against cumulative energy\n")
    print(f"  {'policy':<22}{'AUC (lower better)':>22}{'rank':>8}")
    print("  " + "-" * 55)
    order = sorted(ALL, key=lambda p: res[p]["auc"])
    for p in ALL:
        print(f"  {p:<22}{res[p]['auc']:>22.3e}{order.index(p)+1:>8}")
    print("  " + "-" * 55)
    ra = [order.index(p) for p in ALL]
    rb = [sorted(ALL, key=lambda q: res[q]["final_quality"]).index(p)
          for p in ALL]
    print(f"  Rank agreement with the fixed-budget metric: "
          f"{np.corrcoef(ra, rb)[0,1]:+.3f}")

    print("\n" + "=" * 82)


if __name__ == "__main__":
    main()
