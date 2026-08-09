"""
Reproduction attempt for Table 11 (propensity truncation sweep, Section 8.6).

Table 11 sweeps the propensity-truncation floor c in Eq. 23 against the
untuned (V=1) CATS policy, reporting final ||grad F||^2, convergence rate,
and fitted floor G_inf, at c in {none, 0.05, 0.10, 0.20, 0.35}.

THIS SCRIPT DOES NOT REPRODUCE TABLE 11. It is kept in the repository as a
documented record of what was tried and how far off each attempt landed,
so a future attempt does not have to retrace the same ground blind.

Four variants were tried, none converging to the documented values:

  1. SEEDS=range(3), matching converge_charging.py's own reference main()
     for other debias-related runs. Results were far off (rate ~0.015
     across every c, floor collapsing to ~1e-08 for c in {0.05,0.10,0.20} --
     a suspiciously flat pattern suggesting the fitted floor grid was
     saturating, not that c had no effect).
  2. SEEDS=range(6), matching Table 9/10's convention. Closer for "none"
     and c=0.35 (floor landed within ~15% of documented), but c=0.05 barely
     moved from the "none" baseline where the paper shows its largest drop.
  3. A near-zero floor (c=0.001), tested as an approximation to the
     "unbiased 4.61e-06" reference point named in Section 8.6's prose.
     Result: floor=9.4e-06, roughly double the documented unbiased
     reference, at the theoretical limit where truncation is negligible.
     This is the strongest evidence the gap is structural rather than a
     seed or sampling artifact: even at the limit, this reconstruction
     does not approach the paper's stated reference point.
  4. c=0.01, similarly far off.

CONCLUSION: the discrepancy is consistent across every variant tried and
does not shrink with more seeds, so it is very unlikely to be sampling
noise. The most likely explanation is a setup difference this repository's
current converge_charging.py does not capture -- a different alpha, a
different quality metric, or a debiasing formula that differs from what's
implemented here -- but this has not been identified. Reproducing Table 11
correctly needs either the original script or more specific guidance on
what differs; guessing further variants has diminishing returns and was
stopped here rather than continued indefinitely.

The paper's Table 11 has NOT been changed on the basis of this
investigation. This repository's numbers are not trusted enough to
justify overwriting the paper's printed values with a reconstruction this
uncertain.
"""
import time
import numpy as np
from simulation import make_federated_data
from converge_charging import run, rate_and_floor

DOCUMENTED = {
    "none": (1.255e-05, 0.0194, 1.09e-05),
    0.05:   (9.84e-06,  0.0201, 5.92e-06),
    0.10:   (8.59e-06,  0.0205, 6.83e-06),
    0.20:   (9.60e-06,  0.0199, 7.82e-06),
    0.35:   (1.17e-05,  0.0194, 9.86e-06),
}
UNBIASED_REFERENCE = 4.61e-06   # named in Section 8.6's prose, not a table row


def run_variant(seeds, label):
    clients, NC, DIM = make_federated_data(N=100, K=4, dim=10, n_per=50,
                                            alpha=0.3, seed=0)
    t0 = time.time()
    results = {}

    runs = [run("charger_aware", clients, NC, DIM, n=100, K=30, hours=6.0,
                seed=s, debias=False) for s in seeds]
    g = np.mean([r["grads"] for r in runs], axis=0)
    r_, f_ = rate_and_floor(g)
    results["none"] = (float(g[-1]), r_, f_)

    for c in [0.05, 0.10, 0.20, 0.35]:
        runs = [run("charger_aware", clients, NC, DIM, n=100, K=30, hours=6.0,
                    seed=s, debias=True, propensity_floor=c) for s in seeds]
        g = np.mean([r["grads"] for r in runs], axis=0)
        r_, f_ = rate_and_floor(g)
        results[c] = (float(g[-1]), r_, f_)

    print(f"\n--- {label} ({time.time()-t0:.0f}s) ---")
    print(f"{'c':<10}{'final':>12}{'rate':>9}{'floor':>12}   documented (final/rate/floor)")
    for c in ["none", 0.05, 0.10, 0.20, 0.35]:
        final, rate, floor = results[c]
        df, dr, dfl = DOCUMENTED[c]
        print(f"{str(c):<10}{final:>12.3e}{rate:>9.4f}{floor:>12.3e}   "
              f"({df:.3e}/{dr:.4f}/{dfl:.3e})")
    return results


if __name__ == "__main__":
    print("=" * 78)
    print("TABLE 11 REPRODUCTION ATTEMPT -- see module docstring for full context")
    print("=" * 78)
    run_variant(range(3), "Variant 1: seeds=3 (matches converge_charging.py's own reference)")
    run_variant(range(6), "Variant 2: seeds=6 (matches Table 9/10 convention)")

    print("\n--- Variant 3/4: near-zero floor, approximating the 'unbiased "
          f"{UNBIASED_REFERENCE:.2e}' reference named in Section 8.6 ---")
    clients, NC, DIM = make_federated_data(N=100, K=4, dim=10, n_per=50,
                                            alpha=0.3, seed=0)
    for c in [0.001, 0.01]:
        runs = [run("charger_aware", clients, NC, DIM, n=100, K=30, hours=6.0,
                    seed=s, debias=True, propensity_floor=c) for s in range(6)]
        g = np.mean([r["grads"] for r in runs], axis=0)
        r_, f_ = rate_and_floor(g)
        print(f"c={c}: final={g[-1]:.3e}  rate={r_:.4f}  floor={f_:.3e}  "
              f"(documented unbiased reference: {UNBIASED_REFERENCE:.2e})")

    print("\nNone of the above converges cleanly to Table 11. See docstring "
          "for the conclusion.")
