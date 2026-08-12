"""
Reproduces the three empirical-grounding checks reported in Section 7.7:
session-length distribution, concurrent-availability sampling, and
compute-heterogeneity spread. Run `bash setup_data.sh` first (or place
both files manually; see that script's header for expected paths).

This script exists because these three checks were previously run ad hoc
and never committed as reusable code -- the numbers in the paper are
correct (independently re-verified), but the analysis that produced them
was not reproducible from this repository alone until now.
"""
import os
import pickle
import statistics
from collections import Counter

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BEHAVE_PATH = os.environ.get("FMCL_FEDSCALE_TRACE",
                              os.path.join(DATA_DIR, "client_behave_trace"))
CAPACITY_PATH = os.environ.get("FMCL_FEDSCALE_CAPACITY",
                                os.path.join(DATA_DIR, "client_device_capacity"))


def _require(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{what} not found at {path}. Run `bash setup_data.sh` first."
        )


# ---------------------------------------------------------------------
# Check 1 & 2: session-length distribution and concurrent availability
# ---------------------------------------------------------------------

def session_length_check(behave):
    """First finding: session-length distribution against the six-hour
    simulated window. Returns (total_sessions, median_min, mean_min,
    n_3hr_plus, pct_3hr_plus, mean_3hr_hours, median_3hr_hours)."""
    all_sessions = []
    for sess in behave.values():
        a, i = sess.get("active", []), sess.get("inactive", [])
        n = min(len(a), len(i))
        all_sessions.extend(i[j] - a[j] for j in range(n) if i[j] > a[j])

    sessions_min = [s / 60 for s in all_sessions]
    three_hr = [s for s in all_sessions if s >= 3 * 3600]
    three_hr_h = [s / 3600 for s in three_hr]

    return {
        "total_sessions": len(all_sessions),
        "median_min": statistics.median(sessions_min),
        "mean_min": statistics.mean(sessions_min),
        "n_3hr_plus": len(three_hr),
        "pct_3hr_plus": 100 * len(three_hr) / len(all_sessions),
        "mean_3hr_hours": statistics.mean(three_hr_h),
        "median_3hr_hours": statistics.median(three_hr_h),
    }


def concurrency_check(behave, window_days=6, n_samples=8):
    """Second finding: concurrent-availability fraction, sampled at
    n_samples points across window_days. Only devices whose recorded
    trace still covers a given sample point are counted in that sample's
    denominator (a device whose trace has ended is excluded, not treated
    as unavailable) -- see the paper's Section 7.7 for why this matters
    for reading fleet-size assumptions correctly."""
    device_intervals, finish = {}, {}
    for did, sess in behave.items():
        a, i = sess.get("active", []), sess.get("inactive", [])
        n = min(len(a), len(i))
        intervals = [(a[j], i[j]) for j in range(n) if i[j] - a[j] >= 3 * 3600]
        if intervals:
            device_intervals[did] = intervals
        finish[did] = sess.get("finish_time", 0)

    span = window_days * 86400
    fractions = []
    for k in range(n_samples):
        t = span * (k + 0.5) / n_samples
        covered = [did for did, ft in finish.items() if ft >= t]
        concurrent = sum(
            1 for did in covered
            if any(s <= t <= e for (s, e) in device_intervals.get(did, []))
        )
        fractions.append(concurrent / len(covered) if covered else 0.0)

    return {
        "mean_pct": 100 * statistics.mean(fractions),
        "min_pct": 100 * min(fractions),
        "max_pct": 100 * max(fractions),
        "per_sample_pct": [100 * f for f in fractions],
    }


def model_diversity_check(behave):
    """Device-identity diversity: distinct model strings and the largest
    single model's share of the population."""
    models = [v.get("model", "UNKNOWN") for v in behave.values()]
    counts = Counter(models)
    top_model, top_count = counts.most_common(1)[0]
    return {
        "n_devices": len(models),
        "n_distinct_models": len(set(models)),
        "top_model": top_model,
        "top_model_pct": 100 * top_count / len(models),
    }


# ---------------------------------------------------------------------
# Check 3: compute heterogeneity (separate dataset -- client_device_capacity)
# ---------------------------------------------------------------------

def compute_heterogeneity_check(capacity):
    """Third finding: fastest/slowest quartile-median spread in per-device
    inference latency ('computation' field; lower is faster)."""
    comp = np.array([v["computation"] for v in capacity.values()])
    sorted_comp = np.sort(comp)
    n = len(sorted_comp)
    med_fast = np.median(sorted_comp[: n // 4])
    med_slow = np.median(sorted_comp[3 * n // 4:])
    return {
        "n_measurements": len(comp),
        "fastest_quartile_median": float(med_fast),
        "slowest_quartile_median": float(med_slow),
        "fold_difference": float(med_slow / med_fast),
    }


def main():
    _require(BEHAVE_PATH, "client_behave_trace")
    _require(CAPACITY_PATH, "client_device_capacity")

    with open(BEHAVE_PATH, "rb") as f:
        behave = pickle.load(f)
    with open(CAPACITY_PATH, "rb") as f:
        capacity = pickle.load(f)

    print(f"Devices in behaviour trace: {len(behave)}")
    print()

    print("=== Check 1: session-length distribution ===")
    r1 = session_length_check(behave)
    print(f"  Total sessions:        {r1['total_sessions']}")
    print(f"  Median length:         {r1['median_min']:.1f} min")
    print(f"  Mean length:           {r1['mean_min']:.2f} min")
    print(f"  Sessions >= 3h:        {r1['n_3hr_plus']} "
          f"({r1['pct_3hr_plus']:.2f}% of all sessions)")
    print(f"  Mean/median (>=3h):    {r1['mean_3hr_hours']:.2f}h / "
          f"{r1['median_3hr_hours']:.2f}h")
    print()

    print("=== Check 2: concurrent-availability sampling ===")
    r2 = concurrency_check(behave)
    print(f"  Mean concurrent fraction: {r2['mean_pct']:.2f}%")
    print(f"  Range: {r2['min_pct']:.2f}% - {r2['max_pct']:.2f}%")
    print(f"  Per-sample: {[round(p, 2) for p in r2['per_sample_pct']]}")
    print()

    print("=== Check 3: compute heterogeneity ===")
    r3 = compute_heterogeneity_check(capacity)
    print(f"  Measurements:           {r3['n_measurements']}")
    print(f"  Fastest quartile median: {r3['fastest_quartile_median']:.1f}")
    print(f"  Slowest quartile median: {r3['slowest_quartile_median']:.1f}")
    print(f"  Fold difference:         {r3['fold_difference']:.2f}")
    print()

    print("=== Device model diversity ===")
    r4 = model_diversity_check(behave)
    print(f"  Distinct model strings: {r4['n_distinct_models']}")
    print(f"  Largest share:          {r4['top_model']!r} at "
          f"{r4['top_model_pct']:.2f}%")

    with open("trace_summary_stats_results.pkl", "wb") as f:
        pickle.dump({"session": r1, "concurrency": r2,
                     "compute": r3, "models": r4}, f)


if __name__ == "__main__":
    main()
