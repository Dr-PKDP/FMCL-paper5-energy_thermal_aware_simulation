"""
Convert FedScale's real per-device session data into a round-by-round
availability matrix, for driving fleet_charging's simulation with real
temporal structure instead of i.i.d. Bernoulli draws.

Session data: behave[device_id] = {'active': [...], 'inactive': [...], ...}
Each (active[j], inactive[j]) pair is one continuous eligible session,
in seconds since trace start. A device is available at time t iff t falls
inside one of its sessions.
"""
import os
import pickle
import numpy as np

# Expected location: tracesim/data/client_behave_trace. This file is not
# part of this repository (it is ~25 MB and belongs to FedScale, not to
# this paper) -- see the top-level README for the exact download command.
# BEHAVE_PATH can also be overridden with the FMCL_FEDSCALE_TRACE
# environment variable if you keep the file elsewhere.
BEHAVE_PATH = os.environ.get(
    "FMCL_FEDSCALE_TRACE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "data", "client_behave_trace"),
)


def load_trace():
    if not os.path.exists(BEHAVE_PATH):
        raise FileNotFoundError(
            f"FedScale behaviour trace not found at {BEHAVE_PATH}. "
            "See the top-level README ('Real-trace validation, Section 7.7 "
            "and 7.9') for the download command, or set the "
            "FMCL_FEDSCALE_TRACE environment variable to its location."
        )
    with open(BEHAVE_PATH, "rb") as f:
        return pickle.load(f)


def device_availability_at_times(sessions, query_times):
    """
    sessions: dict with 'active', 'inactive' arrays (session start/end, s)
    query_times: 1-D array of times (s) to evaluate, e.g. round-start times
    Returns: boolean array, same length as query_times.
    """
    a, i = np.asarray(sessions["active"]), np.asarray(sessions["inactive"])
    n = min(len(a), len(i))
    a, i = a[:n], i[:n]
    avail = np.zeros(len(query_times), dtype=bool)
    for j in range(n):
        if i[j] <= a[j]:
            continue
        avail |= (query_times >= a[j]) & (query_times < i[j])
    return avail


def build_availability_matrix(behave, device_ids, t_round=65.0, t_gap=15.0,
                                n_rounds=270, window_starts=None, seed=0):
    """
    device_ids: list of trace device keys to use, one simulated device each
    window_starts: per-device start time (s) into their own trace; if None,
      drawn uniformly from [0, finish_time - campaign_length) per device
    Returns: (n_devices, n_rounds) boolean matrix
    """
    rng = np.random.default_rng(seed)
    round_offsets = np.arange(n_rounds) * (t_round + t_gap)
    campaign_length = n_rounds * (t_round + t_gap)

    mat = np.zeros((len(device_ids), n_rounds), dtype=bool)
    starts = np.zeros(len(device_ids))
    for k, did in enumerate(device_ids):
        sess = behave[did]
        finish = sess.get("finish_time", 0)
        if window_starts is not None:
            t0 = window_starts[k]
        else:
            hi = max(finish - campaign_length, 1)
            t0 = rng.uniform(0, hi)
        starts[k] = t0
        query_times = t0 + round_offsets
        mat[k] = device_availability_at_times(sess, query_times)
    return mat, starts
