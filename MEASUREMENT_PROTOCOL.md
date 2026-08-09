# Device Measurement Protocol — FMCL Paper 5

Designed for **two smartphones and one tablet**, no programming, no lab
equipment. Total effort: roughly one afternoon per device.

The purpose is not to demonstrate federated learning on hardware. It is to
close the coefficients that currently carry the widest priors in the model,
above all the two admission thresholds that the paper's own sensitivity
analysis identifies as the largest remaining source of uncertainty, and to
check that the response *shape* calibrated from 2014–2018 devices still holds
on current hardware.

---

## What this buys

| Coefficient | Current status | After measurement |
|---|---|---|
| `T_ceil`, `T_kill` | T4, unmeasured — **the single largest source of uncertainty in the whole model** | T1 or T2, measured or tightly bounded |
| `tau_cool` | T4, prior 55–400 s (span 2.56, widest in library) | T1, measured directly |
| `k_batt` | T4, prior 0.30–0.85, scales the entire Arrhenius arm | T1, measured directly |
| `R_th` | T3, and currently **inconsistent** (finding F2) | T1, resolves the inconsistency |
| `tau_heat` | T3, derived through a chain of assumptions | T1, measured directly |
| `eta_min` | T1 but from 2014–2018 devices | Confirmed or corrected on current hardware |

Six of the model's weakest parameters, from three devices you already own.
`T_ceil` and `T_kill` were added to this list after the sensitivity analysis
in the paper: they turned out to dominate the variance in the worst-device
wear result more than everything else in the model combined, so they are
listed first rather than last, even though they are also the hardest of the
six to get cleanly.

---

## A note on T_ceil and T_kill before you start

These two are a different kind of measurement from the other four, and it is
worth understanding why before running the protocol.

`R_th`, `tau_heat`, `tau_cool`, `k_batt`, and `eta_min` all describe how fast
the phone heats up, cools down, and slows its CPU clock — this is **frequency
throttling**, and the throttling app in Phase 2 measures it directly and
correctly, because a foreground stress-test app experiences frequency
throttling like any other running app.

`T_ceil` and `T_kill` describe something else: the point at which Android
decides to **suspend background work entirely**, independent of clock speed.
This matters specifically for federated learning because FL training runs in
the background while the device is charging, not as the app the user is
looking at. Android treats background work more harshly under thermal stress
than foreground work — Google's own documentation confirms that background
job scheduling is throttled and eventually paused as thermal severity rises,
separately from CPU frequency scaling. A foreground throttling app will not
reliably show this effect, because foreground apps are given more leeway.

There is no app that reports this cleanly without writing code. There are two
ways to get a usable number anyway, in increasing order of effort.

**Option A — no new tools, uses data you are already collecting.** During
Phase 2 and Phase 4 (below), watch the throttling app's own throughput
reading, not just the temperature log. Frequency throttling shows up as a
*gradual decline* in throughput as temperature rises. If, at some point, the
plotted throughput does not just decline but **drops to exactly zero for a
sustained stretch** and then recovers once the phase ends, that is Android
suspending the task rather than just slowing it — a different phenomenon
from ordinary throttling, and the one this coefficient describes. The
processor temperature at the moment throughput first hits zero is a working
estimate of `T_kill`; if there is a visible two-stage drop, a first knee
where throughput drops sharply but not to zero, and a second where it hits
zero, the first knee is a working estimate of `T_ceil`. This costs nothing
beyond reading the trace you already have more carefully, and it is the
option to use by default.

**Option B — more precise, needs a one-time setup, no programming.** Android
has an official thermal status the phone already tracks internally
(`none → light → moderate → severe → critical`, roughly), and this can be
read from a computer over a USB cable using a small free tool called ADB
(Android Debug Bridge), without writing any code, just running one command
repeatedly. This is the same official mechanism Google's documentation says
background job throttling is tied to, so it maps more directly onto
`T_ceil`/`T_kill` than the throughput-based proxy in Option A. If Option A's
estimate looks unstable across the two device runs, this is worth doing; if
Option A gives a consistent number on both runs, it is probably not
necessary. Ask if you want the exact three commands to run — it takes about
ten minutes to set up per device and no code is involved, only copying
commands into a terminal window exactly as given.

Whichever option is used, report which one it was. A number from Option A
should be described in the manuscript as a throughput-based estimate; a
number from Option B as a directly observed OS thermal state. They are not
claimed to be equally precise, and the paper should not blur that
distinction.

---

## Tools

All free, all Play Store, no root required.

- **A sustained-load throttling test.** Any app that runs a fixed compute
  kernel repeatedly and plots achieved throughput against elapsed time. The
  resulting curve *is* the throttling response η(T) projected onto time.
- **A sensor logger** that records to CSV: SoC/CPU temperature, battery
  temperature, CPU frequency per cluster, battery current, battery voltage.
  Sampling at 1 Hz is ample; the thermal time constants are order 100 s.
- Optional: an app that loads a custom TensorFlow Lite model, if you want the
  load to be a neural workload rather than a generic stress kernel.

**Log both temperature sensors simultaneously.** Battery temperature and
SoC temperature are separate sensors and both are exposed on Android. Their
ratio *is* `k_batt`. This is the single highest-value measurement here and it
costs nothing extra.

---

## Protocol

Run identically on all three devices.

**Setup.** Airplane mode on, screen at minimum brightness, ambient temperature
noted, device flat on a hard surface (not on fabric, not in hand, not in a
case). Start from cold — at least 30 minutes idle. Battery between 50% and
80%, and **not charging**, so charging heat does not contaminate the trace.

**Phase 1 — Baseline (10 min).** Log while idle. Establishes T_amb as the
device sees it, and idle power from battery current × voltage.

**Phase 2 — Sustained load (20 min).** Start the throttling test and let it
run continuously. Captures: heating rate, τ_heat, throttle onset time, the
governor cap for this device, and the sustained/peak throughput ratio η_min.

**Phase 3 — Cooldown (30 min).** Stop the load. Keep logging. **Do not touch
the device.** This is the phase that yields τ_cool, which is not published
anywhere for mobile devices under this kind of load, and on which the entire
cycling argument depends. Give it the full 30 minutes even though it will look
finished after 10.

**Phase 4 — Cycling (30 min).** Alternate 2 minutes load, 3 minutes idle, six
times. This is the pattern an FL schedule actually produces, and no published
measurement covers it. It also directly tests finding F5 — whether short
back-to-back bursts land in the fatigue-worst regime the model predicts.

Repeat the whole sequence once per device on a different day. Two runs is
enough to see whether the time constants are stable; it is not enough for
statistics, and the write-up should not claim otherwise.

---

## Extraction

From the Phase 1–3 traces:

- **R_th** = (plateau temperature − ambient) / (loaded power − idle power),
  using measured power from battery current × voltage. This resolves F2
  directly, since it does not go through the closed-loop inversion that failed.
- **τ_heat** — fit T(t) = T_ss + (T₀ − T_ss)·exp(−t/τ) to the Phase 2 rise, or
  read off the time to reach 63.2% of the total rise.
- **τ_cool** — same fit to Phase 3, with T_ss = ambient.
- **k_batt** = (battery ΔT) / (SoC ΔT) at the Phase 2 plateau.
- **η_min** = sustained throughput / peak throughput from the throttling app.
- **Governor cap** = the temperature at which frequency first drops.
- **T_ceil, T_kill** (Option A) — from the Phase 2 and Phase 4 throughput
  trace: `T_ceil` is the processor temperature at the first sharp drop in
  throughput that is not simple frequency throttling (a knee, not a smooth
  decline); `T_kill` is the processor temperature at which throughput first
  reaches exactly zero for a sustained stretch. Report both device runs
  separately rather than averaging, since two points is not enough to fit a
  distribution and should not be presented as one.
- **T_ceil, T_kill** (Option B) — from the ADB thermal-status log, the
  processor temperature at the first transition away from `none` gives
  `T_ceil`; the temperature at the transition into `severe` gives a working
  `T_kill`, since this is the status level at which Android's own
  documentation says background job scheduling is paused rather than merely
  slowed.

`thermal.py` already contains `tau_from_onset` and `time_to_reach`, which
invert the RC solution both ways, so the fits can be checked against closed
form rather than trusted from a curve-fitting routine.

---

## How to describe this in the manuscript

Not as a testbed. Not as an evaluation. Write it as what it is:

> A single-user characterisation across three consumer devices spanning
> capability tiers, used to recover thermal time constants, the SoC-to-battery
> thermal coupling, and a working estimate of the operating-system thresholds
> at which background training is suspended under thermal stress, and to
> confirm that the throttling response shape calibrated from published
> multi-device measurements remains valid on current-generation hardware.
> Quantitative calibration of the throughput derating and energy coefficients
> is taken from published multi-device measurements rather than from these
> three devices.

Stated that way it is honest, it is defensible, and it is more than most
modelling papers in this area offer. Overstating it is the only way it becomes
a liability.

**Limitations to state plainly:** three devices is not a sample; the load is a
compute-intensity proxy rather than federated training specifically; ambient
conditions are uncontrolled; consumer-grade battery current readings are
coarse; and the admission thresholds, if recovered by Option A, are inferred
from a throughput-based proxy rather than an OS-reported state, and should be
presented as such rather than as directly observed quantities on the same
footing as the other four coefficients. None of these undermine a
time-constant measurement, which is what is being claimed for the first four;
the admission thresholds should be claimed more cautiously than that.
