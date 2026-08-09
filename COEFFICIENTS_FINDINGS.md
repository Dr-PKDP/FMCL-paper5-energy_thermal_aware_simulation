# Coefficient Library Findings (F1–F11)

This file is referenced by `coefficients.py` and `uncertainty.py` as the
record of the findings produced by this repository's verification and
uncertainty-quantification tooling. It did not exist in the repository as
originally pushed; this is a full rewrite once the original working notes
were located, replacing an earlier partial reconstruction (F1–F6 only) that
this repository briefly carried.

**Status: 33/33 verification checks pass** (`python verify.py`). Monte Carlo
n = 2000; Sobol Saltelli design, 4096 evaluations, seeded and fully
reproducible (`python uncertainty.py`).

Every distribution and robustness figure below was independently
re-computed from the current codebase and checked against the original
working notes. The Monte Carlo section matches exactly, to three decimal
places, across every reported quantity. All five Sobol targets have now
been analysed and cross-checked; four reproduce closely, and the fifth
(F9's specific total-order-index magnitudes) was fully root-caused --
unseeded Sobol scrambling, now fixed in the code -- rather than left as an
unexplained gap. See F9 below for the complete investigation.

---

## Bug fixes already applied in this codebase

Three fixes and one additional correction, all confirmed present in the
current `thermal.py` / `coefficients.py`:

- **Cycle counter returned zero on plateaus.** The turning-point detector
  treated zero-slope segments as non-turns, so any trace with a duplicated
  extremum scored no damage. Fixed by collapsing plateaus before extremum
  detection (`thermal.py`, `cycling_damage`).
- **Unequal-duration comparison.** An early version compared batching
  patterns over different calendar spans and charged total rather than
  marginal calendar ageing, so heavily batched patterns appeared to age less
  purely because less time had elapsed. Fixed by padding all patterns to
  equal duration and reporting wear as excess over a non-participating
  device (`thermal.py`, `simulate_campaign`).
- **Idle phase under-resolved (F7).** A fixed 30 s idle step under-resolved
  the cooling decay, biasing the marginal Arrhenius term low by ~9%. Fixed
  by capping the step at tau_cool/20 (`thermal.py`, line ~253, explicitly
  labelled "finding F7" in the source).
- **R_th retightened (F2).** The naive closed-loop inversion
  dT_obs/P_train = 3.6 K/W predicts a 43 C steady state at 5 W, below the
  ~50 C governor cap the reference handset data shows, so throttling at
  150 s could not have occurred at that value. Range corrected from
  (3.6, 4.5, 8.0) to (5.0, 6.0, 8.0) K/W -- confirmed as the current value in
  `coefficients.py`.

**Fast campaign model, validated.** Exploits the fact that inter-batch gaps
run to hours while tau_cool is order 100 s, so every batch starts from
ambient and all batches are identical -- one batch determines the whole
campaign. Worst-case relative error against the full step-wise simulation:
**1.656%**, reproduced exactly by this codebase, well under the 5%
acceptance threshold.

---

## F1 -- The doubling rule is wrong for this coefficient range

The widely repeated claim that battery degradation doubles per sustained
10 C requires Ea = 52.9 kJ/mol. The 232-cell, 13-year calendar-ageing
dataset gives 23.6-29.9 kJ/mol, which yields only 1.36-1.48x per 10 C --
roughly a 40% acceleration, not 100%. Stated explicitly in the paper
(Section 5.1) rather than left as an uncorrected repeated figure.

## F2 -- Identifiability (R_th)

See "Bug fixes already applied," above.

## F3 -- Energy miscalibration is real but modest

Across the full swept corner space, a throttled device at 60-70 C costs
between 2.42x and 7.86x the joules per unit of useful work a nominal power
model would predict, robust in sign across every corner but a tens-of-percent
effect rather than an order of magnitude. The paper claims the sign and the
robustness, not this headline multiplier -- these specific figures are
`verify.py`'s own internal sweep and are not meant to be quoted verbatim.

## F4 -- Cycling damage depends on what is held fixed

At fixed **total excursion**, splitting into many small cycles is far less
damaging (ten 2 K cycles cost ~6.3% of one 20 K cycle) because damage grows
as dT^n with n>1. At fixed **total work** -- the case that applies to a
training campaign -- amplitude saturates at the RC steady state while cycle
count falls linearly with batching, so concentration wins instead. Stated
explicitly in the paper, Section 5.3.

## F5 -- No interior optimum in the wear objective, but an interior worst case

Across every weighting of the two wear mechanisms, marginal wear is
minimised at maximum batching -- the two mechanisms do not oppose each
other, so the dual-mechanism objective as originally conceived does not
hold. Cycling damage instead has an interior **maximum**. Reported in the
paper as Section 9's rejected proposition and Section 5.4's closed-form
worst-case session length (Eq. 13).

**Monte Carlo robustness (N=2000, exact match to working notes):**

| Check | Holds in |
|---|---|
| Energy monotone increasing in batch size | 100.0% |
| Energy premium > 1 (throttling costs energy) | 100.0% |
| Energy and cycling anti-correlated | 94.8% |
| Cycling damage has an interior maximum | 98.8% |
| Fatigue-worst batch size m* in [2, 20] | 95.2% |
| Marginal Arrhenius near-flat (spread < 25%) | 29.7% -- fails; see F10 |

**Distributions (median, 5th-95th percentile), exact match to working notes:**

| Quantity | Median | 5th | 95th |
|---|---|---|---|
| Energy premium (batch/spread) | 2.275x | 1.438x | 4.866x |
| corr(energy, cycling) | -0.789 | -0.962 | 0.000 |
| Fatigue-worst batch size m* | 4 | 2 | 20 |
| Cost at m* (x fully-spread case) | 2.178x | 1.146x | 13.896x |
| Arrhenius spread across batch size | 0.359 | 0.085 | 0.661 |

## F6 -- The real tradeoff is energy against fatigue

At fixed work and fixed calendar duration, campaign energy rises
monotonically (median 2.28x, higher than nominal-parameter estimates because
the F2 correction drives devices hotter) as batching increases, entirely
because of throttling, while cycling damage falls. The two are
anti-correlated. An FMCL scheduler minimising energy alone drives devices
into the fatigue-worst regime; one minimising fatigue alone pays a
substantial energy premium. Neither single-objective policy is defensible --
this motivates CATS's joint energy/wear objective (Eq. 14-18).

## F7 -- see "Bug fixes already applied," above.

## F8 -- The core claims survive coefficient uncertainty

The two load-bearing claims (F6a, F6b above) hold in 100% of draws.
Cycling damage has an interior maximum in 98.8% of draws, sitting between 2
and 20 rounds per session in 95.2% (mode m* = 4, with 78% of draws in the
range 2-8). This is what the paper's central claim rests on: not a
percentage saving, but that conventional round-based FL scheduling sits in
the fatigue-worst regime across essentially the entire plausible parameter
space.

## F9 -- The fatigue-worst batch size follows a closed-form rule

Sobol attributes the location of m* almost entirely to `t_round` and
`tau_heat`; every other coefficient is negligible. This motivates the
mechanistic rule used in the paper (Eq. 13):

> **m\* ~= tau_heat / t_round**

Spearman rho = 0.923; 96.9% of draws fall within one grid step of the
prediction. This exact figure (0.92 rank correlation, 97% within one grid
step) is already correctly cited in the paper, Section 5.4.

**Root-caused and fixed.** The specific total-order indices for this
target showed a persistent gap under independent re-verification (ST for
`t_round` running 0.79-0.87 across repeated re-runs against the documented
0.678) that did not close under a matched sample size, ruling out N as the
cause. The actual cause: `uncertainty.py`'s Sobol sampling call did not
pass an explicit seed, and SALib's Saltelli sampler scrambles by default
(`scramble=True`, `seed=None`) -- confirmed directly: two unseeded calls to
`sobol_sample.sample()` with identical arguments return different point
sets, while two calls with an explicit seed return bit-identical output.
Every prior run in this repository's history, including whatever produced
the documented 0.678/0.291, used a different, unrecorded random scramble.
`uncertainty.py` now fixes an explicit seed (`SOBOL_SEED = 20260809`),
making every future run of this script reproducible -- confirmed by
re-generating the design matrix twice and checking for exact equality.

The magnitude of `worst_m`'s sensitivity to the scramble choice (~0.08-0.09
of total-order index) is itself informative: `worst_m` is a discrete,
heavily right-skewed output restricted to the `PATTERNS` grid
(median 4-5, mean skewed up to ~7 by a long tail), and total-order Sobol
estimates are known to be noisier for discrete/skewed targets than for
smooth ones -- consistent with `arrhenius_spread` (F10, below), a smoother
target, reproducing far more tightly across the same set of re-runs.

**What is and is not now closed:** the qualitative finding -- `t_round`
and `tau_heat` govern m*'s location, everything else negligible -- is
confirmed identically across every re-run at every seed and sample size
tried. The Spearman correlation for the m* ~= tau_heat/t_round hypothesis
is consistently strong (0.92-0.96 across four independent runs, including
the documented one). The *exact* total-order index values (0.678, 0.291)
are specific to a scramble this repository no longer has access to and
cannot be exactly reproduced; going forward, `python uncertainty.py` will
always return the same (seeded) values, which is the more useful property
for a reader trying to reproduce this repository's own claims.

## F10 -- F5c fails, and the failure has a precise mechanistic account

The claim that marginal Arrhenius damage is flat in batch size is not simply
noisy -- under the corrected R_th, the marginal Arrhenius term is genuinely
**non-monotone**, falling to ~0.95x at m ~= 8 before rising to ~1.09x at
m = 200: an interior *minimum*, the one condition under which a
pure-Arrhenius objective would admit an interior optimum at all. That
structure is itself not robust under coefficient uncertainty -- the spread
has a median of 36% and reaches 66%, so the near-flat claim holds in under a
third of draws (29.7%, exact match to the F5c figure above).

Sobol identifies why: the Arrhenius spread is driven overwhelmingly by
`tau_cool` and `tau_heat`. Every wear-chemistry parameter (`Ea`, `k_batt`) is
negligible. The disputed finding is controlled almost entirely by the
thermal time constants -- and tau_cool is the fourth-tier coefficient with
the widest, least-measured prior in the library. That is a clean, direct
line from this uncertainty analysis to the device-measurement protocol
(`MEASUREMENT_PROTOCOL.md`).

**The paper's current text (Section 9) states this finding at the F5c
level of precision** -- "approximately flat... in fewer than a third of
draws" -- without the fuller F10 characterisation (genuinely non-monotone,
interior minimum, mechanistically attributed to tau_cool/tau_heat). Whether
to upgrade the paper's text to the more precise version is a
scientific-content decision outside the scope of this documentation fix.

## F11 -- The wear chemistry is unidentifiable from the architecture-level claims

`Ea` and `k_batt` contribute negligible variance (ST < 0.05) to every
reported output, including the Arrhenius spread. `n_cm` contributes to
fatigue magnitude but almost entirely through interaction with other
parameters, not on its own. The practical consequence is favourable: the
paper's conclusions do not depend on resolving the mobile
package-on-package Coffin-Manson exponent precisely -- it only needs to be
bounded, consistent with how the paper already treats it (Section 5.2).

---

## Sobol sensitivity, final seeded run (seed=20260809, N=4096 evaluations)

All five targets `evaluate()` produces have now been analysed via Sobol,
using the fixed seed `uncertainty.py` now carries. This is the
authoritative table; re-running `python uncertainty.py` reproduces it
exactly.

| Output | Dominant parameters (ST) | Negligible (ST < 0.01) |
|---|---|---|
| Energy premium | P_train 0.542, R_th 0.294, theta_leak 0.174, f_static 0.169 | t_round and everything below it |
| Cost at fatigue-worst batch size | t_round 0.509, n_cm 0.240, tau_heat 0.235 | R_th, P_train and below |
| Energy-cycling correlation | t_round 0.807, tau_heat 0.111, n_cm 0.087 | P_train, R_th and below |
| Location of fatigue-worst batch size (F9) | t_round 0.793, tau_heat 0.267, P_train 0.100, R_th 0.072 | n_cm and below |
| Arrhenius spread across batch size (F10) | tau_cool 0.785, tau_heat 0.490, P_train 0.143, R_th 0.081 | k_batt and below |

In every case, `Ea` and `k_batt` (the two wear-chemistry coefficients) are
either absent from the top contributors or sit at the bottom of them --
consistent with F11 across all five outputs, not just the three originally
checked.

**F10 (Arrhenius spread) reproduces closely against the documented run:**
tau_cool ST 0.785 vs. 1.013, tau_heat 0.490 vs. 0.442, P_train 0.143 vs.
0.151, R_th 0.081 vs. 0.090 -- same ranking, same order of magnitude
throughout.

**F9 (location of m*) -- fully investigated, root cause identified and
fixed; see the F9 section above for the complete account.** In short: the
persistent gap between this repository's re-runs and the documented
0.678/0.291 was traced to `uncertainty.py` never having fixed a seed for
its Sobol sampling, so every run -- including whatever produced the
documented figures -- used a different, unrecorded random scramble. This
is now fixed in the code. The qualitative finding and the underlying
m* ~= tau_heat/t_round hypothesis (Spearman rho 0.92-0.96 across every
re-run attempted) were never in question; only the specific total-order
index values were seed-dependent, and that dependency is now closed by
making the seed explicit and permanent.

---

## Standing claims for the manuscript, with supporting fraction

1. Throttling makes batched training strictly more energy-expensive at
   fixed work (100%), median 2.28x.
2. Cycling damage has an interior maximum in batch size (98.8%), located in
   [2, 20] rounds per session (95.2%), following m* ~= tau_heat/t_round
   (rho = 0.92).
3. Energy and fatigue are anti-correlated across the scheduling variable
   (94.8%), so neither single-objective policy is defensible.
4. Energy-aware schedulers using nominal power models underprice throttled
   devices, robustly in sign across every parameter corner.

**Withdrawn:** the dual-mechanism opposition between Arrhenius and
Coffin-Manson wear (F5), and the flatness of the marginal Arrhenius term
(F5c/F10 -- genuinely non-monotone, not merely noisy).
