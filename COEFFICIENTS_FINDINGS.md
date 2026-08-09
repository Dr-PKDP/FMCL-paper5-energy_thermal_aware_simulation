# Coefficient Library Findings (F1–F6)

This file is referenced by `coefficients.py` and `uncertainty.py` as the
record of the six findings produced by this repository's verification and
uncertainty-quantification tooling. It did not exist in the repository as
pushed; the references were dead links. This file closes that gap with the
actual finding text, reproduced from `verify.py`'s own output, plus the
robustness results from `uncertainty.py`'s Monte Carlo (N=2000) and Sobol
(N=4096 evaluations) sweeps over the 14-coefficient uncertain space.

Regenerate the finding text at any time with `python verify.py`.

---

## F1 — The doubling rule is wrong for this coefficient range

The widely repeated claim that battery degradation doubles per sustained
10 °C requires Ea = 52.9 kJ/mol. The 232-cell, 13-year calendar-ageing
dataset gives 23.6–29.9 kJ/mol, which yields only 1.36–1.48x per 10 °C —
roughly a 40% acceleration, not 100%.

**Consequence:** the Arrhenius arm of the wear objective is substantially
weaker than the popular framing suggests, which shifts relative weight onto
the cycling arm. This is stated explicitly in the paper (Section 5.1) rather
than left as a repeated but unsupported figure.

## F2 — Identifiability

At the lower bound R_th = 3.6 K/W, the model predicts a steady state of only
43 °C, which never reaches the ~50 °C governor cap measured in the reference
handset data — so throttling at 150 s could not occur. The closed-loop
inversion of the observed temperature rise is therefore not merely a lower
bound; it is inconsistent with the observed onset. R_th must exceed roughly
5.00 K/W for a 5 W load to reach a 50 °C cap at all. This tightens the
library: R_th's lower bound was raised accordingly (documented in the paper,
Section 9.5/9.6).

## F3 — Energy miscalibration is real but modest

Across the full swept corner space, a throttled device at 60–70 °C costs
between 2.42x and 7.86x the joules per unit of useful work that a nominal
power model would predict. The effect is robust in sign across every corner
(supporting contribution C2's energy-miscalibration claim), but it is a
tens-of-percent effect rather than an order of magnitude. The paper claims
the sign and the robustness, not a headline multiplier — these specific
figures are this script's own internal sweep and are not meant to be quoted
verbatim in the manuscript.

## F4 — Cycling damage depends on what is held fixed

At fixed **total excursion**, splitting into many small cycles is far less
damaging (ten 2 K cycles cost ~6.3% of one 20 K cycle) because damage grows
as ΔT^n with n>1. But at fixed **total work** — the case that actually
applies to a training campaign — amplitude saturates at the RC steady state
while cycle count falls linearly with batching, so concentration wins
instead. The initial intuition that the two wear mechanisms oppose each
other was drawn from the first case and does not survive the second. This
constraint is stated explicitly in the paper (Section 5.3).

## F5 — No interior optimum in the wear objective, but an interior worst case

Across every weighting of the two wear mechanisms, marginal wear is
minimised at maximum batching — the mechanisms do not oppose each other, so
the dual-mechanism objective as originally conceived does not hold. The
marginal Arrhenius term is close to flat in batch size (0.948–1.089x),
because at these temperatures the rate multiplier is nearly linear in ΔT and
the time-integral of ΔT is fixed by total delivered energy rather than by
how it is distributed. What the sweep does reveal is sharper: cycling damage
has an interior **maximum** at m = 5 rounds per session in the illustrative
parameterisation used by `verify.py`, costing 2.40x the fully-spread case.
Short back-to-back bursts are the worst possible pattern for package
fatigue — precisely where a conventional round-based FL schedule sits. This
is reported in the paper as Section 9's rejected proposition and as the
closed-form worst-case session length of Section 5.4 (Eq. 13).

**Monte Carlo robustness (N=2000 draws, `uncertainty.py`):**

| Check | Fraction holding |
|---|---|
| Energy premium > 1 (throttling costs energy) | 100.0% |
| Energy monotone increasing in batch size | 100.0% |
| Cycling has an interior maximum | 98.8% |
| Worst-case m* in [2, 20] | 95.2% |
| Energy and cycling anti-correlated | 94.8% |
| Arrhenius term near-flat (spread < 25%) | 29.7% — the one finding that does *not* survive broadly; the disagreement is attributed almost entirely to τ_cool (fourth-tier, unmeasured). Reported in the paper as "fewer than a third of draws" (Section 9.3). |

## F6 — The real tradeoff is energy against fatigue

At fixed work and fixed calendar duration, campaign energy rises
monotonically from 26,749 J to 64,263 J (2.40x) as batching increases,
entirely because of throttling, while cycling damage falls. The two are
anti-correlated (r = −0.783). This is the tension the scheduler must
resolve: an FMCL scheduler that minimises energy alone drives devices into
the fatigue-worst regime, and one that minimises fatigue alone pays a 67%
energy premium. Neither single-objective policy is defensible — this
motivates CATS's joint energy/wear objective (Eq. 14–18).

**Sobol sensitivity (N=4096 evaluations, `uncertainty.py`):**

| Target | Dominant parameters (total-order index) | Negligible (< 0.01) |
|---|---|---|
| Energy premium | P_train (0.370), R_th (0.255), theta_leak (0.157), f_static (0.153) | tau_heat, tau_cool, T_amb, T_cap, eta_min, p_dvfs, **Ea**, **k_batt**, n_cm |
| Cost at fatigue-worst batch size | t_round (1.047), n_cm (0.267), tau_heat (0.159), P_train (0.133) | R_th, tau_cool, T_amb, T_cap, eta_min, f_static, theta_leak, p_dvfs, **Ea**, **k_batt** |
| Energy–cycling correlation | t_round (0.826), tau_heat (0.109), n_cm (0.083) | P_train, R_th, tau_cool, T_amb, T_cap, eta_min, f_static, theta_leak, p_dvfs, **Ea**, **k_batt** |

This confirms the paper's Section 9 claim that Ea (activation energy) and
k_batt (SoC-thermal coupling ratio) — the two wear-chemistry coefficients —
contribute negligible variance to every outcome tested, and that the energy
premium specifically is governed by training power, thermal resistance, and
the leakage constant, none of the wear parameters.

**One number this sweep does not reproduce:** the paper's Section 9 claim
that the fatigue-worst *session length* (as opposed to the *cost* at that
session length, which is what this Sobol design targets) is governed by
round duration and heating time constant with total-order indices 0.68 and
0.29. That is a standardised-regression attribution over `worst_m` directly,
computed by a script not present in this repository. This document does not
claim to close that gap.
# Coefficient Library Findings (F1–F6)

This file is referenced by `coefficients.py` and `uncertainty.py` as the
record of the six findings produced by this repository's verification and
uncertainty-quantification tooling. It did not exist in the repository as
pushed; the references were dead links. This file closes that gap with the
actual finding text, reproduced from `verify.py`'s own output, plus the
robustness results from `uncertainty.py`'s Monte Carlo (N=2000) and Sobol
(N=4096 evaluations) sweeps over the 14-coefficient uncertain space.

Regenerate the finding text at any time with `python verify.py`.

---

## F1 — The doubling rule is wrong for this coefficient range

The widely repeated claim that battery degradation doubles per sustained
10 °C requires Ea = 52.9 kJ/mol. The 232-cell, 13-year calendar-ageing
dataset gives 23.6–29.9 kJ/mol, which yields only 1.36–1.48x per 10 °C —
roughly a 40% acceleration, not 100%.

**Consequence:** the Arrhenius arm of the wear objective is substantially
weaker than the popular framing suggests, which shifts relative weight onto
the cycling arm. This is stated explicitly in the paper (Section 5.1) rather
than left as a repeated but unsupported figure.

## F2 — Identifiability

At the lower bound R_th = 3.6 K/W, the model predicts a steady state of only
43 °C, which never reaches the ~50 °C governor cap measured in the reference
handset data — so throttling at 150 s could not occur. The closed-loop
inversion of the observed temperature rise is therefore not merely a lower
bound; it is inconsistent with the observed onset. R_th must exceed roughly
5.00 K/W for a 5 W load to reach a 50 °C cap at all. This tightens the
library: R_th's lower bound was raised accordingly (documented in the paper,
Section 9.5/9.6).

## F3 — Energy miscalibration is real but modest

Across the full swept corner space, a throttled device at 60–70 °C costs
between 2.42x and 7.86x the joules per unit of useful work that a nominal
power model would predict. The effect is robust in sign across every corner
(supporting contribution C2's energy-miscalibration claim), but it is a
tens-of-percent effect rather than an order of magnitude. The paper claims
the sign and the robustness, not a headline multiplier — these specific
figures are this script's own internal sweep and are not meant to be quoted
verbatim in the manuscript.

## F4 — Cycling damage depends on what is held fixed

At fixed **total excursion**, splitting into many small cycles is far less
damaging (ten 2 K cycles cost ~6.3% of one 20 K cycle) because damage grows
as ΔT^n with n>1. But at fixed **total work** — the case that actually
applies to a training campaign — amplitude saturates at the RC steady state
while cycle count falls linearly with batching, so concentration wins
instead. The initial intuition that the two wear mechanisms oppose each
other was drawn from the first case and does not survive the second. This
constraint is stated explicitly in the paper (Section 5.3).

## F5 — No interior optimum in the wear objective, but an interior worst case

Across every weighting of the two wear mechanisms, marginal wear is
minimised at maximum batching — the mechanisms do not oppose each other, so
the dual-mechanism objective as originally conceived does not hold. The
marginal Arrhenius term is close to flat in batch size (0.948–1.089x),
because at these temperatures the rate multiplier is nearly linear in ΔT and
the time-integral of ΔT is fixed by total delivered energy rather than by
how it is distributed. What the sweep does reveal is sharper: cycling damage
has an interior **maximum** at m = 5 rounds per session in the illustrative
parameterisation used by `verify.py`, costing 2.40x the fully-spread case.
Short back-to-back bursts are the worst possible pattern for package
fatigue — precisely where a conventional round-based FL schedule sits. This
is reported in the paper as Section 9's rejected proposition and as the
closed-form worst-case session length of Section 5.4 (Eq. 13).

**Monte Carlo robustness (N=2000 draws, `uncertainty.py`):**

| Check | Fraction holding |
|---|---|
| Energy premium > 1 (throttling costs energy) | 100.0% |
| Energy monotone increasing in batch size | 100.0% |
| Cycling has an interior maximum | 98.8% |
| Worst-case m* in [2, 20] | 95.2% |
| Energy and cycling anti-correlated | 94.8% |
| Arrhenius term near-flat (spread < 25%) | 29.7% — the one finding that does *not* survive broadly; the disagreement is attributed almost entirely to τ_cool (fourth-tier, unmeasured). Reported in the paper as "fewer than a third of draws" (Section 9.3). |

## F6 — The real tradeoff is energy against fatigue

At fixed work and fixed calendar duration, campaign energy rises
monotonically from 26,749 J to 64,263 J (2.40x) as batching increases,
entirely because of throttling, while cycling damage falls. The two are
anti-correlated (r = −0.783). This is the tension the scheduler must
resolve: an FMCL scheduler that minimises energy alone drives devices into
the fatigue-worst regime, and one that minimises fatigue alone pays a 67%
energy premium. Neither single-objective policy is defensible — this
motivates CATS's joint energy/wear objective (Eq. 14–18).

**Sobol sensitivity (N=4096 evaluations, `uncertainty.py`):**

| Target | Dominant parameters (total-order index) | Negligible (< 0.01) |
|---|---|---|
| Energy premium | P_train (0.370), R_th (0.255), theta_leak (0.157), f_static (0.153) | tau_heat, tau_cool, T_amb, T_cap, eta_min, p_dvfs, **Ea**, **k_batt**, n_cm |
| Cost at fatigue-worst batch size | t_round (1.047), n_cm (0.267), tau_heat (0.159), P_train (0.133) | R_th, tau_cool, T_amb, T_cap, eta_min, f_static, theta_leak, p_dvfs, **Ea**, **k_batt** |
| Energy–cycling correlation | t_round (0.826), tau_heat (0.109), n_cm (0.083) | P_train, R_th, tau_cool, T_amb, T_cap, eta_min, f_static, theta_leak, p_dvfs, **Ea**, **k_batt** |

This confirms the paper's Section 9 claim that Ea (activation energy) and
k_batt (SoC-thermal coupling ratio) — the two wear-chemistry coefficients —
contribute negligible variance to every outcome tested, and that the energy
premium specifically is governed by training power, thermal resistance, and
the leakage constant, none of the wear parameters.

**One number this sweep does not reproduce:** the paper's Section 9 claim
that the fatigue-worst *session length* (as opposed to the *cost* at that
session length, which is what this Sobol design targets) is governed by
round duration and heating time constant with total-order indices 0.68 and
0.29. That is a standardised-regression attribution over `worst_m` directly,
computed by a script not present in this repository. This document does not
claim to close that gap.
