# Table 11 Reproduction Note

**Status: Table 11 (propensity truncation sweep, Section 8.6) does not
reproduce from the current `converge_charging.py`, and the paper's printed
values have not been changed on the basis of this investigation.**

Run `python table11_reproduction_attempt.py` to see this directly. That
script documents four independent attempts, none of which converges to
Table 11's printed values:

| Attempt | Result |
|---|---|
| 3 seeds (matches `converge_charging.py`'s own reference `main()`) | Far off; c=0.05/0.10/0.20 collapse to a suspiciously identical ~1e-08 floor |
| 6 seeds (matches Table 9/10's convention) | Closer for `none` and c=0.35 (~15% off), but c=0.05 barely moves from baseline where the paper shows its largest drop |
| Near-zero floor (c=0.001), approximating the "unbiased 4.61e-06" reference named in Section 8.6's prose | Floor lands at ~9.4e-06 — roughly double the documented reference, at the theoretical limit where truncation should be negligible |
| c=0.01 | Same pattern as above |

## Why this is not being treated as "close enough"

Unlike several other reproduction gaps closed during this verification pass
(see `COEFFICIENTS_FINDINGS.md` for the Sobol-seeding bug that explained a
similar-looking gap in F9), this one did not respond to the two most common
explanations:

- **Not a sample-size effect.** Going from 3 to 6 seeds moved some values
  closer but left others (c=0.05 specifically) essentially unchanged. A
  genuine sampling-noise explanation should shrink roughly uniformly with
  more seeds; this didn't.
- **Not resolved at the theoretical limit.** If truncation strength were
  simply mis-scaled, an near-zero floor should approach the paper's own
  cited "unbiased" reference point. It doesn't — it lands at roughly double
  that value.

The pattern across all four attempts (consistently ~1.5-2x off from
documented, in ways that don't converge with more data) points to a
**structural difference** between this reconstruction and whatever produced
Table 11 — a different Dirichlet alpha, a different quality metric, a
different debiasing formula, or a setting not captured in
`converge_charging.py`'s current parameters. None of these has been
identified.

## What would resolve this

Either:
1. The original script or log that produced Table 11 (if it still exists
   somewhere), the same way `COEFFICIENTS_v0.2.md` and `UQ_LOG_v0.2_full.txt`
   resolved the earlier coefficient-library documentation gap, or
2. Specific confirmation of what differs in the experimental setup —
   Dirichlet alpha, seed count, the exact debiasing formula, or anything
   else `converge_charging.py`'s current implementation might not match.

Guessing further variants without either of these has diminishing returns
and was stopped after four attempts rather than continued indefinitely.

## What this note is not

This is not a claim that Table 11 is wrong. No error has been identified —
only an inability to independently reproduce it with the tools currently in
this repository. The paper's printed values reflect the authors' original
computation; this repository's reconstruction is the one held to a lower
standard of trust here, not the other way around.
