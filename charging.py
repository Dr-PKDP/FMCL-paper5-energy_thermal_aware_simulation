"""
FMCL Paper 5 -- Charging-coupled thermal model.

WHY THIS EXISTS
---------------
Earlier versions modelled a device heating from ambient under training load
alone. That is the wrong baseline for FMCL. The architecture schedules
participation during charging by design (Paper 1, Definition 1), and a charging
phone is not at ambient: fast charging raises battery temperature by 8-10 C
relative to standard charging, and thermal governors begin derating charge
current in the mid-30s C. Training therefore starts on a device that is already
elevated and whose governor is already intervening.

Three consequences the earlier model could not represent:

  1. Throttling engages at far lower duty cycle, because the compute heat is
     added to a raised baseline rather than to ambient.
  2. Participation places the device in the high-state-of-charge,
     high-temperature corner, which is the worst region for calendar ageing.
  3. Compute heat and charging heat compete for one thermal budget, so
     participation SLOWS CHARGING -- a user-visible cost the series has never
     priced, and one that bears directly on retention.

MODEL
-----
Two lumped nodes, because the two heat sources enter at different places:
compute heat at the SoC, charging loss at the cell.

    C_s dT_s/dt = P_comp     - (T_s - T_amb)/R_s - (T_s - T_b)/R_sb
    C_b dT_b/dt = P_chg_loss - (T_b - T_amb)/R_b + (T_s - T_b)/R_sb

The SoC-to-battery coupling k_batt, previously a free T4 parameter with a wide
prior, now falls out of the network as R_b/(R_sb + R_b) rather than being
assumed. That is a strict improvement in identifiability.
"""

import numpy as np
import thermal as TH
import coefficients as C

KELVIN = 273.15

# ---------------------------------------------------------------------
# Network parameters, calibrated against three independent constraints
# ---------------------------------------------------------------------
# R_s_eff = R_s || (R_sb + R_b) must reproduce the compute-only steady state
#           (~6 K/W, coefficient library v0.2 after finding F2)
# R_b_eff = R_b || (R_sb + R_s) must reproduce the observed charging rise
# k_batt  = R_b / (R_sb + R_b) must land inside the prior [0.30, 0.85]
R_S = 10.0     # SoC to ambient, K/W
R_B = 10.0     # battery to ambient, K/W
R_SB = 5.0     # SoC to battery, K/W
C_S = 15.0     # SoC heat capacity, J/K
C_B = 60.0     # battery heat capacity, J/K (larger thermal mass)

# Charging
ETA_CHG = 0.93          # electrical efficiency; loss becomes heat at the cell
P_CHG_MAX_FAST = 25.0   # W delivered, fast charger
P_CHG_MAX_STD = 5.0     # W delivered, standard charger
E_BATT_WH = 18.0        # ~4800 mAh at 3.85 V
T_CHG_DERATE = 40.0     # C, governor begins cutting charge current
T_CHG_STOP = 45.0       # C, charging suspended

# Hard compute ceiling. Beyond the frequency derating already in thermal.eta,
# mobile operating systems suspend background work outright to hold skin
# temperature -- the OS-level battery-management behaviour Paper 1's
# limitations section flags via Wang & Wu [115]. Without this the leakage term
# is an unbounded positive feedback and the model diverges.
T_COMPUTE_CEILING = 58.0   # C, SoC temperature at which background work is cut
T_COMPUTE_KILL = 63.0      # C, background work fully suspended


def network_summary():
    r_s_eff = 1.0 / (1.0 / R_S + 1.0 / (R_SB + R_B))
    r_b_eff = 1.0 / (1.0 / R_B + 1.0 / (R_SB + R_S))
    k_batt = R_B / (R_SB + R_B)
    return dict(R_s_eff=r_s_eff, R_b_eff=r_b_eff, k_batt=k_batt)


def charge_derate(T_b, T_lo=T_CHG_DERATE, T_hi=T_CHG_STOP):
    """Fraction of maximum charge current the governor permits."""
    return float(np.clip((T_hi - T_b) / (T_hi - T_lo), 0.0, 1.0))


def compute_admit(T_s, T_lo=T_COMPUTE_CEILING, T_hi=T_COMPUTE_KILL):
    """
    Fraction of requested background compute the OS admits. Falls to zero at
    the kill threshold, so FL participation is self-limiting: the governor
    suspends the training job before the device damages itself.
    """
    return float(np.clip((T_hi - T_s) / (T_hi - T_lo), 0.0, 1.0))


def charge_power(soc, T_b, P_max):
    """
    Delivered charge power: constant-current up to 80% state of charge, then a
    constant-voltage taper, further derated by the thermal governor.
    """
    # Linear CV taper with a floor: a taper reaching exactly zero never
    # terminates, whereas a real CV phase ends at a current cutoff.
    taper = 1.0 if soc < 0.8 else max((1.0 - soc) / 0.2, 0.05)
    return P_max * taper * charge_derate(T_b)


def step(T_s, T_b, soc, P_comp, P_max, dt, T_amb):
    """Advance the two-node system and the state of charge by dt seconds."""
    P_chg = charge_power(soc, T_b, P_max)
    P_loss = (1.0 - ETA_CHG) * P_chg
    dTs = (P_comp - (T_s - T_amb) / R_S - (T_s - T_b) / R_SB) / C_S
    dTb = (P_loss - (T_b - T_amb) / R_B + (T_s - T_b) / R_SB) / C_B
    soc_next = min(soc + (P_chg * ETA_CHG) * dt / (E_BATT_WH * 3600.0), 1.0)
    return T_s + dTs * dt, T_b + dTb * dt, soc_next, P_chg


def simulate_charge_session(participate, duty=0.5, P_comp=5.0,
                            P_max=P_CHG_MAX_FAST, soc0=0.30, T_amb=25.0,
                            dt=1.0, max_s=6 * 3600, T_cap=50.0, eta_min=0.75):
    """
    Charge a device from soc0 to full, optionally running FL training at a
    given duty cycle throughout.

    Returns charge completion time, thermal exposure, and the compute actually
    delivered -- so the charging penalty and the training benefit are measured
    on the same run.
    """
    T_s = T_b = T_amb
    soc = soc0
    t = 0.0
    work_done = 0.0
    energy_comp = 0.0
    above35 = above40 = 0.0
    arr = 0.0
    peak_s = peak_b = T_amb
    derate_integral = 0.0
    suspended_s = 0.0
    n = 0

    while soc < 0.999 and t < max_s:
        active = participate and (np.sin(2 * np.pi * t / 120.0) > (1 - 2 * duty))
        if active:
            eta = float(TH.eta(T_s, T_cap, eta_min))
            P = float(TH.power_at(T_s, eta, P_comp,
                                  C.STATIC_POWER_FRACTION.value,
                                  C.LEAKAGE_TEMP_CONSTANT.value,
                                  C.DVFS_EXPONENT.value, T_amb))
            admit = compute_admit(T_s)
            P *= admit
            work_done += eta * admit * dt
            energy_comp += P * dt
            suspended_s += (1.0 - admit) * dt
        else:
            P = 0.0
        derate_integral += charge_derate(T_b) * dt
        T_s, T_b, soc, _ = step(T_s, T_b, soc, P, P_max, dt, T_amb)
        if T_b > 35.0:
            above35 += dt
        if T_b > 40.0:
            above40 += dt
        arr += (float(TH.arrhenius_ratio(T_b, T_amb, C.EA_CAPACITY_FADE.value))
                - 1.0) * dt
        peak_s = max(peak_s, T_s)
        peak_b = max(peak_b, T_b)
        t += dt
        n += 1

    return dict(charge_time_s=t, completed=soc >= 0.999, soc_final=soc,
                T_s_peak=peak_s, T_b_peak=peak_b,
                s_above35=above35, s_above40=above40,
                arrhenius_excess=arr, work_done_s=work_done,
                energy_comp_J=energy_comp, suspended_s=suspended_s,
                mean_charge_derate=derate_integral / max(t, 1e-9))
