"""
FMCL Paper 5 -- Provenance-tagged coefficient library
Version 0.1  (initial build)

Every coefficient carries: value, plausible range, unit, provenance tier,
source, and an explicit note on what is assumed in transferring it to the
FMCL setting.

PROVENANCE TIERS
----------------
T1  Peer-reviewed primary measurement, directly applicable to consumer
    mobile devices running neural workloads.
T2  Peer-reviewed measurement, but transferred from adjacent hardware,
    workload, or domain. Direction trusted; magnitude bounded, not fixed.
T3  Derived by this work from T1/T2 quantities, or reported by a
    non-peer-reviewed source. Must be entered into UQ as a range.
T4  Not identifiable from published data. Entered as a wide prior and
    flagged for direct measurement.

USAGE RULE (carried over from Paper 3)
--------------------------------------
No T3 or T4 coefficient may appear as a point estimate in any reported
result. They enter Monte Carlo / Sobol as distributions, and any claim
must be shown invariant across their range.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

LIBRARY_VERSION = "0.1"


@dataclass(frozen=True)
class Coefficient:
    symbol: str
    value: float                      # nominal / central estimate
    low: float                        # lower plausible bound
    high: float                       # upper plausible bound
    unit: str
    tier: str                         # T1 | T2 | T3 | T4
    source: str
    note: str = ""
    derived_from: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        assert self.tier in {"T1", "T2", "T3", "T4"}, f"bad tier {self.tier}"
        assert self.low <= self.value <= self.high, (
            f"{self.symbol}: nominal {self.value} outside [{self.low}, {self.high}]"
        )

    @property
    def span(self) -> float:
        """Relative width of the plausible range, for triage."""
        if self.value == 0:
            return float("inf")
        return (self.high - self.low) / abs(self.value)


# =====================================================================
# A. THERMAL DYNAMICS  (lumped first-order RC model)
#
#    C_th dT/dt = P(t) - (T - T_amb)/R_th
#    tau = R_th * C_th
#    T_ss = T_amb + P * R_th
# =====================================================================

P_TRAIN = Coefficient(
    symbol="P_train",
    value=5.0, low=3.5, high=6.5, unit="W",
    tier="T1",
    source="EnFed, arXiv:2412.00768 -- average mobile device power during "
           "federated local training (human activity recognition).",
    note="HAR workload on mid-range Android. Directly applicable to the "
         "HARBox arm; treat as lower bound for vision/transformer workloads.",
)

P_IDLE = Coefficient(
    symbol="P_idle",
    value=0.5, low=0.3, high=0.9, unit="W",
    tier="T3",
    source="Derived. Screen-off idle draw consistent with the marginal-energy "
           "framing of FMCL Paper 1 Sec 4.4 worked example (0.5 W idle).",
    note="Needed only to convert whole-device measured power into the MARGINAL "
         "power charged to the learning task under assumption A1.",
)

DELTA_T_OBSERVED = Coefficient(
    symbol="dT_obs",
    value=18.0, low=14.0, high=22.0, unit="K",
    tier="T1",
    source="Multi-DNN mobile co-execution study, arXiv:2503.21109 -- ~18 C "
           "average temperature rise under sustained mobile inference.",
    note="CRITICAL: this is a CLOSED-LOOP observation. The thermal governor is "
         "already throttling by the time this plateau is reached, so it is the "
         "regulated rise, not the open-loop steady state. R_th derived from it "
         "is therefore a LOWER BOUND.",
)

R_TH = Coefficient(
    symbol="R_th",
    value=6.0, low=5.0, high=8.0, unit="K/W",
    tier="T3",
    source="Derived, then TIGHTENED by verification finding F2 (v0.1). The "
           "naive closed-loop inversion R_th = dT_obs/P_train = 3.6 K/W is "
           "INCONSISTENT: it predicts a 43 C steady state at 5 W, below the "
           "~50 C governor cap Wang et al. (arXiv:2005.12326) measure, so "
           "throttling at 150 s could not occur. Lower bound is therefore set "
           "by the requirement that a 5 W load reach a 50 C cap from 25 C "
           "ambient: R_th >= (50-25)/5 = 5.0 K/W.",
    note="Superseded value in v0.1 was (3.6, 4.5, 8.0). Physically plausible "
         "band for a passively cooled handset is ~2-8 K/W, so the tightened "
         "range sits at the upper end of that band -- consistent with a "
         "device whose governor is actively suppressing the open-loop rise. "
         "Directly measurable; see MEASUREMENT_PROTOCOL.md.",
    derived_from=("dT_obs", "P_train", "T_cap_conservative", "T_amb"),
)

T_AMB = Coefficient(
    symbol="T_amb",
    value=25.0, low=20.0, high=30.0, unit="degC",
    tier="T2",
    source="Reference ambient used throughout the battery-ageing literature "
           "(25 C is the standard Arrhenius reference temperature).",
    note="Indoor overnight charging conditions assumed for the FMCL "
         "participation window (Paper 1 Sec 4.2).",
)

T_THROTTLE_ONSET_TIME = Coefficient(
    symbol="t_onset",
    value=150.0, low=110.0, high=200.0, unit="s",
    tier="T1",
    source="arXiv:2503.21109 -- ~150 s to onset of throttling under sustained "
           "mobile inference.",
    note="Used to back out tau_heat given the throttle threshold. Not itself a "
         "model parameter.",
)

TAU_HEAT = Coefficient(
    symbol="tau_heat",
    value=90.0, low=55.0, high=150.0, unit="s",
    tier="T3",
    source="Derived from t_onset and the RC solution: "
           "t_onset = -tau * ln(1 - dT_throttle/dT_ss).",
    note="Sensitive to where the throttle threshold sits relative to the "
         "open-loop steady state. Wide band retained deliberately.",
    derived_from=("t_onset", "R_th", "P_train"),
)

TAU_COOL = Coefficient(
    symbol="tau_cool",
    value=135.0, low=55.0, high=400.0, unit="s",
    tier="T4",
    source="NOT PUBLISHED for mobile devices under federated training loads.",
    note="This is the single most important unmeasured parameter in the model. "
         "The entire cycling argument depends on how fast a device returns to "
         "baseline between rounds. Prior: tau_cool in [1.0, 2.7] x tau_heat, "
         "since passive cooling has no internal power source and natural "
         "convection makes R_th weakly temperature dependent. "
         "MEASURE THIS FIRST -- see MEASUREMENT_PROTOCOL.md.",
    derived_from=("tau_heat",),
)

# --- Governor thresholds, per device class (Wang et al., arXiv:2005.12326) ---

T_CAP_TOLERANT = Coefficient(
    symbol="T_cap_tolerant",
    value=65.0, low=60.0, high=72.0, unit="degC",
    tier="T1",
    source="Wang, Yang & Zhou, arXiv:2005.12326 Fig. 3(a) -- Nexus 6 permits "
           "SoC temperature above 60 C and past 70 C, trading heat for clock.",
    note="'Tolerant' governor class. Older/aggressive tuning.",
)

T_CAP_CONSERVATIVE = Coefficient(
    symbol="T_cap_conservative",
    value=50.0, low=48.0, high=55.0, unit="degC",
    tier="T1",
    source="Wang et al., arXiv:2005.12326 Fig. 3(b,c,d) -- Nexus 6P holds ~50 C, "
           "Mate 10 and Samsung J8 maintain ~50 C.",
    note="'Conservative' and 'balanced' governor classes share the cap; they "
         "differ in HOW MUCH throughput they give up to hold it (eta_min).",
)

T_SKIN_CAP = Coefficient(
    symbol="T_skin_cap",
    value=45.0, low=43.0, high=48.0, unit="degC",
    tier="T2",
    source="General smartphone thermal-management literature; skin-temperature "
           "limits in the 43-48 C band.",
    note="User-comfort constraint, distinct from the SoC junction cap. Relevant "
         "to FMCL because participation must not degrade the primary user "
         "experience (Paper 1 Sec 4.5 incentive alignment).",
)


# =====================================================================
# B. THROTTLING RESPONSE  (throughput derating)
#
#    eta(T) = 1                                      for T <= T_cap
#    eta(T) = max(eta_min, 1 - slope*(T - T_cap))    for T >  T_cap
# =====================================================================

ETA_MIN_TOLERANT = Coefficient(
    symbol="eta_min_tolerant",
    value=0.75, low=0.70, high=0.80, unit="-",
    tier="T1",
    source="Wang et al., arXiv:2005.12326 -- Nexus 6, Mate 10 and Samsung J8 "
           "run at a 20-30% clock discount under sustained load.",
    note="Sustained/peak throughput ratio for governors that hold clock.",
)

ETA_MIN_SEVERE = Coefficient(
    symbol="eta_min_severe",
    value=0.23, low=0.20, high=0.50, unit="-",
    tier="T1",
    source="arXiv:2503.21109 -- CPU frequency 3 GHz -> 1 GHz, up to 4.3x "
           "slowdown. Corroborated by Wang et al.: Nexus 6P drops big-core "
           "frequency below 50% and takes the big cluster offline entirely.",
    note="Worst-case governor class. Core shutdown, not just DVFS. Upper bound "
         "of the range is the frequency-only case (0.5); lower bound includes "
         "core shutdown.",
)

ETA_MIN_SUSTAINED_LLM = Coefficient(
    symbol="eta_min_llm",
    value=0.56, low=0.45, high=0.70, unit="-",
    tier="T3",
    source="arXiv:2603.23640 -- reported as -44% sustained throughput "
           "degradation. NOT YET VERIFIED against full text.",
    note="FLAGGED. The portion of this paper inspected reports the RTX 4050 "
         "showing NO throttling and stable throughput; the -44% figure applies "
         "to a mobile platform not identified in the inspected excerpt. Verify "
         "which device before citing. Do not use as a load-bearing coefficient "
         "until confirmed.",
)


# =====================================================================
# C. ENERGY
# =====================================================================

E_ROUND_MLP = Coefficient(
    symbol="E_round_mlp",
    value=22.5, low=22.0, high=23.0, unit="J",
    tier="T1",
    source="EnFed, arXiv:2412.00768 -- per-round on-device training energy for "
           "an MLP on activity-recognition data.",
    note="Directly reusable for the HARBox arm.",
)

E_ROUND_LSTM = Coefficient(
    symbol="E_round_lstm",
    value=120.0, low=52.75, high=330.0, unit="J",
    tier="T1",
    source="EnFed, arXiv:2412.00768 -- 52.75 J (dataset 1) to 330 J (dataset 2) "
           "per round for an LSTM.",
    note="The 6x spread across datasets at fixed architecture is itself a "
         "finding: per-round energy is not determined by model size alone.",
)

T_ROUND = Coefficient(
    symbol="t_round",
    value=25.0, low=4.3, high=65.1, unit="s",
    tier="T1",
    source="EnFed, arXiv:2412.00768 -- per-round training time 4.3-65.1 s "
           "depending on model and dataset.",
    note="Compare against tau_heat: a single round is SHORTER than the thermal "
         "time constant for most of this range. Thermal state therefore "
         "accumulates across rounds rather than saturating within one.",
)

E_WIFI_PER_BIT = Coefficient(
    symbol="e_wifi",
    value=130e-9, low=90e-9, high=200e-9, unit="J/bit",
    tier="T1",
    source="Google production federated-learning telemetry, reported via CACM "
           "'Energy and Emissions of ML on Smartphones vs. the Cloud'.",
    note="Production measurement at fleet scale, not a lab estimate. Strongest "
         "single coefficient in the library.",
)

STATIC_POWER_FRACTION = Coefficient(
    symbol="f_static",
    value=0.28, low=0.15, high=0.40, unit="-",
    tier="T3",
    source="Derived. Typical leakage share of mobile SoC power at nominal "
           "operating temperature.",
    note="Drives the energy-miscalibration result (contribution C2): leakage "
         "does not scale down with clock, so J per useful gradient rises under "
         "throttling. Wide band; must be swept.",
)

LEAKAGE_TEMP_CONSTANT = Coefficient(
    symbol="theta_leak",
    value=17.0, low=14.4, high=21.6, unit="K",
    tier="T3",
    source="Derived from the CMOS rule of thumb that leakage roughly doubles "
           "per 10-15 C: theta = dT_double / ln(2).",
    note="P_static(T) = P_static0 * exp((T - T_ref)/theta).",
)

DVFS_EXPONENT = Coefficient(
    symbol="p_dvfs",
    value=2.0, low=1.0, high=3.0, unit="-",
    tier="T3",
    source="Derived. P_dyn ~ eta^p. p=1 if voltage is held fixed and only "
           "frequency scales; p=3 if voltage scales linearly with frequency "
           "(P ~ C V^2 f).",
    note="Real governors sit between the two. Sweep the full range.",
)


# =====================================================================
# D. WEAR -- MECHANISM 1: BATTERY CAPACITY FADE (Arrhenius)
#
#    k(T)/k(Tref) = exp[ (Ea/R) * (1/Tref - 1/T) ]   with T in Kelvin
# =====================================================================

EA_CAPACITY_FADE = Coefficient(
    symbol="Ea",
    value=26750.0, low=23600.0, high=29900.0, unit="J/mol",
    tier="T1",
    source="Calendar-ageing dataset of 232 commercial cells, 8 cell types, "
           "5 manufacturers, up to 13 years; Arrhenius fits give activation "
           "energy for capacity loss of 23.6-29.9 kJ/mol "
           "(J. Energy Storage, S2352152X21011889).",
    note="SEE FINDING F1 in COEFFICIENTS_v0.1.md. This range does NOT reproduce "
         "the widely repeated 'degradation doubles per 10 C' claim. Do not use "
         "the doubling rule.",
)

R_GAS = Coefficient(
    symbol="R",
    value=8.314462618, low=8.314462618, high=8.314462618, unit="J/(mol K)",
    tier="T1",
    source="CODATA molar gas constant.",
    note="Exact by definition for our purposes.",
)

BATT_SOC_COUPLING = Coefficient(
    symbol="k_soc",
    value=1.0, low=0.8, high=1.4, unit="-",
    tier="T2",
    source="Same 232-cell dataset: influence of state of charge on calendar "
           "ageing peaks around 85% SOC.",
    note="FMCL schedules participation during charging (Paper 1 Sec 4.2), which "
         "places devices in the HIGH-SOC, HIGH-TEMPERATURE corner. This is an "
         "uncomfortable interaction the series has not previously priced.",
)

BATT_SOC_THERMAL_COUPLING = Coefficient(
    symbol="k_batt",
    value=0.55, low=0.30, high=0.85, unit="-",
    tier="T4",
    source="NOT PUBLISHED. Fraction of SoC temperature rise that reaches the "
           "battery cell.",
    note="dT_battery = k_batt * dT_SoC. The entire Arrhenius arm is scaled by "
         "this number and it is currently a guess. Both sensors are exposed on "
         "Android, so this is directly measurable. MEASURE THIS SECOND.",
)


# =====================================================================
# E. WEAR -- MECHANISM 2: PACKAGE / SOLDER FATIGUE (Coffin-Manson)
#
#    N_f = A * (dT)^(-n)      damage per cycle D = (dT)^n / A
# =====================================================================

CM_EXPONENT = Coefficient(
    symbol="n_cm",
    value=2.2, low=1.9, high=2.7, unit="-",
    tier="T2",
    source="Solder-fatigue literature; Coffin-Manson exponents for SnAgCu "
           "joints typically 1.9-2.5, occasionally to 2.7.",
    note="TRANSFER WARNING: derived for conventional solder-ball packages. "
         "Mobile SoCs use package-on-package with LPDDR stacked on die -- "
         "different alloys, different CTE mismatch. Mechanism transfers; "
         "coefficient does not. Sweep the full range and report invariance.",
)

CM_CORROBORATION = Coefficient(
    symbol="cm_ratio_obs",
    value=174.0, low=50.0, high=400.0, unit="-",
    tier="T2",
    source="Synchronised thermocouple + LWIR study of a COTS 14 nm mobile CPU "
           "(Intel i7-6600U), IEEE DataPort: gaming workloads show up to 174x "
           "accelerated fatigue vs office use, while constant-load crypto shows "
           "the LOWEST degradation despite high power draw.",
    note="DIRECTIONAL EVIDENCE ONLY. Wrong package class (laptop CPU, not PoP). "
         "Cannot be inverted to recover n because cycle COUNT and cycle "
         "AMPLITUDE both differ between the workloads. Its value to this paper "
         "is the qualitative finding that thermal CYCLING, not peak "
         "temperature, dominates fatigue -- which is the premise of the "
         "dual-mechanism objective.",
)


# =====================================================================
# F. FLEET / TRACE DATA
# =====================================================================

FLASH_DEVICES = Coefficient(
    symbol="N_flash",
    value=136000, low=136000, high=136000, unit="devices",
    tier="T1",
    source="FLASH behaviour dataset via FedScale -- 136k users, one week "
           "(31 Jan - 6 Feb 2020), 180M trace items covering battery charge "
           "state, network condition, and screen lock.",
    note="Used as the availability process. Replaces the semi-Markov "
         "assumption of Paper 1 Assumption 1 with empirical traces.",
)


ALL_COEFFICIENTS = [
    P_TRAIN, P_IDLE, DELTA_T_OBSERVED, R_TH, T_AMB, T_THROTTLE_ONSET_TIME,
    TAU_HEAT, TAU_COOL, T_CAP_TOLERANT, T_CAP_CONSERVATIVE, T_SKIN_CAP,
    ETA_MIN_TOLERANT, ETA_MIN_SEVERE, ETA_MIN_SUSTAINED_LLM,
    E_ROUND_MLP, E_ROUND_LSTM, T_ROUND, E_WIFI_PER_BIT,
    STATIC_POWER_FRACTION, LEAKAGE_TEMP_CONSTANT, DVFS_EXPONENT,
    EA_CAPACITY_FADE, R_GAS, BATT_SOC_COUPLING, BATT_SOC_THERMAL_COUPLING,
    CM_EXPONENT, CM_CORROBORATION, FLASH_DEVICES,
]


def tier_census():
    """Count coefficients by provenance tier."""
    out = {}
    for c in ALL_COEFFICIENTS:
        out[c.tier] = out.get(c.tier, 0) + 1
    return dict(sorted(out.items()))


def unmeasured():
    """T4 coefficients -- the measurement agenda."""
    return [c for c in ALL_COEFFICIENTS if c.tier == "T4"]


def widest(k=5):
    """Coefficients with the widest relative range -- UQ triage."""
    ranked = sorted(ALL_COEFFICIENTS, key=lambda c: -c.span)
    return ranked[:k]
