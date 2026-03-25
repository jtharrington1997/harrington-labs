"""QD-Doped Pulsed Fiber Laser testbed engine.

Models a fiber laser where quantum dots replace rare-earth dopants
as the gain medium. Combines fiber propagation physics with QD
size-dependent gain, absorption cross-sections, and saturation
dynamics. No Streamlit imports.

Key physics:
- Brus equation for QD size-dependent bandgap → emission wavelength
- QD absorption/emission cross-sections from oscillator strength
- Three-level gain model adapted for QD inhomogeneous broadening
- Fiber mode confinement with QD-doped core
- Pulsed operation via Q-switching or mode-locking
- Auger recombination and multiexciton effects
- Thermal management from QD quantum defect heating
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from harrington_labs.domain import SimulationResult, C_M_S, H_J_S, K_B


# ── Physical constants ──────────────────────────────────────────────────

_E_CHARGE = 1.602176634e-19
_ME = 9.10938e-31
_EPSILON_0 = 8.854187817e-12


# ── QD gain medium bulk parameters ─────────────────────────────────────
# (Eg_bulk_eV, me_eff, mh_eff, epsilon_r, a_exciton_nm)
_QD_BULK = {
    "CdSe": (1.74, 0.12, 0.45, 9.4, 5.6),
    "CdSe/ZnS": (1.74, 0.12, 0.45, 9.4, 5.6),
    "PbSe": (0.28, 0.047, 0.040, 227.0, 46.0),
    "PbS": (0.41, 0.085, 0.085, 169.0, 20.0),
    "InP": (1.35, 0.077, 0.60, 12.5, 11.0),
    "InAs": (0.354, 0.023, 0.41, 15.15, 34.0),
    "Perovskite": (1.55, 0.12, 0.15, 25.0, 7.0),
    "Si": (1.12, 0.26, 0.36, 11.7, 4.9),
}


# ── Dataclass ──────────────────────────────────────────────────────────


@dataclass
class QDFiberLaserParams:
    """Parameters for the QD-doped pulsed fiber laser testbed."""
    # QD gain medium
    qd_material: str = "PbS"
    qd_diameter_nm: float = 5.0
    qd_size_distribution_pct: float = 5.0
    qd_concentration_cm3: float = 1e17  # QD volume density in fiber core
    qd_quantum_yield: float = 0.3
    # Fiber
    core_diameter_um: float = 6.0
    cladding_diameter_um: float = 125.0
    fiber_na: float = 0.12
    fiber_length_m: float = 1.0
    background_loss_db_m: float = 0.5  # higher than RE-doped due to QD scattering
    host_refractive_index: float = 1.45  # glass matrix
    # Pump
    pump_wavelength_nm: float = 808.0
    pump_power_mw: float = 500.0
    pump_coupling_efficiency: float = 0.7
    # Pulse parameters
    pulse_mode: str = "Q-switched"  # "Q-switched", "Mode-locked", "CW"
    rep_rate_khz: float = 100.0
    q_switch_hold_time_us: float = 10.0  # for Q-switched
    saturable_absorber_modulation_depth: float = 0.3  # for mode-locked
    # Output coupler
    output_coupler_reflectivity: float = 0.5
    # Thermal
    ambient_temperature_k: float = 293.0


# ── QD physics ─────────────────────────────────────────────────────────


def qd_bandgap_ev(material: str, diameter_nm: float) -> float:
    """Size-dependent bandgap using Brus equation with empirical corrections.

    For lead-salt QDs (PbS, PbSe, InAs), the standard Brus equation
    overestimates confinement at small sizes due to non-parabolicity.
    We use empirical sizing curves from the literature for these materials.
    """
    bulk = _QD_BULK.get(material)
    if bulk is None:
        return 1.0
    eg_bulk, me_eff, mh_eff, eps_r, _ = bulk
    r = (diameter_nm / 2.0) * 1e-9
    if r <= 0:
        return eg_bulk

    # Empirical sizing curves for lead-salt / narrow-gap QDs
    # These override the Brus equation where it's known to fail.
    # Fitted from experimental data (Moreels 2009, Cademartiri 2006, etc.)
    if material in ("PbS", "PbSe"):
        # PbS/PbSe: Eg ≈ Eg_bulk + 1/(a*d^2 + b*d + c) — hyperbolic fit
        d = diameter_nm
        if material == "PbS":
            # PbS empirical: Eg(eV) from Moreels et al. (Chem. Mater. 2009)
            eg = 0.41 + 1.0 / (0.0252 * d**2 + 0.283 * d)
        else:  # PbSe
            eg = 0.28 + 1.0 / (0.0239 * d**2 + 0.340 * d)
        return max(eg, eg_bulk)

    if material == "InAs":
        d = diameter_nm
        eg = 0.354 + 1.0 / (0.022 * d**2 + 0.294 * d)
        return max(eg, eg_bulk)

    if material in ("CdSe", "CdSe/ZnS"):
        # Yu et al. (Chem. Mater. 2003) empirical sizing curve
        d = diameter_nm
        eg = 1.74 + 1.0 / (0.0556 * d**2 + 0.260 * d)
        return max(eg, eg_bulk)

    if material == "InP":
        d = diameter_nm
        eg = 1.35 + 1.0 / (0.0490 * d**2 + 0.225 * d)
        return max(eg, eg_bulk)

    if material == "Si":
        # Meier et al. empirical for Si nanocrystals
        d = diameter_nm
        eg = 1.12 + 3.73 / d**1.39
        return max(eg, eg_bulk)

    if material == "Perovskite":
        # Weak confinement for most perovskite QDs
        d = diameter_nm
        eg = 1.55 + 1.0 / (0.10 * d**2 + 0.50 * d)
        return max(eg, eg_bulk)

    # Standard Brus equation for other materials
    mu = (me_eff * mh_eff) / (me_eff + mh_eff) * _ME
    confinement = (H_J_S**2 * math.pi**2) / (2 * mu * r**2)
    coulomb = (1.8 * _E_CHARGE**2) / (4 * math.pi * _EPSILON_0 * eps_r * r)
    eg = eg_bulk + (confinement - coulomb) / _E_CHARGE
    return max(eg, eg_bulk * 0.5)


def qd_emission_wavelength_nm(material: str, diameter_nm: float) -> float:
    """Peak emission wavelength from bandgap."""
    eg = qd_bandgap_ev(material, diameter_nm)
    return 1240.0 / eg if eg > 0 else 0.0


def qd_absorption_cross_section_cm2(material: str, diameter_nm: float, wavelength_nm: float) -> float:
    """Size-dependent absorption cross-section.

    Scales with QD volume and oscillator strength. Enhanced near
    the 1S absorption peak.
    """
    bulk = _QD_BULK.get(material)
    if bulk is None:
        return 1e-16
    eg_bulk, me_eff, mh_eff, eps_r, _ = bulk

    eg_ev = qd_bandgap_ev(material, diameter_nm)
    photon_ev = 1240.0 / wavelength_nm if wavelength_nm > 0 else 0

    # Base cross-section scales with QD volume
    r_nm = diameter_nm / 2
    volume_nm3 = (4 / 3) * math.pi * r_nm**3
    sigma_base = 1e-15 * (volume_nm3 / 100)  # ~1e-15 cm² for ~5nm QD

    # Resonance enhancement near bandgap
    if photon_ev > 0 and eg_ev > 0:
        detuning = (photon_ev - eg_ev) / eg_ev
        if detuning > 0:
            # Above bandgap: absorption scales as sqrt(E - Eg) for bulk-like
            enhancement = min(5.0, 1.0 + 4.0 * math.sqrt(min(detuning, 1.0)))
        elif detuning > -0.3:
            # Near-resonance: Lorentzian tail
            enhancement = 1.0 / (1.0 + (detuning * 10)**2)
        else:
            enhancement = 0.01  # well below bandgap
    else:
        enhancement = 0.01

    return sigma_base * enhancement


def qd_emission_cross_section_cm2(material: str, diameter_nm: float) -> float:
    """Emission cross-section from McCumber relation (simplified)."""
    abs_sigma = qd_absorption_cross_section_cm2(
        material, diameter_nm, qd_emission_wavelength_nm(material, diameter_nm)
    )
    # Emission cross-section typically 0.5–1x absorption at peak
    return abs_sigma * 0.8 * _QD_BULK.get(material, (1,))[0]  # scale with bulk Eg (proxy for oscillator strength)


def qd_gain_bandwidth_nm(material: str, diameter_nm: float, size_distribution_pct: float) -> float:
    """Gain bandwidth from inhomogeneous broadening due to size distribution.

    QDs have much broader gain than RE ions — the size distribution
    creates a spread of emission wavelengths.
    """
    # Homogeneous linewidth ~20-50 nm at RT
    homogeneous_nm = 30.0

    # Inhomogeneous broadening: δλ/λ ≈ 2 * δd/d for Brus equation
    lam_em = qd_emission_wavelength_nm(material, diameter_nm)
    inhomogeneous_nm = 2 * (size_distribution_pct / 100) * lam_em

    # Total (quadrature)
    return math.sqrt(homogeneous_nm**2 + inhomogeneous_nm**2)


def auger_recombination_rate_s(material: str, diameter_nm: float) -> float:
    """Auger recombination rate for biexciton → hot exciton.

    Scales as ~V (volume) — larger QDs have slower Auger. Literature values:
    CdSe 3nm: ~50 ps, 6nm: ~600 ps
    PbS 3nm: ~10 ps, 5nm: ~100 ps, 8nm: ~500 ps
    PbSe 4nm: ~50 ps, 8nm: ~400 ps
    """
    r_nm = diameter_nm / 2
    volume_nm3 = (4 / 3) * math.pi * r_nm**3

    # Material-dependent Auger coefficient (ps per nm³)
    # Fitted to match experimental data
    _AUGER_COEFF = {
        "CdSe": 1.2, "CdSe/ZnS": 2.0,  # shell suppresses Auger
        "PbS": 0.8, "PbSe": 0.6,
        "InAs": 0.5, "InP": 1.5,
        "Si": 3.0, "Perovskite": 2.5,
    }
    coeff = _AUGER_COEFF.get(material, 1.0)
    tau_auger_ps = coeff * volume_nm3
    tau_auger_ps = max(tau_auger_ps, 5.0)  # physical minimum ~5 ps

    return 1.0 / (tau_auger_ps * 1e-12)


# ── Fiber laser physics ────────────────────────────────────────────────


def fiber_v_number(core_diameter_um: float, na: float, wavelength_nm: float) -> float:
    return math.pi * core_diameter_um * na / (wavelength_nm * 1e-3) if wavelength_nm > 0 else 0


def fiber_mfd_um(core_diameter_um: float, v: float) -> float:
    if v <= 0:
        return core_diameter_um
    return core_diameter_um * (0.65 + 1.619 / v**1.5 + 2.879 / v**6)


def fiber_effective_area_um2(mfd_um: float) -> float:
    return math.pi * (mfd_um / 2)**2


def overlap_factor(core_diameter_um: float, mfd_um: float) -> float:
    """Fraction of mode power overlapping with the doped core."""
    if core_diameter_um <= 0:
        return 0.0
    ratio = (core_diameter_um / mfd_um)**2 if mfd_um > 0 else 1.0
    return 1.0 - math.exp(-2 * ratio)


# ── Main simulation ────────────────────────────────────────────────────


def simulate_qd_fiber_laser(params: QDFiberLaserParams) -> SimulationResult:
    """Run the full QD-doped pulsed fiber laser testbed simulation."""
    warnings = []

    # ── QD properties ──
    eg_ev = qd_bandgap_ev(params.qd_material, params.qd_diameter_nm)
    lam_em_nm = qd_emission_wavelength_nm(params.qd_material, params.qd_diameter_nm)
    sigma_abs = qd_absorption_cross_section_cm2(params.qd_material, params.qd_diameter_nm, params.pump_wavelength_nm)
    sigma_em = qd_emission_cross_section_cm2(params.qd_material, params.qd_diameter_nm)
    gain_bw_nm = qd_gain_bandwidth_nm(params.qd_material, params.qd_diameter_nm, params.qd_size_distribution_pct)
    auger_rate = auger_recombination_rate_s(params.qd_material, params.qd_diameter_nm)
    auger_lifetime_ns = 1e9 / auger_rate if auger_rate > 0 else 1e6

    # QD radiative lifetime (size-dependent, typically 20-1000 ns)
    bulk = _QD_BULK.get(params.qd_material, (1, 0.1, 0.1, 10, 5))
    r_nm = params.qd_diameter_nm / 2
    radiative_lifetime_ns = 20.0 * (r_nm / 2.5)**2  # rough scaling

    # Auger only matters for multiexciton states (biexciton, triexciton)
    # In the single-exciton (linear gain) regime, Auger is irrelevant.
    # The effective QY for single-exciton recombination uses only
    # radiative and non-radiative (surface trap) rates.
    # Surface trap rate scales inversely with QY
    if params.qd_quantum_yield > 0:
        nonrad_lifetime_ns = radiative_lifetime_ns * params.qd_quantum_yield / (1 - params.qd_quantum_yield) if params.qd_quantum_yield < 1 else 1e6
    else:
        nonrad_lifetime_ns = 0.1
    total_lifetime_ns = 1.0 / (1.0 / radiative_lifetime_ns + 1.0 / nonrad_lifetime_ns)
    effective_qy = total_lifetime_ns / radiative_lifetime_ns

    # Auger lifetime (relevant for multi-exciton quenching at high pump)
    auger_lifetime_ns = 1e9 / auger_rate if auger_rate > 0 else 1e6

    # At high pump intensities, average exciton number <N> > 1 and Auger degrades QY
    # <N> ≈ σ_abs × pump_flux × τ_total
    # We'll compute this after knowing the pump flux, and correct QY if needed

    # ── Fiber properties ──
    v = fiber_v_number(params.core_diameter_um, params.fiber_na, lam_em_nm)
    mfd = fiber_mfd_um(params.core_diameter_um, v)
    a_eff = fiber_effective_area_um2(mfd)
    gamma = overlap_factor(params.core_diameter_um, mfd)
    single_mode = v <= 2.405

    if not single_mode:
        warnings.append(f"V = {v:.2f} > 2.405 — fiber is multimode at {lam_em_nm:.0f} nm")

    # ── Gain calculation ──
    # Small-signal gain coefficient: g₀ = N_QD × σ_em × Γ
    n_qd = params.qd_concentration_cm3  # cm⁻³
    g0_cm = n_qd * sigma_em * gamma  # cm⁻¹

    # Pump absorption
    pump_abs_cm = n_qd * sigma_abs * gamma  # cm⁻¹
    fiber_length_cm = params.fiber_length_m * 100
    pump_absorbed_fraction = 1.0 - math.exp(-pump_abs_cm * fiber_length_cm)
    pump_power_w = params.pump_power_mw * 1e-3 * params.pump_coupling_efficiency
    absorbed_pump_w = pump_power_w * pump_absorbed_fraction

    # Inversion fraction (three-level, simplified)
    pump_rate = absorbed_pump_w / (H_J_S * C_M_S / (params.pump_wavelength_nm * 1e-9))
    core_volume_cm3 = math.pi * (params.core_diameter_um * 1e-4 / 2)**2 * fiber_length_cm
    n_total = n_qd * core_volume_cm3
    lifetime_s = total_lifetime_ns * 1e-9
    inversion = pump_rate * lifetime_s / (n_total + pump_rate * lifetime_s) if n_total > 0 else 0
    inversion = min(inversion, 0.95)

    # Net gain
    net_gain_cm = g0_cm * (2 * inversion - 1) - params.background_loss_db_m * 100 / (10 * math.log10(math.e))
    total_gain_db = net_gain_cm * fiber_length_cm * 10 * math.log10(math.e) if net_gain_cm > 0 else 0

    # Saturation power
    sat_power_w = H_J_S * C_M_S / (lam_em_nm * 1e-9) * a_eff * 1e-12 / (sigma_em * lifetime_s) if sigma_em > 0 and lifetime_s > 0 else 1.0

    # ── CW output estimate ──
    quantum_defect = 1 - params.pump_wavelength_nm / lam_em_nm
    oc_loss = -math.log(params.output_coupler_reflectivity)
    internal_loss = params.background_loss_db_m * params.fiber_length_m / (10 * math.log10(math.e))
    total_loss = oc_loss + internal_loss
    slope_efficiency = effective_qy * (1 - quantum_defect) * gamma * oc_loss / total_loss if total_loss > 0 else 0
    slope_efficiency = min(slope_efficiency, 0.6)

    # Threshold
    threshold_pump_w = (total_loss * a_eff * 1e-12) / (sigma_em * lifetime_s * gamma) * H_J_S * C_M_S / (params.pump_wavelength_nm * 1e-9) if sigma_em > 0 else pump_power_w
    threshold_pump_mw = threshold_pump_w * 1e3

    cw_output_w = max(0, slope_efficiency * (absorbed_pump_w - threshold_pump_w))
    cw_output_mw = cw_output_w * 1e3

    if absorbed_pump_w < threshold_pump_w:
        warnings.append(f"Below threshold: absorbed pump {absorbed_pump_w*1e3:.1f} mW < threshold {threshold_pump_mw:.1f} mW")

    # ── Pulsed output ──
    if params.pulse_mode == "Q-switched":
        # Stored energy during hold time
        stored_energy_j = absorbed_pump_w * min(params.q_switch_hold_time_us * 1e-6, lifetime_s * 3) * effective_qy * (1 - quantum_defect)
        # Pulse energy (fraction extracted)
        extraction = min(0.8, oc_loss / total_loss)
        pulse_energy_j = stored_energy_j * extraction
        rep_rate_hz = params.rep_rate_khz * 1e3
        avg_power_w = pulse_energy_j * rep_rate_hz
        # Pulse width estimate: ~cavity round-trip × ln(gain)
        rt_time_s = 2 * params.fiber_length_m * params.host_refractive_index / C_M_S
        pulse_width_ns = max(1.0, rt_time_s * 1e9 * max(1, math.log(max(total_gain_db, 1))))
        peak_power_w = pulse_energy_j / (pulse_width_ns * 1e-9) if pulse_width_ns > 0 else 0

    elif params.pulse_mode == "Mode-locked":
        # Mode-locked: pulse width from gain bandwidth
        delta_nu = gain_bw_nm * 1e-9 * C_M_S / (lam_em_nm * 1e-9)**2
        transform_limit_fs = 0.44 / delta_nu * 1e15
        pulse_width_ns = transform_limit_fs * 1e-6  # convert fs to ns
        fsr_hz = C_M_S / (2 * params.fiber_length_m * params.host_refractive_index)
        rep_rate_hz = fsr_hz
        avg_power_w = cw_output_w * (1 - params.saturable_absorber_modulation_depth)
        pulse_energy_j = avg_power_w / rep_rate_hz if rep_rate_hz > 0 else 0
        peak_power_w = pulse_energy_j / (pulse_width_ns * 1e-9) if pulse_width_ns > 0 else 0

    else:  # CW
        pulse_energy_j = 0
        rep_rate_hz = 0
        avg_power_w = cw_output_w
        pulse_width_ns = 0
        peak_power_w = cw_output_w

    # ── Thermal ──
    heat_load_w = absorbed_pump_w * quantum_defect + absorbed_pump_w * (1 - effective_qy) * (1 - quantum_defect)
    heat_per_length_w_m = heat_load_w / params.fiber_length_m if params.fiber_length_m > 0 else 0

    # ── Gain spectrum ──
    n_spec = 512
    lam_center = lam_em_nm
    lam_range = gain_bw_nm * 3
    lam = np.linspace(lam_center - lam_range, lam_center + lam_range, n_spec)
    sigma_nm = gain_bw_nm / (2 * math.sqrt(2 * math.log(2)))
    gain_spectrum = g0_cm * (2 * inversion - 1) * np.exp(-((lam - lam_center)**2) / (2 * sigma_nm**2))
    gain_spectrum_db = gain_spectrum * fiber_length_cm * 10 * np.log10(math.e)

    # ── Size sweep ──
    diameters = np.linspace(2.0, 15.0, 100)
    emission_wavelengths = np.array([qd_emission_wavelength_nm(params.qd_material, d) for d in diameters])
    bandgaps = np.array([qd_bandgap_ev(params.qd_material, d) for d in diameters])
    gain_bandwidths = np.array([qd_gain_bandwidth_nm(params.qd_material, d, params.qd_size_distribution_pct) for d in diameters])

    # ── Pump sweep ──
    pump_range_mw = np.linspace(0, params.pump_power_mw * 2, 200)
    output_range_mw = np.zeros_like(pump_range_mw)
    for i, p_mw in enumerate(pump_range_mw):
        p_w = p_mw * 1e-3 * params.pump_coupling_efficiency * pump_absorbed_fraction
        out = max(0, slope_efficiency * (p_w - threshold_pump_w))
        output_range_mw[i] = out * 1e3

    return SimulationResult(
        name="QD-Doped Pulsed Fiber Laser",
        data={
            "qd": {
                "material": params.qd_material,
                "diameter_nm": params.qd_diameter_nm,
                "bandgap_ev": eg_ev,
                "emission_nm": lam_em_nm,
                "sigma_abs_cm2": sigma_abs,
                "sigma_em_cm2": sigma_em,
                "gain_bandwidth_nm": gain_bw_nm,
                "auger_lifetime_ns": auger_lifetime_ns,
                "radiative_lifetime_ns": radiative_lifetime_ns,
                "total_lifetime_ns": total_lifetime_ns,
                "effective_qy": effective_qy,
                "concentration_cm3": n_qd,
            },
            "fiber": {
                "v_number": v,
                "mfd_um": mfd,
                "a_eff_um2": a_eff,
                "overlap_factor": gamma,
                "single_mode": single_mode,
            },
            "gain": {
                "g0_cm": g0_cm,
                "pump_abs_cm": pump_abs_cm,
                "pump_absorbed_fraction": pump_absorbed_fraction,
                "absorbed_pump_mw": absorbed_pump_w * 1e3,
                "inversion": inversion,
                "net_gain_cm": net_gain_cm,
                "total_gain_db": total_gain_db,
                "saturation_power_mw": sat_power_w * 1e3,
                "threshold_pump_mw": threshold_pump_mw,
                "slope_efficiency": slope_efficiency,
            },
            "output": {
                "mode": params.pulse_mode,
                "cw_output_mw": cw_output_mw,
                "avg_power_mw": avg_power_w * 1e3,
                "pulse_energy_nj": pulse_energy_j * 1e9,
                "pulse_width_ns": pulse_width_ns,
                "peak_power_w": peak_power_w,
                "rep_rate_hz": rep_rate_hz,
            },
            "thermal": {
                "quantum_defect": quantum_defect,
                "heat_load_mw": heat_load_w * 1e3,
                "heat_per_length_mw_m": heat_per_length_w_m * 1e3,
            },
            "spectra": {
                "wavelength_nm": lam,
                "gain_spectrum_db": gain_spectrum_db,
            },
            "size_sweep": {
                "diameter_nm": diameters,
                "emission_nm": emission_wavelengths,
                "bandgap_ev": bandgaps,
                "gain_bandwidth_nm": gain_bandwidths,
            },
            "pump_sweep": {
                "pump_mw": pump_range_mw,
                "output_mw": output_range_mw,
            },
        },
        warnings=warnings,
    )
