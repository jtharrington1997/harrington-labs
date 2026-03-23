"""Fiber Laser Lab simulator.

Models fiber laser/amplifier gain, ASE buildup, nonlinear limits
(SBS, SRS, SPM), thermal management, and output characteristics.
No Streamlit imports.
"""
from __future__ import annotations

import math
import numpy as np

from harrington_common.compute import jit, parallel_map

from harrington_labs.domain import (
    FiberLaserParams, FiberType, SimulationResult,
    C_M_S, H_J_S, K_B,
)


# ── Fiber mode parameters ────────────────────────────────────────

def v_number(core_diameter_um: float, na: float, wavelength_nm: float) -> float:
    """Compute V-number for step-index fiber."""
    a = (core_diameter_um / 2.0) * 1e-6
    lam = wavelength_nm * 1e-9
    return (2 * math.pi * a * na) / lam


def mode_field_diameter_um(core_diameter_um: float, v: float) -> float:
    """Approximate MFD using Marcuse formula for step-index fiber."""
    if v <= 0:
        return core_diameter_um
    a = core_diameter_um / 2.0
    # Marcuse: w/a ≈ 0.65 + 1.619/V^(3/2) + 2.879/V^6
    w_over_a = 0.65 + 1.619 / v**1.5 + 2.879 / v**6
    return 2 * a * w_over_a


def effective_area_um2(mfd_um: float) -> float:
    """Effective mode area from MFD."""
    return math.pi * (mfd_um / 2.0) ** 2


# ── Gain and amplification ───────────────────────────────────────

def small_signal_gain_db(
    params: FiberLaserParams,
    inversion: float = 0.6,
) -> float:
    """Estimate small-signal gain in dB.

    Simplified rate-equation-derived gain for Yb-doped fiber.
    """
    # Absorption and emission cross sections (typical Yb @ 1064 nm)
    sigma_e = 0.6e-24   # cm²  emission at signal
    sigma_a = 0.01e-24  # cm²  absorption at signal

    # Ion density from doping
    n_ions = params.doping_concentration_ppm * 1e-6 * 2.2e22  # ions/cm³ (silica host)

    # Gain coefficient
    g = n_ions * (inversion * sigma_e - (1 - inversion) * sigma_a)  # cm⁻¹

    # Background loss
    alpha_bg = params.background_loss_db_m / 4.343 * 1e-2  # convert dB/m to cm⁻¹

    net_gain_per_cm = g - alpha_bg
    length_cm = params.fiber_length_m * 100.0

    gain_neper = net_gain_per_cm * length_cm
    return gain_neper * 4.343  # to dB


def amplifier_output(
    params: FiberLaserParams,
    gain_db: float | None = None,
) -> dict:
    """Compute amplifier output power and efficiency."""
    if gain_db is None:
        gain_db = small_signal_gain_db(params)

    # Saturated gain estimation
    gain_linear = 10 ** (gain_db / 10)
    signal_out = params.signal_seed_power_w * gain_linear

    # Quantum efficiency limit
    qe = params.pump_wavelength_nm / params.signal_wavelength_nm
    max_signal = params.pump_power_w * qe

    # Saturation clamping
    if signal_out > max_signal:
        signal_out = max_signal
        gain_db = 10 * math.log10(signal_out / params.signal_seed_power_w) if params.signal_seed_power_w > 0 else 0

    efficiency = signal_out / params.pump_power_w if params.pump_power_w > 0 else 0

    return {
        "signal_out_w": signal_out,
        "gain_db": gain_db,
        "quantum_efficiency_limit": qe,
        "optical_efficiency": efficiency,
        "pump_absorbed_w": params.pump_power_w * 0.95,  # typical absorption
        "heat_load_w": params.pump_power_w * 0.95 - signal_out,
    }


# ── Nonlinear thresholds ────────────────────────────────────────

def sbs_threshold_w(
    fiber_length_m: float,
    effective_area_um2: float,
    linewidth_mhz: float = 10.0,
) -> float:
    """Estimate SBS threshold power.

    P_sbs ≈ 21 * A_eff / (g_B * L_eff)
    """
    g_b = 5e-11  # m/W, typical for silica
    # Linewidth broadening factor
    delta_nu_b = 30e6  # Brillouin linewidth Hz
    broadening = 1 + linewidth_mhz * 1e6 / delta_nu_b

    a_eff = effective_area_um2 * 1e-12  # m²
    l_eff = (1 - math.exp(-0.005 * fiber_length_m)) / 0.005 if fiber_length_m > 0 else 0
    if l_eff <= 0 or g_b <= 0:
        return float("inf")

    return 21 * a_eff * broadening / (g_b * l_eff)


def srs_threshold_w(
    fiber_length_m: float,
    effective_area_um2: float,
) -> float:
    """Estimate SRS threshold power."""
    g_r = 1e-13  # m/W Raman gain coefficient
    a_eff = effective_area_um2 * 1e-12
    l_eff = (1 - math.exp(-0.005 * fiber_length_m)) / 0.005 if fiber_length_m > 0 else 0
    if l_eff <= 0:
        return float("inf")
    return 16 * a_eff / (g_r * l_eff)


def self_phase_modulation_b_integral(
    peak_power_w: float,
    fiber_length_m: float,
    effective_area_um2: float,
    n2: float = 2.6e-20,  # m²/W for silica
) -> float:
    """B-integral from SPM accumulation."""
    a_eff = effective_area_um2 * 1e-12
    if a_eff <= 0:
        return 0.0
    gamma = 2 * math.pi * n2 / (1064e-9 * a_eff)  # nonlinear parameter
    l_eff = (1 - math.exp(-0.005 * fiber_length_m)) / 0.005 if fiber_length_m > 0 else 0
    return gamma * peak_power_w * l_eff


# ── Thermal estimate ─────────────────────────────────────────────

def fiber_thermal_estimate(
    heat_load_w: float,
    fiber_length_m: float,
    cladding_diameter_um: float = 250.0,
    coating_diameter_um: float = 400.0,
    h_conv: float = 50.0,  # W/m²/K convective coefficient
) -> dict:
    """Estimate fiber core temperature rise."""
    if fiber_length_m <= 0:
        return {"core_temp_rise_k": 0.0, "surface_temp_rise_k": 0.0}

    q_linear = heat_load_w / fiber_length_m  # W/m
    r_coat = (coating_diameter_um / 2) * 1e-6

    # Surface temp rise from convection
    circumference = 2 * math.pi * r_coat
    surface_rise = q_linear / (h_conv * circumference) if circumference > 0 else 0

    # Core-to-surface rise (conduction through silica)
    k_silica = 1.38  # W/m/K
    r_clad = (cladding_diameter_um / 2) * 1e-6
    core_rise = q_linear / (4 * math.pi * k_silica)  # simplified

    return {
        "core_temp_rise_k": core_rise + surface_rise,
        "surface_temp_rise_k": surface_rise,
        "linear_heat_load_w_m": q_linear,
    }


# ── Gain-along-fiber profile ────────────────────────────────────


@jit
def _fiber_propagation_kernel(
    n_steps, dz, alpha_p_neper, clad_area,
    n_ions, sigma_e, sigma_a, bg_loss, qe, max_signal,
    pump_init, signal_init,
):
    """Pump/signal co-propagation — JIT-accelerated."""
    pump = np.zeros(n_steps)
    signal = np.zeros(n_steps)
    pump[0] = pump_init
    signal[0] = signal_init

    for i in range(1, n_steps):
        pump[i] = pump[i - 1] * math.exp(-alpha_p_neper * dz)
        local_pump_intensity = pump[i] / clad_area
        inversion = min(0.95, 0.4 + 0.3 * (local_pump_intensity / 1e4))
        g = n_ions * (inversion * sigma_e - (1.0 - inversion) * sigma_a) * 1e-2
        net_g = g - bg_loss
        signal[i] = signal[i - 1] * math.exp(net_g * dz)
        if signal[i] > max_signal:
            signal[i] = max_signal

    return pump, signal


def gain_profile(
    params: FiberLaserParams,
    n_steps: int = 200,
) -> dict:
    """Simplified forward-propagation gain profile along fiber."""
    z = np.linspace(0, params.fiber_length_m, n_steps)
    dz = z[1] - z[0] if n_steps > 1 else params.fiber_length_m

    # Absorption coefficients
    alpha_p = 1.7  # dB/m pump absorption (typical Yb @ 976)
    alpha_p_neper = alpha_p / 4.343

    pump = np.zeros(n_steps)
    signal = np.zeros(n_steps)
    pump[0] = params.pump_power_w
    signal[0] = params.signal_seed_power_w

    sigma_e = 0.6e-24
    sigma_a = 0.01e-24
    n_ions = params.doping_concentration_ppm * 1e-6 * 2.2e22
    qe = params.pump_wavelength_nm / params.signal_wavelength_nm

    bg_loss = params.background_loss_db_m / 4.343
    clad_area = math.pi * (params.cladding_diameter_um / 2 * 1e-4) ** 2
    max_signal = params.pump_power_w * qe

    pump, signal = _fiber_propagation_kernel(
        n_steps, dz, alpha_p_neper, clad_area,
        n_ions, sigma_e, sigma_a, bg_loss, qe, max_signal,
        params.pump_power_w, params.signal_seed_power_w,
    )

    return {"z_m": z, "pump_w": pump, "signal_w": signal}


# ── Full fiber laser simulation bundle ───────────────────────────

def run_fiber_laser_simulation(params: FiberLaserParams) -> SimulationResult:
    """Run complete fiber laser lab simulation."""
    v = v_number(params.core_diameter_um, params.na, params.signal_wavelength_nm)
    mfd = mode_field_diameter_um(params.core_diameter_um, v)
    a_eff = effective_area_um2(mfd)

    gain = small_signal_gain_db(params)
    amp = amplifier_output(params, gain)

    p_sbs = sbs_threshold_w(params.fiber_length_m, a_eff)
    p_srs = srs_threshold_w(params.fiber_length_m, a_eff)

    thermal = fiber_thermal_estimate(
        amp["heat_load_w"], params.fiber_length_m,
        params.cladding_diameter_um,
    )
    profile = gain_profile(params)

    warnings = []
    if v > 2.405:
        n_modes = int(v**2 / 2)
        warnings.append(f"V = {v:.2f} — fiber supports ~{n_modes} modes (not single-mode)")
    if amp["signal_out_w"] > p_sbs:
        warnings.append(f"Output {amp['signal_out_w']:.1f} W exceeds SBS threshold {p_sbs:.1f} W")
    if amp["signal_out_w"] > p_srs:
        warnings.append(f"Output {amp['signal_out_w']:.1f} W exceeds SRS threshold {p_srs:.1f} W")
    if thermal["core_temp_rise_k"] > 200:
        warnings.append(f"Core temp rise {thermal['core_temp_rise_k']:.0f} K — coating damage risk")

    return SimulationResult(
        name="Fiber Laser Lab",
        data={
            "fiber_params": {"v_number": v, "mfd_um": mfd, "a_eff_um2": a_eff},
            "amplifier": amp,
            "nonlinear": {"sbs_threshold_w": p_sbs, "srs_threshold_w": p_srs},
            "thermal": thermal,
            "gain_profile": profile,
        },
        warnings=warnings,
    )
