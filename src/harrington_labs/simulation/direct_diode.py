"""Direct Diode Lab simulator.

Models diode laser L-I characteristics, thermal rollover,
beam quality, spectral behavior, and beam combining scenarios.
No Streamlit imports.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field

from harrington_labs.domain import (
    DiodeLaserParams, BeamParams, BeamProfile, SimulationResult,
    C_M_S, H_J_S, K_B,
)


# ── L-I curve with thermal rollover ─────────────────────────────

def compute_li_curve(
    params: DiodeLaserParams,
    current_max_a: float | None = None,
    n_points: int = 200,
    heatsink_temp_c: float = 25.0,
) -> dict:
    """Compute L-I curve including thermal rollover.

    Returns dict with arrays: current_a, power_w, voltage_v,
    efficiency, junction_temp_c.
    """
    if current_max_a is None:
        current_max_a = params.operating_current_a * 1.5
    current = np.linspace(0, current_max_a, n_points)
    t_hs = heatsink_temp_c + 273.15

    power = np.zeros_like(current)
    voltage = np.zeros_like(current)
    junction_temp = np.zeros_like(current)
    efficiency = np.zeros_like(current)

    # Photon energy
    e_photon = H_J_S * C_M_S / (params.wavelength_nm * 1e-9)
    v_photon = e_photon / 1.602e-19  # voltage equivalent

    for i, I in enumerate(current):
        # Junction temperature rises with dissipated power (iterative)
        t_j = t_hs
        for _ in range(5):
            # Temperature-dependent threshold
            i_th = params.threshold_current_a * math.exp((t_j - t_hs) / params.t0_k)
            # Temperature-dependent slope efficiency
            eta_s = params.slope_efficiency * math.exp(-(t_j - t_hs) / params.t1_k)

            if I > i_th:
                p_out = eta_s * (I - i_th) * v_photon
            else:
                p_out = 0.0

            # Series resistance model
            r_s = 0.05  # Ohms typical
            v_diode = v_photon + 0.3 + r_s * I  # forward voltage
            p_elec = v_diode * I
            p_dissipated = max(p_elec - p_out, 0.0)
            t_j = t_hs + p_dissipated * params.thermal_resistance_k_w

        power[i] = max(p_out, 0.0)
        voltage[i] = v_diode
        junction_temp[i] = t_j - 273.15
        efficiency[i] = power[i] / (voltage[i] * I) if I > 0 and voltage[i] > 0 else 0.0

    return {
        "current_a": current,
        "power_w": power,
        "voltage_v": voltage,
        "efficiency": efficiency,
        "junction_temp_c": junction_temp,
    }


# ── Wavelength vs temperature ────────────────────────────────────

def wavelength_vs_temperature(
    center_nm: float = 976.0,
    temp_range_c: tuple[float, float] = (10.0, 80.0),
    n_points: int = 100,
    shift_nm_per_k: float = 0.3,
) -> dict:
    """Diode wavelength drift with junction temperature."""
    temps = np.linspace(temp_range_c[0], temp_range_c[1], n_points)
    ref_t = 25.0
    wavelengths = center_nm + shift_nm_per_k * (temps - ref_t)
    return {"temperature_c": temps, "wavelength_nm": wavelengths}


# ── Far-field pattern ────────────────────────────────────────────

def far_field_pattern(
    params: DiodeLaserParams,
    angle_range_deg: float = 60.0,
    n_points: int = 361,
) -> dict:
    """Compute fast-axis and slow-axis far-field intensity profiles."""
    angles = np.linspace(-angle_range_deg, angle_range_deg, n_points)
    angles_rad = np.deg2rad(angles)

    # Gaussian approximation for fast and slow axes
    sigma_fast = math.radians(params.beam_divergence_fast_deg / 2.0) / math.sqrt(2 * math.log(2))
    sigma_slow = math.radians(params.beam_divergence_slow_deg / 2.0) / math.sqrt(2 * math.log(2))

    fast = np.exp(-angles_rad**2 / (2 * sigma_fast**2))
    slow = np.exp(-angles_rad**2 / (2 * sigma_slow**2))

    return {
        "angle_deg": angles,
        "fast_axis": fast,
        "slow_axis": slow,
    }


# ── Beam combining efficiency ────────────────────────────────────

def spectral_beam_combining(
    n_emitters: int = 10,
    per_emitter_power_w: float = 10.0,
    grating_efficiency: float = 0.92,
    pointing_error_urad: float = 50.0,
    spectral_fill_factor: float = 0.8,
) -> dict:
    """Estimate spectral beam combining performance."""
    raw_power = n_emitters * per_emitter_power_w
    combining_eff = grating_efficiency * spectral_fill_factor
    # Pointing loss ~ exp(-theta^2 / theta_diff^2)
    # Approximate pointing degradation
    pointing_loss = math.exp(-(pointing_error_urad * 1e-6)**2 / (1e-3)**2)
    combined_power = raw_power * combining_eff * pointing_loss

    return {
        "n_emitters": n_emitters,
        "raw_power_w": raw_power,
        "combined_power_w": combined_power,
        "combining_efficiency": combining_eff * pointing_loss,
        "grating_efficiency": grating_efficiency,
        "pointing_loss": pointing_loss,
    }


# ── Full direct-diode simulation bundle ─────────────────────────

def run_direct_diode_simulation(params: DiodeLaserParams, heatsink_temp_c: float = 25.0) -> SimulationResult:
    """Run complete direct diode lab simulation."""
    li = compute_li_curve(params, heatsink_temp_c=heatsink_temp_c)
    wl_temp = wavelength_vs_temperature(center_nm=params.wavelength_nm)
    ff = far_field_pattern(params)

    warnings = []
    peak_idx = int(np.argmax(li["power_w"]))
    if li["junction_temp_c"][peak_idx] > 80:
        warnings.append(f"Junction temperature reaches {li['junction_temp_c'][peak_idx]:.0f} °C at peak power — thermal rollover likely")
    if li["efficiency"][peak_idx] < 0.3:
        warnings.append(f"Peak wall-plug efficiency only {li['efficiency'][peak_idx]:.1%}")

    return SimulationResult(
        name="Direct Diode Lab",
        parameters={"heatsink_temp_c": heatsink_temp_c},
        data={"li_curve": li, "wavelength_temp": wl_temp, "far_field": ff},
        warnings=warnings,
    )
