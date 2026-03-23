"""Thermal simulation for ultrafast laser-material interaction.

Implements:
- Single-pulse surface temperature rise (1D semi-infinite model)
- Multi-pulse thermal accumulation at rep rate
- Two-temperature model (electron-phonon) for fs pulses
- Melt/ablation threshold estimation

Accelerated via harrington_common.compute:
    CUDA GPU → Numba JIT → NumPy (automatic fallback)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from harrington_common.compute import (
    _thermal_accumulation,
    _euler_two_temp,
)


@dataclass
class ThermalResult:
    """Results of thermal simulation."""

    # Single-pulse
    delta_t_surface_k: float
    thermal_diffusion_length_m: float
    heat_confined: bool

    # Multi-pulse accumulation
    t_surface_vs_pulses: np.ndarray
    n_pulses: np.ndarray
    t_steady_state_k: float

    # Thresholds
    melt_threshold_fluence_j_cm2: float
    estimated_ablation_fluence_j_cm2: float
    operating_above_melt: bool
    operating_above_ablation: bool

    # Depth profile (single pulse)
    z_m: np.ndarray
    delta_t_z_k: np.ndarray


@dataclass
class TwoTempResult:
    """Results of two-temperature model for ultrafast pulses."""

    t_ps: np.ndarray
    t_electron_k: np.ndarray
    t_lattice_k: np.ndarray
    equilibrium_time_ps: float
    peak_electron_temp_k: float
    final_lattice_temp_k: float


def thermal_analysis(
    fluence_j_cm2: float,
    pulse_width_s: float,
    rep_rate_hz: float,
    spot_radius_m: float,
    alpha_cm: float,
    thermal_conductivity_w_mk: float,
    density_kg_m3: float,
    specific_heat_j_kgk: float,
    thermal_diffusivity_m2_s: float,
    melting_point_k: float,
    t_ambient_k: float = 300.0,
    n_pulses_max: int = 1000,
) -> ThermalResult:
    """Compute thermal response for fs/ps pulse irradiation.

    Uses JIT-accelerated kernel for the O(N²) multi-pulse accumulation loop.
    """
    D = thermal_diffusivity_m2_s
    k = thermal_conductivity_w_mk
    rho = density_kg_m3
    cp = specific_heat_j_kgk
    alpha_m = alpha_cm * 100

    l_th = math.sqrt(D * pulse_width_s) if D > 0 and pulse_width_s > 0 else 0.0
    l_opt = 1.0 / alpha_m if alpha_m > 0 else float("inf")
    heat_confined = l_th < l_opt if l_opt < float("inf") else False

    fluence_j_m2 = fluence_j_cm2 * 1e4
    l_heat = max(l_th, l_opt) if l_opt < float("inf") else l_th
    if l_heat > 0 and rho > 0 and cp > 0:
        delta_t_single = fluence_j_m2 / (rho * cp * l_heat)
    else:
        delta_t_single = 0.0

    # Depth profile
    n_z = 200
    z_max = max(5 * l_heat, 5 * l_opt) if l_opt < float("inf") else 5 * l_heat
    z_max = max(z_max, 1e-6)
    z = np.linspace(0, z_max, n_z)

    if alpha_m > 0:
        delta_t_z = delta_t_single * np.exp(-alpha_m * z)
    else:
        delta_t_z = np.full(n_z, delta_t_single)

    # Multi-pulse accumulation — JIT-accelerated O(N²) loop
    n_arr = np.arange(1, n_pulses_max + 1)
    if rep_rate_hz > 0 and k > 0 and D > 0:
        period = 1.0 / rep_rate_hz
        l_between = math.sqrt(D * period)
        t_accum = _thermal_accumulation(
            n_pulses_max, delta_t_single, l_heat, l_between, t_ambient_k
        )
        t_ss = float(t_accum[-1]) if len(t_accum) > 0 else t_ambient_k
    else:
        t_accum = np.full(n_pulses_max, t_ambient_k + delta_t_single)
        t_ss = t_ambient_k + delta_t_single

    # Melt threshold
    delta_t_melt = melting_point_k - t_ambient_k
    if delta_t_single > 0 and delta_t_melt > 0:
        f_melt = fluence_j_cm2 * (delta_t_melt / delta_t_single)
    else:
        f_melt = float("inf")

    f_ablation = f_melt * 3.0

    return ThermalResult(
        delta_t_surface_k=delta_t_single,
        thermal_diffusion_length_m=l_th,
        heat_confined=heat_confined,
        t_surface_vs_pulses=t_accum,
        n_pulses=n_arr,
        t_steady_state_k=t_ss,
        melt_threshold_fluence_j_cm2=f_melt,
        estimated_ablation_fluence_j_cm2=f_ablation,
        operating_above_melt=t_ambient_k + delta_t_single > melting_point_k,
        operating_above_ablation=fluence_j_cm2 > f_ablation,
        z_m=z,
        delta_t_z_k=delta_t_z,
    )


def two_temperature_model(
    fluence_j_cm2: float,
    pulse_width_s: float,
    alpha_cm: float,
    electron_phonon_coupling_w_m3k: float,
    density_kg_m3: float,
    specific_heat_j_kgk: float,
    ce_coeff: float = 100.0,
    t_ambient_k: float = 300.0,
    t_max_ps: float = 50.0,
) -> TwoTempResult:
    """Simplified two-temperature model for ultrafast heating.

    Uses JIT-accelerated Euler integration kernel.
    """
    alpha_m = alpha_cm * 100
    fluence_j_m2 = fluence_j_cm2 * 1e4
    tau = pulse_width_s
    G = electron_phonon_coupling_w_m3k
    Cl = density_kg_m3 * specific_heat_j_kgk

    l_abs = 1.0 / alpha_m if alpha_m > 0 else 1e-6
    s_peak = fluence_j_m2 * alpha_m / tau if tau > 0 else 0.0

    dt_ps = 0.01
    n_steps = int(t_max_ps / dt_ps)
    dt_s = dt_ps * 1e-12

    t_center_s = 3 * tau
    pulse_sigma_s = tau / (2 * math.sqrt(2 * math.log(2)))

    # JIT-accelerated Euler loop
    te, tl = _euler_two_temp(
        n_steps, dt_s, s_peak, t_center_s, pulse_sigma_s,
        G, Cl, ce_coeff, t_ambient_k,
    )

    t_ps = np.linspace(0, t_max_ps, n_steps)

    # Find equilibrium time
    eq_idx = n_steps - 1
    for i in range(n_steps):
        if te[i] > t_ambient_k * 1.01:
            if abs(te[i] - tl[i]) < 0.1 * (tl[i] - t_ambient_k + 1):
                eq_idx = i
                break

    return TwoTempResult(
        t_ps=t_ps,
        t_electron_k=te,
        t_lattice_k=tl,
        equilibrium_time_ps=t_ps[eq_idx],
        peak_electron_temp_k=float(np.max(te)),
        final_lattice_temp_k=float(tl[-1]),
    )
