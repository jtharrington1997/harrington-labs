"""Thermal simulation for ultrafast laser-material interaction.

Implements:
- Single-pulse surface temperature rise (1D semi-infinite model)
- Multi-pulse thermal accumulation at rep rate
- Two-temperature model (electron-phonon) for fs pulses
- Melt/ablation threshold estimation
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ThermalResult:
    """Results of thermal simulation."""

    # Single-pulse
    delta_t_surface_k: float  # surface temperature rise from one pulse
    thermal_diffusion_length_m: float  # sqrt(D * tau)
    heat_confined: bool  # diffusion length < absorption depth?

    # Multi-pulse accumulation
    t_surface_vs_pulses: np.ndarray  # temperature after N pulses
    n_pulses: np.ndarray  # pulse number array
    t_steady_state_k: float  # approximate steady-state surface T

    # Thresholds
    melt_threshold_fluence_j_cm2: float
    estimated_ablation_fluence_j_cm2: float
    operating_above_melt: bool
    operating_above_ablation: bool

    # Depth profile (single pulse)
    z_m: np.ndarray
    delta_t_z_k: np.ndarray  # temperature vs depth


@dataclass
class TwoTempResult:
    """Results of two-temperature model for ultrafast pulses."""

    t_ps: np.ndarray  # time array in ps
    t_electron_k: np.ndarray  # electron temperature vs time
    t_lattice_k: np.ndarray  # lattice temperature vs time
    equilibrium_time_ps: float  # time for Te ~ Tl
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

    Uses the 1D semi-infinite solid model for surface heating,
    appropriate when the thermal diffusion length during the pulse
    is much smaller than the spot size (which it is for fs pulses).

    Parameters
    ----------
    fluence_j_cm2 : float
        Incident fluence per pulse.
    pulse_width_s : float
        Pulse duration (FWHM).
    rep_rate_hz : float
        Repetition rate.
    spot_radius_m : float
        1/e^2 beam radius at the surface.
    alpha_cm : float
        Linear absorption coefficient.
    thermal_conductivity_w_mk : float
        Thermal conductivity.
    density_kg_m3 : float
        Mass density.
    specific_heat_j_kgk : float
        Specific heat capacity.
    thermal_diffusivity_m2_s : float
        Thermal diffusivity D = k / (rho * cp).
    melting_point_k : float
        Melting temperature.
    t_ambient_k : float
        Ambient temperature.
    n_pulses_max : int
        Number of pulses to simulate for accumulation.
    """
    D = thermal_diffusivity_m2_s
    k = thermal_conductivity_w_mk
    rho = density_kg_m3
    cp = specific_heat_j_kgk
    alpha_m = alpha_cm * 100  # 1/m

    # Thermal diffusion length during one pulse
    l_th = math.sqrt(D * pulse_width_s) if D > 0 and pulse_width_s > 0 else 0.0

    # Optical penetration depth
    l_opt = 1.0 / alpha_m if alpha_m > 0 else float("inf")

    # Heat confinement: thermal diffusion length << optical penetration
    heat_confined = l_th < l_opt if l_opt < float("inf") else False

    # Single-pulse surface temperature rise
    # For heat-confined regime: delta_T = F_abs / (rho * cp * l_heat)
    # where l_heat = max(l_th, l_opt) is the characteristic heated depth
    fluence_j_m2 = fluence_j_cm2 * 1e4  # convert to J/m^2
    l_heat = max(l_th, l_opt) if l_opt < float("inf") else l_th
    if l_heat > 0 and rho > 0 and cp > 0:
        delta_t_single = fluence_j_m2 / (rho * cp * l_heat)
    else:
        delta_t_single = 0.0

    # Depth profile for single pulse (exponential absorption)
    n_z = 200
    z_max = max(5 * l_heat, 5 * l_opt) if l_opt < float("inf") else 5 * l_heat
    z_max = max(z_max, 1e-6)  # at least 1 um
    z = np.linspace(0, z_max, n_z)

    if alpha_m > 0:
        # Source term decays exponentially, heat hasn't diffused yet (fs)
        delta_t_z = delta_t_single * np.exp(-alpha_m * z)
    else:
        delta_t_z = np.full(n_z, delta_t_single)

    # Multi-pulse accumulation
    # Between pulses, heat diffuses radially over length sqrt(D / f_rep)
    # Steady-state surface temp rise ~ F_abs * f_rep / (k * sqrt(pi * D / f_rep))
    # (from repeated Gaussian heat sources)
    n_arr = np.arange(1, n_pulses_max + 1)
    if rep_rate_hz > 0 and k > 0 and D > 0:
        period = 1.0 / rep_rate_hz
        l_between = math.sqrt(D * period)

        # Each old pulse contributes delta_T * (l_heat / sqrt(l_heat^2 + n*l_between^2))
        t_accum = np.zeros(n_pulses_max)
        for i in range(n_pulses_max):
            contributions = 0.0
            for j in range(i + 1):
                n_elapsed = i - j  # pulses ago
                if n_elapsed == 0:
                    contributions += delta_t_single
                else:
                    denom = math.sqrt(l_heat**2 + n_elapsed * l_between**2)
                    contributions += delta_t_single * l_heat / denom if denom > 0 else 0
            t_accum[i] = t_ambient_k + contributions

        # Approximate steady state (last value)
        t_ss = t_accum[-1] if len(t_accum) > 0 else t_ambient_k
    else:
        t_accum = np.full(n_pulses_max, t_ambient_k + delta_t_single)
        t_ss = t_ambient_k + delta_t_single

    # Melt threshold fluence (single pulse)
    delta_t_melt = melting_point_k - t_ambient_k
    if delta_t_single > 0 and delta_t_melt > 0:
        f_melt = fluence_j_cm2 * (delta_t_melt / delta_t_single)
    else:
        f_melt = float("inf")

    # Rough ablation threshold (~3x melt for semiconductors)
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

    Solves the coupled electron-lattice energy equations:
        Ce * dTe/dt = S(t) - G*(Te - Tl)
        Cl * dTl/dt = G*(Te - Tl)

    where Ce = ce_coeff * Te (electronic heat capacity scales with Te),
    Cl = rho * cp, G = electron-phonon coupling constant.

    Parameters
    ----------
    fluence_j_cm2 : float
        Absorbed fluence at surface.
    pulse_width_s : float
        Pulse duration (FWHM).
    alpha_cm : float
        Absorption coefficient (determines heated depth).
    electron_phonon_coupling_w_m3k : float
        Electron-phonon coupling constant G.
    density_kg_m3 : float
        Material density.
    specific_heat_j_kgk : float
        Lattice specific heat.
    ce_coeff : float
        Electronic heat capacity coefficient (Ce = ce_coeff * Te), J/(m^3 K^2).
    t_ambient_k : float
        Initial temperature.
    t_max_ps : float
        Simulation duration in ps.
    """
    alpha_m = alpha_cm * 100
    fluence_j_m2 = fluence_j_cm2 * 1e4
    tau = pulse_width_s
    G = electron_phonon_coupling_w_m3k
    Cl = density_kg_m3 * specific_heat_j_kgk  # J/(m^3 K)

    # Heated volume depth
    l_abs = 1.0 / alpha_m if alpha_m > 0 else 1e-6

    # Source term peak intensity (W/m^3)
    # S_peak = F_abs * alpha / tau (absorbed energy density rate)
    s_peak = fluence_j_m2 * alpha_m / tau if tau > 0 else 0.0

    # Time stepping
    dt_ps = 0.01  # 10 fs steps
    n_steps = int(t_max_ps / dt_ps)
    t_ps = np.linspace(0, t_max_ps, n_steps)
    dt_s = dt_ps * 1e-12

    te = np.full(n_steps, float(t_ambient_k))
    tl = np.full(n_steps, float(t_ambient_k))

    # Gaussian pulse temporal profile centered at 3*tau
    t_center_s = 3 * tau
    pulse_sigma_s = tau / (2 * math.sqrt(2 * math.log(2)))

    for i in range(1, n_steps):
        t_s = t_ps[i - 1] * 1e-12

        # Source term (Gaussian pulse envelope)
        s_t = s_peak * math.exp(-((t_s - t_center_s) ** 2) / (2 * pulse_sigma_s**2))

        # Electronic heat capacity
        ce = ce_coeff * te[i - 1]  # J/(m^3 K)
        ce = max(ce, 1.0)  # numerical floor

        # Euler step
        dte = (s_t - G * (te[i - 1] - tl[i - 1])) / ce * dt_s
        dtl = G * (te[i - 1] - tl[i - 1]) / Cl * dt_s if Cl > 0 else 0.0

        te[i] = te[i - 1] + dte
        tl[i] = tl[i - 1] + dtl

        # Prevent unphysical negatives
        te[i] = max(te[i], t_ambient_k)
        tl[i] = max(tl[i], t_ambient_k)

    # Find equilibrium time (Te within 10% of Tl)
    eq_idx = n_steps - 1
    for i in range(n_steps):
        if te[i] > t_ambient_k * 1.01:  # started heating
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
