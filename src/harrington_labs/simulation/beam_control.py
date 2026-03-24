"""Beam Control & Atmospheric Propagation Lab simulator.

Models free-space beam propagation, atmospheric turbulence effects,
Fried parameter, scintillation, adaptive optics correction,
and beam wander/spread. No Streamlit imports.
"""
from __future__ import annotations

import math
import numpy as np

from harrington_common.compute import jit, parallel_map

from harrington_labs.domain import (
    BeamParams, PropagationPath, AdaptiveOpticsParams,
    AtmosphericCondition, SimulationResult, C_M_S,
)


# ── Atmospheric extinction ───────────────────────────────────────

_CONDITION_VISIBILITY_KM = {
    AtmosphericCondition.CLEAR: 23.0,
    AtmosphericCondition.HAZE: 5.0,
    AtmosphericCondition.FOG: 0.5,
    AtmosphericCondition.RAIN_LIGHT: 8.0,
    AtmosphericCondition.RAIN_HEAVY: 2.0,
    AtmosphericCondition.TURBULENCE_WEAK: 23.0,
    AtmosphericCondition.TURBULENCE_MODERATE: 23.0,
    AtmosphericCondition.TURBULENCE_STRONG: 23.0,
}

_CONDITION_CN2 = {
    AtmosphericCondition.CLEAR: 1e-16,
    AtmosphericCondition.HAZE: 5e-16,
    AtmosphericCondition.FOG: 1e-15,
    AtmosphericCondition.RAIN_LIGHT: 5e-16,
    AtmosphericCondition.RAIN_HEAVY: 1e-15,
    AtmosphericCondition.TURBULENCE_WEAK: 1e-16,
    AtmosphericCondition.TURBULENCE_MODERATE: 1e-14,
    AtmosphericCondition.TURBULENCE_STRONG: 1e-13,
}


def atmospheric_extinction_db_km(
    wavelength_nm: float,
    visibility_km: float,
) -> float:
    """Kim model for atmospheric extinction coefficient."""
    # Kim model: α = (3.91/V) * (λ/550)^(-q)
    # where q depends on visibility
    lam_um = wavelength_nm / 1000.0
    if visibility_km > 50:
        q = 1.6
    elif visibility_km > 6:
        q = 1.3
    elif visibility_km > 1:
        q = 0.16 * visibility_km + 0.34
    else:
        q = visibility_km - 0.5
        q = max(q, 0.0)

    alpha_km = (3.91 / visibility_km) * (lam_um / 0.55) ** (-q)
    return alpha_km * 4.343  # to dB/km


def transmission_over_path(
    wavelength_nm: float,
    path: PropagationPath,
) -> float:
    """Total atmospheric transmission over the path."""
    vis = path.visibility_km or _CONDITION_VISIBILITY_KM.get(path.condition, 23.0)
    alpha_db_km = atmospheric_extinction_db_km(wavelength_nm, vis)
    dist_km = path.distance_m / 1000.0
    return 10 ** (-alpha_db_km * dist_km / 10)


# ── Turbulence parameters ────────────────────────────────────────

def fried_parameter_m(
    wavelength_nm: float,
    cn2: float,
    path_length_m: float,
) -> float:
    """Fried coherence length r₀ for plane wave."""
    lam = wavelength_nm * 1e-9
    if cn2 <= 0 or path_length_m <= 0:
        return float("inf")
    r0 = (0.423 * (2 * math.pi / lam) ** 2 * cn2 * path_length_m) ** (-3.0 / 5.0)
    return r0


def coherence_time_s(r0_m: float, wind_speed_m_s: float) -> float:
    """Greenwood time delay τ₀ ≈ 0.314 r₀/v."""
    if wind_speed_m_s <= 0:
        return float("inf")
    return 0.314 * r0_m / wind_speed_m_s


def isoplanatic_angle_rad(r0_m: float, path_length_m: float) -> float:
    """Isoplanatic angle θ₀ ≈ 0.314 r₀/L."""
    if path_length_m <= 0:
        return math.pi
    return 0.314 * r0_m / path_length_m


def rytov_variance(
    wavelength_nm: float,
    cn2: float,
    path_length_m: float,
) -> float:
    """Rytov variance σ²_R for plane wave — scintillation strength."""
    lam = wavelength_nm * 1e-9
    k = 2 * math.pi / lam
    return 1.23 * cn2 * k ** (7.0 / 6.0) * path_length_m ** (11.0 / 6.0)


# ── Beam propagation through turbulence ──────────────────────────

def long_term_beam_spread_m(
    beam: BeamParams,
    path: PropagationPath,
) -> dict:
    """Long-term and short-term beam radius at target."""
    lam = beam.wavelength_m
    w0 = beam.beam_radius_m
    L = path.distance_m
    cn2 = path.cn2 or _CONDITION_CN2.get(path.condition, 1e-16)

    # Vacuum diffraction
    z_r = math.pi * w0**2 / (beam.m_squared * lam) if lam > 0 else 1.0
    w_diff = w0 * math.sqrt(1 + (L / z_r) ** 2)

    # Fried parameter
    r0 = fried_parameter_m(beam.wavelength_nm, cn2, L)

    # Long-term spread includes beam wander
    d_aperture = 2 * w0
    if r0 < float("inf") and r0 > 0:
        w_lt = w_diff * math.sqrt(1 + (d_aperture / r0) ** (5.0 / 3.0))
    else:
        w_lt = w_diff

    # Short-term (tilt-removed)
    w_st = w_diff * math.sqrt(max(1 + 0.56 * (d_aperture / r0) ** (5.0 / 3.0), 1.0)) if r0 > 0 and r0 < float("inf") else w_diff

    # Beam wander variance
    if r0 > 0 and r0 < float("inf"):
        sigma_bw = 0.54 * L**2 * lam**2 / (d_aperture**(1.0/3.0) * r0**(5.0/3.0))
        beam_wander_rms_m = math.sqrt(max(sigma_bw, 0))
    else:
        beam_wander_rms_m = 0.0

    return {
        "w_vacuum_m": w_diff,
        "w_long_term_m": w_lt,
        "w_short_term_m": w_st,
        "beam_wander_rms_m": beam_wander_rms_m,
        "r0_m": r0,
        "d_over_r0": d_aperture / r0 if r0 > 0 and r0 < float("inf") else 0.0,
    }


# ── Propagation profile along path ──────────────────────────────

@jit
def _turbulence_broadening_kernel(z, w_vac, w0, wavelength_nm, cn2):
    """Compute turbulence-broadened beam radius at each z step.

    Uses Fried parameter r₀ evaluated at each propagation distance
    and long-term beam spread model.  Returns (w_turb, r0_local) arrays.
    """
    lam = wavelength_nm * 1e-9
    k = 2.0 * 3.141592653589793 / lam
    n = len(z)
    w_turb = np.empty(n)
    r0_local = np.empty(n)
    d_aperture = 2.0 * w0

    for i in range(n):
        zi = z[i]
        if zi <= 0.0 or cn2 <= 0.0:
            w_turb[i] = w_vac[i]
            r0_local[i] = 1e30  # effectively infinite
            continue
        # Fried parameter at distance z_i
        r0 = (0.423 * k**2 * cn2 * zi) ** (-3.0 / 5.0)
        r0_local[i] = r0
        # Long-term spread including beam wander
        if r0 > 0.0 and r0 < 1e20:
            ratio = (d_aperture / r0) ** (5.0 / 3.0)
            w_turb[i] = w_vac[i] * math.sqrt(1.0 + ratio)
        else:
            w_turb[i] = w_vac[i]

    return w_turb, r0_local


def propagation_profile(
    beam: BeamParams,
    path: PropagationPath,
    n_steps: int = 200,
) -> dict:
    """Beam radius and irradiance along the propagation path."""
    z = np.linspace(0, path.distance_m, n_steps)
    lam = beam.wavelength_m
    w0 = beam.beam_radius_m
    z_r = beam.rayleigh_range_m if beam.rayleigh_range_m > 0 else 1.0
    cn2 = path.cn2 or _CONDITION_CN2.get(path.condition, 1e-16)

    w_vac = w0 * np.sqrt(1 + (z / z_r) ** 2)

    # Turbulence broadening accumulates
    # JIT-accelerated turbulence broadening
    w_turb, r0_local = _turbulence_broadening_kernel(
        z, w_vac, w0, beam.wavelength_nm, cn2,
    )

    # Transmission along path
    vis = path.visibility_km or _CONDITION_VISIBILITY_KM.get(path.condition, 23.0)
    alpha_db_km = atmospheric_extinction_db_km(beam.wavelength_nm, vis)
    alpha_per_m = alpha_db_km / 4.343 / 1000.0
    transmission = np.exp(-alpha_per_m * z)

    irradiance_vac = beam.power_w * transmission / (math.pi * w_vac**2) * 1e-4  # W/cm²
    irradiance_turb = beam.power_w * transmission / (math.pi * w_turb**2) * 1e-4

    return {
        "z_m": z,
        "w_vacuum_m": w_vac,
        "w_turbulence_m": w_turb,
        "r0_m": r0_local,
        "transmission": transmission,
        "irradiance_vacuum_w_cm2": irradiance_vac,
        "irradiance_turbulence_w_cm2": irradiance_turb,
    }


# ── Adaptive optics Strehl estimate ─────────────────────────────

def ao_strehl_estimate(
    beam: BeamParams,
    path: PropagationPath,
    ao: AdaptiveOpticsParams,
) -> dict:
    """Estimate AO-corrected Strehl ratio using Maréchal approximation."""
    cn2 = path.cn2 or _CONDITION_CN2.get(path.condition, 1e-16)
    r0 = fried_parameter_m(beam.wavelength_nm, cn2, path.distance_m)
    tau0 = coherence_time_s(r0, path.wind_speed_m_s)
    theta0 = isoplanatic_angle_rad(r0, path.distance_m)

    d_aperture = 2 * beam.beam_radius_m
    d_over_r0 = d_aperture / r0 if r0 > 0 and r0 < float("inf") else 0.0

    # Fitting error: σ²_fit = a * (d_sub/r0)^(5/3)
    # d_sub = aperture / sqrt(N_actuators)
    n_act = ao.actuator_count
    d_sub = d_aperture / math.sqrt(n_act) if n_act > 0 else d_aperture
    sigma2_fit = 0.28 * (d_sub / r0) ** (5.0 / 3.0) if r0 > 0 and r0 < float("inf") else 0.0

    # Temporal error: σ²_temp = (τ_lag / τ₀)^(5/3)
    tau_lag = ao.latency_ms * 1e-3
    sigma2_temp = (tau_lag / tau0) ** (5.0 / 3.0) if tau0 > 0 and tau0 < float("inf") else 0.0

    # Total residual wavefront error
    sigma2_total = sigma2_fit + sigma2_temp

    # Strehl via Maréchal
    strehl_corrected = math.exp(-sigma2_total) if sigma2_total < 10 else 0.0

    # Uncorrected Strehl
    sigma2_uncorrected = (d_over_r0) ** (5.0 / 3.0) if d_over_r0 > 0 else 0.0
    strehl_uncorrected = math.exp(-sigma2_uncorrected) if sigma2_uncorrected < 10 else 0.0

    return {
        "r0_m": r0,
        "d_over_r0": d_over_r0,
        "tau0_ms": tau0 * 1e3 if tau0 < float("inf") else float("inf"),
        "theta0_urad": theta0 * 1e6 if theta0 < math.pi else float("inf"),
        "sigma2_fit": sigma2_fit,
        "sigma2_temp": sigma2_temp,
        "sigma2_total": sigma2_total,
        "strehl_corrected": strehl_corrected,
        "strehl_uncorrected": strehl_uncorrected,
        "strehl_improvement": strehl_corrected / strehl_uncorrected if strehl_uncorrected > 0 else float("inf"),
    }


# ── Full beam control simulation bundle ──────────────────────────

def run_beam_control_simulation(
    beam: BeamParams,
    path: PropagationPath,
    ao: AdaptiveOpticsParams | None = None,
) -> SimulationResult:
    """Run complete beam control / atmospheric propagation simulation."""
    spread = long_term_beam_spread_m(beam, path)
    profile = propagation_profile(beam, path)
    trans = transmission_over_path(beam.wavelength_nm, path)
    rytov = rytov_variance(beam.wavelength_nm,
                           path.cn2 or _CONDITION_CN2.get(path.condition, 1e-16),
                           path.distance_m)

    data = {
        "beam_spread": spread,
        "profile": profile,
        "path_transmission": trans,
        "rytov_variance": rytov,
    }

    if ao is not None:
        strehl = ao_strehl_estimate(beam, path, ao)
        data["ao_strehl"] = strehl

    warnings = []
    if rytov > 0.3:
        warnings.append(f"Rytov variance σ²_R = {rytov:.2f} — strong scintillation regime")
    if spread["d_over_r0"] > 10:
        warnings.append(f"D/r₀ = {spread['d_over_r0']:.1f} — deeply turbulence-dominated")
    if trans < 0.5:
        warnings.append(f"Path transmission only {trans:.1%} — high extinction")

    return SimulationResult(
        name="Beam Control / Atmospheric Propagation Lab",
        data=data,
        warnings=warnings,
    )
