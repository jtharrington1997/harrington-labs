"""Pulsed Laser Lab simulator.

Models ultrafast pulse characteristics, temporal/spectral profiles,
dispersion management, nonlinear phase accumulation, autocorrelation,
and open-aperture z-scan behavior. No Streamlit imports.
"""
from __future__ import annotations

import math
import numpy as np

from harrington_common.compute import jit, parallel_map

from harrington_labs.domain import (
    PulsedSource, PulseShape, SimulationResult,
    C_M_S, H_J_S,
)


# ── Pulse temporal profile ───────────────────────────────────────

def temporal_profile(
    pulse_width_s: float,
    shape: PulseShape = PulseShape.GAUSSIAN,
    n_points: int = 512,
    window_factor: float = 5.0,
) -> dict:
    """Generate normalized temporal pulse profile."""
    half_window = window_factor * pulse_width_s
    t = np.linspace(-half_window, half_window, n_points)

    if shape == PulseShape.GAUSSIAN:
        # FWHM = pulse_width_s => σ = FWHM / (2*sqrt(2*ln2))
        sigma = pulse_width_s / (2 * math.sqrt(2 * math.log(2)))
        intensity = np.exp(-t**2 / (2 * sigma**2))
    elif shape == PulseShape.SECH2:
        tau = pulse_width_s / (2 * math.acosh(math.sqrt(2)))
        intensity = 1.0 / np.cosh(t / tau) ** 2
    elif shape == PulseShape.LORENTZIAN:
        gamma = pulse_width_s / 2.0
        intensity = gamma**2 / (t**2 + gamma**2)
    elif shape == PulseShape.SQUARE:
        intensity = np.where(np.abs(t) <= pulse_width_s / 2, 1.0, 0.0)
    else:
        sigma = pulse_width_s / (2 * math.sqrt(2 * math.log(2)))
        intensity = np.exp(-t**2 / (2 * sigma**2))

    return {"time_s": t, "intensity": intensity}


# ── Spectral profile (transform limited) ────────────────────────

def spectral_profile(
    pulse_width_s: float,
    center_wavelength_nm: float,
    shape: PulseShape = PulseShape.GAUSSIAN,
    n_points: int = 512,
) -> dict:
    """Transform-limited spectral profile."""
    # Time-bandwidth products
    tbp = {
        PulseShape.GAUSSIAN: 0.4413,
        PulseShape.SECH2: 0.3148,
        PulseShape.LORENTZIAN: 0.2206,
        PulseShape.SQUARE: 0.8859,
    }
    delta_nu = tbp.get(shape, 0.4413) / pulse_width_s  # Hz

    # Convert to wavelength bandwidth
    c = C_M_S
    lam0 = center_wavelength_nm * 1e-9
    delta_lam = lam0**2 * delta_nu / c  # m

    lam = np.linspace(
        lam0 - 5 * delta_lam,
        lam0 + 5 * delta_lam,
        n_points,
    )
    nu = c / lam
    nu0 = c / lam0
    sigma_nu = delta_nu / (2 * math.sqrt(2 * math.log(2)))

    spectrum = np.exp(-(nu - nu0)**2 / (2 * sigma_nu**2))

    return {
        "wavelength_nm": lam * 1e9,
        "spectrum": spectrum,
        "bandwidth_nm": delta_lam * 1e9,
        "tbp": tbp.get(shape, 0.4413),
    }


# ── Autocorrelation ──────────────────────────────────────────────

def intensity_autocorrelation(
    pulse_width_s: float,
    shape: PulseShape = PulseShape.GAUSSIAN,
    n_points: int = 512,
) -> dict:
    """Compute intensity autocorrelation trace."""
    prof = temporal_profile(pulse_width_s, shape, n_points)
    t = prof["time_s"]
    I = prof["intensity"]

    # Numerical autocorrelation via FFT
    ft = np.fft.fft(I)
    ac = np.fft.ifft(ft * np.conj(ft)).real
    ac = np.fft.fftshift(ac)
    ac = ac / ac.max()

    # Autocorrelation time axis
    dt = t[1] - t[0]
    tau = np.arange(-len(ac)//2, len(ac)//2) * dt

    # Deconvolution factors
    deconv = {
        PulseShape.GAUSSIAN: 1.414,
        PulseShape.SECH2: 1.543,
        PulseShape.LORENTZIAN: 2.0,
        PulseShape.SQUARE: 1.0,
    }

    return {
        "delay_s": tau,
        "autocorrelation": ac,
        "deconvolution_factor": deconv.get(shape, 1.414),
    }


# ── Dispersion stretching ───────────────────────────────────────

def chirped_pulse_width(
    pulse_width_s: float,
    gdd_fs2: float,
) -> float:
    """Pulse width after GDD for transform-limited Gaussian input.

    τ_out = τ_0 * sqrt(1 + (4ln2 * GDD / τ_0²)²)
    """
    tau0 = pulse_width_s
    gdd = gdd_fs2 * 1e-30  # convert fs² to s²
    if tau0 <= 0:
        return 0.0
    ratio = 4 * math.log(2) * gdd / tau0**2
    return tau0 * math.sqrt(1 + ratio**2)


def dispersion_scan(
    pulse_width_s: float,
    gdd_range_fs2: tuple[float, float] = (-5000, 5000),
    n_points: int = 200,
) -> dict:
    """Pulse width vs GDD for dispersion management."""
    gdd = np.linspace(gdd_range_fs2[0], gdd_range_fs2[1], n_points)
    widths = np.array([chirped_pulse_width(pulse_width_s, g) for g in gdd])
    return {"gdd_fs2": gdd, "pulse_width_s": widths}


# ── B-integral / nonlinear phase ─────────────────────────────────

def b_integral(
    peak_power_w: float,
    beam_area_cm2: float,
    n2_cm2_w: float,
    length_cm: float,
    wavelength_nm: float,
) -> float:
    """Accumulated B-integral through a material."""
    if beam_area_cm2 <= 0:
        return 0.0
    I0 = peak_power_w / beam_area_cm2
    lam_cm = wavelength_nm * 1e-7
    return (2 * math.pi / lam_cm) * n2_cm2_w * I0 * length_cm


# ── Open-aperture z-scan ─────────────────────────────────────────

def open_aperture_zscan(
    pulse: PulsedSource,
    beta_cm_w: float = 1e-10,
    sample_thickness_cm: float = 0.1,
    z_range_mm: float = 20.0,
    focal_length_mm: float = 100.0,
    n_positions: int = 200,
) -> dict:
    """Simplified open-aperture z-scan transmission vs sample position."""
    z_positions = np.linspace(-z_range_mm, z_range_mm, n_positions)
    transmission = np.zeros(n_positions)

    w0 = pulse.beam.beam_radius_m
    lam = pulse.beam.wavelength_m
    z_r_mm = (math.pi * w0**2 / (pulse.beam.m_squared * lam)) * 1e3 if lam > 0 else 1.0

    for i, z_mm in enumerate(z_positions):
        # Beam radius at sample position
        w_z = w0 * math.sqrt(1 + (z_mm / z_r_mm) ** 2)
        area = math.pi * (w_z * 1e-3) ** 2 * 1e4  # cm²

        if area > 0:
            I0 = pulse.peak_power_w / area
        else:
            I0 = 0.0

        # Nonlinear absorption: T ≈ 1 / (1 + β*I0*L_eff)
        l_eff = sample_thickness_cm
        q0 = beta_cm_w * I0 * l_eff
        if q0 > -1:
            transmission[i] = 1.0 / (1.0 + q0)
        else:
            transmission[i] = 0.0

    return {
        "z_mm": z_positions,
        "transmission": transmission,
        "z_rayleigh_mm": z_r_mm,
    }


# ── Full pulsed laser simulation bundle ──────────────────────────

def run_pulsed_laser_simulation(pulse: PulsedSource) -> SimulationResult:
    """Run complete pulsed laser lab simulation."""
    temp = temporal_profile(pulse.pulse_width_s, pulse.pulse_shape)
    spec = spectral_profile(pulse.pulse_width_s, pulse.beam.wavelength_nm, pulse.pulse_shape)
    ac = intensity_autocorrelation(pulse.pulse_width_s, pulse.pulse_shape)
    disp = dispersion_scan(pulse.pulse_width_s)

    warnings = []
    if pulse.peak_power_w > 1e12:
        warnings.append(f"Peak power {pulse.peak_power_w:.2e} W — self-focusing/damage risk")
    if pulse.fluence_j_cm2 > 1.0:
        warnings.append(f"Fluence {pulse.fluence_j_cm2:.2f} J/cm² — near LIDT for many optics")

    return SimulationResult(
        name="Pulsed Laser Lab",
        data={
            "temporal": temp,
            "spectral": spec,
            "autocorrelation": ac,
            "dispersion_scan": disp,
            "pulse_summary": {
                "pulse_energy_j": pulse.pulse_energy_j,
                "peak_power_w": pulse.peak_power_w,
                "fluence_j_cm2": pulse.fluence_j_cm2,
                "bandwidth_nm": spec["bandwidth_nm"],
                "tbp": spec["tbp"],
            },
        },
        warnings=warnings,
    )
