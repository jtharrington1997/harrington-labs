"""Nonlinear optical effects for ultrafast laser interactions.

Implements:
- Multiphoton absorption (MPA) rate and cross-section estimation
- Self-focusing critical power and collapse distance
- Kerr effect (B-integral, delta_n)
- Nonlinear absorption depth profiling

Accelerated via harrington_common.compute:
    CUDA GPU → Numba JIT → NumPy (automatic fallback)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from harrington_common.compute import (
    jit, _nonlinear_propagation,
)


@dataclass
class MPAResult:
    """Multiphoton absorption analysis."""

    photon_energy_ev: float
    bandgap_ev: float
    photon_order: int
    mpa_rate_cm_2n_1_w_n: float
    mpa_absorption_depth_m: float
    energy_deposited_fraction: float
    is_dominant: bool


@dataclass
class SelfFocusingResult:
    """Self-focusing analysis."""

    n2_cm2_w: float
    critical_power_w: float
    input_power_w: float
    p_over_pcr: float
    self_focusing_occurs: bool
    collapse_distance_m: float
    delta_n_peak: float
    b_integral_rad: float


@dataclass
class NonlinearResult:
    """Combined nonlinear analysis."""

    mpa: MPAResult
    self_focusing: SelfFocusingResult

    z_m: np.ndarray
    irradiance_z_w_cm2: np.ndarray
    delta_n_z: np.ndarray
    carrier_density_z_cm3: np.ndarray


@jit
def _estimate_mpa_coefficient_kernel(photon_order, refractive_index):
    """Keldysh-type MPA coefficient estimate — JIT kernel."""
    base = 1e-18
    reduction_per_photon = 1e-10
    sigma_n = base * reduction_per_photon ** (photon_order - 1)
    sigma_n *= refractive_index ** photon_order
    return sigma_n


def estimate_mpa_coefficient(
    photon_order: int,
    bandgap_ev: float,
    refractive_index: float,
) -> float:
    """Estimate MPA coefficient using Keldysh-type scaling."""
    return float(_estimate_mpa_coefficient_kernel(photon_order, refractive_index))


def mpa_analysis(
    wavelength_nm: float,
    bandgap_ev: float,
    irradiance_w_cm2: float,
    pulse_width_s: float,
    alpha_linear_cm: float,
    refractive_index: float,
    thickness_m: float,
) -> MPAResult:
    """Analyze multiphoton absorption for a given interaction."""
    photon_ev = 1240.0 / wavelength_nm if wavelength_nm > 0 else 0.0
    n_photons = math.ceil(bandgap_ev / photon_ev) if photon_ev > 0 else 0

    if n_photons <= 0:
        return MPAResult(
            photon_energy_ev=photon_ev, bandgap_ev=bandgap_ev,
            photon_order=0, mpa_rate_cm_2n_1_w_n=0.0,
            mpa_absorption_depth_m=float("inf"),
            energy_deposited_fraction=0.0, is_dominant=False,
        )

    sigma_n = estimate_mpa_coefficient(n_photons, bandgap_ev, refractive_index)
    alpha_mpa_cm = sigma_n * irradiance_w_cm2 ** (n_photons - 1)
    alpha_mpa_m = alpha_mpa_cm * 100
    l_mpa = 1.0 / alpha_mpa_m if alpha_mpa_m > 0 else float("inf")

    thickness_cm = thickness_m * 100
    alpha_total_cm = alpha_linear_cm + alpha_mpa_cm
    if alpha_total_cm > 0 and thickness_cm > 0:
        absorbed = 1.0 - math.exp(-alpha_total_cm * thickness_cm)
    else:
        absorbed = 0.0

    is_dominant = alpha_mpa_cm > alpha_linear_cm

    return MPAResult(
        photon_energy_ev=photon_ev, bandgap_ev=bandgap_ev,
        photon_order=n_photons, mpa_rate_cm_2n_1_w_n=sigma_n,
        mpa_absorption_depth_m=l_mpa,
        energy_deposited_fraction=absorbed, is_dominant=is_dominant,
    )


def self_focusing_analysis(
    wavelength_nm: float,
    n2_cm2_w: float,
    refractive_index: float,
    peak_power_w: float,
    beam_radius_m: float,
    irradiance_w_cm2: float,
    thickness_m: float,
) -> SelfFocusingResult:
    """Analyze Kerr self-focusing."""
    lam_m = wavelength_nm * 1e-9
    n0 = refractive_index
    n2_m2_w = n2_cm2_w * 1e-4

    if n2_m2_w > 0 and n0 > 0:
        p_cr = 3.77 * lam_m**2 / (8 * math.pi * n0 * n2_m2_w)
    else:
        p_cr = float("inf")

    p_ratio = peak_power_w / p_cr if p_cr > 0 and p_cr < float("inf") else 0.0
    sf_occurs = p_ratio > 1.0

    z_r = math.pi * beam_radius_m**2 * n0 / lam_m if lam_m > 0 else 0.0
    if sf_occurs and p_ratio > 1.01:
        z_collapse = 0.367 * z_r / math.sqrt(math.sqrt(p_ratio) - 0.852)
    else:
        z_collapse = float("inf")

    delta_n = n2_cm2_w * irradiance_w_cm2

    thickness_cm = thickness_m * 100
    lam_cm = wavelength_nm * 1e-7
    b_integral = (2 * math.pi / lam_cm) * delta_n * thickness_cm if lam_cm > 0 else 0.0

    return SelfFocusingResult(
        n2_cm2_w=n2_cm2_w, critical_power_w=p_cr,
        input_power_w=peak_power_w, p_over_pcr=p_ratio,
        self_focusing_occurs=sf_occurs, collapse_distance_m=z_collapse,
        delta_n_peak=delta_n, b_integral_rad=b_integral,
    )


def nonlinear_analysis(
    wavelength_nm: float,
    bandgap_ev: float,
    n2_cm2_w: float,
    refractive_index: float,
    alpha_linear_cm: float,
    peak_power_w: float,
    irradiance_w_cm2: float,
    pulse_width_s: float,
    beam_radius_m: float,
    thickness_m: float,
) -> NonlinearResult:
    """Full nonlinear analysis combining MPA and self-focusing.

    Uses JIT-accelerated depth-resolved propagation kernel.
    """
    mpa = mpa_analysis(
        wavelength_nm=wavelength_nm, bandgap_ev=bandgap_ev,
        irradiance_w_cm2=irradiance_w_cm2, pulse_width_s=pulse_width_s,
        alpha_linear_cm=alpha_linear_cm, refractive_index=refractive_index,
        thickness_m=thickness_m,
    )

    sf = self_focusing_analysis(
        wavelength_nm=wavelength_nm, n2_cm2_w=n2_cm2_w,
        refractive_index=refractive_index, peak_power_w=peak_power_w,
        beam_radius_m=beam_radius_m, irradiance_w_cm2=irradiance_w_cm2,
        thickness_m=thickness_m,
    )

    n_z = 200
    z = np.linspace(0, thickness_m, n_z)
    dz_cm = (thickness_m * 100) / n_z

    photon_ev = 1240.0 / wavelength_nm if wavelength_nm > 0 else 1.0
    photon_j = photon_ev * 1.602e-19

    # JIT-accelerated propagation kernel
    irr, dn, carriers = _nonlinear_propagation(
        n_z, dz_cm, alpha_linear_cm,
        mpa.mpa_rate_cm_2n_1_w_n, mpa.photon_order,
        n2_cm2_w, irradiance_w_cm2, pulse_width_s, photon_j,
    )

    return NonlinearResult(
        mpa=mpa, self_focusing=sf,
        z_m=z, irradiance_z_w_cm2=irr,
        delta_n_z=dn, carrier_density_z_cm3=carriers,
    )
