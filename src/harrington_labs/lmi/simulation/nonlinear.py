"""Nonlinear optical effects for ultrafast laser interactions.

Implements:
- Multiphoton absorption (MPA) rate and cross-section estimation
- Self-focusing critical power and collapse distance
- Kerr effect (B-integral, delta_n)
- Nonlinear absorption depth profiling
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class MPAResult:
    """Multiphoton absorption analysis."""

    photon_energy_ev: float
    bandgap_ev: float
    photon_order: int  # number of photons needed
    mpa_rate_cm_2n_1_w_n: float  # effective MPA coefficient (estimated)
    mpa_absorption_depth_m: float  # effective nonlinear penetration depth
    energy_deposited_fraction: float  # fraction of pulse absorbed via MPA
    is_dominant: bool  # MPA dominates over linear absorption?


@dataclass
class SelfFocusingResult:
    """Self-focusing analysis."""

    n2_cm2_w: float
    critical_power_w: float
    input_power_w: float
    p_over_pcr: float  # P / P_critical
    self_focusing_occurs: bool
    collapse_distance_m: float  # Marburger formula
    delta_n_peak: float
    b_integral_rad: float  # B-integral over material thickness


@dataclass
class NonlinearResult:
    """Combined nonlinear analysis."""

    mpa: MPAResult
    self_focusing: SelfFocusingResult

    # Depth-resolved nonlinear effects
    z_m: np.ndarray
    irradiance_z_w_cm2: np.ndarray  # includes nonlinear absorption
    delta_n_z: np.ndarray
    carrier_density_z_cm3: np.ndarray  # free carriers from MPA


def estimate_mpa_coefficient(
    photon_order: int,
    bandgap_ev: float,
    refractive_index: float,
) -> float:
    """Estimate MPA coefficient using Keldysh-type scaling.

    This is an order-of-magnitude estimate. For precise values,
    literature data should be used. Units: cm^(2n-1) / W^(n-1)
    where n is the photon order.

    Typical values (literature):
    - Si at 8.5 um, 8-photon: ~1e-95 cm^15/W^7 (extremely weak)
    - Ge at 8.5 um, 5-photon: ~1e-55 cm^9/W^4
    """
    # Scaling law: sigma_n ~ (alpha_0 / (n * h_nu))^n * (tau_phonon)^(n-1)
    # Simplified Keldysh estimate
    # For semiconductors in the multiphoton regime:
    n = photon_order

    # Base scale from tunneling limit
    # This is approximate -- real values need experimental calibration
    base = 1e-18  # cm^2 single-photon cross-section scale
    reduction_per_photon = 1e-10  # each additional photon reduces by ~10 orders

    sigma_n = base * reduction_per_photon ** (n - 1)

    # Refractive index correction (higher n -> stronger field confinement)
    sigma_n *= refractive_index ** n

    return sigma_n


def mpa_analysis(
    wavelength_nm: float,
    bandgap_ev: float,
    irradiance_w_cm2: float,
    pulse_width_s: float,
    alpha_linear_cm: float,
    refractive_index: float,
    thickness_m: float,
) -> MPAResult:
    """Analyze multiphoton absorption for a given interaction.

    Parameters
    ----------
    wavelength_nm : float
        Laser wavelength.
    bandgap_ev : float
        Material bandgap.
    irradiance_w_cm2 : float
        Peak irradiance at focus.
    pulse_width_s : float
        Pulse duration.
    alpha_linear_cm : float
        Linear absorption coefficient.
    refractive_index : float
        Material refractive index at this wavelength.
    thickness_m : float
        Material thickness.
    """
    photon_ev = 1240.0 / wavelength_nm if wavelength_nm > 0 else 0.0
    n_photons = math.ceil(bandgap_ev / photon_ev) if photon_ev > 0 else 0

    if n_photons <= 0:
        return MPAResult(
            photon_energy_ev=photon_ev,
            bandgap_ev=bandgap_ev,
            photon_order=0,
            mpa_rate_cm_2n_1_w_n=0.0,
            mpa_absorption_depth_m=float("inf"),
            energy_deposited_fraction=0.0,
            is_dominant=False,
        )

    # Estimate MPA coefficient
    sigma_n = estimate_mpa_coefficient(n_photons, bandgap_ev, refractive_index)

    # Effective nonlinear absorption coefficient: alpha_MPA = sigma_n * I^(n-1)
    alpha_mpa_cm = sigma_n * irradiance_w_cm2 ** (n_photons - 1)

    # Nonlinear absorption depth
    alpha_mpa_m = alpha_mpa_cm * 100
    l_mpa = 1.0 / alpha_mpa_m if alpha_mpa_m > 0 else float("inf")

    # Energy deposited fraction (Beer-Lambert for effective alpha)
    thickness_cm = thickness_m * 100
    alpha_total_cm = alpha_linear_cm + alpha_mpa_cm
    if alpha_total_cm > 0 and thickness_cm > 0:
        absorbed = 1.0 - math.exp(-alpha_total_cm * thickness_cm)
    else:
        absorbed = 0.0

    # MPA dominant if nonlinear absorption > linear
    is_dominant = alpha_mpa_cm > alpha_linear_cm

    return MPAResult(
        photon_energy_ev=photon_ev,
        bandgap_ev=bandgap_ev,
        photon_order=n_photons,
        mpa_rate_cm_2n_1_w_n=sigma_n,
        mpa_absorption_depth_m=l_mpa,
        energy_deposited_fraction=absorbed,
        is_dominant=is_dominant,
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
    """Analyze Kerr self-focusing.

    Parameters
    ----------
    wavelength_nm : float
        Laser wavelength.
    n2_cm2_w : float
        Nonlinear refractive index (cm^2/W).
    refractive_index : float
        Linear refractive index.
    peak_power_w : float
        Peak power of the pulse.
    beam_radius_m : float
        1/e^2 beam radius.
    irradiance_w_cm2 : float
        Peak irradiance.
    thickness_m : float
        Propagation distance in material.
    """
    lam_m = wavelength_nm * 1e-9
    n0 = refractive_index
    n2_m2_w = n2_cm2_w * 1e-4  # convert to m^2/W

    # Critical power for self-focusing (Gaussian beam)
    # P_cr = 3.77 * lambda^2 / (8*pi*n0*n2)
    if n2_m2_w > 0 and n0 > 0:
        p_cr = 3.77 * lam_m**2 / (8 * math.pi * n0 * n2_m2_w)
    else:
        p_cr = float("inf")

    p_ratio = peak_power_w / p_cr if p_cr > 0 and p_cr < float("inf") else 0.0
    sf_occurs = p_ratio > 1.0

    # Marburger collapse distance
    # z_sf = 0.367 * z_R / (sqrt((P/Pcr)^0.5 - 0.852)^2 - 0.0219)
    # Simplified: z_sf ~ z_R / sqrt(P/Pcr - 1) for P >> Pcr
    z_r = math.pi * beam_radius_m**2 * n0 / lam_m if lam_m > 0 else 0.0
    if sf_occurs and p_ratio > 1.01:
        z_collapse = 0.367 * z_r / math.sqrt(math.sqrt(p_ratio) - 0.852)
    else:
        z_collapse = float("inf")

    # Peak delta_n
    delta_n = n2_cm2_w * irradiance_w_cm2

    # B-integral over thickness
    thickness_cm = thickness_m * 100
    lam_cm = wavelength_nm * 1e-7
    b_integral = (2 * math.pi / lam_cm) * delta_n * thickness_cm if lam_cm > 0 else 0.0

    return SelfFocusingResult(
        n2_cm2_w=n2_cm2_w,
        critical_power_w=p_cr,
        input_power_w=peak_power_w,
        p_over_pcr=p_ratio,
        self_focusing_occurs=sf_occurs,
        collapse_distance_m=z_collapse,
        delta_n_peak=delta_n,
        b_integral_rad=b_integral,
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

    Returns depth-resolved irradiance including nonlinear absorption
    and free carrier generation estimates.
    """
    mpa = mpa_analysis(
        wavelength_nm=wavelength_nm,
        bandgap_ev=bandgap_ev,
        irradiance_w_cm2=irradiance_w_cm2,
        pulse_width_s=pulse_width_s,
        alpha_linear_cm=alpha_linear_cm,
        refractive_index=refractive_index,
        thickness_m=thickness_m,
    )

    sf = self_focusing_analysis(
        wavelength_nm=wavelength_nm,
        n2_cm2_w=n2_cm2_w,
        refractive_index=refractive_index,
        peak_power_w=peak_power_w,
        beam_radius_m=beam_radius_m,
        irradiance_w_cm2=irradiance_w_cm2,
        thickness_m=thickness_m,
    )

    # Depth-resolved propagation with nonlinear effects
    n_z = 200
    z = np.linspace(0, thickness_m, n_z)
    dz_cm = (thickness_m * 100) / n_z

    irr = np.zeros(n_z)
    dn = np.zeros(n_z)
    carriers = np.zeros(n_z)

    irr[0] = irradiance_w_cm2

    photon_ev = 1240.0 / wavelength_nm if wavelength_nm > 0 else 1.0
    photon_j = photon_ev * 1.602e-19

    for i in range(1, n_z):
        I = irr[i - 1]

        # Linear absorption
        dI_linear = -alpha_linear_cm * I * dz_cm

        # MPA absorption
        if mpa.photon_order > 1 and mpa.mpa_rate_cm_2n_1_w_n > 0:
            dI_mpa = -mpa.mpa_rate_cm_2n_1_w_n * I ** mpa.photon_order * dz_cm
        else:
            dI_mpa = 0.0

        irr[i] = max(I + dI_linear + dI_mpa, 0.0)
        dn[i] = n2_cm2_w * irr[i]

        # Free carrier density from MPA
        # dn_fc/dz ~ alpha_MPA * I / (n_photons * h_nu)
        if mpa.photon_order > 0 and photon_j > 0:
            abs_rate = abs(dI_mpa) * 1e4  # W/m^3 equivalent
            carriers[i] = carriers[i - 1] + abs_rate * pulse_width_s / (
                mpa.photon_order * photon_j
            ) * (dz_cm / 100)

    return NonlinearResult(
        mpa=mpa,
        self_focusing=sf,
        z_m=z,
        irradiance_z_w_cm2=irr,
        delta_n_z=dn,
        carrier_density_z_cm3=carriers,
    )
