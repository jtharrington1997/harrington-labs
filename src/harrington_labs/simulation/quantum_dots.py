"""Quantum Dots Lab simulator.

Models size-dependent electronic structure, photoluminescence,
absorption spectra, exciton dynamics, and quantum yield for
colloidal quantum dots. No Streamlit imports.
"""
from __future__ import annotations

import math
import numpy as np

from harrington_common.compute import jit, parallel_map

from harrington_labs.domain import (
    QuantumDotParams, QDMaterial, SimulationResult,
    C_M_S, H_J_S, K_B,
)


# ── Material bulk parameters ─────────────────────────────────────

# (Eg_bulk_eV, me_eff, mh_eff, epsilon_r, a_exciton_nm)
_QD_BULK = {
    QDMaterial.CDSE:       (1.74, 0.12, 0.45, 9.4, 5.6),
    QDMaterial.CDSE_ZNS:   (1.74, 0.12, 0.45, 9.4, 5.6),  # core-shell, use CdSe core
    QDMaterial.PBSE:       (0.28, 0.047, 0.040, 227.0, 46.0),
    QDMaterial.PBS:        (0.41, 0.085, 0.085, 169.0, 20.0),
    QDMaterial.INP:        (1.35, 0.077, 0.60, 12.5, 11.0),
    QDMaterial.PEROVSKITE: (1.55, 0.12, 0.15, 25.0, 7.0),
    QDMaterial.SI:         (1.12, 0.26, 0.36, 11.7, 4.9),
}

_ME = 9.10938e-31  # electron mass kg
_E_CHARGE = 1.602176634e-19


# ── Size-dependent bandgap (Brus equation) ───────────────────────

def brus_bandgap_ev(
    material: QDMaterial,
    diameter_nm: float,
) -> float:
    """Effective bandgap via Brus equation including Coulomb correction."""
    bulk = _QD_BULK.get(material)
    if bulk is None:
        return 2.0  # fallback
    eg_bulk, me_eff, mh_eff, eps_r, _ = bulk

    r = (diameter_nm / 2.0) * 1e-9  # radius in meters
    if r <= 0:
        return eg_bulk

    # Kinetic confinement term
    mu = 1.0 / (1.0/me_eff + 1.0/mh_eff)  # reduced effective mass
    kinetic = (H_J_S**2 * math.pi**2) / (2 * mu * _ME * r**2)

    # Coulomb attraction
    coulomb = 1.8 * _E_CHARGE**2 / (4 * math.pi * 8.854e-12 * eps_r * r)

    eg = eg_bulk + kinetic / _E_CHARGE - coulomb / _E_CHARGE
    return max(eg, eg_bulk)



@jit
def _brus_vectorized(eg_bulk, me_ratio, mh_ratio, eps_r, diameters):
    """Brus equation over diameter array — JIT-accelerated."""
    _H_BAR = 1.0546e-34
    _E_CHARGE = 1.602e-19
    _M_E = 9.109e-31
    _EPS_0 = 8.854e-12
    _PI = 3.141592653589793

    n = len(diameters)
    bandgaps = np.empty(n)
    me = me_ratio * _M_E
    mh = mh_ratio * _M_E
    mu = me * mh / (me + mh) if (me + mh) > 0.0 else _M_E

    for i in range(n):
        r = diameters[i] * 0.5e-9  # nm to m, radius
        if r <= 0.0:
            bandgaps[i] = eg_bulk
            continue
        kinetic = (_H_BAR ** 2 * _PI ** 2) / (2.0 * mu * r ** 2)
        coulomb = 1.8 * _E_CHARGE ** 2 / (4.0 * _PI * _EPS_0 * eps_r * r)
        eg = eg_bulk + kinetic / _E_CHARGE - coulomb / _E_CHARGE
        bandgaps[i] = max(eg, eg_bulk)
    return bandgaps


def bandgap_vs_size(
    material: QDMaterial,
    diameter_range_nm: tuple[float, float] = (1.5, 12.0),
    n_points: int = 100,
) -> dict:
    """Bandgap as function of QD diameter."""
    diameters = np.linspace(diameter_range_nm[0], diameter_range_nm[1], n_points)
    # Vectorized Brus equation (JIT-accelerated inner kernel)
    bulk = _QD_BULK.get(material)
    if bulk is None:
        # Fallback for unknown materials
        return {
            "diameter_nm": diameters,
            "bandgap_ev": np.full(n_points, 2.0),
            "peak_wavelength_nm": np.full(n_points, 620.0),
        }
    eg_bulk, me_ratio, mh_ratio, eps_r, _ = bulk
    bandgaps = _brus_vectorized(
        eg_bulk, me_ratio, mh_ratio, eps_r,
        np.asarray(diameters, dtype=np.float64),
    )
    wavelengths = 1240.0 / bandgaps  # nm
    return {
        "diameter_nm": diameters,
        "bandgap_ev": bandgaps,
        "peak_wavelength_nm": wavelengths,
    }


# ── Emission spectrum ────────────────────────────────────────────

def emission_spectrum(
    params: QuantumDotParams,
    n_points: int = 500,
) -> dict:
    """Model PL emission spectrum as Gaussian with size broadening."""
    eg = brus_bandgap_ev(params.material, params.diameter_nm)
    peak_nm = 1240.0 / eg

    # Inhomogeneous broadening from size distribution
    # dE/dR contribution to FWHM
    fwhm_nm = params.fwhm_emission_nm
    # Add size-distribution broadening
    size_broadening = params.size_distribution_pct / 100.0 * peak_nm * 0.5
    total_fwhm = math.sqrt(fwhm_nm**2 + size_broadening**2)

    sigma = total_fwhm / (2 * math.sqrt(2 * math.log(2)))
    wavelengths = np.linspace(
        peak_nm - 4 * total_fwhm,
        peak_nm + 4 * total_fwhm,
        n_points,
    )
    intensity = np.exp(-(wavelengths - peak_nm)**2 / (2 * sigma**2))
    intensity *= params.quantum_yield

    return {
        "wavelength_nm": wavelengths,
        "intensity": intensity,
        "peak_nm": peak_nm,
        "fwhm_nm": total_fwhm,
        "bandgap_ev": eg,
    }


# ── Absorption spectrum ──────────────────────────────────────────

def absorption_spectrum(
    params: QuantumDotParams,
    wavelength_range_nm: tuple[float, float] = (300, 800),
    n_points: int = 500,
) -> dict:
    """Model absorption spectrum with excitonic peaks and continuum."""
    wavelengths = np.linspace(wavelength_range_nm[0], wavelength_range_nm[1], n_points)
    energies = 1240.0 / wavelengths  # eV

    eg = brus_bandgap_ev(params.material, params.diameter_nm)

    # First excitonic peak
    sigma_exc = 0.05  # eV broadening
    peak_1s = np.exp(-(energies - eg)**2 / (2 * sigma_exc**2))

    # Second excitonic peak (1P)
    e_1p = eg * 1.15
    peak_1p = 0.6 * np.exp(-(energies - e_1p)**2 / (2 * (sigma_exc * 1.2)**2))

    # Continuum absorption above bandgap
    continuum = np.where(energies > eg, ((energies - eg) / eg) ** 0.5, 0.0)
    continuum = np.clip(continuum, 0, 3)

    # Combined
    absorbance = params.absorption_cross_section_cm2 * (peak_1s + peak_1p + 0.3 * continuum)
    absorbance *= params.concentration_nmol_ml * 6.022e14  # scale by number density

    return {
        "wavelength_nm": wavelengths,
        "absorbance": absorbance,
        "bandgap_ev": eg,
    }


# ── Exciton dynamics ─────────────────────────────────────────────

def exciton_decay(
    params: QuantumDotParams,
    time_range_ns: float = 100.0,
    n_points: int = 500,
) -> dict:
    """Model exciton population decay (single exponential + biexciton)."""
    t = np.linspace(0, time_range_ns, n_points)

    # Radiative rate
    k_rad = params.quantum_yield / params.exciton_lifetime_ns  # ns⁻¹
    k_nr = (1 - params.quantum_yield) / params.exciton_lifetime_ns
    k_total = k_rad + k_nr

    # Single exciton decay
    single = np.exp(-k_total * t)

    # Biexciton (Auger) — typically ~4x faster
    k_bx = 4 * k_total
    biexciton = np.exp(-k_bx * t)

    return {
        "time_ns": t,
        "single_exciton": single,
        "biexciton": biexciton,
        "k_rad_ns": k_rad,
        "k_nr_ns": k_nr,
        "k_total_ns": k_total,
    }


# ── Temperature-dependent PL ────────────────────────────────────

def temperature_dependent_pl(
    params: QuantumDotParams,
    temp_range_k: tuple[float, float] = (4, 400),
    n_points: int = 100,
) -> dict:
    """PL intensity and peak shift vs temperature."""
    temps = np.linspace(temp_range_k[0], temp_range_k[1], n_points)

    # Varshni bandgap shift: Eg(T) = Eg(0) - α*T²/(T+β)
    eg0 = brus_bandgap_ev(params.material, params.diameter_nm)
    alpha = 4e-4  # eV/K typical
    beta = 200.0  # K typical

    eg_t = eg0 - alpha * temps**2 / (temps + beta)
    peak_nm = 1240.0 / eg_t

    # Thermal quenching of QY
    e_act = 0.05  # eV activation energy
    qy = params.quantum_yield / (1 + 50 * np.exp(-e_act * _E_CHARGE / (K_B * temps)))

    return {
        "temperature_k": temps,
        "bandgap_ev": eg_t,
        "peak_wavelength_nm": peak_nm,
        "quantum_yield": qy,
    }


# ── Full QD simulation bundle ────────────────────────────────────

def run_quantum_dot_simulation(params: QuantumDotParams) -> SimulationResult:
    """Run complete quantum dots lab simulation."""
    eg = brus_bandgap_ev(params.material, params.diameter_nm)
    size_scan = bandgap_vs_size(params.material)
    emission = emission_spectrum(params)
    absorption = absorption_spectrum(params)
    decay = exciton_decay(params)
    temp_pl = temperature_dependent_pl(params)

    bulk = _QD_BULK.get(params.material)
    a_bohr = bulk[4] if bulk else 5.0

    warnings = []
    if params.diameter_nm < 1.5:
        warnings.append("QD diameter < 1.5 nm — deep confinement regime, Brus model less reliable")
    if params.diameter_nm > 3 * a_bohr:
        warnings.append(f"QD diameter > 3× Bohr radius ({a_bohr:.1f} nm) — weak confinement")
    if params.quantum_yield > 0.95:
        warnings.append("QY > 95% claimed — verify measurement conditions")

    return SimulationResult(
        name="Quantum Dots Lab",
        data={
            "bandgap_ev": eg,
            "peak_emission_nm": emission["peak_nm"],
            "size_scan": size_scan,
            "emission": emission,
            "absorption": absorption,
            "exciton_decay": decay,
            "temp_dependence": temp_pl,
        },
        warnings=warnings,
    )
