"""Advanced Spectroscopy Lab simulator.

Physics engines for Raman (spontaneous, stimulated, DUVRR),
Brillouin (spontaneous, SBS), LIBS, FTIR, and hyperspectral
imaging. All functions return plain dicts/arrays. No Streamlit imports.
"""
from __future__ import annotations

import math
import numpy as np

from harrington_labs.domain import SimulationResult, C_M_S, H_J_S, K_B
from harrington_labs.domain.spectroscopy import (
    RamanParams, BrillouinParams, DUVRRParams,
    LIBSParams, FTIRParams, HyperspectralParams,
)


# ── Physical constants ──────────────────────────────────────────────────

_EV_TO_CM_INV = 8065.544  # 1 eV in cm⁻¹


def _wavenumber_to_wavelength_nm(excitation_nm: float, shift_cm_inv: float) -> float:
    """Convert Raman shift (cm⁻¹) to scattered wavelength (nm)."""
    exc_cm_inv = 1e7 / excitation_nm
    scattered_cm_inv = exc_cm_inv - shift_cm_inv
    return 1e7 / scattered_cm_inv if scattered_cm_inv > 0 else float("inf")


def _lorentzian(x: np.ndarray, center: float, fwhm: float, amplitude: float) -> np.ndarray:
    gamma = fwhm / 2.0
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)


def _gaussian(x: np.ndarray, center: float, fwhm: float, amplitude: float) -> np.ndarray:
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return amplitude * np.exp(-((x - center)**2) / (2.0 * sigma**2))


def _voigt_approx(x: np.ndarray, center: float, fwhm_g: float, fwhm_l: float, amplitude: float) -> np.ndarray:
    """Pseudo-Voigt approximation."""
    fwhm = (fwhm_g**5 + 2.69 * fwhm_g**4 * fwhm_l + 2.43 * fwhm_g**3 * fwhm_l**2 +
            4.47 * fwhm_g**2 * fwhm_l**3 + 0.07 * fwhm_g * fwhm_l**4 + fwhm_l**5) ** 0.2
    eta = 1.36603 * (fwhm_l / fwhm) - 0.47719 * (fwhm_l / fwhm)**2 + 0.11116 * (fwhm_l / fwhm)**3
    eta = max(0.0, min(1.0, eta))
    return amplitude * (eta * _lorentzian(x, center, fwhm, 1.0) + (1 - eta) * _gaussian(x, center, fwhm, 1.0))


def _bose_einstein(shift_cm_inv: float, temperature_k: float) -> float:
    """Bose-Einstein population factor for Stokes scattering."""
    if temperature_k <= 0 or shift_cm_inv <= 0:
        return 1.0
    energy_j = shift_cm_inv * 100 * H_J_S * C_M_S  # cm⁻¹ to J
    x = energy_j / (K_B * temperature_k)
    if x > 500:
        return 1.0
    return 1.0 / (1.0 - math.exp(-x))


# ── Spontaneous Raman ──────────────────────────────────────────────────


def spontaneous_raman(params: RamanParams, n_points: int = 1024) -> dict:
    """Simulate spontaneous Raman spectrum.

    Models: Lorentzian peak shapes, thermal (Bose-Einstein) population,
    ν⁴ scattering efficiency, shot noise, fluorescence background.
    """
    shift_min, shift_max = -200.0, 4000.0
    shifts = np.linspace(shift_min, shift_max, n_points)

    exc_wl = params.excitation_wavelength_nm
    exc_nu = 1e7 / exc_wl  # cm⁻¹

    # Build spectrum from defined modes
    spectrum = np.zeros(n_points)
    for pos, width, rel_int in zip(params.raman_shifts_cm_inv, params.raman_widths_cm_inv, params.raman_intensities):
        # Stokes line
        thermal = _bose_einstein(pos, params.temperature_k)
        scattered_nu = exc_nu - pos
        nu4_factor = (scattered_nu / exc_nu)**4 if exc_nu > 0 else 1.0
        amp = rel_int * thermal * nu4_factor * params.laser_power_mw * params.integration_time_s
        spectrum += _lorentzian(shifts, pos, width, amp)

        # Anti-Stokes (weaker)
        if params.temperature_k > 0:
            as_thermal = math.exp(-pos * 100 * H_J_S * C_M_S / (K_B * params.temperature_k)) if pos > 0 else 0
            as_scattered_nu = exc_nu + pos
            as_nu4 = (as_scattered_nu / exc_nu)**4 if exc_nu > 0 else 1.0
            spectrum += _lorentzian(shifts, -pos, width, rel_int * as_thermal * as_nu4 * params.laser_power_mw * params.integration_time_s)

    # Fluorescence background (broad hump)
    bg_center = 1500.0
    bg_width = 2000.0
    bg_amp = 0.02 * params.laser_power_mw * params.integration_time_s
    background = _gaussian(shifts, bg_center, bg_width, bg_amp)

    # Shot noise
    total = spectrum + background
    noise_level = np.sqrt(np.maximum(total, 0)) * 0.1
    noise = np.random.default_rng(42).normal(0, 1, n_points) * noise_level
    noisy_spectrum = total + noise

    # Stokes / anti-Stokes ratio for temperature determination
    stokes_anti_stokes_ratio = None
    if len(params.raman_shifts_cm_inv) > 0:
        nu_s = params.raman_shifts_cm_inv[0]
        if params.temperature_k > 0 and nu_s > 0:
            energy_ratio = nu_s * 100 * H_J_S * C_M_S / (K_B * params.temperature_k)
            sas = ((exc_nu - nu_s) / (exc_nu + nu_s))**4 * math.exp(energy_ratio)
            stokes_anti_stokes_ratio = sas

    return {
        "shift_cm_inv": shifts,
        "spectrum": noisy_spectrum,
        "clean_spectrum": total,
        "background": background,
        "stokes_anti_stokes_ratio": stokes_anti_stokes_ratio,
        "peak_positions_cm_inv": list(params.raman_shifts_cm_inv),
        "excitation_nm": exc_wl,
    }


# ── Stimulated Raman (SRS) ────────────────────────────────────────────


def stimulated_raman(params: RamanParams, fiber_length_m: float = 1.0, n_points: int = 512) -> dict:
    """Simulate stimulated Raman scattering gain spectrum.

    Models: Lorentzian gain profile, pump depletion, Stokes amplification.
    """
    shifts = np.linspace(0, 2000, n_points)

    # Raman gain coefficient (typical silica fiber ~1e-13 m/W at 1 µm)
    g_r_peak = 1e-11  # m/W, scaled for illustrative purposes

    # Gain profile from modes
    gain_profile = np.zeros(n_points)
    for pos, width, rel_int in zip(params.raman_shifts_cm_inv, params.raman_widths_cm_inv, params.raman_intensities):
        gain_profile += _lorentzian(shifts, pos, width, rel_int * g_r_peak)

    # Effective area
    beam_area_m2 = math.pi * (30e-6)**2  # ~30 µm beam radius
    pump_intensity = params.pump_power_mw * 1e-3 / beam_area_m2

    # Stokes gain: G = exp(g_R * I_p * L_eff)
    stokes_gain = np.exp(gain_profile * pump_intensity * fiber_length_m)
    stokes_power = params.stokes_seed_power_mw * stokes_gain

    # Pump depletion (simplified)
    total_stokes_gain = float(np.max(stokes_gain))
    pump_remaining_frac = max(0.0, 1.0 - (total_stokes_gain - 1) * params.stokes_seed_power_mw / params.pump_power_mw)

    return {
        "shift_cm_inv": shifts,
        "gain_coefficient": gain_profile,
        "stokes_gain": stokes_gain,
        "stokes_power_mw": stokes_power,
        "pump_remaining_fraction": pump_remaining_frac,
        "peak_gain_m_per_w": float(np.max(gain_profile)),
    }


# ── Spontaneous Brillouin ──────────────────────────────────────────────


def spontaneous_brillouin(params: BrillouinParams, n_points: int = 512) -> dict:
    """Simulate spontaneous Brillouin spectrum.

    Models: acoustic phonon frequency from Bragg condition, Lorentzian linewidth
    from acoustic attenuation, Stokes/anti-Stokes doublet.
    """
    exc_wl_m = params.excitation_wavelength_nm * 1e-9
    theta = math.radians(params.scattering_angle_deg)

    # Brillouin frequency shift: ν_B = 2n·v_s·sin(θ/2) / λ
    nu_b_hz = 2 * params.refractive_index * params.sound_velocity_m_s * math.sin(theta / 2) / exc_wl_m
    nu_b_ghz = nu_b_hz * 1e-9

    # Linewidth from acoustic attenuation
    # Γ_B ≈ α_acoustic · v_s / π (simplified)
    alpha_neper = params.acoustic_attenuation_db_cm_ghz2 * nu_b_ghz**2 * 100 / 8.686
    linewidth_ghz = alpha_neper * params.sound_velocity_m_s / (math.pi * 1e9)
    linewidth_ghz = max(linewidth_ghz, 0.05)  # minimum resolvable

    # Build spectrum (GHz axis)
    freq_ghz = np.linspace(-nu_b_ghz * 2, nu_b_ghz * 2, n_points)

    # Stokes and anti-Stokes peaks
    stokes = _lorentzian(freq_ghz, -nu_b_ghz, linewidth_ghz, 1.0)
    anti_stokes = _lorentzian(freq_ghz, nu_b_ghz, linewidth_ghz, 1.0)

    # Rayleigh (elastic) peak
    rayleigh = _lorentzian(freq_ghz, 0.0, linewidth_ghz * 0.3, 5.0)

    total = stokes + anti_stokes + rayleigh

    # Brillouin shift in cm⁻¹
    nu_b_cm_inv = nu_b_hz / (C_M_S * 100)

    # Elastic modulus from Brillouin data
    # M = ρ · v² (longitudinal modulus)
    longitudinal_modulus_gpa = params.density_kg_m3 * params.sound_velocity_m_s**2 * 1e-9

    return {
        "frequency_ghz": freq_ghz,
        "spectrum": total,
        "stokes": stokes,
        "anti_stokes": anti_stokes,
        "rayleigh": rayleigh,
        "brillouin_shift_ghz": nu_b_ghz,
        "brillouin_shift_cm_inv": nu_b_cm_inv,
        "linewidth_ghz": linewidth_ghz,
        "longitudinal_modulus_gpa": longitudinal_modulus_gpa,
        "sound_velocity_m_s": params.sound_velocity_m_s,
    }


# ── Stimulated Brillouin (SBS) ─────────────────────────────────────────


def stimulated_brillouin(params: BrillouinParams, n_points: int = 256) -> dict:
    """Simulate SBS threshold and gain in optical fiber.

    Models: Lorentzian gain bandwidth, SBS threshold power,
    backward Stokes amplification.
    """
    spont = spontaneous_brillouin(params, n_points)

    # SBS gain coefficient (typical silica ~5e-11 m/W)
    # g_B = 2π n⁷ p₁₂² / (c λ² ρ v_a Γ_B)
    lam = params.excitation_wavelength_nm * 1e-9
    gamma_b = spont["linewidth_ghz"] * 1e9 * 2 * math.pi
    n = params.refractive_index
    p12 = params.elasto_optic_coefficient

    g_b = (2 * math.pi * n**7 * p12**2) / (C_M_S * lam**2 * params.density_kg_m3 * params.sound_velocity_m_s * gamma_b)
    g_b = max(g_b, 1e-12)

    # Effective area
    core_r = params.fiber_core_diameter_um * 1e-6 / 2
    a_eff = math.pi * core_r**2

    # SBS threshold: P_th ≈ 21 · A_eff / (g_B · L_eff)
    l_eff = params.interaction_length_m
    p_threshold_w = 21 * a_eff / (g_b * l_eff) if g_b * l_eff > 0 else float("inf")

    # Gain vs input power
    powers_mw = np.linspace(0, params.laser_power_mw * 2, n_points)
    gain = g_b * (powers_mw * 1e-3) * l_eff / a_eff
    reflectivity = np.where(gain < 25, np.exp(gain) * 1e-9, 1.0)  # seed from noise
    reflectivity = np.minimum(reflectivity, 1.0)

    return {
        **spont,
        "gain_coefficient_m_per_w": g_b,
        "threshold_power_w": p_threshold_w,
        "threshold_power_mw": p_threshold_w * 1e3,
        "effective_area_um2": a_eff * 1e12,
        "input_power_mw": powers_mw,
        "sbs_reflectivity": reflectivity,
    }


# ── DUVRR ──────────────────────────────────────────────────────────────


def duvrr_spectrum(params: DUVRRParams, n_points: int = 1024) -> dict:
    """Simulate Deep-UV Resonance Raman spectrum.

    Models: resonance enhancement via Albrecht A-term, amide band
    vibrational modes, UV absorption pre-resonance profile.
    """
    shifts = np.linspace(800, 1800, n_points)
    exc_wl = params.excitation_wavelength_nm

    # Resonance enhancement: A-term ∝ 1/(ν_e² - ν_0²)²
    nu_e = 1e7 / params.electronic_transition_nm  # electronic transition cm⁻¹
    nu_0 = 1e7 / exc_wl  # excitation cm⁻¹
    denom = (nu_e**2 - nu_0**2)**2
    enhancement = params.resonance_enhancement_factor * (nu_e**4 / denom) if denom > 0 else params.resonance_enhancement_factor

    # Amide bands (protein secondary structure markers)
    modes = [
        (params.amide_I_cm_inv, 25.0, 1.0, "Amide I"),
        (params.amide_II_cm_inv, 30.0, 0.7, "Amide II"),
        (params.amide_III_cm_inv, 35.0, 0.5, "Amide III"),
        (1005.0, 10.0, 0.3, "Phe ring"),
        (1340.0, 15.0, 0.25, "Cα-H def"),
        (1620.0, 20.0, 0.4, "Tyr/Trp"),
    ]

    spectrum = np.zeros(n_points)
    mode_labels = []
    for pos, width, rel_int, label in modes:
        amp = rel_int * enhancement * params.laser_power_uw * 1e-3 * params.integration_time_s * params.concentration_mg_ml
        spectrum += _voigt_approx(shifts, pos, width * 0.6, width * 0.4, amp)
        mode_labels.append({"position": pos, "label": label, "intensity": rel_int})

    # Add noise
    noise = np.random.default_rng(42).normal(0, np.sqrt(np.maximum(spectrum, 0)) * 0.05, n_points)
    noisy = spectrum + noise

    # Excitation profile (enhancement vs wavelength)
    exc_scan_nm = np.linspace(200, 300, 100)
    exc_profile = np.array([
        params.resonance_enhancement_factor * (nu_e**4 / ((nu_e**2 - (1e7 / wl)**2)**2 + (500)**2))
        for wl in exc_scan_nm
    ])

    return {
        "shift_cm_inv": shifts,
        "spectrum": noisy,
        "clean_spectrum": spectrum,
        "mode_assignments": mode_labels,
        "resonance_enhancement": enhancement,
        "excitation_profile_nm": exc_scan_nm,
        "excitation_profile": exc_profile / exc_profile.max(),
    }


# ── LIBS ───────────────────────────────────────────────────────────────

# NIST line database (subset) — wavelength_nm, element, relative_intensity, upper_eV
_LIBS_LINES = {
    "Fe": [(248.3, 0.8, 5.0), (259.9, 0.9, 4.8), (274.9, 0.6, 4.5), (302.1, 0.7, 4.1),
           (344.1, 0.5, 3.6), (373.7, 1.0, 3.3), (382.0, 0.7, 3.2), (404.6, 0.6, 3.1)],
    "Cr": [(267.7, 0.7, 4.6), (283.6, 0.9, 4.4), (357.9, 1.0, 3.5), (425.4, 0.8, 2.9)],
    "Ni": [(231.6, 0.6, 5.4), (300.2, 0.8, 4.1), (341.5, 1.0, 3.6), (352.5, 0.9, 3.5)],
    "Mn": [(257.6, 1.0, 4.8), (279.5, 0.7, 4.4), (403.1, 0.9, 3.1)],
    "Si": [(251.6, 1.0, 4.9), (288.2, 0.8, 4.3)],
    "C":  [(247.9, 1.0, 5.0)],
    "Ca": [(393.4, 1.0, 3.2), (396.8, 0.9, 3.1), (422.7, 0.7, 2.9)],
    "Na": [(589.0, 1.0, 2.1), (589.6, 0.5, 2.1)],
    "Al": [(308.2, 0.9, 4.0), (309.3, 1.0, 4.0), (394.4, 0.7, 3.1), (396.2, 0.8, 3.1)],
    "Mg": [(279.6, 0.8, 4.4), (280.3, 1.0, 4.4), (285.2, 0.9, 4.3)],
    "Cu": [(324.8, 1.0, 3.8), (327.4, 0.5, 3.8)],
    "Ti": [(334.9, 0.9, 3.7), (336.1, 1.0, 3.7), (337.3, 0.8, 3.7)],
    "O":  [(777.2, 1.0, 1.6), (777.4, 0.8, 1.6), (844.6, 0.6, 1.5)],
    "N":  [(742.4, 0.7, 1.7), (744.2, 1.0, 1.7), (746.8, 0.8, 1.7)],
    "H":  [(656.3, 1.0, 1.9)],  # H-alpha
}


def libs_spectrum(params: LIBSParams, n_points: int = 2048) -> dict:
    """Simulate LIBS emission spectrum.

    Models: plasma emission lines (NIST database subset), Boltzmann population,
    Stark broadening, continuum bremsstrahlung, self-absorption.
    """
    wl = np.linspace(200, 900, n_points)

    # Plasma temperature estimate from irradiance
    spot_area_cm2 = math.pi * (params.spot_diameter_um * 1e-4 / 2)**2
    irradiance = params.pulse_energy_mj * 1e-3 / (params.pulse_width_ns * 1e-9 * spot_area_cm2)
    # Empirical: T_plasma ~ 8000-15000 K for typical LIBS
    t_plasma_k = min(15000, max(8000, 8000 + 2000 * math.log10(max(irradiance, 1e6) / 1e9)))

    # Build spectrum
    spectrum = np.zeros(n_points)
    line_data = []

    for element, weight_frac in params.composition.items():
        if element not in _LIBS_LINES:
            continue
        for line_wl, rel_int, upper_ev in _LIBS_LINES[element]:
            # Boltzmann factor
            boltz = math.exp(-upper_ev * 1.602e-19 / (K_B * t_plasma_k))
            # Stark broadening (approximate: ~0.01–0.1 nm per 10¹⁷ cm⁻³ electron density)
            stark_width_nm = 0.05 + 0.02 * (t_plasma_k / 10000)
            # Line intensity
            intensity = rel_int * weight_frac * boltz * params.pulse_energy_mj * 10
            spectrum += _voigt_approx(wl, line_wl, stark_width_nm * 0.5, stark_width_nm * 0.5, intensity)
            line_data.append({
                "element": element,
                "wavelength_nm": line_wl,
                "intensity": intensity,
                "upper_ev": upper_ev,
            })

    # Continuum (bremsstrahlung)
    continuum = 0.01 * params.pulse_energy_mj * np.exp(-(wl - 300)**2 / (2 * 200**2))
    total = spectrum + continuum

    # Shot noise
    noise = np.random.default_rng(42).normal(0, np.sqrt(np.maximum(total, 0)) * 0.03, n_points)

    return {
        "wavelength_nm": wl,
        "spectrum": total + noise,
        "clean_spectrum": total,
        "continuum": continuum,
        "line_data": sorted(line_data, key=lambda x: -x["intensity"]),
        "plasma_temperature_k": t_plasma_k,
        "irradiance_gw_cm2": irradiance * 1e-9,
    }


# ── FTIR ───────────────────────────────────────────────────────────────


def ftir_spectrum(params: FTIRParams, n_points: int = 2048) -> dict:
    """Simulate FTIR absorbance/transmittance spectrum.

    Models: Beer-Lambert absorption, Lorentzian band shapes,
    baseline tilt, noise reduction from co-added scans.
    """
    wn = np.linspace(params.wavenumber_min_cm_inv, params.wavenumber_max_cm_inv, n_points)

    absorbance = np.zeros(n_points)
    for pos, width, peak_abs in params.ir_modes:
        # Scale with thickness (Beer-Lambert)
        abs_scaled = peak_abs * (params.thickness_um / 10.0)
        absorbance += _lorentzian(wn, pos, width, abs_scaled)

    # Baseline tilt
    baseline = 0.02 * (wn - wn.min()) / (wn.max() - wn.min())
    absorbance += baseline

    # Noise (decreases with sqrt(n_scans))
    noise_level = 0.005 / math.sqrt(params.n_scans)
    noise = np.random.default_rng(42).normal(0, noise_level, n_points)
    absorbance_noisy = absorbance + noise

    transmittance = 10**(-absorbance_noisy)
    transmittance_clean = 10**(-absorbance)

    return {
        "wavenumber_cm_inv": wn,
        "absorbance": absorbance_noisy,
        "absorbance_clean": absorbance,
        "transmittance": transmittance,
        "transmittance_clean": transmittance_clean,
        "baseline": baseline,
        "snr": 1.0 / noise_level,
    }


# ── Hyperspectral Imaging ──────────────────────────────────────────────


def hyperspectral_image(params: HyperspectralParams) -> dict:
    """Simulate a hyperspectral Raman/FTIR image datacube.

    Models: spatial distribution of N chemical components, spectral
    mixing, Poisson noise, diffraction-limited spatial resolution.
    """
    n_px = params.image_size_px
    wn_min, wn_max = params.spectral_range_cm_inv
    n_spec = 256
    wn = np.linspace(wn_min, wn_max, n_spec)

    rng = np.random.default_rng(42)

    # Generate N component spectra
    component_spectra = []
    component_maps = []
    component_names = []

    for i in range(params.n_components):
        # Random peaks for each component
        n_peaks = rng.integers(2, 6)
        spec = np.zeros(n_spec)
        for _ in range(n_peaks):
            pos = rng.uniform(wn_min + 100, wn_max - 100)
            width = rng.uniform(10, 40)
            amp = rng.uniform(0.3, 1.0)
            spec += _lorentzian(wn, pos, width, amp)
        component_spectra.append(spec)
        component_names.append(f"Component {i + 1}")

        # Spatial distribution (blobs)
        spatial = np.zeros((n_px, n_px))
        n_blobs = rng.integers(1, 4)
        for _ in range(n_blobs):
            cx, cy = rng.uniform(0.2, 0.8, 2) * n_px
            sigma = rng.uniform(3, n_px / 4)
            yy, xx = np.mgrid[0:n_px, 0:n_px]
            spatial += np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        spatial /= spatial.max() + 1e-10
        component_maps.append(spatial)

    # Build datacube
    datacube = np.zeros((n_px, n_px, n_spec))
    for spec, spatial_map in zip(component_spectra, component_maps):
        datacube += spatial_map[:, :, np.newaxis] * spec[np.newaxis, np.newaxis, :]

    # Add noise
    snr_linear = 10**(params.snr_db / 20)
    noise = rng.normal(0, datacube.max() / snr_linear, datacube.shape)
    datacube_noisy = datacube + noise

    # Integrated intensity image
    intensity_image = datacube_noisy.sum(axis=2)

    return {
        "wavenumber_cm_inv": wn,
        "datacube": datacube_noisy,
        "datacube_clean": datacube,
        "intensity_image": intensity_image,
        "component_spectra": component_spectra,
        "component_maps": component_maps,
        "component_names": component_names,
        "image_size_px": n_px,
        "pixel_size_um": params.pixel_size_um,
    }


# ── Full simulation bundle ─────────────────────────────────────────────


def run_spectroscopy_simulation(technique: str, params) -> SimulationResult:
    """Run a spectroscopy simulation based on technique type."""
    data = {}
    warnings = []

    if technique == "Spontaneous Raman":
        data = spontaneous_raman(params)
    elif technique == "Stimulated Raman (SRS)":
        data = stimulated_raman(params)
    elif technique == "Spontaneous Brillouin":
        data = spontaneous_brillouin(params)
    elif technique == "Stimulated Brillouin (SBS)":
        data = stimulated_brillouin(params)
    elif technique == "Deep-UV Resonance Raman":
        data = duvrr_spectrum(params)
    elif technique == "LIBS":
        data = libs_spectrum(params)
    elif technique == "FTIR":
        data = ftir_spectrum(params)
    elif technique == "Hyperspectral Imaging":
        data = hyperspectral_image(params)
    else:
        warnings.append(f"Unknown technique: {technique}")

    return SimulationResult(
        name=f"Spectroscopy — {technique}",
        data=data,
        warnings=warnings,
    )
