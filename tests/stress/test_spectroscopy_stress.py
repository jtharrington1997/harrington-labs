from __future__ import annotations

import numpy as np

from harrington_labs.domain.spectroscopy import (
    BrillouinParams,
    DUVRRParams,
    FTIRParams,
    HyperspectralParams,
    LIBSParams,
    RamanParams,
    SamplePhase,
)
from harrington_labs.simulation.spectroscopy import (
    duvrr_spectrum,
    ftir_spectrum,
    hyperspectral_image,
    libs_spectrum,
    spontaneous_brillouin,
    spontaneous_raman,
    stimulated_brillouin,
    stimulated_raman,
)


def _finite_array(values) -> np.ndarray:
    arr = np.asarray(values)
    assert arr.size > 0
    assert np.all(np.isfinite(arr))
    return arr


def _strictly_increasing(values) -> np.ndarray:
    arr = _finite_array(values)
    assert np.all(np.diff(arr) > 0)
    return arr


def test_spontaneous_raman_stress_cases_remain_finite() -> None:
    cases = [
        RamanParams(
            excitation_wavelength_nm=532.0,
            laser_power_mw=50.0,
            integration_time_s=1.0,
            sample_phase=SamplePhase.SOLID,
            raman_shifts_cm_inv=[520.0],
            raman_widths_cm_inv=[8.0],
            raman_intensities=[1.0],
        ),
        RamanParams(
            excitation_wavelength_nm=785.0,
            laser_power_mw=120.0,
            integration_time_s=5.0,
            sample_phase=SamplePhase.LIQUID,
            temperature_k=320.0,
            raman_shifts_cm_inv=[1640.0, 3250.0, 3450.0],
            raman_widths_cm_inv=[80.0, 180.0, 180.0],
            raman_intensities=[0.4, 0.8, 1.0],
        ),
        RamanParams(
            excitation_wavelength_nm=244.0,
            laser_power_mw=10.0,
            integration_time_s=10.0,
            sample_phase=SamplePhase.THIN_FILM,
            temperature_k=500.0,
            raman_shifts_cm_inv=[156.0, 282.0, 712.0, 1086.0, 1436.0],
            raman_widths_cm_inv=[10.0, 8.0, 12.0, 5.0, 15.0],
            raman_intensities=[0.3, 0.5, 0.4, 1.0, 0.15],
        ),
    ]

    for params in cases:
        data = spontaneous_raman(params, n_points=2048)
        shifts = _strictly_increasing(data["shift_cm_inv"])
        spectrum = _finite_array(data["spectrum"])
        clean = _finite_array(data["clean_spectrum"])
        background = _finite_array(data["background"])
        assert shifts.shape == spectrum.shape == clean.shape == background.shape
        assert np.any(clean > 0.0)
        ratio = data["stokes_anti_stokes_ratio"]
        if ratio is not None:
            assert np.isfinite(ratio)
            assert ratio > 0.0


def test_stimulated_raman_stress_cases_remain_bounded() -> None:
    cases = [
        (
            RamanParams(
                excitation_wavelength_nm=532.0,
                pump_power_mw=100.0,
                stokes_seed_power_mw=0.01,
                raman_shifts_cm_inv=[440.0, 490.0, 800.0, 1060.0],
                raman_widths_cm_inv=[40.0, 30.0, 50.0, 60.0],
                raman_intensities=[0.7, 0.3, 0.5, 0.4],
            ),
            1.0,
        ),
        (
            RamanParams(
                excitation_wavelength_nm=1064.0,
                pump_power_mw=1500.0,
                stokes_seed_power_mw=0.1,
                raman_shifts_cm_inv=[440.0, 1332.0],
                raman_widths_cm_inv=[40.0, 4.0],
                raman_intensities=[0.8, 1.0],
            ),
            10.0,
        ),
        (
            RamanParams(
                excitation_wavelength_nm=633.0,
                pump_power_mw=500.0,
                stokes_seed_power_mw=0.05,
                raman_shifts_cm_inv=[621.0, 1001.0, 1602.0, 2904.0],
                raman_widths_cm_inv=[12.0, 6.0, 6.0, 15.0],
                raman_intensities=[0.3, 1.0, 0.8, 0.4],
            ),
            25.0,
        ),
    ]

    for params, fiber_length_m in cases:
        data = stimulated_raman(params, fiber_length_m=fiber_length_m, n_points=1024)
        shifts = _strictly_increasing(data["shift_cm_inv"])
        gain_coeff = _finite_array(data["gain_coefficient"])
        stokes_gain = _finite_array(data["stokes_gain"])
        stokes_power = _finite_array(data["stokes_power_mw"])
        assert shifts.shape == gain_coeff.shape == stokes_gain.shape == stokes_power.shape
        assert np.all(gain_coeff >= 0.0)
        assert np.all(stokes_gain >= 1.0)
        assert np.all(stokes_power >= 0.0)
        assert np.isfinite(data["peak_gain_m_per_w"])
        assert data["peak_gain_m_per_w"] >= 0.0
        assert np.isfinite(data["pump_remaining_fraction"])
        assert 0.0 <= data["pump_remaining_fraction"] <= 1.0


def test_brillouin_and_sbs_stress_cases_remain_finite() -> None:
    spont_cases = [
        BrillouinParams(
            excitation_wavelength_nm=532.0,
            scattering_angle_deg=180.0,
            sound_velocity_m_s=5960.0,
            refractive_index=1.46,
            density_kg_m3=2200.0,
        ),
        BrillouinParams(
            excitation_wavelength_nm=780.0,
            scattering_angle_deg=90.0,
            sound_velocity_m_s=3200.0,
            refractive_index=1.33,
            density_kg_m3=1000.0,
            acoustic_attenuation_db_cm_ghz2=0.2,
        ),
        BrillouinParams(
            excitation_wavelength_nm=405.0,
            scattering_angle_deg=150.0,
            sound_velocity_m_s=11000.0,
            refractive_index=2.1,
            density_kg_m3=3980.0,
            acoustic_attenuation_db_cm_ghz2=1.5,
        ),
    ]

    for params in spont_cases:
        data = spontaneous_brillouin(params, n_points=1024)
        freq = _strictly_increasing(data["frequency_ghz"])
        stokes = _finite_array(data["stokes"])
        anti_stokes = _finite_array(data["anti_stokes"])
        rayleigh = _finite_array(data["rayleigh"])
        total = _finite_array(data["spectrum"])
        assert freq.shape == stokes.shape == anti_stokes.shape == rayleigh.shape == total.shape
        assert data["brillouin_shift_ghz"] > 0.0
        assert data["linewidth_ghz"] > 0.0
        assert data["longitudinal_modulus_gpa"] > 0.0

    sbs_cases = [
        BrillouinParams(
            excitation_wavelength_nm=1064.0,
            laser_power_mw=500.0,
            interaction_length_m=10.0,
            fiber_core_diameter_um=8.0,
        ),
        BrillouinParams(
            excitation_wavelength_nm=1550.0,
            laser_power_mw=2000.0,
            interaction_length_m=100.0,
            fiber_core_diameter_um=10.0,
            refractive_index=1.45,
            density_kg_m3=2200.0,
        ),
    ]

    for params in sbs_cases:
        data = stimulated_brillouin(params, n_points=1024)
        input_power = _strictly_increasing(data["input_power_mw"])
        reflectivity = _finite_array(data["sbs_reflectivity"])
        assert input_power.shape == reflectivity.shape
        assert np.all(reflectivity >= 0.0)
        assert np.all(reflectivity <= 1.0)
        assert data["gain_coefficient_m_per_w"] > 0.0
        assert data["threshold_power_w"] > 0.0
        assert data["threshold_power_mw"] > 0.0
        assert data["effective_area_um2"] > 0.0


def test_duvrr_libs_ftir_and_hyperspectral_stress_cases_remain_finite() -> None:
    duvrr_cases = [
        DUVRRParams(excitation_wavelength_nm=244.0, electronic_transition_nm=260.0, integration_time_s=60.0),
        DUVRRParams(excitation_wavelength_nm=229.0, electronic_transition_nm=235.0, integration_time_s=120.0, resonance_enhancement_factor=5e4),
        DUVRRParams(excitation_wavelength_nm=266.0, electronic_transition_nm=255.0, concentration_mg_ml=25.0, laser_power_uw=1000.0),
    ]
    for params in duvrr_cases:
        data = duvrr_spectrum(params, n_points=2048)
        shifts = _strictly_increasing(data["shift_cm_inv"])
        spectrum = _finite_array(data["spectrum"])
        clean = _finite_array(data["clean_spectrum"])
        exc_nm = _strictly_increasing(data["excitation_profile_nm"])
        exc_profile = _finite_array(data["excitation_profile"])
        assert shifts.shape == spectrum.shape == clean.shape
        assert exc_nm.shape == exc_profile.shape
        assert np.any(clean > 0.0)
        assert np.all(exc_profile >= 0.0)
        assert np.max(exc_profile) <= 1.0 + 1e-12
        assert data["resonance_enhancement"] > 0.0
        assert len(data["mode_assignments"]) >= 3

    libs_cases = [
        LIBSParams(composition={"Fe": 0.70, "Cr": 0.18, "Ni": 0.08, "Mn": 0.02, "Si": 0.01, "C": 0.01}),
        LIBSParams(pulse_energy_mj=150.0, pulse_width_ns=5.0, spot_diameter_um=50.0, composition={"Al": 0.97, "Mg": 0.01, "Si": 0.006, "Cu": 0.003}),
        LIBSParams(pulse_energy_mj=25.0, pulse_width_ns=12.0, spot_diameter_um=200.0, composition={"Si": 0.30, "Al": 0.08, "Fe": 0.05, "Ca": 0.04, "Mg": 0.02, "Na": 0.01}),
    ]
    for params in libs_cases:
        data = libs_spectrum(params, n_points=4096)
        wl = _strictly_increasing(data["wavelength_nm"])
        spectrum = _finite_array(data["spectrum"])
        clean = _finite_array(data["clean_spectrum"])
        continuum = _finite_array(data["continuum"])
        assert wl.shape == spectrum.shape == clean.shape == continuum.shape
        assert np.any(clean > 0.0)
        assert np.all(continuum >= 0.0)
        assert data["plasma_temperature_k"] >= 8000.0
        assert data["irradiance_gw_cm2"] > 0.0
        assert len(data["line_data"]) > 0

    ftir_cases = [
        FTIRParams(),
        FTIRParams(n_scans=256, thickness_um=100.0, ir_modes=[(3300.0, 250.0, 0.6), (1650.0, 30.0, 1.0), (1540.0, 30.0, 0.8)]),
        FTIRParams(wavenumber_min_cm_inv=650.0, wavenumber_max_cm_inv=1800.0, resolution_cm_inv=1.0, n_scans=64, thickness_um=5.0, ir_modes=[(1260.0, 15.0, 1.0), (1090.0, 60.0, 0.9), (800.0, 20.0, 0.7)]),
    ]
    for params in ftir_cases:
        data = ftir_spectrum(params, n_points=4096)
        wn = _strictly_increasing(data["wavenumber_cm_inv"])
        absorbance = _finite_array(data["absorbance"])
        absorbance_clean = _finite_array(data["absorbance_clean"])
        transmittance = _finite_array(data["transmittance"])
        transmittance_clean = _finite_array(data["transmittance_clean"])
        baseline = _finite_array(data["baseline"])
        assert wn.shape == absorbance.shape == absorbance_clean.shape == transmittance.shape == transmittance_clean.shape == baseline.shape
        assert np.all(transmittance > 0.0)
        assert np.all(transmittance_clean > 0.0)
        assert np.all(transmittance_clean <= 1.0)
        assert data["snr"] > 0.0

    hyperspectral_cases = [
        HyperspectralParams(image_size_px=32, n_components=2, snr_db=20.0),
        HyperspectralParams(image_size_px=64, n_components=3, snr_db=30.0),
        HyperspectralParams(image_size_px=96, n_components=5, snr_db=40.0, pixel_dwell_time_ms=250.0),
    ]
    for params in hyperspectral_cases:
        data = hyperspectral_image(params)
        wn = _strictly_increasing(data["wavenumber_cm_inv"])
        datacube = _finite_array(data["datacube"])
        datacube_clean = _finite_array(data["datacube_clean"])
        intensity_image = _finite_array(data["intensity_image"])
        cube = np.asarray(data["datacube"])
        cube_clean = np.asarray(data["datacube_clean"])
        image = np.asarray(data["intensity_image"])
        assert cube.shape == cube_clean.shape
        assert cube.shape[0] == cube.shape[1] == params.image_size_px
        assert cube.shape[2] == wn.shape[0]
        assert image.shape == (params.image_size_px, params.image_size_px)
        assert intensity_image.size == params.image_size_px * params.image_size_px
        assert len(data["component_spectra"]) == params.n_components
        assert len(data["component_maps"]) == params.n_components
        assert len(data["component_names"]) == params.n_components
        for spec in data["component_spectra"]:
            spec_arr = _finite_array(spec)
            assert spec_arr.shape == wn.shape
        for comp_map in data["component_maps"]:
            map_arr = _finite_array(comp_map)
            assert map_arr.shape == (params.image_size_px, params.image_size_px)
