"""Domain models for the Advanced Spectroscopy Lab.

Dataclasses for configuring each spectroscopy technique simulator.
No Streamlit imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ── Enums ───────────────────────────────────────────────────────────────


class SpectroscopyTechnique(str, Enum):
    RAMAN_SPONTANEOUS = "Spontaneous Raman"
    RAMAN_STIMULATED = "Stimulated Raman (SRS)"
    BRILLOUIN_SPONTANEOUS = "Spontaneous Brillouin"
    BRILLOUIN_STIMULATED = "Stimulated Brillouin (SBS)"
    DUVRR = "Deep-UV Resonance Raman"
    LIBS = "LIBS"
    FTIR = "FTIR"
    HYPERSPECTRAL = "Hyperspectral Imaging"


class RamanExcitation(str, Enum):
    UV_244 = "244 nm"
    UV_257 = "257 nm"
    VIS_488 = "488 nm"
    VIS_514 = "514 nm"
    VIS_532 = "532 nm"
    VIS_633 = "633 nm"
    NIR_785 = "785 nm"
    NIR_830 = "830 nm"
    NIR_1064 = "1064 nm"


class SamplePhase(str, Enum):
    SOLID = "Solid"
    LIQUID = "Liquid"
    GAS = "Gas"
    THIN_FILM = "Thin Film"
    POWDER = "Powder"
    BIOLOGICAL = "Biological Tissue"


# ── Raman / Brillouin ──────────────────────────────────────────────────


@dataclass
class RamanParams:
    """Parameters for Raman spectroscopy simulation."""
    excitation_wavelength_nm: float = 532.0
    laser_power_mw: float = 50.0
    integration_time_s: float = 1.0
    numerical_aperture: float = 0.75
    spectral_resolution_cm_inv: float = 4.0
    sample_phase: SamplePhase = SamplePhase.SOLID
    temperature_k: float = 293.0
    # Material Raman properties
    raman_cross_section_cm2_sr: float = 1e-29  # differential cross-section
    raman_shifts_cm_inv: list[float] = field(default_factory=lambda: [520.0])  # peak positions
    raman_widths_cm_inv: list[float] = field(default_factory=lambda: [8.0])    # peak FWHM
    raman_intensities: list[float] = field(default_factory=lambda: [1.0])       # relative
    concentration_mol_l: float = 1.0
    path_length_mm: float = 1.0
    # Stimulated Raman
    pump_power_mw: float = 100.0
    stokes_seed_power_mw: float = 0.01


@dataclass
class BrillouinParams:
    """Parameters for Brillouin spectroscopy simulation."""
    excitation_wavelength_nm: float = 532.0
    laser_power_mw: float = 100.0
    scattering_angle_deg: float = 180.0  # backscattering default
    temperature_k: float = 293.0
    sample_phase: SamplePhase = SamplePhase.SOLID
    # Material acoustic properties
    sound_velocity_m_s: float = 5960.0      # longitudinal, e.g. fused silica
    refractive_index: float = 1.46
    acoustic_attenuation_db_cm_ghz2: float = 0.5
    elasto_optic_coefficient: float = 0.27  # p12
    density_kg_m3: float = 2200.0
    # SBS
    interaction_length_m: float = 1.0
    fiber_core_diameter_um: float = 8.0


@dataclass
class DUVRRParams:
    """Parameters for Deep-UV Resonance Raman simulation."""
    excitation_wavelength_nm: float = 244.0
    laser_power_uw: float = 500.0  # µW typical for DUVRR
    integration_time_s: float = 60.0
    spectral_resolution_cm_inv: float = 8.0
    sample_phase: SamplePhase = SamplePhase.BIOLOGICAL
    temperature_k: float = 293.0
    # Resonance enhancement
    electronic_transition_nm: float = 260.0  # e.g. protein amide π→π*
    resonance_enhancement_factor: float = 1e4
    # Amide band modes (protein secondary structure)
    amide_I_cm_inv: float = 1655.0    # C=O stretch
    amide_II_cm_inv: float = 1555.0   # N-H bend + C-N stretch
    amide_III_cm_inv: float = 1300.0  # C-N stretch + N-H bend
    concentration_mg_ml: float = 10.0


# ── LIBS ────────────────────────────────────────────────────────────────


@dataclass
class LIBSParams:
    """Parameters for LIBS simulation."""
    wavelength_nm: float = 1064.0
    pulse_energy_mj: float = 50.0
    pulse_width_ns: float = 8.0
    spot_diameter_um: float = 100.0
    rep_rate_hz: float = 10.0
    gate_delay_us: float = 1.0
    gate_width_us: float = 10.0
    temperature_k: float = 293.0
    ambient_pressure_atm: float = 1.0
    ambient_gas: str = "Air"
    # Sample composition (element -> weight fraction)
    composition: dict[str, float] = field(default_factory=lambda: {
        "Fe": 0.70, "Cr": 0.18, "Ni": 0.08, "Mn": 0.02, "Si": 0.01, "C": 0.01,
    })
    sample_phase: SamplePhase = SamplePhase.SOLID


# ── FTIR ────────────────────────────────────────────────────────────────


@dataclass
class FTIRParams:
    """Parameters for FTIR spectroscopy simulation."""
    wavenumber_min_cm_inv: float = 400.0
    wavenumber_max_cm_inv: float = 4000.0
    resolution_cm_inv: float = 4.0
    n_scans: int = 32
    sample_phase: SamplePhase = SamplePhase.SOLID
    thickness_um: float = 10.0
    temperature_k: float = 293.0
    # Vibrational modes: (position_cm_inv, width_cm_inv, peak_absorbance)
    ir_modes: list[tuple[float, float, float]] = field(default_factory=lambda: [
        (3400.0, 200.0, 0.8),   # O-H stretch
        (2920.0, 30.0, 0.6),    # C-H asymmetric
        (2850.0, 25.0, 0.4),    # C-H symmetric
        (1740.0, 20.0, 0.9),    # C=O stretch
        (1460.0, 15.0, 0.3),    # C-H bend
        (1050.0, 40.0, 0.7),    # C-O stretch
    ])


# ── Hyperspectral Imaging ──────────────────────────────────────────────


@dataclass
class HyperspectralParams:
    """Parameters for hyperspectral / Raman imaging simulation."""
    technique: str = "Raman"  # "Raman", "FTIR", "fluorescence"
    excitation_wavelength_nm: float = 532.0
    laser_power_mw: float = 10.0
    pixel_dwell_time_ms: float = 100.0
    image_size_px: int = 64
    pixel_size_um: float = 1.0
    spectral_range_cm_inv: tuple[float, float] = (200.0, 3200.0)
    spectral_resolution_cm_inv: float = 8.0
    # Spatial distribution: list of (component_name, {peak_cm_inv: intensity}, spatial_pattern)
    n_components: int = 3
    snr_db: float = 30.0
