"""Domain models for Harrington Labs simulators.

Canonical data structures for laser sources, optical components,
beam parameters, and lab-specific configurations. No Streamlit imports.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Physical constants ───────────────────────────────────────────

C_M_S = 299_792_458.0          # speed of light [m/s]
H_J_S = 6.62607015e-34         # Planck constant [J·s]
K_B = 1.380649e-23             # Boltzmann constant [J/K]
EPSILON_0 = 8.854187817e-12    # vacuum permittivity [F/m]
SIGMA_SB = 5.670374419e-8      # Stefan-Boltzmann [W/m²/K⁴]


# ── Enums ────────────────────────────────────────────────────────

class LaserType(str, Enum):
    CW = "CW"
    PULSED = "Pulsed"
    QUASI_CW = "Quasi-CW"
    MODE_LOCKED = "Mode-Locked"


class BeamProfile(str, Enum):
    GAUSSIAN = "Gaussian (TEM00)"
    TOP_HAT = "Top-Hat"
    MULTIMODE = "Multimode"
    ANNULAR = "Annular"
    BESSEL = "Bessel"


class PolarizationState(str, Enum):
    LINEAR_H = "Linear (H)"
    LINEAR_V = "Linear (V)"
    CIRCULAR_R = "Circular (RHC)"
    CIRCULAR_L = "Circular (LHC)"
    UNPOLARIZED = "Unpolarized"
    ELLIPTICAL = "Elliptical"


class AtmosphericCondition(str, Enum):
    CLEAR = "Clear"
    HAZE = "Haze"
    FOG = "Fog"
    RAIN_LIGHT = "Light Rain"
    RAIN_HEAVY = "Heavy Rain"
    TURBULENCE_WEAK = "Weak Turbulence"
    TURBULENCE_MODERATE = "Moderate Turbulence"
    TURBULENCE_STRONG = "Strong Turbulence"


class CoatingType(str, Enum):
    AR = "Anti-Reflection"
    HR = "High-Reflection"
    BBAR = "Broadband AR"
    DICHROIC = "Dichroic"
    METALLIC = "Metallic"
    DIELECTRIC_STACK = "Dielectric Stack"


class SubstrateType(str, Enum):
    BK7 = "BK7"
    FUSED_SILICA = "Fused Silica"
    SAPPHIRE = "Sapphire"
    ZNS = "ZnS"
    ZNSE = "ZnSe"
    CAF2 = "CaF2"
    SILICON = "Silicon"
    GERMANIUM = "Germanium"


class FiberType(str, Enum):
    SMF = "Single-Mode"
    LMA = "Large Mode Area"
    PCF = "Photonic Crystal"
    DCF = "Double-Clad"
    PM = "Polarization-Maintaining"
    HOLLOW_CORE = "Hollow-Core"


class QDMaterial(str, Enum):
    CDSE = "CdSe"
    CDSE_ZNS = "CdSe/ZnS"
    PBSE = "PbSe"
    PBS = "PbS"
    INP = "InP"
    PEROVSKITE = "Perovskite"
    SI = "Si"


class PulseShape(str, Enum):
    GAUSSIAN = "Gaussian"
    SECH2 = "sech²"
    LORENTZIAN = "Lorentzian"
    SQUARE = "Square"


# ── Core beam dataclass ──────────────────────────────────────────

@dataclass
class BeamParams:
    """Fundamental beam parameters at a reference plane."""
    wavelength_nm: float
    power_w: float
    beam_diameter_mm: float = 1.0
    m_squared: float = 1.0
    profile: BeamProfile = BeamProfile.GAUSSIAN
    polarization: PolarizationState = PolarizationState.LINEAR_H
    divergence_mrad: float = 0.0

    @property
    def wavelength_m(self) -> float:
        return self.wavelength_nm * 1e-9

    @property
    def beam_radius_m(self) -> float:
        return (self.beam_diameter_mm / 2.0) * 1e-3

    @property
    def area_cm2(self) -> float:
        r_cm = (self.beam_diameter_mm / 2.0) / 10.0
        return math.pi * r_cm ** 2

    @property
    def irradiance_w_cm2(self) -> float:
        a = self.area_cm2
        return self.power_w / a if a > 0 else 0.0

    @property
    def rayleigh_range_m(self) -> float:
        w0 = self.beam_radius_m
        return math.pi * w0**2 / (self.m_squared * self.wavelength_m) if self.wavelength_m > 0 else 0.0


# ── Pulsed source extension ─────────────────────────────────────

@dataclass
class PulsedSource:
    """Pulsed laser source parameters."""
    beam: BeamParams
    rep_rate_hz: float = 1e3
    pulse_width_s: float = 100e-15
    pulse_shape: PulseShape = PulseShape.GAUSSIAN
    laser_type: LaserType = LaserType.PULSED

    @property
    def pulse_energy_j(self) -> float:
        return self.beam.power_w / self.rep_rate_hz if self.rep_rate_hz > 0 else 0.0

    @property
    def peak_power_w(self) -> float:
        return self.pulse_energy_j / self.pulse_width_s if self.pulse_width_s > 0 else 0.0

    @property
    def fluence_j_cm2(self) -> float:
        a = self.beam.area_cm2
        return self.pulse_energy_j / a if a > 0 else 0.0

    @property
    def peak_irradiance_w_cm2(self) -> float:
        a = self.beam.area_cm2
        return self.peak_power_w / a if a > 0 else 0.0


# ── Direct diode specifics ──────────────────────────────────────

@dataclass
class DiodeLaserParams:
    """Direct diode laser parameters."""
    wavelength_nm: float = 976.0
    power_w: float = 50.0
    slope_efficiency: float = 0.65
    threshold_current_a: float = 0.5
    operating_current_a: float = 5.0
    beam_divergence_fast_deg: float = 35.0
    beam_divergence_slow_deg: float = 10.0
    emitter_width_um: float = 100.0
    emitter_count: int = 1
    fill_factor: float = 0.3
    thermal_resistance_k_w: float = 1.5
    t0_k: float = 150.0     # characteristic temperature
    t1_k: float = 400.0     # slope efficiency char. temp


# ── Fiber laser specifics ────────────────────────────────────────

@dataclass
class FiberLaserParams:
    """Fiber laser / amplifier parameters."""
    fiber_type: FiberType = FiberType.DCF
    core_diameter_um: float = 25.0
    cladding_diameter_um: float = 250.0
    na: float = 0.065
    fiber_length_m: float = 3.0
    pump_wavelength_nm: float = 976.0
    pump_power_w: float = 50.0
    signal_wavelength_nm: float = 1064.0
    signal_seed_power_w: float = 0.01
    doping_concentration_ppm: float = 1000.0
    dopant: str = "Yb"
    background_loss_db_m: float = 0.005
    output_power_w: float = 0.0  # computed


# ── Beam control / propagation ───────────────────────────────────

@dataclass
class PropagationPath:
    """Atmospheric / free-space propagation path."""
    distance_m: float = 1000.0
    condition: AtmosphericCondition = AtmosphericCondition.CLEAR
    cn2: float = 1e-15           # refractive index structure constant [m^{-2/3}]
    visibility_km: float = 23.0
    temperature_k: float = 293.0
    pressure_hpa: float = 1013.25
    relative_humidity: float = 0.5
    wind_speed_m_s: float = 5.0
    altitude_m: float = 0.0


@dataclass
class AdaptiveOpticsParams:
    """Adaptive optics system parameters."""
    actuator_count: int = 97
    actuator_pitch_mm: float = 5.0
    stroke_um: float = 3.5
    bandwidth_hz: float = 1000.0
    wfs_type: str = "Shack-Hartmann"
    subaperture_count: int = 77
    dm_type: str = "Continuous Facesheet"
    loop_gain: float = 0.3
    latency_ms: float = 1.0


# ── Quantum dot specifics ────────────────────────────────────────

@dataclass
class QuantumDotParams:
    """Quantum dot sample parameters."""
    material: QDMaterial = QDMaterial.CDSE_ZNS
    diameter_nm: float = 5.0
    size_distribution_pct: float = 5.0
    shell_thickness_nm: float = 2.0
    concentration_nmol_ml: float = 1.0
    solvent: str = "toluene"
    temperature_k: float = 293.0
    quantum_yield: float = 0.5
    peak_emission_nm: float = 620.0
    fwhm_emission_nm: float = 30.0
    absorption_cross_section_cm2: float = 1e-15
    exciton_lifetime_ns: float = 20.0


# ── Coating specifics ────────────────────────────────────────────

@dataclass
class ThinFilmLayer:
    """Single layer in a thin-film coating stack."""
    material: str = "SiO2"
    thickness_nm: float = 100.0
    refractive_index: float = 1.46
    extinction_coefficient: float = 0.0


@dataclass
class CoatingDesign:
    """Multi-layer coating stack specification."""
    name: str = "Quarter-Wave AR"
    coating_type: CoatingType = CoatingType.AR
    substrate: SubstrateType = SubstrateType.BK7
    substrate_n: float = 1.52
    design_wavelength_nm: float = 1064.0
    layers: list[ThinFilmLayer] = field(default_factory=list)
    angle_of_incidence_deg: float = 0.0
    notes: str = ""


# ── Result containers ────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Generic container for simulation outputs."""
    name: str
    parameters: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: str = ""
