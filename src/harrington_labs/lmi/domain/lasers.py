"""Laser source definitions and database.

Provides default commercial laser configurations and a builder
for custom sources (gain media, seed light, etc.).  Supports
arbitrary spatial beam modes and OPA parent-child relationships.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import math
from pathlib import Path

LASER_DB_PATH = Path("data/manual/lasers.json")


# ── Spatial beam modes ────────────────────────────────────────────

class SpatialMode(str, Enum):
    """Supported transverse spatial beam profiles."""

    TEM00 = "TEM00"                   # Fundamental Gaussian
    TOP_HAT = "Top-Hat"               # Flat-top / super-Gaussian
    HERMITE_GAUSSIAN = "Hermite-Gaussian"  # HG_mn higher-order
    LAGUERRE_GAUSSIAN = "Laguerre-Gaussian"  # LG_pl (includes vortex)
    BESSEL = "Bessel"                 # Non-diffracting Bessel beam

    @property
    def default_m_squared(self) -> float:
        """Characteristic M² for typical use of each mode family."""
        return {
            self.TEM00: 1.0,
            self.TOP_HAT: 1.0,  # varies with fill factor
            self.HERMITE_GAUSSIAN: 2.0,
            self.LAGUERRE_GAUSSIAN: 2.0,
            self.BESSEL: 1.0,  # core ring; effective M² depends on aperture
        }[self]


@dataclass
class BeamModeParams:
    """Extra parameters that qualify the chosen spatial mode.

    Only the fields relevant to the selected mode need to be non-zero.
    """

    hg_m: int = 0           # Hermite-Gaussian x-order
    hg_n: int = 0           # Hermite-Gaussian y-order
    lg_p: int = 0           # Laguerre-Gaussian radial index
    lg_l: int = 0           # Laguerre-Gaussian azimuthal index (OAM)
    bessel_order: int = 0   # Bessel beam order (J_n)
    bessel_half_angle_mrad: float = 0.0  # Axicon half-angle
    tophat_order: int = 10  # Super-Gaussian order (higher = flatter)


@dataclass
class LaserSource:
    """A laser source with full beam parameters."""

    name: str
    wavelength_nm: float
    power_w: float
    rep_rate_hz: float = 0.0          # 0 = CW
    pulse_width_s: float = 0.0        # 0 = CW
    beam_diameter_mm: float = 1.0
    m_squared: float = 1.0
    polarization: str = "linear"       # linear, circular, unpolarized
    spatial_mode: str = "TEM00"
    beam_mode_params: dict = field(default_factory=dict)
    gain_medium: str = ""
    pump_source: str = ""              # parent laser name (e.g. for OPA)
    tunable_range_nm: tuple[float, float] | None = None
    notes: str = ""

    @property
    def is_cw(self) -> bool:
        return self.rep_rate_hz == 0 or self.pulse_width_s == 0

    @property
    def peak_power_w(self) -> float:
        if self.is_cw:
            return self.power_w
        return self.power_w / (self.rep_rate_hz * self.pulse_width_s)

    @property
    def pulse_energy_j(self) -> float:
        if self.is_cw:
            return 0.0
        return self.power_w / self.rep_rate_hz

    @property
    def fluence_j_cm2(self) -> float:
        """Fluence at beam center (assuming Gaussian, 1/e² radius)."""
        radius_cm = (self.beam_diameter_mm / 2) / 10
        area_cm2 = math.pi * radius_cm ** 2
        energy = self.pulse_energy_j if not self.is_cw else self.power_w
        return energy / area_cm2 if area_cm2 > 0 else 0.0

    @property
    def irradiance_w_cm2(self) -> float:
        """Peak irradiance (W/cm²)."""
        radius_cm = (self.beam_diameter_mm / 2) / 10
        area_cm2 = math.pi * radius_cm ** 2
        return self.peak_power_w / area_cm2 if area_cm2 > 0 else 0.0

    @property
    def wavelength_um(self) -> float:
        """Wavelength in micrometers."""
        return self.wavelength_nm / 1000.0

    @property
    def wavelength_m(self) -> float:
        """Wavelength in meters."""
        return self.wavelength_nm * 1e-9

    @property
    def photon_energy_ev(self) -> float:
        """Single-photon energy in eV."""
        return 1240.0 / self.wavelength_nm if self.wavelength_nm > 0 else 0.0

    @property
    def is_tunable(self) -> bool:
        return self.tunable_range_nm is not None

    def get_mode_params(self) -> BeamModeParams:
        """Deserialise beam_mode_params dict into BeamModeParams."""
        return BeamModeParams(**self.beam_mode_params) if self.beam_mode_params else BeamModeParams()


# ── Default commercial lasers ──────────────────────────────────────

DEFAULT_LASERS: list[LaserSource] = [
    # ── Ultrafast ──
    LaserSource(
        name="Light Conversion Pharos 2mJ",
        wavelength_nm=1030.0,
        power_w=20.0,             # 2 mJ @ 10 kHz = 20 W avg
        rep_rate_hz=10_000.0,
        pulse_width_s=190e-15,    # < 190 fs (compressible to < 100 fs)
        beam_diameter_mm=5.0,
        m_squared=1.2,
        gain_medium="Yb:KGW",
        spatial_mode="TEM00",
        tunable_range_nm=(257, 2600),  # with harmonics
        notes=(
            "Light Conversion Pharos industrial fs laser. "
            "2 mJ / pulse, < 190 fs, tunable rep rate single-shot to 1 MHz. "
            "Max avg power 20 W. Harmonics: 515, 343, 257 nm. "
            "Typical pump source for Orpheus OPA. Supports burst mode (up to 10 pulses per burst) and bi-burst mode (burst of bursts) for enhanced ablation and material processing."
        ),
    ),
    LaserSource(
        name="Light Conversion Orpheus OPA",
        wavelength_nm=8500.0,       # current lab default
        power_w=0.2,              # ~20 uJ @ 10 kHz = 200 mW
        rep_rate_hz=10_000.0,
        pulse_width_s=170e-15,    # ~ pump pulse width
        beam_diameter_mm=5.0,
        m_squared=1.5,
        gain_medium="OPA (BBO/LGS)",
        pump_source="Light Conversion Pharos 2mJ",
        spatial_mode="TEM00",
        tunable_range_nm=(630, 16000),  # signal + idler combined
        notes=(
            "Light Conversion Orpheus optical parametric amplifier. "
            "Pumped by Pharos. Tunable 630 nm - 16 um (signal + idler). "
            "Pulse energy depends on wavelength: ~20 uJ typ at 8.5 um. "
            "MIR extension via DFG for > 4 um."
        ),
    ),
    LaserSource(
        name="Ti:Sapphire 800nm Ultrafast",
        wavelength_nm=800.0,
        power_w=1.0,
        rep_rate_hz=1000.0,
        pulse_width_s=100e-15,
        beam_diameter_mm=5.0,
        m_squared=1.2,
        gain_medium="Ti:Sapphire",
        notes="1 kHz, 100 fs, 1 mJ/pulse",
    ),
    # ── CW Near-IR ──
    LaserSource(
        name="Nd:YAG 1064nm CW",
        wavelength_nm=1064.0,
        power_w=10.0,
        beam_diameter_mm=2.0,
        m_squared=1.1,
        gain_medium="Nd:YAG",
        notes="Standard CW near-IR laser",
    ),
    LaserSource(
        name="Nd:YAG 1064nm Q-switched",
        wavelength_nm=1064.0,
        power_w=5.0,
        rep_rate_hz=10.0,
        pulse_width_s=10e-9,
        beam_diameter_mm=3.0,
        m_squared=1.5,
        gain_medium="Nd:YAG",
        notes="10 Hz, 10 ns pulses, 500 mJ/pulse",
    ),
    # ── Mid-IR / Far-IR ──
    LaserSource(
        name="CO2 10.6um CW",
        wavelength_nm=10600.0,
        power_w=100.0,
        beam_diameter_mm=10.0,
        m_squared=1.3,
        gain_medium="CO2",
        notes="Industrial CW CO2 laser",
    ),
    # ── Telecom / Fiber ──
    LaserSource(
        name="Er:Fiber 1550nm CW",
        wavelength_nm=1550.0,
        power_w=0.5,
        beam_diameter_mm=0.01,  # fiber output
        m_squared=1.05,
        gain_medium="Er:Glass fiber",
        notes="Telecom-grade single-mode fiber laser",
    ),
    LaserSource(
        name="Yb:Fiber 1070nm High-Power",
        wavelength_nm=1070.0,
        power_w=1000.0,
        beam_diameter_mm=14.0,
        m_squared=1.1,
        gain_medium="Yb:Glass fiber",
        notes="kW-class industrial fiber laser",
    ),
    # ── UV ──
    LaserSource(
        name="Excimer KrF 248nm",
        wavelength_nm=248.0,
        power_w=20.0,
        rep_rate_hz=100.0,
        pulse_width_s=25e-9,
        beam_diameter_mm=20.0,
        m_squared=50.0,
        gain_medium="KrF",
        spatial_mode="Top-Hat",  # excimers are naturally flat-top
        notes="UV excimer, 100 Hz, 25 ns, highly multimode",
    ),
    # ── Alignment / Visible ──
    LaserSource(
        name="HeNe 632.8nm",
        wavelength_nm=632.8,
        power_w=0.005,
        beam_diameter_mm=0.8,
        m_squared=1.0,
        gain_medium="HeNe",
        notes="Alignment laser, 5 mW",
    ),
]


def load_custom_lasers() -> list[LaserSource]:
    """Load user-defined lasers from JSON."""
    if not LASER_DB_PATH.exists():
        return []
    data = json.loads(LASER_DB_PATH.read_text())
    return [LaserSource(**d) for d in data.get("custom_lasers", [])]


def save_custom_lasers(lasers: list[LaserSource]):
    LASER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if LASER_DB_PATH.exists():
        existing = json.loads(LASER_DB_PATH.read_text())
    existing["custom_lasers"] = [asdict(l) for l in lasers]
    LASER_DB_PATH.write_text(json.dumps(existing, indent=2))


def all_lasers() -> list[LaserSource]:
    """Return defaults + custom lasers."""
    return DEFAULT_LASERS + load_custom_lasers()


def get_laser_by_name(name: str) -> LaserSource | None:
    """Look up a laser by name from defaults + custom."""
    for laser in all_lasers():
        if laser.name == name:
            return laser
    return None
